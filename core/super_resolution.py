from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


Resolution = Tuple[int, int]


@dataclass
class SuperResolutionConfig:
    backend: str = "auto"
    model_path: Optional[str] = None
    scale_factor: int = 2
    device: str = "cuda"
    half_precision: bool = True
    tile_size: int = 0
    tile_overlap: int = 16
    quality: str = "balanced"
    warmup_size: Resolution = (128, 128)
    model_dim: int = 48
    model_depth: int = 4
    model_heads: int = 4
    model_window_size: int = 8
    residual_scale: float = 0.1


@dataclass
class SuperResolutionStats:
    backend: str = "unknown"
    device: str = "cpu"
    scale_factor: int = 1
    latency_ms: float = 0.0
    input_resolution: Resolution = (0, 0)
    output_resolution: Resolution = (0, 0)
    gpu_memory_mb: float = 0.0


class SuperResolutionEngine:
    """Super-resolution runtime used by the app and the processing pipeline."""

    INTERPOLATION_BACKENDS = {"nearest", "bilinear", "bicubic"}

    def __init__(self, config: Optional[SuperResolutionConfig] = None):
        self.config = config or SuperResolutionConfig()
        self.device = self._resolve_device(self.config.device)
        self.half_precision = self.config.half_precision and self.device.type == "cuda"
        self.backend = self._resolve_backend(self.config.backend, self.config.model_path)
        self.model = None
        self.last_stats = SuperResolutionStats(
            backend=self.backend,
            device=str(self.device),
            scale_factor=self.config.scale_factor,
        )
        self._load_backend()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device)

    @staticmethod
    def _resolve_backend(backend: str, model_path: Optional[str]) -> str:
        if backend == "auto":
            return "sr_transformer" if model_path else "bicubic"
        return backend

    def _load_backend(self) -> None:
        if self.backend in self.INTERPOLATION_BACKENDS:
            return
        if self.backend != "sr_transformer":
            raise ValueError(f"Unsupported super-resolution backend: {self.backend}")

        from models.transformer.sr_transformer import SRTransformer

        self.model = SRTransformer(
            dim=self.config.model_dim,
            depth=self.config.model_depth,
            num_heads=self.config.model_heads,
            scale_factor=self.config.scale_factor,
            window_size=self.config.model_window_size,
            residual_scale=self.config.residual_scale,
        ).to(self.device)
        if self.config.model_path:
            checkpoint_path = Path(self.config.model_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"SR model not found: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
            self.model.load_state_dict(state_dict, strict=False)

        if self.half_precision:
            self.model = self.model.half()
        self.model.eval()

    def warmup(self) -> None:
        if self.model is None:
            return
        width, height = self.config.warmup_size
        tensor = torch.zeros((1, 3, height, width), device=self.device)
        if self.half_precision:
            tensor = tensor.half()
        with torch.inference_mode():
            for _ in range(3):
                _ = self.model(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def upscale_frame(
        self,
        frame: np.ndarray,
        target_resolution: Optional[Resolution] = None,
    ) -> np.ndarray:
        result, _ = self.upscale_with_stats(frame, target_resolution)
        return result

    def upscale_with_stats(
        self,
        frame: np.ndarray,
        target_resolution: Optional[Resolution] = None,
    ) -> tuple[np.ndarray, SuperResolutionStats]:
        start = time.perf_counter()
        input_height, input_width = frame.shape[:2]
        if target_resolution is None:
            target_resolution = (
                input_width * self.config.scale_factor,
                input_height * self.config.scale_factor,
            )

        if self.config.tile_size > 0 and self.backend not in self.INTERPOLATION_BACKENDS:
            result = self._upscale_tiled(frame)
        else:
            tensor = self._preprocess(frame)
            output = self._infer_tensor(tensor)
            result = self._postprocess(output)

        if result.shape[1] != target_resolution[0] or result.shape[0] != target_resolution[1]:
            result = self._align_numpy(result, target_resolution)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0
        gpu_memory_mb = (
            torch.cuda.memory_allocated(self.device) / 1024**2
            if self.device.type == "cuda"
            else 0.0
        )
        self.last_stats = SuperResolutionStats(
            backend=self.backend,
            device=str(self.device),
            scale_factor=self.config.scale_factor,
            latency_ms=latency_ms,
            input_resolution=(input_width, input_height),
            output_resolution=(result.shape[1], result.shape[0]),
            gpu_memory_mb=gpu_memory_mb,
        )
        return result, self.last_stats

    def benchmark(self, frame: np.ndarray, runs: int = 10) -> SuperResolutionStats:
        runs = max(1, runs)
        self.warmup()
        total = 0.0
        last_stats = self.last_stats
        for _ in range(runs):
            _, last_stats = self.upscale_with_stats(frame)
            total += last_stats.latency_ms
        last_stats.latency_ms = total / runs
        self.last_stats = last_stats
        return last_stats

    def release(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise ValueError("frame must be an HWC RGB/RGBA array")
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        frame = np.ascontiguousarray(frame.astype(np.float32) / 255.0)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor.half() if self.half_precision else tensor

    def _postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        frame = tensor.squeeze(0).permute(1, 2, 0)
        frame = (frame.float() * 255.0).clamp(0, 255).byte().cpu().numpy()
        return np.ascontiguousarray(frame)

    def _infer_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            if self.backend in self.INTERPOLATION_BACKENDS:
                mode = self.backend
                kwargs = {"scale_factor": self.config.scale_factor, "mode": mode}
                if mode in {"bilinear", "bicubic"}:
                    kwargs["align_corners"] = False
                return F.interpolate(tensor, **kwargs).clamp(0.0, 1.0)
            return self.model(tensor)

    def _upscale_tiled(self, frame: np.ndarray) -> np.ndarray:
        scale = self.config.scale_factor
        tile_size = max(16, self.config.tile_size)
        overlap = max(0, min(self.config.tile_overlap, tile_size // 2))
        height, width = frame.shape[:2]
        output = np.zeros((height * scale, width * scale, 3), dtype=np.uint8)

        for y0 in range(0, height, tile_size):
            for x0 in range(0, width, tile_size):
                y1 = min(height, y0 + tile_size)
                x1 = min(width, x0 + tile_size)
                read_y0 = max(0, y0 - overlap)
                read_x0 = max(0, x0 - overlap)
                read_y1 = min(height, y1 + overlap)
                read_x1 = min(width, x1 + overlap)
                tile = frame[read_y0:read_y1, read_x0:read_x1]

                tile_tensor = self._preprocess(tile)
                tile_result = self._postprocess(self._infer_tensor(tile_tensor))

                crop_y0 = (y0 - read_y0) * scale
                crop_x0 = (x0 - read_x0) * scale
                crop_y1 = crop_y0 + (y1 - y0) * scale
                crop_x1 = crop_x0 + (x1 - x0) * scale
                output[y0 * scale : y1 * scale, x0 * scale : x1 * scale] = tile_result[
                    crop_y0:crop_y1,
                    crop_x0:crop_x1,
                ]

        return output

    def _resize_numpy(self, frame: np.ndarray, target_resolution: Resolution) -> np.ndarray:
        tensor = self._preprocess(frame)
        with torch.inference_mode():
            resized = F.interpolate(
                tensor,
                size=(target_resolution[1], target_resolution[0]),
                mode="bilinear",
                align_corners=False,
            )
        return self._postprocess(resized)

    def _align_numpy(self, frame: np.ndarray, target_resolution: Resolution) -> np.ndarray:
        target_width, target_height = target_resolution
        height, width = frame.shape[:2]
        if width >= target_width and height >= target_height:
            left = (width - target_width) // 2
            top = (height - target_height) // 2
            return np.ascontiguousarray(
                frame[top : top + target_height, left : left + target_width]
            )
        return self._resize_numpy(frame, target_resolution)


class SuperResolution(SuperResolutionEngine):
    """Backward-compatible wrapper for existing pipeline code."""

    def __init__(
        self,
        model_path: str = None,
        scale_factor: int = 2,
        device: str = "cuda",
        half_precision: bool = True,
        backend: str = "auto",
        tile_size: int = 0,
        tile_overlap: int = 16,
        model_dim: int = 48,
        model_depth: int = 4,
        model_heads: int = 4,
        model_window_size: int = 8,
    ):
        super().__init__(
            SuperResolutionConfig(
                backend=backend,
                model_path=model_path,
                scale_factor=scale_factor,
                device=device,
                half_precision=half_precision,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                model_dim=model_dim,
                model_depth=model_depth,
                model_heads=model_heads,
                model_window_size=model_window_size,
            )
        )
