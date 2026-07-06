import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.super_resolution import SuperResolutionConfig, SuperResolutionEngine
from models.transformer.sr_transformer import SRTransformer, build_sr_model
from training.data import SRFramePairDataset


class SuperResolutionTests(unittest.TestCase):
    def test_sr_transformer_respects_scale_and_odd_sizes(self):
        model = SRTransformer(dim=12, depth=1, num_heads=3, scale_factor=2, window_size=8)
        model.eval()
        x = torch.rand(1, 3, 15, 17)
        with torch.inference_mode():
            y = model(x)
        self.assertEqual(tuple(y.shape), (1, 3, 30, 34))

    def test_all_model_variants_respect_scale(self):
        x = torch.rand(1, 3, 16, 16)
        for variant in ["baseline", "hybrid", "shared_attention", "detail_aware"]:
            model = build_sr_model(
                variant=variant,
                dim=12,
                depth=1,
                num_heads=3,
                scale_factor=3,
                window_size=8,
            )
            model.eval()
            with torch.inference_mode():
                y = model(x)
            self.assertEqual(tuple(y.shape), (1, 3, 48, 48), variant)

    def test_new_model_starts_from_bicubic_base(self):
        x = torch.rand(1, 3, 16, 16)
        model = build_sr_model(
            variant="detail_aware",
            dim=12,
            depth=1,
            num_heads=3,
            scale_factor=3,
            window_size=8,
        )
        model.eval()
        with torch.inference_mode():
            y = model(x)
            expected = F.interpolate(
                x,
                scale_factor=3,
                mode="bicubic",
                align_corners=False,
            ).clamp(0.0, 1.0)
        self.assertTrue(torch.allclose(y, expected, atol=1e-6))

    def test_model_accepts_exact_target_size(self):
        x = torch.rand(1, 3, 16, 17)
        model = build_sr_model(
            variant="detail_aware",
            dim=8,
            depth=1,
            num_heads=2,
            scale_factor=3,
            window_size=8,
        )
        model.eval()
        with torch.inference_mode():
            y = model(x, target_size=(48, 50))
            expected = F.interpolate(
                x,
                size=(48, 50),
                mode="bicubic",
                align_corners=False,
            ).clamp(0.0, 1.0)
        self.assertEqual(tuple(y.shape), (1, 3, 48, 50))
        self.assertTrue(torch.allclose(y, expected, atol=1e-6))

    def test_engine_bicubic_fallback_without_model(self):
        frame = np.zeros((17, 19, 3), dtype=np.uint8)
        frame[:, :, 0] = 128
        engine = SuperResolutionEngine(
            SuperResolutionConfig(backend="bicubic", scale_factor=2, device="cpu")
        )
        output, stats = engine.upscale_with_stats(frame)
        self.assertEqual(output.shape, (34, 38, 3))
        self.assertEqual(stats.backend, "bicubic")
        self.assertEqual(stats.output_resolution, (38, 34))
        self.assertGreaterEqual(stats.latency_ms, 0.0)

    def test_target_resolution_resize(self):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        engine = SuperResolutionEngine(
            SuperResolutionConfig(backend="bilinear", scale_factor=2, device="cpu")
        )
        output = engine.upscale_frame(frame, target_resolution=(40, 30))
        self.assertEqual(output.shape, (30, 40, 3))

    def test_480p_to_1440p_alignment(self):
        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        engine = SuperResolutionEngine(
            SuperResolutionConfig(backend="bilinear", scale_factor=3, device="cpu")
        )
        output = engine.upscale_frame(frame, target_resolution=(2560, 1440))
        self.assertEqual(output.shape, (1440, 2560, 3))

    def test_engine_loads_model_variant(self):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        engine = SuperResolutionEngine(
            SuperResolutionConfig(
                backend="sr_transformer",
                scale_factor=3,
                device="cpu",
                half_precision=False,
                model_variant="hybrid",
                model_dim=12,
                model_depth=1,
                model_heads=3,
                model_window_size=8,
            )
        )
        output = engine.upscale_frame(frame, target_resolution=(48, 48))
        self.assertEqual(output.shape, (48, 48, 3))

    def test_engine_uses_checkpoint_model_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "detail_aware.pt"
            model = build_sr_model(
                variant="detail_aware",
                dim=12,
                depth=1,
                num_heads=3,
                scale_factor=3,
                window_size=8,
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": {
                        "variant": "detail_aware",
                        "scale_factor": 3,
                        "dim": 12,
                        "depth": 1,
                        "num_heads": 3,
                        "window_size": 8,
                        "residual_scale": 0.1,
                    },
                },
                checkpoint_path,
            )
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            engine = SuperResolutionEngine(
                SuperResolutionConfig(
                    backend="sr_transformer",
                    model_path=str(checkpoint_path),
                    scale_factor=2,
                    device="cpu",
                    half_precision=False,
                )
            )
            output = engine.upscale_frame(frame)
            self.assertEqual(output.shape, (48, 48, 3))

    def test_dataset_pairs_by_stem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lr_dir = root / "train" / "lr"
            hr_dir = root / "train" / "hr"
            lr_dir.mkdir(parents=True)
            hr_dir.mkdir(parents=True)
            cv2.imwrite(str(lr_dir / "sample.png"), np.zeros((8, 8, 3), dtype=np.uint8))
            cv2.imwrite(str(hr_dir / "sample.png"), np.zeros((24, 24, 3), dtype=np.uint8))
            dataset = SRFramePairDataset(root, split="train", scale=3)
            item = dataset[0]
            self.assertEqual(item["lr"].shape, (3, 8, 8))
            self.assertEqual(item["hr"].shape, (3, 24, 24))


if __name__ == "__main__":
    unittest.main()
