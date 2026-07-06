from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torch.amp import autocast


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.super_resolution import SuperResolutionConfig, SuperResolutionEngine
from training.data import SRFramePairDataset
from training.metrics import crop_or_resize_to_match, psnr


def tensor_to_bgr(tensor: torch.Tensor) -> object:
    image = tensor.squeeze(0).permute(1, 2, 0).float().clamp(0, 1)
    image = (image.cpu().numpy() * 255.0).round().astype("uint8")
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an SR checkpoint on a paired dataset.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--save-images", type=int, default=4)
    parser.add_argument("--fp32", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_enabled = (not args.fp32) and device.type == "cuda"
    dataset = SRFramePairDataset(args.data_root, split=args.split, scale=3, random_crop=False)
    engine = SuperResolutionEngine(
        SuperResolutionConfig(
            backend="sr_transformer",
            model_path=str(args.checkpoint),
            device=str(device),
            half_precision=amp_enabled,
        )
    )
    model = engine.model
    if model is None:
        raise RuntimeError("Checkpoint did not load an SR model")

    output_dir = args.output_dir
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    model_scores = []
    bicubic_scores = []
    limit = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for index in range(limit):
            item = dataset[index]
            lr = item["lr"].unsqueeze(0).to(device)
            hr = item["hr"].unsqueeze(0).to(device)
            if amp_enabled:
                lr = lr.half()
            with autocast(device_type="cuda", enabled=amp_enabled):
                pred = crop_or_resize_to_match(model(lr, target_size=hr.shape[-2:]), hr)
            bicubic = F.interpolate(
                lr.float(),
                size=hr.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).clamp(0.0, 1.0)

            model_psnr = psnr(pred.float(), hr.float())
            bicubic_psnr = psnr(bicubic.float(), hr.float())
            model_scores.append(model_psnr)
            bicubic_scores.append(bicubic_psnr)
            rows.append(
                {
                    "stem": item["stem"],
                    "model_psnr": f"{model_psnr:.4f}",
                    "bicubic_psnr": f"{bicubic_psnr:.4f}",
                    "delta": f"{model_psnr - bicubic_psnr:.4f}",
                }
            )

            if output_dir and index < args.save_images:
                stem = str(item["stem"])
                cv2.imwrite(str(output_dir / f"{stem}_sr.png"), tensor_to_bgr(pred.float()))
                cv2.imwrite(str(output_dir / f"{stem}_bicubic.png"), tensor_to_bgr(bicubic))
                cv2.imwrite(str(output_dir / f"{stem}_hr.png"), tensor_to_bgr(hr.float()))

    if output_dir:
        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["stem", "model_psnr", "bicubic_psnr", "delta"])
            writer.writeheader()
            writer.writerows(rows)

    model_avg = sum(model_scores) / max(1, len(model_scores))
    bicubic_avg = sum(bicubic_scores) / max(1, len(bicubic_scores))
    peak_vram = (
        torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    )
    print(f"samples: {limit}")
    print(f"model_psnr: {model_avg:.4f}")
    print(f"bicubic_psnr: {bicubic_avg:.4f}")
    print(f"delta: {model_avg - bicubic_avg:.4f}")
    print(f"peak_vram_mb: {peak_vram:.1f}")
    if output_dir:
        print(f"output_dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
