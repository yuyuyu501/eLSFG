from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.super_resolution import SuperResolutionConfig, SuperResolutionEngine


def make_frame(width: int, height: int) -> np.ndarray:
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    frame = np.stack(
        [
            255 * xx,
            255 * yy,
            255 * (1.0 - xx * yy),
        ],
        axis=-1,
    )
    return frame.astype(np.uint8)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SR inference.")
    parser.add_argument("--backend", default="sr_transformer")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--width", type=int, default=854)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--target-width", type=int, default=2560)
    parser.add_argument("--target-height", type=int, default=1440)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dim", type=int, default=48)
    parser.add_argument("--model-depth", type=int, default=4)
    parser.add_argument("--model-heads", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--fp32", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = make_frame(args.width, args.height)
    config = SuperResolutionConfig(
        backend=args.backend,
        model_path=args.model_path,
        scale_factor=args.scale,
        device=args.device,
        half_precision=not args.fp32,
        warmup_size=(args.width, args.height),
        model_dim=args.model_dim,
        model_depth=args.model_depth,
        model_heads=args.model_heads,
        model_window_size=args.window_size,
    )
    engine = SuperResolutionEngine(config)

    if torch.cuda.is_available() and engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(engine.device)

    for _ in range(max(0, args.warmup)):
        engine.upscale_frame(frame, (args.target_width, args.target_height))

    latencies = []
    output_shape = None
    for _ in range(max(1, args.runs)):
        output, stats = engine.upscale_with_stats(
            frame,
            (args.target_width, args.target_height),
        )
        output_shape = output.shape
        latencies.append(stats.latency_ms)

    avg = statistics.mean(latencies)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)
    fps = 1000.0 / avg if avg > 0 else 0.0
    peak_mb = (
        torch.cuda.max_memory_allocated(engine.device) / 1024**2
        if torch.cuda.is_available() and engine.device.type == "cuda"
        else 0.0
    )

    print(f"backend: {engine.backend}")
    print(f"device: {engine.device}")
    print(f"input: {args.width}x{args.height}")
    print(f"target: {args.target_width}x{args.target_height}")
    print(f"output_shape: {output_shape}")
    print(f"avg_ms: {avg:.3f}")
    print(f"p95_ms: {p95:.3f}")
    print(f"p99_ms: {p99:.3f}")
    print(f"fps: {fps:.2f}")
    print(f"peak_vram_mb: {peak_mb:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
