from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.transformer.sr_transformer import build_sr_model


def align_tensor(output: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    _, _, height, width = output.shape
    if height == target_height and width == target_width:
        return output
    return F.interpolate(
        output,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def run_tensor_benchmark(args: argparse.Namespace) -> dict[str, object]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device)
    if device.type == "cuda" and args.require_gpu_name:
        gpu_name = torch.cuda.get_device_name(device)
        if args.require_gpu_name.lower() not in gpu_name.lower():
            raise RuntimeError(
                "Performance benchmarks must run on the configured training GPU. "
                f"Required name fragment: {args.require_gpu_name!r}; current GPU: {gpu_name!r}. "
                "Use --require-gpu-name \"\" only for non-performance smoke tests."
            )
    dtype = torch.float32 if args.fp32 or device.type != "cuda" else torch.float16

    model = build_sr_model(
        variant=args.variant,
        dim=args.model_dim,
        depth=args.model_depth,
        num_heads=args.model_heads,
        scale_factor=args.scale,
        window_size=args.window_size,
    ).to(device=device, dtype=dtype)
    model.eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)

    input_tensor = torch.rand(
        (1, 3, args.height, args.width),
        device=device,
        dtype=dtype,
    )
    if args.channels_last:
        input_tensor = input_tensor.contiguous(memory_format=torch.channels_last)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(max(0, args.warmup)):
            output = model(input_tensor)
            output = align_tensor(output, args.target_height, args.target_width)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        latencies = []
        output_shape = None
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(max(1, args.runs)):
                start.record()
                output = model(input_tensor)
                output = align_tensor(output, args.target_height, args.target_width)
                end.record()
                torch.cuda.synchronize(device)
                latencies.append(start.elapsed_time(end))
                output_shape = tuple(output.shape)
        else:
            import time

            for _ in range(max(1, args.runs)):
                t0 = time.perf_counter()
                output = model(input_tensor)
                output = align_tensor(output, args.target_height, args.target_width)
                latencies.append((time.perf_counter() - t0) * 1000.0)
                output_shape = tuple(output.shape)

    avg = statistics.mean(latencies)
    result = {
        "variant": args.variant,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "input": f"{args.width}x{args.height}",
        "target": f"{args.target_width}x{args.target_height}",
        "output_shape": output_shape,
        "avg_ms": avg,
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "fps": 1000.0 / avg if avg > 0 else 0.0,
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
        "params_m": sum(p.numel() for p in model.parameters()) / 1_000_000,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU tensor-only SR benchmark.")
    parser.add_argument(
        "--variant",
        default="baseline",
        choices=["baseline", "hybrid", "shared_attention", "detail_aware"],
    )
    parser.add_argument("--width", type=int, default=854)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--target-width", type=int, default=2560)
    parser.add_argument("--target-height", type=int, default=1440)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dim", type=int, default=32)
    parser.add_argument("--model-depth", type=int, default=2)
    parser.add_argument("--model-heads", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument(
        "--require-gpu-name",
        default="2080 Ti",
        help="Required CUDA device name fragment for valid performance benchmarks.",
    )
    return parser.parse_args()


def main() -> int:
    result = run_tensor_benchmark(parse_args())
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
