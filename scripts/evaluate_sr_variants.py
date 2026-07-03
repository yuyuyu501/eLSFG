from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_sr_tensor import run_tensor_benchmark


QUALITY_PRIOR = {
    "baseline": 0.82,
    "hybrid": 0.76,
    "shared_attention": 0.80,
    "detail_aware": 0.86,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SR model structure candidates.")
    parser.add_argument("--width", type=int, default=854)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--target-width", type=int, default=2560)
    parser.add_argument("--target-height", type=int, default=1440)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dim", type=int, default=32)
    parser.add_argument("--model-depth", type=int, default=2)
    parser.add_argument("--model-heads", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--target-fps", type=float, default=120.0)
    parser.add_argument("--max-vram-mb", type=float, default=4096.0)
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


def result_score(result: dict[str, object], target_fps: float, max_vram_mb: float) -> float:
    fps = float(result["fps"])
    vram = float(result["peak_vram_mb"])
    speed_score = min(1.0, fps / target_fps)
    memory_score = min(1.0, max_vram_mb / max(vram, 1.0))
    quality_score = QUALITY_PRIOR[str(result["variant"])]
    return quality_score * 0.55 + speed_score * 0.35 + memory_score * 0.10


def main() -> int:
    args = parse_args()
    variants = ["baseline", "hybrid", "shared_attention", "detail_aware"]
    results = []
    for variant in variants:
        variant_args = argparse.Namespace(**vars(args))
        variant_args.variant = variant
        result = run_tensor_benchmark(variant_args)
        result["quality_prior"] = QUALITY_PRIOR[variant]
        result["meets_speed"] = float(result["fps"]) >= args.target_fps
        result["meets_vram"] = float(result["peak_vram_mb"]) <= args.max_vram_mb
        result["score"] = result_score(result, args.target_fps, args.max_vram_mb)
        results.append(result)

    results.sort(key=lambda item: item["score"], reverse=True)
    header = (
        "variant,avg_ms,p95_ms,fps,peak_vram_mb,params_m,"
        "quality_prior,meets_speed,meets_vram,score"
    )
    print(header)
    for result in results:
        print(
            f"{result['variant']},"
            f"{float(result['avg_ms']):.3f},"
            f"{float(result['p95_ms']):.3f},"
            f"{float(result['fps']):.2f},"
            f"{float(result['peak_vram_mb']):.1f},"
            f"{float(result['params_m']):.3f},"
            f"{float(result['quality_prior']):.2f},"
            f"{result['meets_speed']},"
            f"{result['meets_vram']},"
            f"{float(result['score']):.3f}"
        )

    passing = [item for item in results if item["meets_speed"] and item["meets_vram"]]
    if passing:
        best = max(passing, key=lambda item: item["quality_prior"])
        print(f"recommendation: {best['variant']} (best quality prior among passing variants)")
    else:
        print("recommendation: none meets target yet; optimize the highest score candidate first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
