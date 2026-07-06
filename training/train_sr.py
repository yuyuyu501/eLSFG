from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.transformer.sr_transformer import build_sr_model
from training.checkpoint import load_checkpoint, save_checkpoint
from training.data import SRFramePairDataset, collate_sr_batch
from training.losses import CombinedSRLoss, SRLossConfig
from training.metrics import crop_or_resize_to_match, psnr


@dataclass
class ModelConfig:
    variant: str = "baseline"
    scale_factor: int = 3
    dim: int = 48
    depth: int = 4
    num_heads: int = 4
    window_size: int = 8
    residual_scale: float = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train eLSFG Transformer SR.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/sr_transformer"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument(
        "--variant",
        default="baseline",
        choices=["baseline", "hybrid", "shared_attention", "detail_aware"],
    )
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--edge-weight", type=float, default=0.05)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def setup_distributed() -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ:
        return False, 0, 0, 1
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def cleanup_distributed(enabled: bool) -> None:
    if enabled:
        dist.destroy_process_group()


def build_loader(args, split: str, distributed: bool, shuffle: bool) -> tuple[DataLoader, object]:
    dataset = SRFramePairDataset(
        args.data_root,
        split=split,
        patch_size=args.patch_size if split == "train" else None,
        scale=args.scale,
        random_crop=split == "train",
    )
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle and sampler is None,
        "sampler": sampler,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": collate_sr_batch,
        "drop_last": split == "train",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(dataset, **loader_kwargs)
    return loader, sampler


@torch.no_grad()
def validate(model, loader, device, amp_enabled: bool, channels_last: bool) -> tuple[float, int]:
    model.eval()
    score_sum = 0.0
    count = 0
    for batch in loader:
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        if channels_last:
            lr = lr.contiguous(memory_format=torch.channels_last)
            hr = hr.contiguous(memory_format=torch.channels_last)
        with autocast(device_type="cuda", enabled=amp_enabled):
            pred = model(lr)
            pred = crop_or_resize_to_match(pred, hr)
        score_sum += psnr(pred.float(), hr.float())
        count += 1
    model.train()
    return score_sum, count


def distributed_average(total: float, count: int, device, distributed: bool) -> float:
    stats = torch.tensor([total, float(count)], device=device, dtype=torch.float64)
    if distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float(stats[0].item() / max(1.0, stats[1].item()))


def main() -> int:
    args = parse_args()
    distributed, rank, local_rank, _ = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    amp_enabled = (not args.no_amp) and device.type == "cuda"
    channels_last = args.channels_last and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model_config = ModelConfig(
        variant=args.variant,
        scale_factor=args.scale,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.heads,
        window_size=args.window_size,
        residual_scale=args.residual_scale,
    )
    model = build_sr_model(
        variant=model_config.variant,
        dim=model_config.dim,
        depth=model_config.depth,
        num_heads=model_config.num_heads,
        scale_factor=model_config.scale_factor,
        window_size=model_config.window_size,
        residual_scale=model_config.residual_scale,
    ).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler("cuda", enabled=amp_enabled)
    criterion = CombinedSRLoss(SRLossConfig(edge_weight=args.edge_weight)).to(device)
    start_epoch = 0
    step = 0
    best_val_psnr = float("-inf")
    last_val_psnr = 0.0

    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scaler, map_location=device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        step = int(checkpoint.get("step", 0))
        metrics = checkpoint.get("metrics", {})
        best_val_psnr = float(metrics.get("best_val_psnr", metrics.get("val_psnr", best_val_psnr)))
        last_val_psnr = float(metrics.get("val_psnr", last_val_psnr))

    train_loader, train_sampler = build_loader(args, "train", distributed, shuffle=True)
    val_loader, _ = build_loader(args, "val", distributed, shuffle=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        running_count = 0
        for batch in train_loader:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            if channels_last:
                lr = lr.contiguous(memory_format=torch.channels_last)
                hr = hr.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=amp_enabled):
                pred = model(lr)
                pred = crop_or_resize_to_match(pred, hr)
                losses = criterion(pred, hr)
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(losses["loss"].detach().cpu())
            running_count += 1
            step += 1

        avg_loss = distributed_average(running_loss, running_count, device, distributed)
        should_validate = args.val_every > 0 and (
            epoch % args.val_every == 0 or epoch == args.epochs - 1
        )
        if should_validate:
            val_sum, val_count = validate(model, val_loader, device, amp_enabled, channels_last)
            last_val_psnr = distributed_average(val_sum, val_count, device, distributed)
        if rank == 0:
            best_val_psnr = max(best_val_psnr, last_val_psnr)
            print(
                f"epoch={epoch} step={step} loss={avg_loss:.6f} val_psnr={last_val_psnr:.3f}",
                flush=True,
            )
            if epoch % args.save_every == 0:
                metrics = {
                    "loss": avg_loss,
                    "val_psnr": last_val_psnr,
                    "best_val_psnr": best_val_psnr,
                }
                latest_path = save_checkpoint(
                    args.output_dir / "latest.pt",
                    model,
                    optimizer,
                    scaler,
                    epoch=epoch,
                    step=step,
                    model_config=asdict(model_config),
                    metrics=metrics,
                )
                if should_validate and last_val_psnr >= best_val_psnr:
                    save_checkpoint(
                        args.output_dir / "best.pt",
                        model,
                        optimizer,
                        scaler,
                        epoch=epoch,
                        step=step,
                        model_config=asdict(model_config),
                        metrics=metrics,
                    )
        if distributed:
            dist.barrier()

    cleanup_distributed(distributed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
