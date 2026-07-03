from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scaler: Any = None,
    epoch: int = 0,
    step: int = 0,
    model_config: Optional[dict[str, Any]] = None,
    metrics: Optional[dict[str, Any]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "epoch": epoch,
        "step": step,
        "model_config": model_config or {},
        "metrics": metrics or {},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"], strict=False)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint
