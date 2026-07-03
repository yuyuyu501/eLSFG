from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((prediction - target) ** 2 + self.eps**2).mean()


class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_edges = self._edges(prediction)
        target_edges = self._edges(target)
        return F.l1_loss(pred_edges, target_edges)

    def _edges(self, image: torch.Tensor) -> torch.Tensor:
        channels = image.shape[1]
        kernel_x = self.kernel_x.to(image.dtype).repeat(channels, 1, 1, 1)
        kernel_y = self.kernel_y.to(image.dtype).repeat(channels, 1, 1, 1)
        grad_x = F.conv2d(image, kernel_x, padding=1, groups=channels)
        grad_y = F.conv2d(image, kernel_y, padding=1, groups=channels)
        return torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)


@dataclass
class SRLossConfig:
    reconstruction_weight: float = 1.0
    edge_weight: float = 0.05


class CombinedSRLoss(nn.Module):
    def __init__(self, config: SRLossConfig | None = None):
        super().__init__()
        self.config = config or SRLossConfig()
        self.reconstruction = CharbonnierLoss()
        self.edge = SobelEdgeLoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        reconstruction = self.reconstruction(prediction, target)
        edge = self.edge(prediction, target)
        total = (
            reconstruction * self.config.reconstruction_weight
            + edge * self.config.edge_weight
        )
        return {
            "loss": total,
            "reconstruction": reconstruction.detach(),
            "edge": edge.detach(),
        }
