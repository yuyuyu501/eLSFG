from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class SRPair:
    lr: Path
    hr: Path


def list_image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_pairs(lr_dir: Path, hr_dir: Path) -> list[SRPair]:
    lr_files = {path.stem: path for path in list_image_files(lr_dir)}
    hr_files = {path.stem: path for path in list_image_files(hr_dir)}
    stems = sorted(set(lr_files) & set(hr_files))
    return [SRPair(lr=lr_files[stem], hr=hr_files[stem]) for stem in stems]


def read_rgb_tensor(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
    return tensor


class SRFramePairDataset(Dataset):
    """Paired LR/HR frame dataset.

    Expected layout:

    datasets/sr_game/train/lr/*.png
    datasets/sr_game/train/hr/*.png
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        patch_size: Optional[int] = None,
        scale: int = 3,
        random_crop: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self.scale = scale
        self.random_crop = random_crop
        self.lr_dir = self.root / split / "lr"
        self.hr_dir = self.root / split / "hr"
        if not self.lr_dir.exists() or not self.hr_dir.exists():
            raise FileNotFoundError(f"Dataset split not found: {self.root / split}")
        self.pairs = find_pairs(self.lr_dir, self.hr_dir)
        if not self.pairs:
            raise ValueError(f"No LR/HR image pairs found in {self.root / split}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        pair = self.pairs[index]
        lr = read_rgb_tensor(pair.lr)
        hr = read_rgb_tensor(pair.hr)
        if self.patch_size:
            lr, hr = self._crop_pair(lr, hr)
        return {"lr": lr, "hr": hr, "stem": pair.lr.stem}

    def _crop_pair(
        self,
        lr: torch.Tensor,
        hr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lr_crop = max(1, self.patch_size // self.scale)
        _, lr_h, lr_w = lr.shape
        _, hr_h, hr_w = hr.shape
        if lr_h < lr_crop or lr_w < lr_crop:
            return lr, hr

        if self.random_crop:
            lr_top = random.randint(0, lr_h - lr_crop)
            lr_left = random.randint(0, lr_w - lr_crop)
        else:
            lr_top = (lr_h - lr_crop) // 2
            lr_left = (lr_w - lr_crop) // 2

        hr_top = min(hr_h - self.patch_size, lr_top * self.scale)
        hr_left = min(hr_w - self.patch_size, lr_left * self.scale)
        hr_top = max(0, hr_top)
        hr_left = max(0, hr_left)

        lr = lr[:, lr_top : lr_top + lr_crop, lr_left : lr_left + lr_crop]
        hr = hr[:, hr_top : hr_top + self.patch_size, hr_left : hr_left + self.patch_size]
        return lr, hr


def collate_sr_batch(batch: Iterable[dict[str, torch.Tensor | str]]) -> dict[str, object]:
    items = list(batch)
    return {
        "lr": torch.stack([item["lr"] for item in items]),
        "hr": torch.stack([item["hr"] for item in items]),
        "stem": [item["stem"] for item in items],
    }
