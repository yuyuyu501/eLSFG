from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = PROJECT_ROOT / "profiles"


@dataclass
class AppProfile:
    name: str = "Default"
    backend: str = "bicubic"
    scale_factor: int = 2
    target_width: int = 2560
    target_height: int = 1440
    quality: str = "balanced"
    model_path: str = ""
    tile_size: int = 0
    tile_overlap: int = 16
    half_precision: bool = True
    capture_api: str = "debug-preview"
    hotkey: str = "Alt+S"
    always_on_top: bool = False
    model_variant: str = "baseline"
    model_dim: int = 48
    model_depth: int = 4
    model_heads: int = 4
    model_window_size: int = 8

    @property
    def filename(self) -> str:
        safe = "".join(ch for ch in self.name.lower() if ch.isalnum() or ch in ("-", "_"))
        return f"{safe or 'profile'}.json"


class ProfileStore:
    def __init__(self, directory: Path = PROFILE_DIR):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> List[AppProfile]:
        profiles = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                profiles.append(AppProfile(**data))
            except (TypeError, json.JSONDecodeError):
                continue
        if not profiles:
            profiles = self._default_profiles()
            for profile in profiles:
                self.save(profile)
        return profiles

    def save(self, profile: AppProfile) -> Path:
        path = self.directory / profile.filename
        path.write_text(
            json.dumps(asdict(profile), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _default_profiles() -> List[AppProfile]:
        return [
            AppProfile(name="Default", backend="bicubic", scale_factor=2),
            AppProfile(name="Game Fast", backend="bilinear", scale_factor=2, quality="fast"),
            AppProfile(
                name="AI Quality",
                backend="sr_transformer",
                scale_factor=3,
                quality="quality",
                model_path="checkpoints/elsfg_sr_detail_aware.pt",
                tile_size=0,
                model_variant="detail_aware",
                model_dim=12,
                model_depth=1,
                model_heads=3,
            ),
        ]
