from __future__ import annotations

import argparse
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def iter_sources(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.rglob("*") if path.is_file())


def resize_frame(frame, width: int, height: int):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def write_pair(frame, output_root: Path, split: str, stem: str, hr_size, lr_size) -> None:
    hr_width, hr_height = hr_size
    lr_width, lr_height = lr_size
    hr = resize_frame(frame, hr_width, hr_height)
    lr = resize_frame(hr, lr_width, lr_height)
    hr_path = output_root / split / "hr" / f"{stem}.png"
    lr_path = output_root / split / "lr" / f"{stem}.png"
    hr_path.parent.mkdir(parents=True, exist_ok=True)
    lr_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(hr_path), hr)
    cv2.imwrite(str(lr_path), lr)


def process_image(path: Path, output_root: Path, args, index: int) -> int:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        return index
    split = "val" if index % args.val_every == 0 else "train"
    write_pair(
        frame,
        output_root,
        split,
        f"{path.stem}_{index:08d}",
        (args.hr_width, args.hr_height),
        (args.lr_width, args.lr_height),
    )
    return index + 1


def process_video(path: Path, output_root: Path, args, index: int) -> int:
    capture = cv2.VideoCapture(str(path))
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % args.frame_step == 0:
            split = "val" if index % args.val_every == 0 else "train"
            write_pair(
                frame,
                output_root,
                split,
                f"{path.stem}_{frame_index:08d}",
                (args.hr_width, args.hr_height),
                (args.lr_width, args.lr_height),
            )
            index += 1
        frame_index += 1
    capture.release()
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare paired LR/HR SR frames.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hr-width", type=int, default=2560)
    parser.add_argument("--hr-height", type=int, default=1440)
    parser.add_argument("--lr-width", type=int, default=854)
    parser.add_argument("--lr-height", type=int, default=480)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--val-every", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = iter_sources(args.input)
    index = 0
    for source in sources:
        suffix = source.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            index = process_image(source, args.output, args, index)
        elif suffix in VIDEO_EXTENSIONS:
            index = process_video(source, args.output, args, index)
    print(f"Wrote {index} LR/HR pairs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
