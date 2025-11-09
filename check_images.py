"""Utility to count valid vs corrupted images in the dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, UnidentifiedImageError

from utils import load_config

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_image_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTS:
            yield path


def is_image_ok(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def main() -> None:
    cfg = load_config()
    root = Path(cfg["data"]["root_dir"]).expanduser()
    total = 0
    ok = 0
    bad_files = []

    for img_path in iter_image_files(root):
        total += 1
        if is_image_ok(img_path):
            ok += 1
        else:
            bad_files.append(img_path)

    print(f"Dataset root: {root}")
    print(f"Total image files: {total}")
    print(f"Valid images: {ok}")
    print(f"Corrupted images: {total - ok}")
    if bad_files:
        print("Corrupted file list:")
        for path in bad_files:
            print(f" - {path}")


if __name__ == "__main__":
    main()
