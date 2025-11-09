"""Dataset splitting helper for PetImages."""
from __future__ import annotations

import random
import shutil
from pathlib import Path

from utils import load_config


def copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            return
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def split_class(files, val_ratio, test_ratio, rng: random.Random):
    files = list(files)
    rng.shuffle(files)
    n = len(files)
    val_count = int(n * val_ratio)
    test_count = int(n * test_ratio)
    train_count = n - val_count - test_count
    train_files = files[:train_count]
    val_files = files[train_count : train_count + val_count]
    test_files = files[train_count + val_count :]
    return train_files, val_files, test_files


def main() -> None:
    cfg = load_config()
    root = Path(cfg["data"]["root_dir"]).expanduser()
    target_root = Path(f"{root}_split")
    if target_root.exists():
        print(f"Split directory already exists at {target_root}. Nothing to do.")
        return

    classes = [d for d in root.iterdir() if d.is_dir()]
    if not classes:
        raise RuntimeError(f"No class folders found under {root}")

    val_ratio = cfg["data"]["val_split"]
    test_ratio = cfg["data"]["test_split"]
    seed = cfg["data"]["shuffle_seed"]

    for split in ("train", "val", "test"):
        for cls in classes:
            (target_root / split / cls.name).mkdir(parents=True, exist_ok=True)

    for idx, cls in enumerate(classes):
        files = list(cls.glob("*"))
        rng = random.Random(seed + idx)
        train_files, val_files, test_files = split_class(files, val_ratio, test_ratio, rng)
        for f in train_files:
            copy_or_link(f, target_root / "train" / cls.name / f.name)
        for f in val_files:
            copy_or_link(f, target_root / "val" / cls.name / f.name)
        for f in test_files:
            copy_or_link(f, target_root / "test" / cls.name / f.name)

    print(f"Created split dataset at {target_root}")


if __name__ == "__main__":
    main()
