"""Inspect saved checkpoints without command-line arguments."""
from __future__ import annotations

from pathlib import Path

import torch

from utils import load_config


def summarize_state_dict(state_dict, label):
    print(f"\nState dict under '{label}':")
    for name, tensor in state_dict.items():
        shape = tuple(tensor.shape)
        dtype = tensor.dtype
        print(f"  {name}: shape={shape}, dtype={dtype}")


def main() -> None:
    cfg = load_config()
    ckpt_path = Path(cfg["project"]["output_root"]) / cfg["project"]["run_name"] / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    print(f"Loaded checkpoint: {ckpt_path}")
    if isinstance(checkpoint, dict):
        print(f"Top-level keys: {list(checkpoint.keys())}")
        if "epoch" in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                summarize_state_dict(checkpoint[key], key)
                break
    else:
        print("Checkpoint is a raw state dict without metadata.")
        summarize_state_dict(checkpoint, "state_dict")


if __name__ == "__main__":
    main()
