"""DataLoader factory for the ConvNeXt dog-vs-cat project."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder

from augmentation import build_transforms


class TransformSubset(Dataset):
    def __init__(self, dataset: ImageFolder, indices: List[int], transform) -> None:
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.target_transform = dataset.target_transform

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.indices)

    def __getitem__(self, idx: int):  # type: ignore[override]
        path, target = self.dataset.samples[self.indices[idx]]
        sample = self.dataset.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


def _split_indices(num_samples: int, val_split: float, test_split: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_samples, generator=generator).tolist()
    test_size = int(num_samples * test_split)
    val_size = int(num_samples * val_split)
    train_size = num_samples - val_size - test_size
    train_indices = perm[:train_size]
    val_indices = perm[train_size : train_size + val_size]
    test_indices = perm[train_size + val_size :]
    return train_indices, val_indices, test_indices


def _build_loader(dataset: Dataset, batch_size: int, sampler, shuffle: bool, num_workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _maybe_build_sampler(dataset: Dataset, shuffle: bool, ddp_enabled: bool):
    if not ddp_enabled:
        return None
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return None
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    return DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)


def _get_presplit_dirs(root_dir: str) -> Dict[str, str]:
    """Return directories for existing splits when data is already partitioned."""
    splits = {split: os.path.join(root_dir, split) for split in ("train", "val", "test")}
    if not all(os.path.isdir(splits[split]) for split in ("train", "val")):
        return {}
    return {split: path for split, path in splits.items() if os.path.isdir(path)}


def create_dataloaders(cfg) -> Tuple[Dict[str, DataLoader], Dict[str, Optional[DistributedSampler]]]:
    train_tfms, eval_tfms = build_transforms(cfg)
    root_dir = cfg["data"]["root_dir"]
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["project"]["num_workers"]
    pin_memory = cfg["project"]["pin_memory"]
    ddp_enabled = cfg["ddp"].get("enabled", False) and int(os.environ.get("WORLD_SIZE", "1")) > 1

    split_dirs = _get_presplit_dirs(root_dir)

    if split_dirs:
        train_dataset = ImageFolder(split_dirs["train"], transform=train_tfms)
        val_dataset = ImageFolder(split_dirs["val"], transform=eval_tfms)
        test_dir = split_dirs.get("test")
        if test_dir:
            test_dataset = ImageFolder(test_dir, transform=eval_tfms)
        else:
            # No dedicated test split provided; reuse validation set for evaluation.
            test_dataset = ImageFolder(split_dirs["val"], transform=eval_tfms)
    else:
        base_dataset = ImageFolder(root_dir, transform=None)
        train_idx, val_idx, test_idx = _split_indices(
            len(base_dataset), cfg["data"]["val_split"], cfg["data"]["test_split"], cfg["data"]["shuffle_seed"]
        )
        train_dataset = TransformSubset(base_dataset, train_idx, train_tfms)
        val_dataset = TransformSubset(base_dataset, val_idx, eval_tfms)
        test_dataset = TransformSubset(base_dataset, test_idx, eval_tfms)

    samplers: Dict[str, Optional[DistributedSampler]] = {
        "train": _maybe_build_sampler(train_dataset, True, ddp_enabled),
        "val": _maybe_build_sampler(val_dataset, False, ddp_enabled),
        "test": _maybe_build_sampler(test_dataset, False, ddp_enabled),
    }

    loaders = {
        "train": _build_loader(train_dataset, batch_size, samplers["train"], True, num_workers, pin_memory),
        "val": _build_loader(val_dataset, batch_size, samplers["val"], False, num_workers, pin_memory),
        "test": _build_loader(test_dataset, batch_size, samplers["test"], False, num_workers, pin_memory),
    }
    return loaders, samplers
