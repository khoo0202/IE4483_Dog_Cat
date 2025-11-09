"""Transform builders and mixup/cutmix helpers."""
from __future__ import annotations

import math
import random
from typing import Tuple

import torch
from torchvision import set_image_backend, transforms
from torchvision.transforms import InterpolationMode

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

set_image_backend("PIL")


def build_transforms(cfg) -> Tuple[transforms.Compose, transforms.Compose]:
    size = cfg["data"]["img_size"]
    train_tfms = transforms.Compose(
        [
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )
    return train_tfms, eval_tfms


def _rand_bbox(width: int, height: int, lam: float):
    cut_ratio = math.sqrt(1 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = random.randint(0, width)
    cy = random.randint(0, height)

    x1 = int(max(cx - cut_w // 2, 0))
    y1 = int(max(cy - cut_h // 2, 0))
    x2 = int(min(cx + cut_w // 2, width))
    y2 = int(min(cy + cut_h // 2, height))
    return x1, y1, x2, y2


def apply_mixup_cutmix(images: torch.Tensor, targets: torch.Tensor, alpha_mixup: float, alpha_cutmix: float):
    batch_size = images.size(0)
    if batch_size <= 1 or (alpha_mixup <= 0 and alpha_cutmix <= 0):
        return images, targets, targets, 1.0

    indices = torch.randperm(batch_size, device=images.device)
    shuffled_targets = targets[indices]

    if alpha_mixup > 0 and (alpha_cutmix <= 0 or random.random() < 0.5):
        lam = torch.distributions.Beta(alpha_mixup, alpha_mixup).sample().item()
        mixed = lam * images + (1 - lam) * images[indices]
        return mixed, targets, shuffled_targets, lam

    lam = torch.distributions.Beta(alpha_cutmix, alpha_cutmix).sample().item()
    x1, y1, x2, y2 = _rand_bbox(images.size(3), images.size(2), lam)
    images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (images.size(2) * images.size(3)))
    return images, targets, shuffled_targets, lam
