"""Custom ConvNeXt implementation based on the official paper."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic depth regularization."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor


class LayerNorm2d(nn.Module):
    """LayerNorm applied over the channel dimension for NCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        h = x.permute(0, 2, 3, 1)
        h = self.norm(h)
        return h.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """Single ConvNeXt block (depthwise conv + MLP)."""

    def __init__(self, dim: int, drop_path: float, layer_scale_init_value: float = 1e-6) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        else:
            self.layer_scale = None
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.layer_scale is not None:
            x = self.layer_scale * x
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    """ConvNeXt backbone with configurable depths and widths."""

    def __init__(
        self,
        depths: Tuple[int, int, int, int],
        dims: Tuple[int, int, int, int],
        num_classes: int = 2,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
                LayerNorm2d(dims[0]),
            )
        )
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            blocks = []
            for _ in range(depths[i]):
                blocks.append(
                    ConvNeXtBlock(
                        dim=dims[i],
                        drop_path=dp_rates[cur],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
                cur += 1
            self.stages.append(nn.Sequential(*blocks))

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.downsample_layers[0](x)
        for i in range(4):
            x = self.stages[i](x)
            if i < 3:
                x = self.downsample_layers[i + 1](x)
        x = x.mean(dim=(2, 3))
        x = self.norm(x)
        return self.head(x)


_MODEL_CONFIGS: Dict[str, Dict[str, Tuple[int, ...]]] = {
    "convnext_tiny": {"depths": (3, 3, 9, 3), "dims": (96, 192, 384, 768)},
    "convnext_base": {"depths": (3, 3, 27, 3), "dims": (128, 256, 512, 1024)},
    "convnext_large": {"depths": (3, 3, 27, 3), "dims": (192, 384, 768, 1536)},
}


def _load_torchvision_weights(model: nn.Module, variant: str) -> None:
    try:
        from torchvision.models import (
            ConvNeXt_Base_Weights,
            ConvNeXt_Large_Weights,
            ConvNeXt_Tiny_Weights,
            convnext_base,
            convnext_large,
            convnext_tiny,
        )
    except Exception as exc:  # pragma: no cover - torchvision optional
        raise RuntimeError("torchvision is required for pretrained ConvNeXt weights.") from exc

    if variant == "convnext_tiny":
        tv_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    elif variant == "convnext_base":
        tv_model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
    elif variant == "convnext_large":
        tv_model = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"No torchvision weights available for variant '{variant}'.")

    tv_state = tv_model.state_dict()
    missing, unexpected = model.load_state_dict(tv_state, strict=False)
    print(
        "[ConvNeXt] Loaded torchvision pretrained weights "
        f"(missing={len(missing)} unexpected={len(unexpected)})."
    )


def get_model(cfg: Dict) -> nn.Module:
    model_cfg = cfg["model"]
    name = model_cfg["name"]
    if name not in _MODEL_CONFIGS:
        raise ValueError(f"Unsupported ConvNeXt variant '{name}'.")

    variant = _MODEL_CONFIGS[name]
    layer_scale = float(model_cfg.get("layer_scale_init_value", 1e-6))
    model = ConvNeXt(
        depths=variant["depths"],
        dims=variant["dims"],
        num_classes=model_cfg["num_classes"],
        drop_path_rate=float(model_cfg.get("drop_path_rate", 0.0)),
        layer_scale_init_value=layer_scale,
    )

    if model_cfg.get("pretrained", False):
        try:
            _load_torchvision_weights(model, name)
        except Exception as exc:
            print(f"[ConvNeXt] Failed to load pretrained weights: {exc}. Training from scratch.")

    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    sample_cfg = {
        "model": {
            "name": "convnext_tiny",
            "pretrained": False,
            "num_classes": 2,
            "drop_path_rate": 0.1,
        }
    }
    net = get_model(sample_cfg)
    tot, trainable = count_parameters(net)
    print(f"Total params: {tot:,}")
    print(f"Trainable params: {trainable:,}")
