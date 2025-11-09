"""Utility helpers for configuration, logging, and distributed training."""
from __future__ import annotations

import os
import random
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import yaml
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

_LAST_CFG: Dict | None = None


def load_config(path: str = "configs/config.yaml") -> Dict:
    cfg_path = Path(os.environ.get("CFG_PATH", path))
    if not cfg_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        cfg_path = (project_root / cfg_path).resolve()
    with open(cfg_path, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    global _LAST_CFG
    _LAST_CFG = cfg
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_experiment_folders(cfg: Dict) -> Dict[str, Path]:
    root = Path(cfg["project"]["output_root"]) / cfg["project"]["run_name"]
    checkpoints = root / "checkpoints"
    logs = root / "logs"
    results = root / "results"
    for folder in (root, checkpoints, logs, results):
        folder.mkdir(parents=True, exist_ok=True)
    return {"root": root, "checkpoints": checkpoints, "logs": logs, "results": results}


@dataclass
class AverageMeter:
    name: str
    fmt: str = ".4f"

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def __str__(self) -> str:
        return f"{self.name} {self.val:{self.fmt}} (avg: {self.avg:{self.fmt}})"


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Iterable[int] = (1,)) -> Tuple[torch.Tensor, ...]:
    maxk = max(topk)
    with torch.no_grad():
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            correct_k.mul_(100.0 / target.size(0))
            res.append(correct_k)
    return tuple(res)


def save_checkpoint(state: Dict, is_best: bool, ckpt_dir: Path, filename: str = "last.pt", best_filename: str = "best.pt") -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / filename
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copy2(ckpt_path, ckpt_dir / best_filename)


def get_device_and_ddp(cfg: Dict):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    cfg_ddp = cfg.get("ddp", {}).get("enabled", False)
    backend = cfg.get("ddp", {}).get("backend", "nccl")
    is_ddp = cfg_ddp or world_size_env > 1
    if not is_ddp or world_size_env <= 1:
        return device, False, 0, 1

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    world_size = dist.get_world_size()
    return device, True, local_rank, world_size


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def get_tb_writer(log_dir: Path):
    enabled = bool(_LAST_CFG and _LAST_CFG.get("logging", {}).get("tensorboard", False))
    if not enabled:
        return nullcontext()
    return SummaryWriter(log_dir=str(log_dir))


def save_loss_curve(train_losses, val_losses, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_accuracy_curve(train_accs, val_accs, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_accs) + 1), train_accs, label="Train Accuracy")
    plt.plot(range(1, len(val_accs) + 1), val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names: List[str], output_path: Path) -> None:
    if not class_names:
        class_names = [str(i) for i in sorted(set(y_true))]
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    thresh = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                format(matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "black",
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def get_class_names(dataset) -> List[str]:
    if hasattr(dataset, "classes"):
        return list(dataset.classes)
    if hasattr(dataset, "dataset"):
        return get_class_names(dataset.dataset)
    return []
