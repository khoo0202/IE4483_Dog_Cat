"""Evaluation script for the ConvNeXt dog-vs-cat classifier."""
from __future__ import annotations

import csv
import torch
import torch.nn as nn

from dataloader import create_dataloaders
from model import get_model
from utils import get_class_names, load_config, save_confusion_matrix, setup_experiment_folders


def ensure_test_loader(loaders):
    test_loader = loaders.get("test")
    if test_loader is not None and len(test_loader.dataset) > 0:
        return test_loader
    return loaders["val"]


def main() -> None:
    cfg = load_config()
    folders = setup_experiment_folders(cfg)
    loaders, _ = create_dataloaders(cfg)
    test_loader = ensure_test_loader(loaders)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).to(device)

    ckpt_path = folders["checkpoints"] / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.CrossEntropyLoss().to(device)
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    num_classes = cfg["model"]["num_classes"]
    class_correct = torch.zeros(num_classes)
    class_counts = torch.zeros(num_classes)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    all_targets, all_preds = [], []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()

            targets_cpu = targets.cpu()
            preds_cpu = preds.cpu()

            for cls in range(num_classes):
                cls_mask = targets_cpu == cls
                class_counts[cls] += cls_mask.sum().item()
                if cls_mask.any():
                    class_correct[cls] += (preds_cpu[cls_mask] == targets_cpu[cls_mask]).sum().item()

            for t, p in zip(targets_cpu.view(-1), preds_cpu.view(-1)):
                confusion[t.long(), p.long()] += 1
            all_targets.extend(targets_cpu.tolist())
            all_preds.extend(preds_cpu.tolist())

    avg_loss = total_loss / max(total_samples, 1)
    top1 = 100.0 * total_correct / max(total_samples, 1)
    per_class_acc = [
        100.0 * (class_correct[i].item() / class_counts[i].item()) if class_counts[i] > 0 else 0.0
        for i in range(num_classes)
    ]

    class_names = get_class_names(test_loader.dataset)
    if not class_names:
        class_names = ["Cat", "Dog"][:num_classes]

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Top-1 Accuracy: {top1:.2f}%")
    for name, acc in zip(class_names, per_class_acc):
        print(f"{name} Accuracy: {acc:.2f}%")
    print("Confusion Matrix:")
    print(confusion.tolist())

    results_dir = folders["results"]
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "test_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["metric", "value"])
        writer.writerow(["loss", f"{avg_loss:.4f}"])
        writer.writerow(["top1_accuracy", f"{top1:.2f}"])
        for name, acc in zip(class_names, per_class_acc):
            writer.writerow([f"acc_{name.lower()}", f"{acc:.2f}"])
        writer.writerow(["confusion_matrix", confusion.flatten().tolist()])

    save_confusion_matrix(
        all_targets,
        all_preds,
        class_names,
        results_dir / "confusion_matrix.png",
    )

if __name__ == "__main__":
    main()
