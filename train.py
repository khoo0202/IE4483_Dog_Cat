"""Training entrypoint for ConvNeXt dog-vs-cat classification."""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ReduceLROnPlateau, SequentialLR
from torch.utils.data import DataLoader

from augmentation import apply_mixup_cutmix
from dataloader import create_dataloaders
from model import count_parameters, get_model
from utils import (
    AverageMeter,
    accuracy,
    cleanup_ddp,
    get_device_and_ddp,
    get_tb_writer,
    load_config,
    save_accuracy_curve,
    save_checkpoint,
    save_confusion_matrix,
    save_loss_curve,
    get_class_names,
    set_seed,
    setup_experiment_folders,
)


def build_scheduler(optimizer, cfg):
    sched_cfg = cfg["sched"]
    name = sched_cfg["name"].lower()
    if name == "cosine":
        warmup_epochs = sched_cfg["warmup_epochs"]
        cosine_epochs = max(sched_cfg["max_epochs"] - warmup_epochs, 1)
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=sched_cfg["min_lr"])
        if warmup_epochs > 0:
            def warmup_lambda(epoch: int) -> float:
                return min(1.0, (epoch + 1) / warmup_epochs)

            warmup = LambdaLR(optimizer, lr_lambda=warmup_lambda)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        else:
            scheduler = cosine
        plateau = None
    else:
        scheduler = None
        plateau = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=sched_cfg["patience"],
            factor=sched_cfg["factor"],
            min_lr=sched_cfg["min_lr"],
        )
    return scheduler, plateau


def sync_meter(loss_meter: AverageMeter, acc_meter: AverageMeter, device: torch.device, is_ddp: bool) -> Tuple[float, float]:
    if not is_ddp:
        return loss_meter.avg, acc_meter.avg
    tensor = torch.tensor(
        [loss_meter.sum, loss_meter.count, acc_meter.sum, acc_meter.count], device=device
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    loss_avg = (tensor[0] / max(tensor[1], 1.0)).item()
    acc_avg = (tensor[2] / max(tensor[3], 1.0)).item()
    return loss_avg, acc_avg


def train_one_epoch(
    epoch: int,
    model,
    loader,
    sampler,
    criterion,
    optimizer,
    scaler,
    device,
    cfg,
    use_amp: bool,
    is_ddp: bool,
) -> Tuple[float, float]:
    model.train()
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    loss_meter = AverageMeter("train_loss")
    acc_meter = AverageMeter("train_acc")
    mix_alpha = cfg["train"]["mixup_alpha"]
    cutmix_alpha = cfg["train"]["cutmix_alpha"]
    grad_clip = cfg["train"]["grad_clip_norm"]

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        images, targets_a, targets_b, lam = apply_mixup_cutmix(images, targets, mix_alpha, cutmix_alpha)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = lam * criterion(outputs, targets_a)
            loss += (1.0 - lam) * criterion(outputs, targets_b)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        loss_meter.update(loss.item(), targets.size(0))
        acc1 = accuracy(outputs, targets)[0]
        acc_meter.update(acc1.item(), targets.size(0))

    return sync_meter(loss_meter, acc_meter, device, is_ddp)


def validate(model, loader, criterion, device, is_ddp: bool) -> Tuple[float, float]:
    model.eval()
    loss_meter = AverageMeter("val_loss")
    acc_meter = AverageMeter("val_acc")
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss_meter.update(loss.item(), targets.size(0))
            acc1 = accuracy(outputs, targets)[0]
            acc_meter.update(acc1.item(), targets.size(0))
    return sync_meter(loss_meter, acc_meter, device, is_ddp)


def should_log(rank: int) -> bool:
    return rank == 0


def collect_predictions(model, dataset, cfg, device):
    loader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["project"]["num_workers"],
        pin_memory=cfg["project"]["pin_memory"],
    )
    preds, targets = [], []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds.extend(outputs.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    if was_training:
        model.train()
    return targets, preds


def main() -> None:
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    folders = setup_experiment_folders(cfg)
    loaders, samplers = create_dataloaders(cfg)
    class_names = get_class_names(loaders["train"].dataset)

    device, is_ddp, local_rank, _ = get_device_and_ddp(cfg)
    model = get_model(cfg).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    base_model = model.module if isinstance(model, DDP) else model
    total_params, trainable_params = count_parameters(base_model)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        betas=tuple(cfg["optim"]["betas"]),
        weight_decay=cfg["optim"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler, plateau = build_scheduler(optimizer, cfg)
    use_amp = cfg["project"].get("amp", False) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    epochs_without_improve = 0
    max_epochs = cfg["sched"]["max_epochs"]
    save_every = max(1, cfg["project"].get("save_every_n_epochs", 1))
    rank = dist.get_rank() if is_ddp else 0

    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []

    tb_writer_ctx = get_tb_writer(folders["logs"])
    with tb_writer_ctx as tb_writer:
        for epoch in range(max_epochs):
            train_loss, train_acc = train_one_epoch(
                epoch,
                model,
                loaders["train"],
                samplers.get("train"),
                criterion,
                optimizer,
                scaler,
                device,
                cfg,
                use_amp,
                is_ddp,
            )

            val_loss, val_acc = validate(model, loaders["val"], criterion, device, is_ddp)
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            if scheduler is not None:
                scheduler.step()
            elif plateau is not None:
                plateau.step(val_loss)

            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if should_log(rank):
                state = {
                    "epoch": epoch + 1,
                    "model_state_dict": base_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if use_amp else None,
                    "cfg": cfg,
                }

                save_checkpoint(state, is_best, folders["checkpoints"])
                if (epoch + 1) % save_every == 0:
                    extra_path = folders["checkpoints"] / f"epoch_{epoch + 1}.pt"
                    torch.save(state, extra_path)

                print(
                    f"Epoch {epoch + 1}/{max_epochs} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%"
                )
                if tb_writer is not None:
                    tb_writer.add_scalar("Loss/Train", train_loss, epoch)
                    tb_writer.add_scalar("Loss/Val", val_loss, epoch)
                    tb_writer.add_scalar("Acc/Train", train_acc, epoch)
                    tb_writer.add_scalar("Acc/Val", val_acc, epoch)

            if cfg["train"]["early_stopping_patience"] > 0 and epochs_without_improve >= cfg["train"]["early_stopping_patience"]:
                if should_log(rank):
                    print("Early stopping triggered.")
                break

    if should_log(rank):
        print(f"Total params: {total_params:,} | Trainable params: {trainable_params:,}")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        save_loss_curve(train_loss_history, val_loss_history, folders["results"] / "loss_curve.png")
        save_accuracy_curve(train_acc_history, val_acc_history, folders["results"] / "accuracy_curve.png")
        val_targets_full, val_preds_full = collect_predictions(base_model, loaders["val"].dataset, cfg, device)
        save_confusion_matrix(
            val_targets_full,
            val_preds_full,
            class_names if class_names else [str(i) for i in sorted(set(val_targets_full))],
            folders["results"] / "confusion_matrix.png",
        )

    cleanup_ddp()


if __name__ == "__main__":
    main()
