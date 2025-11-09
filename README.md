# IE4483 Dog vs Cat (ConvNeXt)

ConvNeXt-based binary classifier for the classic Dogs vs Cats problem used in IE4483. The training code supports single-GPU debugging as well as multi-node jobs on NSCC via PyTorch Distributed Data Parallel (DDP).

## Data Flow

1. **Raw images** live under `data.root_dir` (see `config.yaml`). If the directory already contains `train/`, `val/`, and `test/`, the code uses them directly; otherwise, `splitter.py` produces deterministic splits from a flat `Cat/` + `Dog/` folder.
2. **Transforms** from `augmentation.py` normalize images to ImageNet stats, add random crops/jitter/flip during training, and use deterministic center crops for validation/test.
3. **Dataloaders** in `dataloader.py` wrap the torchvision `ImageFolder` datasets, create per-split `DataLoader`s, and build `DistributedSampler`s when `ddp.enabled` is true or when Slurm provides `WORLD_SIZE`.
4. **Model** is instantiated in `model.py` (ConvNeXt Tiny by default) and optionally loaded with ImageNet weights.
5. **Training loop** (`train.py`) applies mixup/cutmix, mixed precision (AMP), cosine LR scheduling, early stopping, and periodically saves checkpoints, metrics, and TensorBoard logs inside `project.output_root/run_name`.
6. **Evaluation** (`test.py`) reloads `best.pt`, reports aggregate/per-class accuracy, and writes CSV + plots to the same results directory (which stays local because `results/` is listed in `.gitignore`).

## Environment Setup

```bash
conda create -n dogcat python=3.10 -y
conda activate dogcat
pip install -r requirements.txt
```

Key configuration lives in `config.yaml`. Before running, update:

- `project.output_root` – where checkpoints/logs/results are written (keep it on a fast filesystem on NSCC).
- `data.root_dir` – path to the PetImages folder (either flat `Cat/` & `Dog/` or the `_split` version created by `splitter.py`).
- `sched.max_epochs`, `optim.lr`, etc., to match your experiment.

## Running Locally

1. (Optional) Create an explicit train/val/test split:
   ```bash
   python splitter.py
   ```
   This generates `<root_dir>_split` with three subfolders per class; point `data.root_dir` there afterward.
2. Kick off training:
   ```bash
   python train.py
   ```
   Checkpoints, TensorBoard logs, accuracy/loss curves, and confusion matrices are written under `project.output_root/run_name/`.
3. Evaluate the best checkpoint:
   ```bash
   python test.py
   ```

## Running on NSCC (SLURM)

1. Copy the project and dataset to your NSCC workspace (match the paths in `config.yaml`).
2. Load your environment (the sample `train.sh` activates a `ai_project` Conda env—adjust as needed) and ensure NCCL env vars match the interconnect provided to you.
3. Submit the distributed job:
   ```bash
   sbatch train.sh
   ```
   The script derives `MASTER_ADDR`, `WORLD_SIZE`, and launches `srun python train.py`, so DDP automatically shards each split via `DistributedSampler`. Logs and stdout/stderr end up in the files declared at the top of `train.sh`.
4. After the job finishes, run `python test.py` on the head/login node (or within another batch job) to generate evaluation metrics from `best.pt`.

## Notes

- The `results/` directory in this repo is ignored by Git so local experiment outputs are never committed by accident.
- For reproducibility across nodes, every run sets `project.seed` and controls torch/cuDNN determinism; keep transforms/mixup configs synchronized when comparing NSCC results.
- Use `tensorboard --logdir /path/to/output_root/<run>/logs` to inspect training curves interactively.
