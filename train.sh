#!/bin/bash
#SBATCH --job-name=alexnet_pretrained_cifar10_%j  # Job name
#SBATCH --output=alexnet_pretrained_%j.txt    # Output file (%j is replaced with job ID)
#SBATCH --error=error_%j.txt      # Error file (%j is replaced with job ID)
#SBATCH --ntasks-per-node=2                # Number of tasks (jobs)
#SBATCH --time=03:00:00           # Maximum runtime (1 hour)
#SBATCH --partition=Lab2080       # Partition to submit to
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4

# module load Miniconda3
# conda activate ai_project

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=enp179s0f1
export NCCL_P2P_DISABLE=1

export MASTER_PORT=58100
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# Commands to execute
source ~/miniconda3/bin/activate
conda activate ai_project
echo "Starting the job..."
srun python train.py