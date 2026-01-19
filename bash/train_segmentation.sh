#!/bin/bash
#SBATCH -A hyperk
#SBATCH -p gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=256G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=20:00:00
#SBATCH -J multiring-seg
#SBATCH -o logs/slurm-%x-%j.out
#SBATCH -e logs/slurm-%x-%j.err

set -euo pipefail

echo "Job: ${SLURM_JOB_NAME:-} (${SLURM_JOB_ID:-}) on $(hostname)"
nvidia-smi || true

IMAGE="/sps/hyperk/zhu/CAVERNS/env/ml_image.sif"
BIND_CODE="/sps/hyperk/zhu/CAVERNS/NeuralNetworks_Software:/workspace/work/ml"
BIND_DATA="/sps/hyperk/melbaz:/workspace/work/data"
WORKDIR="/workspace/work/ml"

export WANDB_API_KEY="a8ee790752c18322d9a8b3fae9fb54313106a3ec"

WANDB_ROOT="/sps/hyperk/zhu/CAVERNS/NeuralNetworks_Software/bash/wandb"

mkdir -p "$WANDB_ROOT" \
         "$WANDB_ROOT/cache" \
         "$WANDB_ROOT/config" \
         "$WANDB_ROOT/data" \
         "$WANDB_ROOT/artifacts"

export WANDB_DIR="$WANDB_ROOT"
export WANDB_CACHE_DIR="$WANDB_ROOT/cache"
export WANDB_CONFIG_DIR="$WANDB_ROOT/config"
export WANDB_DATA_DIR="$WANDB_ROOT/data"
export WANDB_ARTIFACT_DIR="$WANDB_ROOT/artifacts"
export WANDB_MODE=offline

srun apptainer exec --nv \
  --bind "$BIND_CODE" \
  --bind "$BIND_DATA" \
  --pwd  "$WORKDIR" \
  "$IMAGE" \
  bash -lc 'SPCONV_ALGO=native CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python -m main_multiring --config-name segmentation'
