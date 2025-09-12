#!/bin/bash
#SBATCH --job-name=FlatBundleNgbrsMatch
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-1

source /home/parentd1/.bashrc
conda activate cooperative_sheaves

export WANDB_API_KEY=ab6e83a9d99012b14cff411df4e314597d9e6898
export WANDB__SERVICE_WAIT=300

SWEEP_IDS=(
  "andrerg00/NeighborsMatch/jtj3pgbv"  # r=7
  "andrerg00/NeighborsMatch/09rt2gwl"  # r=8
)

NUM_SWEEPS=${#SWEEP_IDS[@]}

IDX=$(( SLURM_ARRAY_TASK_ID % NUM_SWEEPS ))
export SWEEP_ID=${SWEEP_IDS[$IDX]}

wandb agent "$SWEEP_ID"
