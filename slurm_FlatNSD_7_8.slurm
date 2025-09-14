#!/bin/bash
#SBATCH --job-name=FlatBundleNgbrsMatch
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=7
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-1

source /home/parentd1/.bashrc
conda activate cooperative_sheaves

export WANDB_API_KEY=ab6e83a9d99012b14cff411df4e314597d9e6898
export WANDB__SERVICE_WAIT=300

DEPTHS=(7 8)
DEPTH=${DEPTHS[$SLURM_ARRAY_TASK_ID]}

srun python run-fnsd-2-8.py --depth "$DEPTH"
