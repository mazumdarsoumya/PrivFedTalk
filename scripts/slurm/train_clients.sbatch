#!/bin/bash
#SBATCH --job-name=privfedtalk_clients
#SBATCH --array=0-19
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -e
source ~/.bashrc
conda activate privfedtalk
python scripts/run_client.py --config configs/default.yaml --client_id $SLURM_ARRAY_TASK_ID
