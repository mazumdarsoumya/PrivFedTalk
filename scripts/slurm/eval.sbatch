#!/bin/bash
#SBATCH --job-name=privfedtalk_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -e
source ~/.bashrc
conda activate privfedtalk
python -m privfedtalk.cli.eval --config configs/default.yaml
