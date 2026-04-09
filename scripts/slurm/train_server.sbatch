#!/bin/bash
#SBATCH --job-name=privfedtalk_server
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -e
source ~/.bashrc
conda activate privfedtalk
python -m privfedtalk.cli.train_federated --config configs/default.yaml
