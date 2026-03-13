#!/bin/bash
#SBATCH --job-name=forgetting
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd /home/users/agromyko/websleuths-229
export TOKENIZERS_PARALLELISM=false

python3 scripts/run_dreaming_forgetting.py --model Qwen/Qwen2.5-3B
