#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --output=/mnt/scratch/jcmt0720/job_logs/%x_%j.out
#SBATCH --error=/mnt/scratch/jcmt0720/job_logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load miniforge/24.7.1
conda activate crystalsizer3d

echo "Node: $HOSTNAME"
nvidia-smi
python -c "
import torch
print('torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda version:', torch.version.cuda)
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    x = torch.randn(1000, 1000, device='cuda')
    print('matmul ok:', (x @ x).shape)
"
