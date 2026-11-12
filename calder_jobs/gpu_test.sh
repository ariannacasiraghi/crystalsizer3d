#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --partition=calder-gpu-hopper
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=00:10:00
#SBATCH --output=/scratch-calder/users/jcmt0720/job_logs/%x_%j.out

module load calder/cpu miniforge3/26.1.1-3
conda activate crystalsizer3d

hostname
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
