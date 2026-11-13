#!/bin/bash
#SBATCH --job-name=import_test
#SBATCH --partition=calder-gpu-hopper
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=00:15:00
#SBATCH --output=/scratch-calder/users/%u/job_logs/%x_%j.out

module load calder/cpu miniforge3/26.1.1-3
conda activate crystalsizer3d

cd ~/crystalsizer3d
export USE_CUDA=1
export USE_MLAB=0
export DATA_PATH=/scratch-calder/users/$USER/crystalsizer3d_data

python calder_jobs/import_test.py
