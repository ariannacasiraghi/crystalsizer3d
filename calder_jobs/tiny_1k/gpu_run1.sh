#!/bin/bash
#SBATCH --job-name=gpu_run1
#SBATCH --partition=calder-gpu-hopper
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=/scratch-calder/users/%u/job_logs/%x_%j.out

module load calder/cpu miniforge3/26.1.1-3
conda activate crystalsizer3d

cd ~/crystalsizer3d
export USE_CUDA=1
export USE_MLAB=0
export DATA_PATH=/scratch-calder/users/$USER/crystalsizer3d_data

python scripts/train_synthetic.py \
  --dataset-path $HOME/data_calder/subset_1k \
  --use-gpu true --seed 0 \
  --batch-size 32 --n-epochs 5 --n-dataloader-workers 4 \
  --augment true \
  --lr-init 1e-4 --lr-pretrained-init 1e-4 --lr-min 1e-5 --lr-decay-epochs 10 \
  --weight-decay 1e-3 --clip-grad-norm 1 \
  --w-material 0.1 --w-light 0.1 --w-3d 0 \
  --train-zingg false --train-generator false \
  --use-canonical-rotations true \
  timm --model-name regnetz_d8.ra3_in1k --resize-input false