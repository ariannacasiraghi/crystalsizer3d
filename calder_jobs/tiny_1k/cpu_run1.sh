#!/bin/bash
#SBATCH --job-name=cpu_run1
#SBATCH --partition=calder-cpu-turin
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch-calder/users/%u/job_logs/%x_%j.out

module load calder/cpu miniforge3/26.1.1-3
conda activate crystalsizer3d

cd ~/crystalsizer3d
export USE_CUDA=0
export USE_MLAB=0
export DATA_PATH=/scratch-calder/users/$USER/crystalsizer3d_data
export DRJIT_LIBLLVM_PATH=/lib64/libLLVM.so.21.1

python scripts/train_synthetic.py \
  --dataset-path $HOME/data_calder/subset_1k \
  --use-gpu false --seed 0 \
  --batch-size 8 --n-epochs 15 --plot-every-n-batches 1000 --n-dataloader-workers 8 \
  --augment true \
  --lr-init 1e-4 --lr-pretrained-init 1e-4 --lr-min 1e-5 --lr-decay-epochs 10 \
  --weight-decay 1e-3 --clip-grad-norm 1 \
  --w-material 0.1 --w-light 0.1 --w-3d 0 \
  --train-zingg false --train-generator false \
  --use-canonical-rotations true \
  timm --model-name regnetz_d8.ra3_in1k --resize-input false