#!/bin/bash
#SBATCH --job-name=full_100k_w3d0
#SBATCH --partition=calder-gpu-hopper
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch-calder/users/%u/job_logs/%x_%j.out

module load calder/cpu miniforge3/26.1.1-3
conda activate crystalsizer3d

cd ~/crystalsizer3d
export USE_CUDA=1
export USE_MLAB=0
export DATA_PATH=/scratch-calder/users/$USER/crystalsizer3d_data

nvidia-smi
python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

python scripts/train_synthetic.py \
  --dataset-path $HOME/data_calder/support_materials \
  --use-gpu true --seed 0 \
  --batch-size 32 --n-epochs 100 \
  --n-dataloader-workers 8 --prefetch-factor 3 \
  --log-every-n-batches 100 \
  --plot-n-examples 0 --plot-every-n-batches=-1 \
  --augment true \
  --lr-init 1e-4 --lr-pretrained-init 1e-4 --lr-min 1e-5 --lr-decay-epochs 10 \
  --weight-decay 1e-3 --clip-grad-norm 1 \
  --w-material 0.1 --w-light 0.1 --w-3d 0 \
  --train-zingg false --train-generator false \
  --use-canonical-rotations true \
  timm --model-name regnetz_d8.ra3_in1k --resize-input false
