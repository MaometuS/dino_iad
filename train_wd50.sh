#!/bin/bash
#SBATCH --job-name=DION_IAD
###########RESOURCES###########
#SBATCH --partition=48-4
#SBATCH --gres=gpu:4
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
###############################
#SBATCH --output=TEST.out
#SBATCH --error=TEST.err
#SBATCH -v

source ~/anaconda3/etc/profile.d/conda.sh
conda activate dino_iad
export NCCL_DEBUG=INFO
srun torchrun --nnodes=1 --nproc_per_node=4 main_dino.py --arch wide_resnet50_2 --optimizer sgd --lr 0.03 --weight_decay 1e-4 --weight_decay_end 1e-4 --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14 --data_path /po1/rakhimov/extracted --output_dir /po1/rakhimov/result_wd50 --epochs 10
conda deactivate
