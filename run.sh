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
srun torchrun --nnodes=1 --nproc_per_node=4 main_dino.py --arch vit_base --data_path /po1/rakhimov/extracted --output_dir /po1/rakhimov/result --epochs 100
conda deactivate
