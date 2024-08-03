#!/bin/bash
    
# NOTE: Lines starting with "#SBATCH" are valid SLURM commands or statements,
#       while those starting with "#" and "##SBATCH" are comments.  

exp_name="distill_vitl14_192M_exclude_split1_cls_only"
module load Anaconda3

source activate distill
export PYTHONPATH=/home/jmabq/dinov2-distill
python dinov2/run/train/train.py \
    --nodes 2 \
    --gpus 8 \
    --partition project \
    --config-file "dinov2/configs/train/"$exp_name".yaml" \
    --output-dir /project/vcompath/storage/Pathology/result/$exp_name \

