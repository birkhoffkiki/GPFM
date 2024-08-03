export OPENCV_IO_MAX_IMAGE_PIXELS=10995116277760

item="UBC-OCEAN"
save_dir="/storage/Pathology/Patches/"$item
# source_dir="/jhcnas3/Pathology/original_data/"$item
source_dir="/jhcnas3/Pathology/original_data/UBC-OCEAN/WSIs"
wsi_format="png"
patch_size=512

nohup python create_patches_fp.py \
        --source $source_dir \
        --save_dir $save_dir\
        --preset tcga.csv \
        --patch_level 0 \
        --patch_size $patch_size \
        --step_size $patch_size \
        --wsi_format $wsi_format \
        --seg \
        --patch \
        --stitch  > $item".txt" 2>&1 &
