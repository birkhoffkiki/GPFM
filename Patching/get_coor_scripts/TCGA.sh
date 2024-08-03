export OPENCV_IO_MAX_IMAGE_PIXELS=10995116277760
datatype="UCEC"


save_dir="/storage/Pathology/Patches/TCGA__"$datatype
source_dir="/jhcnas3/Pathology/original_data/TCGA/"$datatype"/"
wsi_format="svs"
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
        --stitch > $datatype".txt" 2>&1 &
