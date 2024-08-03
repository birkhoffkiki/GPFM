# AGGC2022
export OPENCV_IO_MAX_IMAGE_PIXELS=10995116277760
# configuration
root="/storage/Pathology/Patches/AGGC2022"
datatype="auto"
wsi_format="tiff"
level=0
size=512
cpu_cores=48
h5_root=$root"/patches"
save_root=$root"/images"
wsi_root="/jhcnas3/Pathology/original_data/AGGC2022"

python extract_images.py \
        --datatype $datatype \
        --wsi_format $wsi_format \
        --level $level \
        --cpu_cores $cpu_cores \
        --h5_root $h5_root \
        --save_root $save_root \
        --wsi_root $wsi_root
