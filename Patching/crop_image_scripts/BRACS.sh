export OPENCV_IO_MAX_IMAGE_PIXELS=10995116277760
# configuration
root="/storage/Pathology/Patches/BRACS"
datatype="auto"
wsi_format="svs"
level=0
size=512
cpu_cores=48
h5_root=$root"/patches"
save_root=$root"/images"
wsi_root="/jhcnas3/BRACS/BRACS_WSI"

python extract_images.py \
        --datatype $datatype \
        --wsi_format $wsi_format \
        --level $level \
        --cpu_cores $cpu_cores \
        --h5_root $h5_root \
        --save_root $save_root \
        --wsi_root $wsi_root
