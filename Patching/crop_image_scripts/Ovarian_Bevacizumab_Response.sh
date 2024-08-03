# template
# configuration
dataset="Ovarian_Bevacizumab_Response"
wsi_root="/jhcnas3/Pathology/original_data/Ovarian_Bevacizumab_Response"
wsi_format="svs"

log_path="./crop_image_scripts/"$dataset".log"
root="/storage/Pathology/Patches/"$dataset
datatype="auto"
level=0
size=512
cpu_cores=48
h5_root=$root"/patches"
save_root=$root"/images"
export OPENCV_IO_MAX_IMAGE_PIXELS=10995116277760

nohup python extract_images.py \
        --datatype $datatype \
        --wsi_format $wsi_format \
        --level $level \
        --cpu_cores $cpu_cores \
        --h5_root $h5_root \
        --save_root $save_root \
        --wsi_root $wsi_root > $log_path 2>&1 &