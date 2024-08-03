

cpu_cores=48
size=512
save_root="/jhcnas3/Pathology/Patches/Osteosarcoma_Tumor/images"
wsi_root="/jhcnas3/Pathology/original_data/Osteosarcoma_Tumor/Osteosarcoma-UT"
prefix="crop_"
format="jpg"
resize_flag="no"

python extract_images_4_non_wsi.py \
        --cpu_cores $cpu_cores \
        --size $size \
        --save_root $save_root \
        --wsi_root $wsi_root \
        --prefix $prefix \
        --format $format \
        --resize_flag $resize_flag