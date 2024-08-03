export OPENCV_IO_MAX_IMAGE_PIXELS=10995116277760
# configuration

declare -A wsi_roots
wsi_roots["CAMELYON16"]="/jhcnas3/Pathology/original_data/CAMELYON16/WSIs"
wsi_roots["CAMELYON17"]="/jhcnas3/Pathology/original_data/CAMELYON17/images"


datasets="CAMELYON16 CAMELYON17"

for dataset_name in $datasets
do
        root="/storage/Pathology/Patches/"$dataset_name
        datatype="auto"
        wsi_format="tif"
        level=0
        size=512
        cpu_cores=48
        h5_root=$root"/patches"
        save_root=$root"/images"
        wsi_root=${wsi_roots[$dataset_name]}

        python extract_images.py \
                --datatype $datatype \
                --wsi_format $wsi_format \
                --level $level \
                --cpu_cores $cpu_cores \
                --h5_root $h5_root \
                --save_root $save_root \
                --wsi_root $wsi_root
done