# KIRC
# save log
root_dir="/storage/Pathology/codes/CLAM/extract_scripts/logs/KIRC_log_"

ramdisk_cache="/home/gzr/tmp/tmp_kirc"

DIR_TO_COORDS="/storage/Pathology/Patches/TCGA__KIRC"
DATA_DIRECTORY="/jhcnas3/Pathology/original_data/TCGA/KIRC/slides"
CSV_FILE_NAME="/storage/Pathology/codes/CLAM/dataset_csv/KIRC.csv"
FEATURES_DIRECTORY="/storage/Pathology/Patches/TCGA__KIRC"
ext=".svs"
use_cache="no"
save_storage="yes"

models="dinov2_vitl"


datatype="tcga" # extra path process for TCGA dataset, direct mode do not care use extra path

declare -A gpus
gpus["mae_vit_l_1000slides_19epoch"]=2
gpus["mae_vit_l_10000slides_3epoch"]=3
gpus["dinov2_vitl"]=2

# for model in $models
# do
#         echo $model", GPU is:"${gpus[$model]}
#         export CUDA_VISIBLE_DEVICES=${gpus[$model]}

#         nohup python3 extract_features_fp_fast.py \
#                 --data_h5_dir $DIR_TO_COORDS \
#                 --data_slide_dir $DATA_DIRECTORY \
#                 --csv_path $CSV_FILE_NAME \
#                 --feat_dir $FEATURES_DIRECTORY \
#                 --batch_size 64 \
#                 --model $model \
#                 --datatype $datatype \
#                 --slide_ext $ext \
#                 --save_storage $save_storage \
#                 --ramdisk_cache $ramdisk_cache > $root_dir"$model.txt" 2>&1 &
# done

for model in $models
do
        echo $model", GPU is:"${gpus[$model]}
        export CUDA_VISIBLE_DEVICES=${gpus[$model]}

        nohup python3 extract_features_fp.py \
                --data_h5_dir $DIR_TO_COORDS \
                --data_slide_dir $DATA_DIRECTORY \
                --csv_path $CSV_FILE_NAME \
                --feat_dir $FEATURES_DIRECTORY \
                --batch_size 64 \
                --use_cache $use_cache \
                --model $model \
                --datatype $datatype \
                --slide_ext $ext \
                --save_storage $save_storage > $root_dir"$model.txt" 2>&1 &

done
