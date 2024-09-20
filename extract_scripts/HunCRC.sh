# LUAD 
root_dir="/storage/Pathology/codes/CLAM/extract_scripts/logs/HunCRC_log_"
DIR_TO_COORDS="/storage/Pathology/Patches/HunCRC"
DATA_DIRECTORY="/jhcnas3/Pathology/original_data/HunCRC/files"
CSV_FILE_NAME="/storage/Pathology/codes/CLAM/dataset_csv/temporty_csv/HunCRC.csv"
FEATURES_DIRECTORY="/storage/Pathology/Patches/HunCRC"
# DIR_TO_COORDS="/jhcnas3/Pathology/Patches/TCGA__BRCA"
# DATA_DIRECTORY="/jhcnas3/Pathology/original_data/TCGA/BRCA/slides"
# CSV_FILE_NAME="/jhcnas3/Pathology/CLAM/dataset_csv/BRCA.csv"
# FEATURES_DIRECTORY="/jhcnas3/Pathology/Patches/TCGA__BRCA"
ext=".mrxs"
save_storage="yes"
# ramdisk_cache="/home/gzr/tmp/tmp_brca"
# ramdisk_cache="/mnt/ramdisk/tmp"

# model="vit_large_patch16_224_21k"
# model="vit_base_patch16_224_21k"
# model="resnet101"
# model="resnet50"
# models="ctranspath"
# models="mae_vit_large_patch16-1-140000"
# models="mae_vit_l_1000slides_19epoch"
# models="mae_vit_huge_patch14_1000slides_22epoch"
models="dinov2_vitl"

declare -A gpus
gpus["mae_vit_l_1000slides_19epoch"]=0
gpus['ctranspath']=0
gpus['mae_vit_huge_patch14_1000slides_22epoch']=7
gpus["dinov2_vitl"]=2
# models="mae_vit_large_patch16-1-140000"

datatype="direct" # extra path process for TCGA dataset, direct mode do not care use extra path

# for model in $models
# do

#         echo $model", GPU is:"${gpus[$model]}
#         export CUDA_VISIBLE_DEVICES=${gpus[$model]}
#         cache_root=$ramdisk_cache"/"$model
#         nohup python3 extract_features_fp_fast.py \
#                 --data_h5_dir $DIR_TO_COORDS \
#                 --data_slide_dir $DATA_DIRECTORY \
#                 --csv_path $CSV_FILE_NAME \
#                 --feat_dir $FEATURES_DIRECTORY \
#                 --batch_size 32 \
#                 --model $model \
#                 --datatype $datatype \
#                 --slide_ext $ext \
#                 --save_storage $save_storage \
#                 --ramdisk_cache $cache_root > $root_dir"$model.txt" 2>&1 &
# done

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
                --batch_size 32 \
                --model $model \
                --datatype $datatype \
                --slide_ext $ext \
                --save_storage $save_storage > $root_dir"$model.txt" 2>&1 &

done