# Camelyon
# save log
prefix="/jhcnas3"

root_dir="extract_scripts/"

tasks="CAMELYON17"

# models="conch"
models="dinov2_vitl14_split1 dinov2_vitl16_split1"

use_cache="no"

declare -A gpus
gpus["phikon"]=2
gpus["plip"]=6
gpus["conch"]=3
gpus["dinov2_vitl"]=5
gpus["dinov2_vitl16_split1"]=1
gpus["dinov2_vitl14_split1"]=0

declare -A wsi_roots
wsi_roots["CAMELYON16"]="/jhcnas3/Pathology/original_data/CAMELYON16/WSIs"
wsi_roots["CAMELYON17"]="/jhcnas3/Pathology/original_data/CAMELYON17/images"

for model in $models
do
        for task in $tasks
        do
                ramdisk_cache="/mnt/ramdisk/"$task
                DIR_TO_COORDS=$prefix"/Pathology/Patches/"$task
                DATA_DIRECTORY=${wsi_roots[$task]}
                CSV_FILE_NAME="dataset_csv/camelyon.csv"
                FEATURES_DIRECTORY=$prefix"/Pathology/Patches/"$task
                ext=".tif"
                save_storage="yes"
                datatype="direct" # extra path process for TCGA dataset, direct mode do not care use extra path

                echo $model", GPU is:"${gpus[$model]}
                export CUDA_VISIBLE_DEVICES=${gpus[$model]}

                nohup python3 extract_features_fp_fast.py \
                        --data_h5_dir $DIR_TO_COORDS \
                        --data_slide_dir $DATA_DIRECTORY \
                        --csv_path $CSV_FILE_NAME \
                        --feat_dir $FEATURES_DIRECTORY \
                        --batch_size 24 \
                        --model $model \
                        --datatype $datatype \
                        --use_cache $use_cache \
                        --slide_ext $ext \
                        --save_storage $save_storage \
                        --ramdisk_cache $ramdisk_cache > $root_dir"/logs/"$task"_log_$model.log" 2>&1 &
        done
done
