# Camelyon
# save log
root_dir="/storage/Pathology/codes/CLAM/extract_scripts"
ramdisk_cache="/mnt/ramdisk/AGGC2022"

models="dinov2_vitl"
tasks="AGGC2022"

declare -A gpus
gpus["dinov2_vitl"]=2

for model in $models
do
        for task in $tasks
        do
                DIR_TO_COORDS="/storage/Pathology/Patches/"$task
                DATA_DIRECTORY="/jhcnas3/Pathology/original_data/AGGC2022"
                CSV_FILE_NAME="/storage/Pathology/codes/CLAM/dataset_csv/temporty_csv/AGGC2022.csv"
                FEATURES_DIRECTORY="/storage/Pathology/Patches/"$task
                ext=".tiff"
                save_storage="yes"
                datatype="auto" # extra path process for TCGA dataset, direct mode do not care use extra path

                echo $model", GPU is:"${gpus[$model]}
                export CUDA_VISIBLE_DEVICES=${gpus[$model]}

                nohup python3 extract_features_fp_fast.py \
                        --data_h5_dir $DIR_TO_COORDS \
                        --data_slide_dir $DATA_DIRECTORY \
                        --csv_path $CSV_FILE_NAME \
                        --feat_dir $FEATURES_DIRECTORY \
                        --batch_size 64 \
                        --model $model \
                        --datatype $datatype \
                        --slide_ext $ext \
                        --save_storage $save_storage \
                        --ramdisk_cache $ramdisk_cache > $root_dir"/logs/"$task"_log_$model.txt" 2>&1 &
        done
done
