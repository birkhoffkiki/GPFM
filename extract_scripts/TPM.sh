# save log
dataset="TPM"
gpu=1
ext=".png"
use_cache="no"
save_storage="yes"
DATA_DIRECTORY="/jhcnas5/gzr/data/TPM/slides_png"

#---------------------------------------
root_dir="/storage/Pathology/codes/EasyMIL/extract_scripts/logs"
# ramdisk_cache="/mnt/ramdisk/"$dataset
models="resnet50"
gpus["resnet50"]=$gpu
for model in $models
do
        DIR_TO_COORDS="/jhcnas5/gzr/data/"$dataset
        CSV_FILE_NAME="/storage/Pathology/codes/EasyMIL/dataset_csv/temporty_csv/"$dataset".csv"
        FEATURES_DIRECTORY="/jhcnas5/gzr/data/"$dataset
        save_storage="yes"
        datatype="auto" # extra path process for TCGA dataset, direct mode do not care use extra path

        echo $model", GPU is:"${gpus[$model]}
        export CUDA_VISIBLE_DEVICES=${gpus[$model]}
        cache_root=$ramdisk_cache"/"$model
        nohup python3 extract_features_fp_fast.py \
                --data_h5_dir $DIR_TO_COORDS \
                --data_slide_dir $DATA_DIRECTORY \
                --csv_path $CSV_FILE_NAME \
                --feat_dir $FEATURES_DIRECTORY \
                --batch_size 64 \
                --use_cache $use_cache \
                --model $model \
                --datatype $datatype \
                --slide_ext $ext \
                --save_storage $save_storage > $root_dir"/"$dataset"_"$model".txt" 2>&1 &
done
