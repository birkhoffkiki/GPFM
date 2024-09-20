# save log
dataset="BACH"
gpu=2
ext=".svs"
DATA_DIRECTORY="/jhcnas3/Pathology/original_data/BACH/ICIAR2018_BACH_Challenge/WSI"


#---------------------------------------
root_dir="/storage/Pathology/codes/CLAM/extract_scripts/"
ramdisk_cache="/mnt/ramdisk/"$dataset
models="dinov2_vitl"
gpus["dinov2_vitl"]=$gpu
for model in $models
do
        DIR_TO_COORDS="/storage/Pathology/Patches/"$dataset
        CSV_FILE_NAME="/storage/Pathology/codes/CLAM/dataset_csv/temporty_csv/"$dataset".csv"
        FEATURES_DIRECTORY="/storage/Pathology/Patches/"$dataset
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
                --batch_size 32 \
                --model $model \
                --datatype $datatype \
                --slide_ext $ext \
                --save_storage $save_storage \
                --ramdisk_cache $cache_root > $root_dir"/logs/"$dataset"_"$model".txt" 2>&1 &
done
