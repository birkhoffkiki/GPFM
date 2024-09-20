# COAD
task="COAD"
root_dir="/jhcnas3/Pathology/CLAM/extract_scripts/"
ramdisk_cache="/mnt/ramdisk/tmp"

DIR_TO_COORDS="/jhcnas3/Pathology/Patches/TCGA__COAD"
DATA_DIRECTORY="/jhcnas2/home/zhoufengtao/data/TCGA/TCGA-COAD/slides"
CSV_FILE_NAME="/jhcnas3/Pathology/CLAM/dataset_csv/COAD.csv"
FEATURES_DIRECTORY="/jhcnas3/Pathology/Patches/TCGA__COAD"
ext=".svs"
save_storage="yes"

models="ctranspath"
declare -A gpus
gpus["ctranspath"]=0

datatype="tcga" # extra path process for TCGA dataset, direct mode do not care use extra path


for model in $models
do
        echo $model", GPU is:"${gpus[$model]}
        export CUDA_VISIBLE_DEVICES=${gpus[$model]}

        nohup python3 extract_features_fp_fast.py \
                --data_h5_dir $DIR_TO_COORDS \
                --data_slide_dir $DATA_DIRECTORY \
                --csv_path $CSV_FILE_NAME \
                --feat_dir $FEATURES_DIRECTORY \
                --batch_size 128 \
                --model $model \
                --datatype $datatype \
                --slide_ext $ext \
                --save_storage $save_storage \
                --ramdisk_cache $ramdisk_cache > $root_dir$task"_log_$model.txt" 2>&1 &

done