# LUAD 
# DIR_TO_COORDS="/home/jmabq/DATA/Pathology/TCGA__READ-rectum"
DIR_TO_COORDS="/storage/Pathology/Patches/TCGA__READ"
DATA_DIRECTORY="/jhcnas3/Pathology/original_data/TCGA/READ"
CSV_FILE_NAME="/storage/Pathology/codes/EasyMIL/dataset_csv/READ.csv"
FEATURES_DIRECTORY="/storage/Pathology/Patches/TCGA__READ"
ext=".svs"
save_storage="yes"
# ramdisk_cache="/mnt/home/gzr/tmp"
ramdisk_cache="/mnt/ramdisk/read"
root_dir="/storage/Pathology/codes/EasyMIL/extract_scripts/logs/READ_log_"
use_cache='no'

models="distill_379999_cls_only"

declare -A gpus
gpus["resnet50"]=2
gpus["ctranspath"]=0
gpus["plip"]=4
gpus["phikon"]=1
gpus["dinov2_vitl"]=1
gpus["distill_87499"]=2
gpus["distill_99999"]=0
gpus["distill_379999_cls_only"]=0
datatype="tcga" # extra path process for TCGA dataset, direct mode do not care use extra path

# for model in $models
# do
#         echo $model", GPU is:"${gpus[$model]}
#         export CUDA_VISIBLE_DEVICES=${gpus[$model]}

#         nohup python3 extract_features_fp_fast.py \
#                 --data_h5_dir $DIR_TO_COORDS \
#                 --data_slide_dir $DATA_DIRECTORY \
#                 --csv_path $CSV_FILE_NAME \
#                 --feat_dir $FEATURES_DIRECTORY \
#                 --batch_size 128 \
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

        nohup python3 extract_features_fp_fast.py \
                --data_h5_dir $DIR_TO_COORDS \
                --data_slide_dir $DATA_DIRECTORY \
                --csv_path $CSV_FILE_NAME \
                --feat_dir $FEATURES_DIRECTORY \
                --batch_size 32 \
                --use_cache $use_cache \
                --model $model \
                --datatype $datatype \
                --slide_ext $ext \
                --save_storage $save_storage \
                --ramdisk_cache '' > $root_dir$task"_log_$model.txt" 2>&1 &

done
