DIR_TO_COORDS="/storage/Pathology/wsi-report/wsi4report"
DATA_DIRECTORY="/jhcnas3/Pathology/original_data/wsi-report/slides"
CSV_FILE_NAME="/storage/Pathology/codes/EasyMIL/dataset_csv/wsi-report-data_no_duplicate.csv"
# FEATURES_DIRECTORY="/jhcnas1/gzr/wsi4report"
FEATURES_DIRECTORY="/jhcnas4/gzr/wsi4report"
# ramdisk_cache="/home/gzr/tmp/plip/"
ext=".svs"
use_cache="no"
save_storage="yes"
root_dir="extract_scripts/logs/WSI-Report_log_"

# models="resnet50 resnet101 vit_base_patch16_224_21k vit_large_patch16_224_21k mae_vit_large_patch16-1-40000 mae_vit_large_patch16-1-140000"
# models="mae_vit_large_patch16-1-40000 mae_vit_large_patch16-1-140000"
# models="ctranspath"
# models="mae_vit_l_1000slides_19epoch"
# model="vit_base_patch16_224_21k"
# model="resnet101"
# models="resnet50"
# models="resnet50"
# models="ctranspath"
# models="uni phikon plip" # also change to conch
models="distill_87499"
# models="phikon"

declare -A gpus
gpus["resnet50"]=4
gpus["resnet101"]=0
gpus["ctranspath"]=5
gpus["dinov2_vitl"]=1
gpus['plip']=6
gpus['conch']=4
gpus['uni']=7
gpus['phikon']=5
gpus['distill_87499']=0

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
                --batch_size 64 \
                --use_cache $use_cache \
                --model $model \
                --datatype $datatype \
                --slide_ext $ext \
                --save_storage $save_storage > $root_dir$model".txt" 2>&1 &

done

# for model in $models
# do
#         echo $model", GPU is:"${gpus[$model]}
#         export CUDA_VISIBLE_DEVICES=${gpus[$model]}

#         nohup python3 extract_features_fp.py \
#                 --data_h5_dir $DIR_TO_COORDS \
#                 --data_slide_dir $DATA_DIRECTORY \
#                 --csv_path $CSV_FILE_NAME \
#                 --feat_dir $FEATURES_DIRECTORY \
#                 --batch_size 16 \
#                 --model $model \
#                 --datatype $datatype \
#                 --slide_ext $ext \
#                 --save_storage $save_storage > $root_dir$model".txt" 2>&1 &

# done