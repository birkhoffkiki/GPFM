# LUAD 
root_dir="/storage/Pathology/codes/EasyMIL/extract_scripts/logs/COAD_log_"
DIR_TO_COORDS="/storage/Pathology/Patches/TCGA__COAD"
DATA_DIRECTORY="/jhcnas3/Pathology/original_data/TCGA/COAD"
CSV_FILE_NAME="/storage/Pathology/codes/EasyMIL/dataset_csv/COAD.csv"
FEATURES_DIRECTORY="/storage/Pathology/Patches/TCGA__COAD"
ext=".svs"
save_storage="yes"
use_cache="no"

models="distill_379999_cls_only"

declare -A gpus
gpus["phikon"]=4
gpus['ctranspath']=0
gpus['plip']=7
gpus["dinov2_vitl"]=1
gpus["distill_87499"]=0
gpus["distill_99999"]=0
gpus["distill_379999_cls_only"]=0
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
                --use_cache $use_cache \
                --batch_size 64 \
                --model $model \
                --datatype $datatype \
                --slide_ext $ext \
                --save_storage $save_storage > $root_dir$task"_log_$model.txt" 2>&1 &

done