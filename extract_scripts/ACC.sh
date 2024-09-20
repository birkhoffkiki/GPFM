# BRCA
task="ACC"
root_dir="/storage/Pathology/codes/CLAM/extract_scripts/logs/"$task"_log_"
ramdisk_cache="/home/gzr/tmp/acc"

DIR_TO_COORDS="/storage/Pathology/Patches/TCGA__ACC"
DATA_DIRECTORY="/jhcnas3/Pathology/original_data/TCGA/ACC"
CSV_FILE_NAME="/storage/Pathology/codes/CLAM/dataset_csv/ACC.csv"
FEATURES_DIRECTORY="/storage/Pathology/Patches/TCGA__ACC"
ext=".svs"
save_storage="yes"

models="dinov2_vitl"

declare -A gpus
gpus["mae_vit_l_1000slides_19epoch"]=0
gpus['ctranspath']=0
gpus['mae_vit_huge_patch14_1000slides_22epoch']=7
gpus["dinov2_vitl"]=1


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
                --model $model \
                --datatype $datatype \
                --slide_ext $ext \
                --save_storage $save_storage \
                --ramdisk_cache $ramdisk_cache > $root_dir"$model.txt" 2>&1 &

done