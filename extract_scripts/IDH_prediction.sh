# save log
dataset="IDH_prediction"
# gpu=1
ext=".svs"
DATA_DIRECTORY="/jhcnas3/Pathology/original_data/IDH_prediction/"
use_cache="no"

#---------------------------------------
root_dir="/storage/Pathology/codes/EasyMIL/extract_scripts/"
ramdisk_cache="/home/gzr/"$dataset
# models="resnet50 uni phikon plip" #dinov2_vitl is saved on /storage!!!
models="distill_87499"
declare -A gpus
gpus["dinov2_vitl"]=1
gpus["resnet50"]=1
gpus["uni"]=1
gpus["phikon"]=1
gpus["plip"]=1
gpus["conch"]=1
gpus["ctranspath"]=7
gpus['distill_87499']=3
for model in $models
do
        # DIR_TO_COORDS="/storage/Pathology/Patches/"$dataset
        DIR_TO_COORDS="/jhcnas3/Pathology/Patches/"$dataset
        CSV_FILE_NAME="/storage/Pathology/codes/EasyMIL/dataset_csv/temporty_csv/"$dataset".csv"
        # FEATURES_DIRECTORY="/storage/Pathology/Patches/"$dataset
        FEATURES_DIRECTORY="/jhcnas3/Pathology/Patches/"$dataset
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
                --batch_size 128 \
                --model $model \
                --use_cache $use_cache \
                --datatype $datatype \
                --slide_ext $ext \
                --save_storage $save_storage \
                --ramdisk_cache $cache_root > $root_dir"/logs/"$dataset"_"$model".txt" 2>&1 &
done
