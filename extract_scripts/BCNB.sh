# save log
dataset="BCNB"
gpu=5
ext=".jpg"
DATA_DIRECTORY="/jhcnas3/Pathology/original_data/BCNB/WSIs"


#---------------------------------------
export OPENCV_IO_MAX_IMAGE_PIXELS=10995116277760
root_dir="/storage/Pathology/codes/EasyMIL/extract_scripts/"
# ramdisk_cache="/mnt/ramdisk/"$dataset
ramdisk_cache="/home/gzr/tmp/"$dataset
# models="resnet50 uni phikon plip" #dinov2_vitl is saved on /storage!!!
models="distill_87499"
# models="conch"
declare -A gpus
gpus["dinov2_vitl"]=1
gpus["resnet50"]=2
gpus["uni"]=2
gpus["phikon"]=2
gpus["plip"]=2
gpus["conch"]=6
gpus["ctranspath"]=7
gpus['distill_87499']=3
for model in $models
do
        # DIR_TO_COORDS="/storage/Pathology/Patches/"$dataset
        DIR_TO_COORDS="/jhcnas3/Pathology/Patches/"$dataset
        CSV_FILE_NAME="/storage/Pathology/codes/CLAM/dataset_csv/temporty_csv/"$dataset".csv"
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
                --datatype $datatype \
                --slide_ext $ext \
                --save_storage $save_storage \
                --ramdisk_cache $cache_root > $root_dir"/logs/"$dataset"_"$model".txt" 2>&1 &
done
