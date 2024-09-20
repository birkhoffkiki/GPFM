# KIRC
# save log
root_dir="/storage/Pathology/codes/CLAM/extract_scripts/logs/KIRP_log_"

ramdisk_cache="/home/gzr/tmp/tmp_kirp"

DIR_TO_COORDS="/storage/Pathology/Patches/TCGA__KIRP"
DATA_DIRECTORY="/jhcnas3/Pathology/original_data/TCGA/KIRP/slides"
CSV_FILE_NAME="/storage/Pathology/codes/CLAM/dataset_csv/KIRP.csv"
FEATURES_DIRECTORY="/storage/Pathology/Patches/TCGA__KIRP"
ext=".svs"
save_storage="yes"

# mae_checkpoint='/jhcnas3/Pathology/outputs_vit_l_resume/checkpoint-1-40000.pth'

# model="ctranspath"
# model="mae_vit_large_patch16"

# model="vit_large_patch16_224_21k"
# model="vit_base_patch16_224_21k"
# model="resnet101"
# model="resnet50"

# models="mae_vit_l_1000slides_19epoch mae_vit_l_10000slides_3epoch"

models="dinov2_vitl"

datatype="tcga" # extra path process for TCGA dataset, direct mode do not care use extra path

declare -A gpus
gpus["mae_vit_l_1000slides_19epoch"]=2
gpus["mae_vit_l_10000slides_3epoch"]=3
gpus["dinov2_vitl"]=3

# for model in $models
# do
#         echo $model", GPU is:"${gpus[$model]}
#         export CUDA_VISIBLE_DEVICES=${gpus[$model]}

#         nohup python3 extract_features_fp_fast.py \
#                 --data_h5_dir $DIR_TO_COORDS \
#                 --data_slide_dir $DATA_DIRECTORY \
#                 --csv_path $CSV_FILE_NAME \
#                 --feat_dir $FEATURES_DIRECTORY \
#                 --batch_size 64 \
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

        nohup python3 extract_features_fp.py \
                --data_h5_dir $DIR_TO_COORDS \
                --data_slide_dir $DATA_DIRECTORY \
                --csv_path $CSV_FILE_NAME \
                --feat_dir $FEATURES_DIRECTORY \
                --batch_size 64 \
                --model $model \
                --datatype $datatype \
                --slide_ext $ext \
                --save_storage $save_storage > $root_dir"$model.txt" 2>&1 &

done
