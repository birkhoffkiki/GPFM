# BRCA
# save log
prefix="/jhcnas3"

root_dir="extract_scripts/logs/"
use_cache="no"
# mkdir $ramdisk_cache

# models="phikon"
models="distill_379999_cls_only"

tasks="BRCA"
declare -A gpus
gpus["dinov2_vitl"]=0
gpus["dinov2_vitl16_split1"]=4
gpus["dinov2_vitl14_split1"]=0
gpus["phikon"]=1
gpus["plip"]=2
gpus["uni"]=1
gpus["conch"]=0
gpus["distill_87499"]=0
gpus["distill_99999"]=0
gpus["distill_379999_cls_only"]=0

for model in $models
do
        for task in $tasks
        do
                DIR_TO_COORDS=$prefix"/Pathology/Patches/TCGA__"$task
                DATA_DIRECTORY="/jhcnas3/Pathology/original_data/TCGA/"$task"/slides"
                CSV_FILE_NAME="dataset_csv/BRCA.csv"
                FEATURES_DIRECTORY=$prefix"/Pathology/Patches/TCGA__"$task
                ext=".svs"
                save_storage="yes"
                datatype="tcga" # extra path process for TCGA dataset, direct mode do not care use extra path

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
                        --save_storage $save_storage > $root_dir$task"_log_$model.log" 2>&1 &
        done
done
