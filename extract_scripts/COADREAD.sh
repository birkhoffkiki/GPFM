# COADREAD
# save log
prefix="/jhcnas3"

root_dir="extract_scripts/logs/"
use_cache="no"

# models="phikon"
models="distill_87499"

tasks="COAD"
declare -A gpus
gpus["dinov2_vitl"]=0
gpus["dinov2_vitl16_split1"]=1
gpus["dinov2_vitl14_split1"]=6
gpus["phikon"]=6
gpus["plip"]=2
gpus["conch"]=2
gpus["uni"]=7
gpus["distill_87499"]=0

for model in $models
do
        for task in $tasks
        do
                DIR_TO_COORDS=$prefix"/Pathology/Patches/TCGA__COADREAD"
                DATA_DIRECTORY="/jhcnas3/Pathology/original_data/TCGA/"$task
                CSV_FILE_NAME="dataset_csv/"$task".csv"
                FEATURES_DIRECTORY=$prefix"/Pathology/Patches/TCGA__COADREAD"
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
