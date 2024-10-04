# BRCA
prefix="/jhcnas3"

task="LGG"
root_dir="extract_scripts/logs/"$task"_log_"
use_cache="no"


DIR_TO_COORDS=$prefix"/Pathology/Patches/TCGA__LGG"
DATA_DIRECTORY=$prefix"/Pathology/original_data/TCGA/LGG/slides"
CSV_FILE_NAME="dataset_csv/temporty_csv/TCGA__LGG.csv"
FEATURES_DIRECTORY=$prefix"/Pathology/Patches/TCGA__LGG"

ext=".svs"
save_storage="yes"

models="conch distill_87499"

declare -A gpus
gpus["conch"]=1
gpus['distill_87499']=1

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
                --use_cache $use_cache \
                --datatype $datatype \
                --slide_ext $ext \
                --save_storage $save_storage > $root_dir"$model.log" 2>&1 &

done