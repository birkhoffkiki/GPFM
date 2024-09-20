

root_dir="extract_scripts/logs/"
use_cache="no"

# models="ctranspath"
models="conch"

declare -A gpus
gpus["ctranspath"]=4
gpus["conch"]=5


for model in $models
do
        DIR_TO_COORDS=/jhcnas5/jmabq/Pathology/NanfangHospital
        DATA_DIRECTORY=/jhcnas5/jmabq/Pathology/NanfangHospital/WSIs
        CSV_FILE_NAME=dataset_csv/NanFangHospital.csv
        FEATURES_DIRECTORY=/jhcnas5/jmabq/Pathology/NanfangHospital
        ext=".svs"
        save_storage="yes"
        datatype="auto" # extra path process for TCGA dataset, direct mode do not care use extra path

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
                --save_storage $save_storage > $root_dir"Nanfang_log_$model.log" 2>&1 &
done
