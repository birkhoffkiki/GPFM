# save log
# subtypes="UCEC BRCA COAD GBM"
subtypes="BRCA"
# models="conch"
models="ctranspath"

declare -A gpus
gpus["conch"]=5
gpus["ctranspath"]=4

for subtype in $subtypes
do
        dataset="CPTAC__"$subtype
        ext=".svs"
        prefix='/jhcnas3'
        DATA_DIRECTORY="/jhcnas3/Pathology/original_data/CPTAC/"$subtype
        #---------------------------------------
        export OPENCV_IO_MAX_IMAGE_PIXELS=10995116277760
        root_dir="extract_scripts/"
        for model in $models
        do
                DIR_TO_COORDS=$prefix"/Pathology/Patches/"$dataset
                CSV_FILE_NAME="dataset_csv/temporty_csv/"$dataset".csv"
                FEATURES_DIRECTORY=$prefix"/Pathology/Patches/"$dataset
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
                        --use_cache 'no' \
                        --batch_size 32 \
                        --model $model \
                        --datatype $datatype \
                        --slide_ext $ext \
                        --save_storage $save_storage > $root_dir"/logs/"$dataset"_"$model".txt" 2>&1 &
        done
done