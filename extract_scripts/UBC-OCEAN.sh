
prefix="/jhcnas3"
dataset="UBC-OCEAN"
skip_partial="no" # yes to skip partial file

# models="conch uni distill_87499"
models="distill_379999_cls_only"

declare -A gpus
gpus["distill_379999_cls_only"]=0
gpus["distill_87499"]=7
gpus["uni"]=6
gpus["phikon"]=6
gpus["conch"]=5


for model in $models
do
        DIR_TO_COORDS=$prefix"/Pathology/Patches/"$dataset
        CSV_FILE_NAME="dataset_csv/"$dataset".csv"
        FEATURES_DIRECTORY=$prefix"/Pathology/Patches/"$dataset

        echo $model", GPU is:"${gpus[$model]}
        export CUDA_VISIBLE_DEVICES=${gpus[$model]}

        nohup python3 extract_features_fp_from_patch.py \
                --data_h5_dir $DIR_TO_COORDS \
                --csv_path $CSV_FILE_NAME \
                --feat_dir $FEATURES_DIRECTORY \
                --batch_size 128 \
                --model $model \
                --skip_partial $skip_partial > "extract_scripts/logs/UBC-OCEAN_log_$model.log" 2>&1 &
done

