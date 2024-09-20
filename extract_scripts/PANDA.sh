
# save log
prefix="/jhcnas3"
skip_partial="no" # yes to skip partial file
models="distill_379999_cls_only"

declare -A gpus
gpus["phikon"]=6
gpus["plip"]=2
gpus["conch"]=5
gpus["ctranspath"]=2
gpus["resnet50"]=7
gpus["distill_87499"]=4
gpus["distill_379999_cls_only"]=1


for model in $models
do
        DIR_TO_COORDS=$prefix"/Pathology/Patches/PANDA"
        CSV_FILE_NAME="dataset_csv/PANDA.csv"
        FEATURES_DIRECTORY=$prefix"/Pathology/Patches/PANDA"

        echo $model", GPU is:"${gpus[$model]}
        export CUDA_VISIBLE_DEVICES=${gpus[$model]}

        nohup python3 extract_features_fp_from_patch.py \
                --data_h5_dir $DIR_TO_COORDS \
                --csv_path $CSV_FILE_NAME \
                --feat_dir $FEATURES_DIRECTORY \
                --batch_size 128 \
                --model $model \
                --skip_partial $skip_partial > "extract_scripts/logs/PANDA_log_$model.log" 2>&1 &
done
