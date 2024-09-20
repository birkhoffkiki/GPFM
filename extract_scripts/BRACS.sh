# save log
# save log
prefix="/jhcnas3"
# models="phikon"
models="distill_379999_cls_only"
skip_partial="no" # yes to skip partial file

declare -A gpus
gpus["distill_379999_cls_only"]=0
gpus["distill_87499"]=4
gpus["phikon"]=7
gpus["plip"]=6
gpus["uni"]=0
gpus["conch"]=1
gpus["ctranspath"]=5
gpus["resnet50"]=6


for model in $models
do
        DIR_TO_COORDS=$prefix"/Pathology/Patches/BRACS"
        CSV_FILE_NAME="dataset_csv/BRACS.csv"
        FEATURES_DIRECTORY=$prefix"/Pathology/Patches/BRACS"

        echo $model", GPU is:"${gpus[$model]}
        export CUDA_VISIBLE_DEVICES=${gpus[$model]}

        nohup python3 extract_features_fp_from_patch.py \
                --data_h5_dir $DIR_TO_COORDS \
                --csv_path $CSV_FILE_NAME \
                --feat_dir $FEATURES_DIRECTORY \
                --batch_size 32 \
                --model $model \
                --skip_partial $skip_partial > "extract_scripts/logs/BRACS_log_$model.log" 2>&1 &
done
