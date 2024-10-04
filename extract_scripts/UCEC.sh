# LUAD LUSC
# save log
prefix="/jhcnas3"


models="distill_379999_cls_only"
# models="ctranspath"
skip_partial="no" # yes to skip partial file

tasks="UCEC"
declare -A gpus

gpus["distill_87499"]=5
gpus["distill_379999_cls_only"]=1
gpus["plip"]=6
gpus["uni"]=6
gpus["conch"]=6
gpus["ctranspath"]=0
gpus["resnet50"]=5

for model in $models
do
        for task in $tasks
        do
                DIR_TO_COORDS=$prefix"/Pathology/Patches/TCGA__"$task
                CSV_FILE_NAME="dataset_csv/temporty_csv/TCGA__UCEC.csv"
                FEATURES_DIRECTORY=$prefix"/Pathology/Patches/TCGA__"$task

                echo $model", GPU is:"${gpus[$model]}
                export CUDA_VISIBLE_DEVICES=${gpus[$model]}

                nohup python3 extract_features_fp_from_patch.py \
                        --data_h5_dir $DIR_TO_COORDS \
                        --csv_path $CSV_FILE_NAME \
                        --feat_dir $FEATURES_DIRECTORY \
                        --batch_size 64 \
                        --model $model \
                        --skip_partial $skip_partial > "extract_scripts/logs/"$task"_log_$model.log" 2>&1 &
        done
done