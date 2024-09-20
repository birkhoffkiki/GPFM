# LUAD LUSC
# save log
prefix="/jhcnas3"


# models="phikon"
models="distill_379999_cls_only"
skip_partial="no" # yes to skip partial file

tasks="LUSC"
declare -A gpus
gpus["dinov2_vitl"]=0
gpus["distill_379999_cls_only"]=0
gpus["distill_87499"]=0
gpus["phikon"]=6
gpus["plip"]=2
gpus["uni"]=1
gpus["conch"]=2

for model in $models
do
        for task in $tasks
        do
                DIR_TO_COORDS=$prefix"/Pathology/Patches/TCGA__"$task
                CSV_FILE_NAME="dataset_csv/LUAD_LUSC.csv"
                FEATURES_DIRECTORY=$prefix"/Pathology/Patches/TCGA__"$task

                echo $model", GPU is:"${gpus[$model]}
                export CUDA_VISIBLE_DEVICES=${gpus[$model]}

                nohup python3 extract_features_fp_from_patch.py \
                        --data_h5_dir $DIR_TO_COORDS \
                        --csv_path $CSV_FILE_NAME \
                        --feat_dir $FEATURES_DIRECTORY \
                        --batch_size 32 \
                        --model $model \
                        --skip_partial $skip_partial > "extract_scripts/logs/"$task"_log_$model.log" 2>&1 &
        done
done
