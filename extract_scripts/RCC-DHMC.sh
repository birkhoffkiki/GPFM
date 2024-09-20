
prefix="/jhcnas3"
dataset="RCC-DHMC"
skip_partial="no" # yes to skip partial file

# models="resnet50 conch uni distill_87499 phikon plip"
# models="conch"
models="distill_379999_cls_only"


declare -A gpus
gpus["distill_87499"]=0
gpus["distill_379999_cls_only"]=1
gpus["resnet50"]=3
gpus["plip"]=4
gpus["uni"]=7
gpus["phikon"]=5
gpus["conch"]=6


for model in $models
do
        DIR_TO_COORDS="/storage/Pathology/Patches/"$dataset
        CSV_FILE_NAME="dataset_csv/RCC-DHMC.csv"
        FEATURES_DIRECTORY=$prefix"/Pathology/Patches/"$dataset

        echo $model", GPU is:"${gpus[$model]}
        export CUDA_VISIBLE_DEVICES=${gpus[$model]}

        nohup python3 extract_features_fp_from_patch.py \
                --data_h5_dir $DIR_TO_COORDS \
                --csv_path $CSV_FILE_NAME \
                --feat_dir $FEATURES_DIRECTORY \
                --batch_size 64 \
                --model $model \
                --skip_partial $skip_partial > "extract_scripts/logs/RCC-DHMC_log_$model.log" 2>&1 &
done

