# model_names="clam_mb clam_sb mean_mil max_mil att_mil trans_mil"

# camely MIL scripts
model_names="att_mil"

# backbones="resnet50"
backbones="resnet50 phikon conch ctranspath plip uni distill_87499"
# backbones="mae_vit_large_patch16-1epoch-180M"
# backbones="ctranspath plip"
# backbones="dinov2_vitl"

declare -A in_dim
in_dim["resnet50"]=1024
in_dim["ctranspath"]=768
in_dim["phikon"]=768
in_dim["distill_87499"]=1024
in_dim["uni"]=1024
in_dim["plip"]=512
in_dim["conch"]=512


declare -A gpus
gpus["clam_sb"]=3
gpus["mean_mil"]=6
gpus["max_mil"]=6
gpus["att_mil"]=5
gpus["trans_mil"]=6
gpus["dtfd"]=3
gpus['simple']=6


log_dir="eval_scripts/logs"
task="UBC-OCEAN"
results="/storage/Pathology/results/experiments/train/splits712/"
save_dir="/jhcnas3/Pathology/experiments/eval/splits712"

splits_dir="splits712/UBC-OCEAN_100"
size=512
n_classes=5

for model in $model_names
do
    for backbone in $backbones
    do
        export CUDA_VISIBLE_DEVICES=${gpus[$model]}

        exp=$model"/"$backbone
        echo "processing:"$exp
        model_exp_code=$task"/"$model"/"$backbone"_s1024"
        save_exp_code=$task"/"$model"/"$backbone"_s1024_512"
        python eval.py \
            --bootstrap \
            --drop_out \
            --k 1 \
            --task_type subtyping \
            --models_exp_code $model_exp_code \
            --save_exp_code $save_exp_code \
            --n_classes $n_classes \
            --task $task \
            --model_type $model \
            --results_dir $results \
            --backbone $backbone \
            --save_dir $save_dir \
            --splits_dir $splits_dir \
            --in_dim ${in_dim[$backbone]} > $log_dir"/"$task"_"$model"_"$backbone".txt"
    done
done

