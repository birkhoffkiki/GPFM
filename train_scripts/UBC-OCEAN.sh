
model_names="att_mil"

backbones="resnet50 phikon conch ctranspath plip uni distill_87499"

declare -A in_dim
in_dim["resnet50"]=1024
in_dim["uni"]=1024
in_dim["ctranspath"]=768
in_dim["distill_87499"]=1024
in_dim["phikon"]=768
in_dim["plip"]=512
in_dim["conch"]=512


declare -A gpus
gpus["clam_sb"]=3
gpus["max_mil"]=0
gpus["att_mil"]=6
gpus["mean_mil"]=0
gpus["ds_mil"]=0
gpus['dtfd']=0
gpus['moe_a2o']=2
gpus['moe']=0
gpus['trans_mil']=0
gpus['hat_encoder_256_32']=3 #√
gpus['hat_encoder_512_512']=1 #√
gpus['hat_encoder_256_256']=1 #√
gpus['96_hat_encoder_512_512']=4 #√
gpus['512_hat_encoder_512_512']=5 #√
gpus['128_hat_encoder_512_512']=7 #√
gpus['64_hat_encoder_512_512']=0 #√
gpus['256_hat_encoder_512_512']=0 #√ 
gpus['96_hat_encoder_512_512_nomem']=5
gpus['hat_encoder_256_32_nomem']=5

n_classes=5 # the class number of dataset
task="UBC-OCEAN" # the name of the dataset, it is defined at `datasets/__init__.py`.
root_log="train_scripts/logs/train_log_"$task"_" # log file path
results_dir="/storage/Pathology/results/experiments/train/splits712/"$task # path to save results
split_dir="splits712/UBC-OCEAN_100" # which splits to use
seed=1024       # random seed
model_size="small" # since the dim of feature of vit-base is 768  `ONLY FOR CLAM`, ignore this for other methods
preloading="no"   # load all data into memory and then train the model
patch_size="512"    # the patch size of instances


for model in $model_names
do
    for backbone in $backbones
    do
        exp=$model"/"$backbone
        echo $exp", GPU is:"${gpus[$model]}
        export CUDA_VISIBLE_DEVICES=${gpus[$model]}
        # k_start and k_end, only for resuming, default is -1
        k_start=0
        k_end=-1
        nohup python main.py \
            --seed $seed \
            --split_dir $split_dir \
            --drop_out \
            --task_type subtyping \
            --early_stopping \
            --lr 2e-4 \
            --reg 1e-4 \
            --k 1 \
            --k_start $k_start \
            --k_end $k_end \
            --label_frac 1.0 \
            --exp_code $exp \
            --patch_size $patch_size \
            --weighted_sample \
            --task $task \
            --backbone $backbone \
            --results_dir $results_dir \
            --model_type $model \
            --log_data \
            --preloading $preloading \
            --model_size $model_size \
            --n_classes $n_classes \
            --in_dim ${in_dim[$backbone]} > "$root_log""$model"_"$backbone.log" 2>&1 &
    done
done

