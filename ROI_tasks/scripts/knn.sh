task="PanCancer-TCGA"

output_dir="/home/jmabq/data/results"
data_root="/home/jmabq/data/"$task"/features"

models="conch uni ctranspath resnet50 plip distill_87499 distill_99999 phikon"
# models="ctranspath"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:./"

for model in $models
do
    echo "processing: $model"
    python downstream_tasks/knn.py \
        --output_dir $output_dir"/"$task"/knn/"$model \
        --data_dir $data_root"/"$model \
        --model_name $model \
        --nb_knn 1 20 
done