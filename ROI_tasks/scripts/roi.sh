# tasks="PanCancer-TCGA"
# tasks="PanCancer-TIL"
# tasks="ESCA"
# tasks="UniToPatho"
# tasks="BACH"
# tasks="PCAM"
# tasks="WSSS4LUAD"
# tasks="CCRCC-TCGA_HEL"
tasks="CRC-100K"
# tasks="CRC-MSI"
# tasks="BreakHis"
# tasks="chaoyang"


models="conch uni ctranspath resnet50 plip distill_87499 phikon"


export CUDA_VISIBLE_DEVICES=6

for task in $tasks
do
    output_dir="/home/jmabq/data/results"
    data_root="/home/jmabq/data/"$task"/features"

    export PYTHONPATH="${PYTHONPATH}:./"

    for model in $models
    do
        echo "processing: $model"
        python downstream_tasks/roi_retrieval.py \
            --metric_file_path $output_dir"/"$task"/roi/"$model"/results_eval_roi.json.json" \
            --train_feat_path $data_root"/"$model"/train.pt" \
            --val_feat_path $data_root"/"$model"/val.pt" \
            --batch_size 256 > "./downstream_tasks/scripts/"roi-$task"-"$model".log" 2>&1
    done
done
