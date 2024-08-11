# tasks="PanCancer-TCGA"
# tasks="PanCancer-TIL"
# tasks="ESCA"
# tasks="UniToPatho"
# tasks="BACH"
# tasks="PCAM"
# tasks="WSSS4LUAD"
# tasks="CCRCC-TCGA_HEL"
# tasks="CRC-100K"
# tasks="CRC-MSI"
# tasks="BreakHis"
# tasks="chaoyang"
tasks="DRYAD"

# tasks="PanCancer-TCGA PanCancer-TIL ESCA UniToPatho PCAM WSSS4LUAD CCRCC-TCGA_HEL CRC-100K CRC-MSI BreakHis chaoyang"

models="conch distill_87499"
# models="conch uni ctranspath resnet50 plip distill_87499 phikon dinov2_vitl"


declare -A gpus
gpus["distill_87499"]=5
gpus["dinov2_vitl"]=4
gpus["phikon"]=7
gpus["plip"]=1
gpus["uni"]=0
gpus["conch"]=1
gpus["ctranspath"]=5
gpus["resnet50"]=6

for task in $tasks
do
    output_dir="/home/jmabq/data/results_withlogits"
    data_root="/home/jmabq/data/"$task"/features"

    export PYTHONPATH="${PYTHONPATH}:./"

    for model in $models
    do
        export CUDA_VISIBLE_DEVICES=${gpus[$model]}
        echo "processing: $model"
        nohup python ROI_tasks/linear.py \
            --output_dir $output_dir"/"$task"/linear/"$model \
            --data_dir $data_root"/"$model \
            --batch_size 256 \
            --epochs 3000 > "./ROI_tasks/scripts/"linear-$task"-"$model".log" 2>&1 &
    done
done
