#!/bin/bash

cancer_names=('ACC' 'BRCA' 'BLCA' 'CESC' 'CHOL' 'COAD' 'DLBC' 'ESCA' 'GBM' 'HNSC' 'KICH' 'KIRC' 'KIRP' 'LGG' 'LIHC' 'LUAD' 'LUSC' 'MESO' 'OV' 'PAAD' 'PCPG' 'PRAD' 'READ' 'SARC' 'SKCM' 'STAD' 'TGCT' 'THCA' 'THYM' 'UCEC' 'UCS' 'UVM')

for cancer in "${cancer_names[@]}"; do
    source_dir="/storage/Pathology/Patches/TCGA__${cancer}/pt_files/dinov2_vitl/"
    destination_dir="/jhcnas2/home/wangyihui/MultiModal/Pathology/TCGA-${cancer}/pt_files/"

    rsync -av "$source_dir" "$destination_dir"

    echo "Batch copy for ${cancer} completed."
done
