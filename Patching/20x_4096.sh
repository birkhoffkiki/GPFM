# BCNB
export OPENCV_IO_MAX_IMAGE_PIXELS=10995116277760

# python create_patches_fp.py \
#         --source /jhcnas3/Pathology/original_data/BCNB/WSIs \
#         --save_dir /home/jmabq/DATA/Pathology/BCNB\
#         --preset BCNB.csv \
#         --patch_level 0 \
#         --patch_size 512 \
#         --step_size 512 \
#         --seg \
#         --patch \
#         --stitch

# ----TCGA---------
# arr[0]="BRCA"
# arr[0]="COAD"
# arr[0]="LUAD"
# arr[0]="STAD"
# arr[0]="KIRP"
# arr[0]="KIRC"
arr[0]="READ"

save_dir="/jhcnas3/Pathology/Patches/TCGA__"
source_dir="/jhcnas2/home/zhoufengtao/data/TCGA/TCGA-"
# source_dir="/jhcnas3/Pathology/original_data/TCGA/READ"
wsi_format="svs"
size=4096
#Array Loop  
for i in "${!arr[@]}" 
do  
subtype="${arr[$i]}"
echo Processing $subtype
python create_patches_fp.py \
        --source $source_dir$subtype"/slides" \
        --save_dir $save_dir$subtype"/4096" \
        --preset tcga.csv \
        --patch_level 1 \
        --patch_size $size \
        --step_size $size \
        --wsi_format $wsi_format \
        --seg \
        --patch \
        --stitch 
done


# save_dir="/jhcnas3/Pathology/Patches/BCNB"
# source_dir="/jhcnas3/Pathology/original_data/BCNB/WSIs"
# wsi_format="jpg"
# patch_size=512

# python create_patches_fp.py \
#         --source $source_dir \
#         --save_dir $save_dir\
#         --preset tcga.csv \
#         --patch_level 0 \
#         --patch_size $patch_size \
#         --step_size $patch_size \
#         --wsi_format $wsi_format \
#         --seg \
#         --patch \
#         --stitch

# TCGA flash frozen
# save_dir="/jhcnas3/Pathology/Patches/BRACS"
# source_dir="/jhcnas3/BRACS"
# wsi_format="svs"

# python create_patches_fp.py \
#         --source $source_dir \
#         --save_dir $save_dir\
#         --preset tcga.csv \
#         --patch_level 0 \
#         --patch_size 512 \
#         --step_size 512 \
#         --wsi_format $wsi_format \
#         --seg \
#         --patch \
#         --stitch

# save_dir="/jhcnas3/Pathology/Patches/Ovarian_Bevacizumab_Response"
# source_dir="/jhcnas3/Pathology/original_data/Ovarian_Bevacizumab_Response"
# wsi_format="svs"

# python create_patches_fp.py \
#         --source $source_dir \
#         --save_dir $save_dir\
#         --preset tcga.csv \
#         --patch_level 0 \
#         --patch_size 512 \
#         --step_size 512 \
#         --wsi_format $wsi_format \
#         --seg \
#         --patch \
#         --stitch

# save_dir="/jhcnas3/Pathology/Patches/SLN-Breast"
# source_dir="/jhcnas3/Pathology/original_data/SLN-Breast"
# wsi_format="svs"

# python create_patches_fp.py \
#         --source $source_dir \
#         --save_dir $save_dir\
#         --preset tcga.csv \
#         --patch_level 0 \
#         --patch_size 512 \
#         --step_size 512 \
#         --wsi_format $wsi_format \
#         --seg \
#         --patch \
#         --stitch