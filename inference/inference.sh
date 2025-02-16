cd /workspace-SR006.nfs2/mkuznetsov/mkuznetsov/GenPlans/house_diffusion/house_diffusion_trainer

export CUDA_VISIBLE_DEVICES=2


FID1="/home/aalanov/mkkuznetsov/hair_editing/HairFlash_23042024/input/metric/CelebA-HQ-img/fid.pkl"
FID2="/home/aalanov/mkkuznetsov/hair_editing/HairFlash_23042024/input/metric/CelebA-HQ-img/w/fid.pkl"

FID_CLIP1="/home/aalanov/mkkuznetsov/hair_editing/HairFlash_23042024/input/metric/CelebA-HQ-img/fid-clip.pkl"
FID_CLIP2="/home/aalanov/mkkuznetsov/hair_editing/HairFlash_23042024/input/metric/CelebA-HQ-img/w/fid-clip.pkl"

WANDB_DIR="/workspace-SR006.nfs2/mkuznetsov/mkuznetsov/GenPlans/house_diffusion/house_diffusion_trainer/wandb/"

WANDB_ID1="run-20250124_061028-e6sanez4"
WANDB_ID2="run-20250124_195018-o1zhwx3h"
WANDB_ID3="run-20250124_195600-1ys6pgck"
WANDB_ID4="run-20250130_065821-5oaapjv3"
WANDB_ID5="run-20250130_065822-uyho4qjk"
WANDB_ID6="run-original_chp"
# WANDB_ID7="run-20241011_151644-erjseudt"
# WANDB_ID8="run-20241011_151644-9c9yh0ax"

DATA1="rplan_inf"
DATA2="rplan_inf"
DATA3="rplan_inf"
DATA4="rplan_square_inf"
DATA5="rplan_square_inf"
DATA6="rplan_inf"


OUTPUT_DIR1="/workspace-SR006.nfs2/mkuznetsov/mkuznetsov/GenPlans/house_diffusion/house_diffusion_trainer/output"

WANDB_ID_LIST=($WANDB_ID1 $WANDB_ID2 $WANDB_ID3 $WANDB_ID4 $WANDB_ID5 $WANDB_ID6)
# WANDB_ID_LIST=($WANDB_ID1 $WANDB_ID2 $WANDB_ID3 $WANDB_ID4 $WANDB_ID5 $WANDB_ID6 $WANDB_ID7 $WANDB_ID8)
DATA_LIST=($DATA1 $DATA2 $DATA3 $DATA4 $DATA5 $DATA6)

declare -i cur_indx=0
declare -i idx=5

for i in "${!WANDB_ID_LIST[@]}"; do
    WANDB_ID=${WANDB_ID_LIST[i]}
    DATA=${DATA_LIST[i]}

    if (( idx == cur_indx )); then
        python3 inference/inference.py --inferencer "BaseInferencer" --dataset $DATA \
        --output_dir "${OUTPUT_DIR1}/${DATA}/${WANDB_ID}/" --wandb_folder "${WANDB_DIR}/${WANDB_ID}/" --save_images #--epoch4load $EPOCH
        fi
    cur_indx+=1
done

# for i in "${!WANDB_ID_LIST[@]}"; do
#     if (( idx == cur_indx )); then
#         WANDB_ID=${WANDB_ID_LIST[i]}
#         DATA=${DATA_LIST[i]}

#         # python3 inference/align2.py --inferencer "AlignmentInference" --dataset "inference|Celeb|full" --fids $FID1 $FID2 $FID_CLIP1 $FID_CLIP2\
#         # --output_dir "${OUTPUT_DIR1}/${WANDB_ID}/" --output_filename "metrics.csv" --wandb_folder "${WANDB_DIR}/${WANDB_ID}/" --save_fids #--epoch4load $EPOCH

#         # python3 inference/align2.py --inferencer "AlignmentInference" --dataset "inference|Celeb|full|w_inv" --fids $FID1 $FID2 $FID_CLIP1 $FID_CLIP2\
#         # --output_dir "${OUTPUT_DIR1}/${WANDB_ID}/" --output_filename "metrics-inv.csv" --wandb_folder "${WANDB_DIR}/${WANDB_ID}/" --save_fids #--epoch4load $EPOCH

#         python3 inference/inference.py --inferencer "BaseInferencer" --dataset $DATA \
#         --output_dir "${OUTPUT_DIR1}/${DATA}/${WANDB_ID}/" --wandb_folder "${WANDB_DIR}/${WANDB_ID}/" --save_images #--epoch4load $EPOCH

#         # python3 inference/inference.py --inferencer "BaseInferencer" --dataset "rplan_inf" \
#         # --output_dir "${OUTPUT_DIR1}/${WANDB_ID}/" --wandb_folder "${WANDB_DIR}/${WANDB_ID}/" --save_images #--epoch4load $EPOCH
    
#         # python3 inference/inference.py --inferencer "BaseInferencer" --dataset "rplan_square_inf" \
#         # --output_dir "${OUTPUT_DIR1}/${WANDB_ID}/" --wandb_folder "${WANDB_DIR}/${WANDB_ID}/" --save_images #--epoch4load $EPOCH

#         # python3 inference/align2.py --inferencer "AlignmentInference" --dataset "inference|compare_celeb" \
#         # --output_dir "${OUTPUT_DIR1}/${WANDB_ID}/" --wandb_folder "${WANDB_DIR}/${WANDB_ID}/" --save_images #--epoch4load $EPOCH
    
#         # python3 inference/align2.py --inferencer "AlignmentInference" --dataset "inference|compare_ffhq" \
#         # --output_dir "${OUTPUT_DIR1}/${WANDB_ID}/" --wandb_folder "${WANDB_DIR}/${WANDB_ID}/" --save_images #--epoch4load $EPOCH
    
#         # python3 inference/align2.py --inferencer "AlignmentInference" --dataset "inference|compare_intro_aligned|w_inv" \
#         # --output_dir "${OUTPUT_DIR1}/${WANDB_ID}/w_inv" --wandb_folder "${WANDB_DIR}/${WANDB_ID}/" --save_images #--epoch4load $EPOCH
    
#         # python3 inference/align2.py --inferencer "AlignmentInference" --dataset "inference|compare_celeb|w_inv" \
#         # --output_dir "${OUTPUT_DIR1}/${WANDB_ID}/w_inv" --wandb_folder "${WANDB_DIR}/${WANDB_ID}/" --save_images #--epoch4load $EPOCH
    
#         # python3 inference/align2.py --inferencer "AlignmentInference" --dataset "inference|compare_ffhq|w_inv" \
#         # --output_dir "${OUTPUT_DIR1}/${WANDB_ID}/w_inv" --wandb_folder "${WANDB_DIR}/${WANDB_ID}/" --save_images #--epoch4load $EPOCH

#     fi
#     cur_indx+=1
# done

# python3 inference/align2.py --inferencer "AlignmentInferenceOld" --dataset "inference|Celeb|full" --fid_cache "/home/aalanov/mkkuznetsov/hair_editing/HairFlash_23042024/input/metric/CelebA-HQ-img/fid.pkl" --fid_clip_cache "/home/aalanov/mkkuznetsov/hair_editing/HairFlash_23042024/input/metric/CelebA-HQ-img/fid-clip.pkl" \
#         --output_dir "/home/aalanov/mkkuznetsov/hair_editing/output/HairFastTrainer/all_configs/run-20240410_152034-x4mpwgj4/transfer3_celeba/" --wandb_folder "/home/aalanov/mkkuznetsov/hair_editing/HairFlash_25032024/wandb/run-20240410_152034-x4mpwgj4/"
