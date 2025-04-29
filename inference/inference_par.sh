# a100-4
# export CUDA_VISIBLE_DEVICES=3

WANDB_DIR="/home/jovyan/shares/SR008.fs2/karanny/hdif_trainer/wandb"

WANDB_ID1="run-20250318_231705-8xbxw10n" #blue
WANDB_ID2="run-20250315_160510-rinelewx" #purple
WANDB_ID3="run-20250317_121110-a4dbx7w8" #grey
WANDB_ID4="run-20250417_155045-xoz6cfgl" #msd+centroids 
WANDB_ID5="run-20250425_121111-iixfktnm" #msd_base 
WANDB_ID6='run-20250324_190503-jkk28kt8' #base run 



DATA1="rplan_inf"
DATA2="rplan_inf"
DATA3="rplan_inf"
DATA4="rplan_square_inf"
DATA5="rplan_square_inf"
DATA6="run-original_chp"
DATA7="rplan-msd-test-full"
DATA8="rplan-msd-test-full"
DATA9="rplan-msd-sq-test-full"

OUTPUT_DIR1='your_output_dir'

# WANDB_ID_LIST=($WANDB_ID3 $WANDB_ID4 $WANDB_ID5 $WANDB_ID6 $WANDB_ID7 $WANDB_ID8)
WANDB_ID_LIST=($WANDB_ID6)
# WANDB_ID_LIST=($WANDB_ID1 $WANDB_ID2 $WANDB_ID3 $WANDB_ID4 $WANDB_ID5 $WANDB_ID6 $WANDB_ID7 $WANDB_ID8)
# DATA_LIST=($DATA7 $DATA8 $DATA9)

declare -i cur_indx=0
declare -i idx=0

for i in "${!WANDB_ID_LIST[@]}"; do
    WANDB_ID=${WANDB_ID_LIST[i]}
    # DATA=${DATA_LIST[i]}
    # DATA="rplan-msd-sq-test-full"
    DATA="rplan-msd-param-test-small"
    # DATA = "rplan-msd-param-test-full"

    python3 inference/inference.py --inferencer "BaseInferencer" --dataset $DATA \
    --output_dir "${OUTPUT_DIR1}/${DATA}/${WANDB_ID}/" --wandb_folder "${WANDB_DIR}/${WANDB_ID}/" --save_images #--epoch4load $EPOCH
done
