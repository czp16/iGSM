set -ex

PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2

# ======= Base Models =======

OUTPUT_DIR=${MODEL_NAME_OR_PATH}/igsm_eval # "rgsm,rgsm_original" #
DATA_NAMES="iGSM"

SPLIT="iGSM_test_1K" 
NUM_TEST_SAMPLE=-1


# Can only be tested on single-gpu
CUDA_VISIBLE_DEVICES=7 TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --data_names ${DATA_NAMES} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
