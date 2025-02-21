export TRANSFORMERS_CACHE=/nfs/scistore19/alistgrp/huggingface/hub
export CUDA_VISIBLE_DEVICES=0,1,3,4,5
export NCCL_TIMEOUT=10800  # Set to 3000 seconds or any value as neede

MODEL="meta-llama/Llama-3.1-8B"
SEQUENCE_LENGTH=8192
CALIB_DATA="fineweb_edu" 

NUM_TOKENS=16777216 #8388608 # 1024*8192
# NUM_TOKENS=40960
COLS_REMOVED_ATTN="0 512 896 1408 1792 2304 2816 3328 3584 3968 4096" # How many columns are removed for each level (here: 17 levels)
COLS_REMOVED_MLP="0 1440 2880 4288 5728 7168 8608 10048 11456 12896 14336"


SAVE_DIR="/nfs/scistore19/alistgrp/stang/StructEvoPress/db/ziplm_llama3.1-8B_2048/${MODEL##*/}" 

torchrun --nnodes=1 --nproc-per-node=5 --master_port 29501 struct_prune.py \
    \
    --model_name_or_path $MODEL \
    --prunable_modules '.*layers.*((q|k|v|o|gate|up|down)_proj)$' \
    --pre_block_modules model.embed_tokens model.rotary_emb \
    --block_modules model.layers \
    \
    --calibration_data $CALIB_DATA \
    --calibration_tokens $NUM_TOKENS \
    --calibration_sequence_length $SEQUENCE_LENGTH \
    \
    --cols_removed_attn $COLS_REMOVED_ATTN \
    --cols_removed_mlp $COLS_REMOVED_MLP \
    --mlp_prune_name "down_proj" \
    --attn_prune_name "o_proj" \
     \
    --low_cpu_mem_usage \
    --cpu_offload_modules \
    --cpu_offload_activations \
    --verbose \
    \
    --dtype float16 \
    --attn_implementation flash_attention_2 \
    \
    --save_dir $SAVE_DIR \