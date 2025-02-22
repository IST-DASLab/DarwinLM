export TRANSFORMERS_CACHE=/nfs/scistore19/alistgrp/huggingface/hub
export CUDA_VISIBLE_DEVICES=4,5,6,7

MODEL="meta-llama/Llama-2-7b-hf" #"Qwen/Qwen2.5-14B-Instruct" "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-3.1-8B"
SEQUENCE_LENGTH=4096
CALIB_DATA="fineweb_edu" 

NUM_TOKENS=8388608 #8388608 # 1024*8192

SAVE_DIR="/nfs/scistore19/alistgrp/stang/StructEvoPress/db/struct_gradual_database_from_level-5/${MODEL##*/}" 

torchrun --nnodes=1 --nproc-per-node=4 --master_port 29501 struct_prune.py \
    \
    --model_name_or_path $MODEL \
    --sparse_weight_path $SAPRSE_WEIGHT_PATH \
    --sparse_config_path $SAPRSE_CONFIG_PATH \
    --sparse_finetuned_weight_path $SAPRSE_FINETUNED_WEIGHT_PATH \
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