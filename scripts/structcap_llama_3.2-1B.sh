export TRANSFORMERS_CACHE=/nfs/scistore19/alistgrp/huggingface/hub
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export NCCL_TIMEOUT=10800  # Set to 3000 seconds or any value as neede

MODEL="meta-llama/Llama-3.2-1B"
SEQUENCE_LENGTH=4096
CALIB_DATA="fineweb_edu" 

NUM_TOKENS=16777216
#NUM_TOKENS=8388608 #8388608 # 1024*8192
# NUM_TOKENS=40960
# COLS_REMOVED_ATTN="0 256 512 768 1024 1280 1536 1792 2048 2304 2560 2816 3072 3328 3584 3840 4096" # How many columns are removed for each level (here: 17 levels)
# COLS_REMOVED_MLP="0 688 1376 2064 2752 3440 4128 4816 5504 6192 6880 7568 8256 8944 9632 10320 11008"

COLS_REMOVED_ATTN="0 192 384 640 832 1024 1216 1408 1664 1856 2048" # How many columns are removed for each level (here: 17 levels)
COLS_REMOVED_MLP="0 832 1632 2464 3264 4096 4928 5728 6560 7360 8192"

# SAPRSE_CONFIG_PATH=/nfs/scistore19/alistgrp/stang/StructEvoPress/evo-kl-configuration-5.0-finetune-multistep.txt
# SAPRSE_WEIGHT_PATH=/nfs/scistore19/alistgrp/osieberl/structEvoPress/EvoPress/struct_database_2048/Llama-2-7b-hf
# SAPRSE_FINETUNED_WEIGHT_PATH=/nfs/scistore19/alistgrp/stang/llm-foundry/srun_logs/evopress_search_llama2_with_multistep-finetune_10B/ep0-ba2500-rank0_hf/pytorch_model.bin


SAVE_DIR="/nfs/scistore19/alistgrp/stang/StructEvoPress/db/structCAP_32768/${MODEL##*/}" 

torchrun --nnodes=1 --nproc-per-node=7 --master_port 29501 struct_prune.py \
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
    --verbose \
    --use_cap \
    \
    --dtype float16 \
    --attn_implementation flash_attention_2 \
    \
    --save_dir $SAVE_DIR \
    \
    --module_regex '.*layers.*((o|down)_proj)$' \
    \
    --grad_sparsity 0.95 \
    --rel_damp 1.0e-7 \
    --group_size_config_path /nfs/scistore19/alistgrp/stang/StructEvoPress/scripts/config/llama-3.2-1B.yaml \
    --rows_in_parallel 32 \
    --find_block_size \
    --block_size 512 \