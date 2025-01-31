#!/bin/bash
export TRANSFORMERS_CACHE=/nfs/scistore19/alistgrp/huggingface/hub
export WANDB_PROJECT=Evo_structure_prune
export CUDA_VISIBLE_DEVICES=5,6,7,8,9
MODEL="meta-llama/Llama-2-7b-hf" # "meta-llama/Meta-Llama-3-8B" "meta-llama/Llama-2-7b-hf" "mistralai/Mistral-7B-v0.3"

LEVEL=5 # target compression level 
GRADUAL_LEVEL=6 # target compression level for gradual pruning

CALIB_DATA="fineweb_edu" 
SEQUENCE_LENGTH=4096
CALIB_TOKENS=100000
EVAL_TOKENS=100000
TRAIN_TOKENS=200000

COMPR_PATH="/nfs/scistore19/alistgrp/osieberl/structEvoPress/EvoPress/struct_database_2048/${MODEL##*/}" 

CONFIG_PATH="/nfs/scistore19/alistgrp/stang/StructEvoPress/evo-kl-configuration-5.0-finetune-multistep.txt"
SAPRSE_FINETUNED_WEIGHT_PATH="/nfs/scistore19/alistgrp/stang/llm-foundry/srun_logs/evopress_search_llama2_with_multistep-finetune_10B/ep0-ba2500-rank0_hf/pytorch_model.bin"

GRADUAL_DATABASE_PATH="/nfs/scistore19/alistgrp/stang/StructEvoPress/db/struct_gradual_database_from_level-5/${MODEL##*/}"

export WANDB_NAME="StructPruneSearch_${MODEL}_level_${LEVEL}_step_${STEP}"


python evo_struct_prune_search.py \
    --calibration_data  $CALIB_DATA \
    --model_name_or_path $MODEL \
    --calibration_tokens $CALIB_TOKENS\
    --training_tokens $TRAIN_TOKENS \
    --eval_tokens $EVAL_TOKENS \
    --offspring 16 \
    --eval_every 20 \
    --eval_datasets "fineweb_edu" \
    --sparse_weights_path $COMPR_PATH \
    --sparse_finetuned_weight_path $SAPRSE_FINETUNED_WEIGHT_PATH \
    --sparse_config_path $CONFIG_PATH \
    --gradual_database $GRADUAL_DATABASE_PATH \
    --eval_sequence_length $SEQUENCE_LENGTH \
    --calibration_sequence_length $SEQUENCE_LENGTH \
    --survivors_per_selection 4 2 1 \
    --tokens_per_selection  2048 4096 8192 \
    --training_tokens_per_selection 10000 50000 100000 \
    --generations 500 \
    --target_level $LEVEL \
    --gradual_target_level $GRADUAL_LEVEL\
    --dtype bfloat16 \
    --fitness_fn kl \
    --log_wandb 