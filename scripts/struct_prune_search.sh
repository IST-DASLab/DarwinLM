#!/bin/bash
export TRANSFORMERS_CACHE=/nfs/scistore19/alistgrp/huggingface/hub
export WANDB_PROJECT=Evo_structure_prune
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
MODEL="meta-llama/Llama-2-7b-hf" # "Qwen/Qwen2.5-14B-Instruct" "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-3.1-8B"

LEVEL=5 # target compression level 

CALIB_DATA="fineweb_edu" 
SEQUENCE_LENGTH=8196
CALIB_TOKENS=100000
EVAL_TOKENS=100000
TRAIN_TOKENS=500000

#Path to the database
COMPR_PATH="/nfs/scistore19/alistgrp/osieberl/structEvoPress/EvoPress/struct_database_2048/${MODEL##*/}"
# COMPR_PATH="/nfs/scistore19/alistgrp/stang/StructEvoPress/db/ziplm_llama3.1-8B_2048/Llama-3.1-8B"

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
    --eval_sequence_length $SEQUENCE_LENGTH \
    --calibration_sequence_length $SEQUENCE_LENGTH \
    --survivors_per_selection 8 4 2 1 \
    --tokens_per_selection 1024 2048 4096 8192 \
    --training_tokens_per_selection 10000 50000 100000 200000 \
    --generations 200 \
    --target_level $LEVEL \
    --dtype bfloat16 \
    --fitness_fn kl \
    --log_wandb 