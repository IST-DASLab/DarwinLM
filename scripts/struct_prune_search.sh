#!/bin/bash
export TRANSFORMERS_CACHE=/nfs/scistore19/alistgrp/huggingface/hub
export WANDB_PROJECT=darwinlm
export CUDA_VISIBLE_DEVICES=6
MODEL="meta-llama/Llama-2-7b-hf" # "meta-llama/Meta-Llama-3-8B" "meta-llama/Llama-2-7b-hf" "mistralai/Mistral-7B-v0.3"

LEVEL=8 # target compression level 

CALIB_DATA="fineweb_edu" 
SEQUENCE_LENGTH=4096
CALIB_TOKENS=100000
EVAL_TOKENS=100000

COMPR_PATH="/nfs/scistore19/alistgrp/osieberl/structEvoPress/EvoPress/struct_database_weightsDiff/${MODEL##*/}" 

export WANDB_NAME="StructPruneSearch_${MODEL}_level_${LEVEL}_step_${STEP}"


python evo_struct_prune_search.py \
    --calibration_data  $CALIB_DATA \
    --model_name_or_path $MODEL \
    --calibration_tokens $CALIB_TOKENS\
    --eval_tokens $EVAL_TOKENS \
    --offspring 8 \
    --eval_every 20 \
    --eval_datasets "fineweb_edu" \
    --sparse_weights_path $COMPR_PATH \
    --eval_sequence_length $SEQUENCE_LENGTH \
    --calibration_sequence_length $SEQUENCE_LENGTH \
    --log_wandb \
    --survivors_per_selection 1 1 \
    --tokens_per_selection 1024 8192 \
    --generations 200 \
    --target_level $LEVEL \
    --dtype float16 \
    --fitness_fn kl