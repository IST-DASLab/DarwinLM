#!/bin/bash
export TRANSFORMERS_CACHE=/nfs/scistore19/alistgrp/huggingface/hub
export CUDA_VISIBLE_DEVICES=7
MODEL="Llama-2-7B"
LMEVAL=${LMEVAL:-"SIMPLE_ZERO_SHOTS"}

if [[ $MODEL == Llama-2-7B ]]; then
    MODEL_ID=meta-llama/Llama-2-7b-hf
elif [[ $MODEL == Llama-3-8B ]]; then
    MODEL_ID=meta-llama/Meta-Llama-3-8B
elif [[ $MODEL == Meta-Llama-3.1-8B ]]; then
    MODEL_ID=meta-llama/Llama-3.1-8B
elif [[ $MODEL == Mistral-7B ]]; then
    MODEL_ID=mistralai/Mistral-7B-v0.3
elif [[ $MODEL == Llama-3.1-8B ]]; then
    MODEL_ID=meta-llama/Llama-3.1-8B
else
    echo "Unknown model"
    exit 1
fi

# ./z_scripts/z_run_lmeval.sh

MODEL_LOADING_KWARGS=${MODEL_LOADING_KWARGS:-"--sparse_weights_path /nfs/scistore19/alistgrp/osieberl/structEvoPress/EvoPress/struct_database/Llama-2-7b-hf --sparse_config_path /nfs/scistore19/alistgrp/osieberl/structEvoPress/EvoPress/struct_database/Llama-2-7b-hf/darwin-5.txt"}
if [[ $LMEVAL == "SIMPLE_ZERO_SHOTS" ]]; then
    echo "Running 0-shots"
    # simple 0-shots
    #--tasks arc_easy,arc_challenge,winogrande,hellaswag,piqa \
    python lmeval.py \
        --model hf \
        --model_args pretrained=$MODEL_ID,low_cpu_mem_usage=True,dtype=float16 \
        $MODEL_LOADING_KWARGS \
        --tasks sciq,piqa,winogrande,arc_easy,arc_challenge,hellaswag,logiqa,boolq \
        --batch_size 8
elif [[ $LMEVAL == "MMLU" ]]; then
    NUM_FEWSHOT=${NUM_FEWSHOT:-5}
    echo "Running 5-shot MMLU"
    # 5-shot MMLU
    python lmeval.py \
        --model hf \
        --model_args pretrained=$MODEL_ID,low_cpu_mem_usage=True,dtype=float16 \
        $MODEL_LOADING_KWARGS \
        --tasks mmlu \
        --batch_size 4 \
        --num_fewshot 5
elif [[ $LMEVAL == "GSM8K" ]]; then
    NUM_FEWSHOT=${NUM_FEWSHOT:-8}
    echo "Running ${C}-shot GSM8k"
    # 5-shot MMLU
    python lmeval.py \
        --model hf \
        --model_args pretrained=$MODEL_ID,low_cpu_mem_usage=True,dtype=float16 \
        $MODEL_LOADING_KWARGS \
        --tasks gsm8k \
        --batch_size 8 \
        --num_fewshot $NUM_FEWSHOT
else
    echo "Unknown $LMEVAL"
    exit 1
fi

