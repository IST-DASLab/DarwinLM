#!/bin/bash
export TRANSFORMERS_CACHE=/nfs/scistore19/alistgrp/huggingface/hub
# export CUDA_VISIBLE_DEVICES=6,7,8,9
MODEL="Llama-3.1-8B" #Llama-2-7B 
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
elif [[ $MODEL == Llama-3.2-1B ]]; then
    MODEL_ID=meta-llama/Llama-3.2-1B
else
    echo "Unknown model"
    exit 1
fi

# MODEL_ID=nvidia/Llama-3.1-Minitron-4B-Depth-Base
port=23001
output_path=/nfs/scistore19/alistgrp/stang/StructEvoPress/eval_results
# ./z_scripts/z_run_lmeval.sh

MODEL_LOADING_KWARGS=${MODEL_LOADING_KWARGS:-"--sparse_weights_path /nfs/scistore19/alistgrp/stang/StructEvoPress/db/ziplm_llama3.1-8B_2048/Llama-3.1-8B --sparse_config_path /nfs/scistore19/alistgrp/stang/StructEvoPress/llama-3.1-8B-intermediate.txt"}


if [[ $LMEVAL == "SIMPLE_ZERO_SHOTS" ]]; then
    echo "Running 0-shots"
    # simple 0-shots
    # --tasks arc_easy,arc_challenge,winogrande,hellaswag,piqa \
    rm -rf ${output_path}/*
    accelerate launch --main_process_port ${port} lmeval.py \
        --model hf \
        --model_args pretrained=$MODEL_ID,low_cpu_mem_usage=True,dtype=float16 \
        $MODEL_LOADING_KWARGS \
        --tasks mmlu \
        --batch_size 4 \
        --num_fewshot 5 \
        --output_path ${output_path}/results_mmlu.json
    # accelerate launch --main_process_port ${port} lmeval.py\
    #     --model hf \
    #     --model_args pretrained=$MODEL_ID,low_cpu_mem_usage=True,dtype=float16 \
    #     $MODEL_LOADING_KWARGS \
    #     --tasks sciq \
    #     --batch_size 8\
    #     --output_path ${output_path}/results_sciq.json

    accelerate launch --main_process_port ${port} lmeval.py\
        --model hf \
        --model_args pretrained=$MODEL_ID,low_cpu_mem_usage=True,dtype=float16 \
        $MODEL_LOADING_KWARGS \
        --tasks sciq,arc_easy,piqa,logiqa,boolq \
        --batch_size 8\
        --output_path ${output_path}/results_sciq.json
    
    accelerate launch --main_process_port ${port} lmeval.py\
        --model hf \
        --model_args pretrained=$MODEL_ID,low_cpu_mem_usage=True,dtype=float16 \
        $MODEL_LOADING_KWARGS \
        --tasks arc_challenge \
        --num_fewshot 25\
        --batch_size 8\
        --output_path ${output_path}/results_arc_challenge.json
    
    accelerate launch --main_process_port ${port} lmeval.py\
        --model hf \
        --model_args pretrained=$MODEL_ID,low_cpu_mem_usage=True,dtype=float16 \
        $MODEL_LOADING_KWARGS \
        --tasks winogrande \
        --num_fewshot 5\
        --batch_size 8\
        --output_path ${output_path}/results_winogrande.json
    
    accelerate launch --main_process_port ${port} lmeval.py\
        --model hf \
        --model_args pretrained=$MODEL_ID,low_cpu_mem_usage=True,dtype=float16 \
        $MODEL_LOADING_KWARGS \
        --tasks hellaswag \
        --num_fewshot 10\
        --batch_size 8\
        --output_path ${output_path}/results_hellaswag.json

elif [[ $LMEVAL == "MMLU" ]]; then
    NUM_FEWSHOT=${NUM_FEWSHOT:-5}
    echo "Running 5-shot MMLU"
    # 5-shot MMLU
    accelerate launch --main_process_port ${port} lmeval.py \
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
    accelerate launch --main_process_port ${port} lmeval.py \
        --model hf \
        --model_args pretrained=$MODEL_ID,low_cpu_mem_usage=True,dtype=float16 \
        $MODEL_LOADING_KWARGS \
        --tasks gsm8k \
        --batch_size 8 \
        --num_fewshot $NUM_FEWSHOT
elif [[ $LMEVAL == "wiki" ]]; then
    NUM_FEWSHOT=${NUM_FEWSHOT:-8}
    echo "Running ${C}-shot GSM8k"
    # 5-shot MMLU
    accelerate launch --main_process_port ${port} lmeval.py \
        --model hf \
        --model_args pretrained=$MODEL_ID,low_cpu_mem_usage=True,dtype=float16 \
        $MODEL_LOADING_KWARGS \
        --tasks wiki \
        --batch_size 8 \
        --num_fewshot $NUM_FEWSHOT
else
    echo "Unknown $LMEVAL"
    exit 1
fi

