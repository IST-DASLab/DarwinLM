#!/bin/bash
export TRANSFORMERS_CACHE=/nfs/scistore19/alistgrp/huggingface/hub
export CUDA_VISIBLE_DEVICES=0,1
# Please modify your model accordingly
MODEL_ID="Shengkun/DarwinLM-2.7B-Pruned"
LMEVAL=${LMEVAL:-"SIMPLE_ZERO_SHOTS"}

port=23001
#Path to save the evaluation results
output_path=/nfs/scistore19/alistgrp/stang/StructEvoPress/eval_results


if [[ $LMEVAL == "SIMPLE_ZERO_SHOTS" ]]; then
    echo "Running 0-shots"
    # simple 0-shots
    rm -rf ${output_path}/*

    accelerate launch --main_process_port ${port} lmeval.py\
        --model hf \
        --model_args pretrained=$MODEL_ID,low_cpu_mem_usage=True,dtype=float16,trust_remote_code=True \
        $MODEL_LOADING_KWARGS \
        --tasks sciq,arc_easy,piqa,logiqa,boolq \
        --batch_size 8\
        --output_path ${output_path}/results_sciq.json
    
    accelerate launch --main_process_port ${port} lmeval.py\
        --model hf \
        --model_args pretrained=$MODEL_ID,low_cpu_mem_usage=True,dtype=float16,trust_remote_code=True \
        $MODEL_LOADING_KWARGS \
        --tasks arc_challenge \
        --num_fewshot 25\
        --batch_size 4\
        --output_path ${output_path}/results_arc_challenge.json
    
    accelerate launch --main_process_port ${port} lmeval.py\
        --model hf \
        --model_args pretrained=$MODEL_ID,low_cpu_mem_usage=True,dtype=float16,trust_remote_code=True \
        $MODEL_LOADING_KWARGS \
        --tasks winogrande \
        --num_fewshot 5\
        --batch_size 8\
        --output_path ${output_path}/results_winogrande.json
    
    accelerate launch --main_process_port ${port} lmeval.py\
        --model hf \
        --model_args pretrained=$MODEL_ID,low_cpu_mem_usage=True,dtype=float16,trust_remote_code=True \
        $MODEL_LOADING_KWARGS \
        --tasks hellaswag \
        --num_fewshot 10\
        --batch_size 4\
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

