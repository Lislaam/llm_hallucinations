#!/bin/bash

# Activate the python environment.
source .venv/bin/activate
echo "Activated the virtual environment."

CUDA_LAUNCH_BLOCKING=1
# Set cuda visible devices
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=1,3
export CUDA_VISIBLE_DEVICES=$1
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Run main file which includes evaluation.
# "mistralai/Mistral-7B-Instruct-v0.3"
# "meta-llama/Meta-Llama-3-8B-Instruct"
# "google/gemma-1.1-7b-it"
#python3 sft_main.py --llm="mistralai/Mistral-7B-Instruct-v0.3"
python3 sft_main.py --llm="meta-llama/Meta-Llama-3-8B-Instruct"
#python3 sft_main.py --llm="google/gemma-1.1-7b-it"