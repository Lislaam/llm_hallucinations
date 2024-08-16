#!/bin/bash

# Activate the python environment.
source .venv/bin/activate
echo "Activated the virtual environment."

# Set cuda visible devices
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=1,3
export CUDA_VISIBLE_DEVICES=$1
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Run main file which includes evaluation.
# "mistralai/Mistral-7B-Instruct-v0.3"
# "meta-llama/Meta-Llama-3-8B-Instruct"
# "google/gemma-1.1-7b-it"
python3 main.py --llms="mistralai/Mistral-7B-Instruct-v0.3" --num_icl_examples=0 --num_beams=3 --llm_device='cuda'