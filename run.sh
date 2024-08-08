#!/bin/bash

# Activate the python environment.
source .venv/bin/activate
echo "Activated the virtual environment."

# Set cuda visible devices
export CUDA_VISIBLE_DEVICES=$1

# Run main file which includes evaluation.
# "mistralai/Mistral-7B-Instruct-v0.3"
# "meta-llama/Meta-Llama-3-8B-Instruct"
python3 main.py --llms="meta-llama/Meta-Llama-3-8B-Instruct" --num_icl_examples=0