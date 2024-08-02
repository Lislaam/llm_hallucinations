#!/bin/bash

# Activate the python environment.
source .venv/bin/activate
echo "Activated the virtual environment."

# Set cuda visible devices
export CUDA_VISIBLE_DEVICES=$1

# Run main file which includes evaluation.
python3 main.py --llms="meta-llama/Meta-Llama-3-8B-Instruct"