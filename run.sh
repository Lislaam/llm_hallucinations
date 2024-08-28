#!/bin/bash

# Activate the python environment.
source .venv/bin/activate
echo "Activated the virtual environment."

CUDA_LAUNCH_BLOCKING=1
# Set cuda visible devices
export CUDA_VISIBLE_DEVICES=$1
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Run main file which includes evaluation.
# "mistralai/Mistral-7B-Instruct-v0.3"
# "meta-llama/Meta-Llama-3-8B-Instruct"
# "google/gemma-1.1-7b-it"

# Maximum number of retries for each command
max_retries=5

# Function to run a command with retries
run_with_retries() {
    local command="$1"
    local count=0

    # Run the command in a while loop
    while ! eval "$command"; do
        ((count++))
        if [[ $count -ge $max_retries ]]; then
            echo "Command failed after $count attempts: $command"
            return
        fi
        echo "Attempt $count failed. Retrying in 2 seconds..."
        sleep 2
    done

    echo "Command succeeded after $count attempts: $command"
    return 0
}

# List of commands to run
commands=(
    "python3 sft_main.py --llm='meta-llama/Meta-Llama-3-8B-Instruct' --sampling='undersampling'"
    "python3 sft_main.py --llm='google/gemma-1.1-7b-it' --sampling='undersampling'"
    "python3 sft_main.py --llm='google/gemma-1.1-7b-it'"
    "python3 sft_main.py --llm='meta-llama/Meta-Llama-3-8B-Instruct' --sampling='oversampling'"
    "python3 sft_main.py --llm='google/gemma-1.1-7b-it' --sampling='oversampling'"
)

# Iterate over each command in the list
for cmd in "${commands[@]}"; do
    run_with_retries "$cmd"
done


echo "All commands executed successfully."