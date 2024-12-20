#!/bin/bash

# Activate the python environment.
source .venv/bin/activate
echo "Activated the virtual environment."

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
    # "python3 sft_main.py --llm='meta-llama/Meta-Llama-3-8B-Instruct' --do_fine_tune='n' --dir='extrinsic_intrinsic_no_error_span'"
    # "python3 sft_main.py --llm='mistralai/Mistral-7B-Instruct-v0.3' --do_fine_tune='n' --dir='extrinsic_intrinsic_no_error_span'"
    # "python3 sft_main.py --llm='google/gemma-1.1-7b-it' --do_fine_tune='n' --dir='extrinsic_intrinsic_no_error_span'"

    # "python3 sft_main.py --llm='meta-llama/Meta-Llama-3-8B-Instruct' --do_fine_tune='y' --dir='extrinsic_intrinsic_no_error_span'"
    # "python3 sft_main.py --llm='mistralai/Mistral-7B-Instruct-v0.3' --do_fine_tune='y' --dir='extrinsic_intrinsic_no_error_span'"
    #"python3 sft_main.py --llm='google/gemma-1.1-7b-it' --do_fine_tune='y' --dir='extrinsic_intrinsic_no_error_span'"

    "python3 sft_main2.py --llm='meta-llama/Meta-Llama-3-8B-Instruct' --do_fine_tune='n' --dir='extrinsic_intrinsic_with_error_span'"
    "python3 sft_main2.py --llm='mistralai/Mistral-7B-Instruct-v0.3' --do_fine_tune='n' --dir='extrinsic_intrinsic_with_error_span'"
    # "python3 sft_main2.py --llm='google/gemma-1.1-7b-it' --do_fine_tune='n' --dir='extrinsic_intrinsic_with_error_span'"

    "python3 sft_main2.py --llm='meta-llama/Meta-Llama-3-8B-Instruct' --do_fine_tune='y' --dir='extrinsic_intrinsic_with_error_span'"
    "python3 sft_main2.py --llm='mistralai/Mistral-7B-Instruct-v0.3' --do_fine_tune='y' --dir='extrinsic_intrinsic_with_error_span'"
)

# Iterate over each command in the list
for cmd in "${commands[@]}"; do
    run_with_retries "$cmd"
done


echo "All commands executed successfully."