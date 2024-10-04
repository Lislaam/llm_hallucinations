import pip_system_certs
import os
from transformers import AutoModel, AutoTokenizer
from sft import *
from constants import SYSTEM_INSTRUCTION, BINARY_INSTRUCTION, SYSTEM_INSTRUCTION2
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, SFTConfig
from peft import LoraConfig
import ast
import torch

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

response = requests.get('https://demoapi.demo.clear.ml', verify=False)

# os.environ["CURL_CA_BUNDLE"] = ''
model = AutoModel.from_pretrained("google/gemma-1.1-7b-it")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-7b-it")


def formatting_prompts_func(example, training=True):
    instruction = SYSTEM_INSTRUCTION
    output_texts = []
    for i in range(len(example["error_type"])):
        text = f"{instruction}\n ### ORIGINAL_TEXT: {example['doc'][i]}\n ### SUMMARY: {example['summ'][i]}\n ### Output: " #  ### ERROR_LOCATIONS: {example['annotated_span'][i]}\n ### ERROR_CORRECTIONS: {example['annotated_corrections'][i]}\n
        if training:
            if instruction == SYSTEM_INSTRUCTION or instruction == BINARY_INSTRUCTION:
                text += (
                    f"{LABEL_CONVERSIONS[example['error_type'][i]]}." + tokenizer.eos_token
                )
            else:
                label = ast.literal_eval(example["error_type"][i])
                text += f"{str(len(label))}, "
                for l in label:
                    text += f"{LABEL_CONVERSIONS[l]}, "
                text.removesuffix(', ')
                text += '.'
                text += tokenizer.eos_token

        output_texts.append(text)
    
    return output_texts

# # Load the dataset
dataset = Dataset.from_file('correct_incorrect_data/data-00000-of-00001.arrow')
dataset = dataset.remove_columns([col for col in dataset.column_names if dataset.filter(lambda x: x[col] is None or x[col] == '').num_rows > 0])

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = int(0.1 * len(dataset))

train_dataset = dataset.select(range(train_size))
val_dataset = dataset.select(range(train_size, train_size + val_size))
test_dataset = dataset.select(range(train_size + val_size, train_size + val_size + test_size))

tokenizer.padding_side = "right"
tokenizer.pad_token_id = tokenizer.eos_token_id

lora_config = LoraConfig(
    r=8,  # Was 16 for mistral and llama trainings oversample/undersample. NOT during binary dataset
    lora_alpha=8,
    bias="none",
    task_type="CAUSAL_LM",
)

# Test formatting for 1st and 2nd example:
# print(formatting_prompts_func(dataset['train'][:1], True))

# Tokenize the labels for constrained decoding
force_tokens = list(LABEL_CONVERSIONS.values())
force_tokens = [f" {token}" for token in force_tokens]
force_token_ids = tokenizer(
    force_tokens, add_special_tokens=False
).input_ids

response_template = " ### Output:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    do_train=True,
    num_train_epochs=1,
    max_seq_length=2500,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=250,
    save_steps=250,
    bf16=True,
    metric_for_best_model="eval_loss",
    per_device_train_batch_size=2,
    load_best_model_at_end = True,
)

trainer = SFTTrainer( 
    model=model,
    train_dataset= train_dataset, #dataset['train'],
    eval_dataset= val_dataset, #dataset['validation'],
    args=sft_config,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    peft_config=lora_config,
    tokenizer=tokenizer,
)

trainer.train()