import random
from constants import (
    PROMPT_INSTRUCTIONS,
    UNCERTAINTY_DOMAINS,
    BASELINE_METRICS,
    DATASET_LABELS,
)
import numpy as np
import pyarrow as pa
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import torch
from constants import PRE_POST_LABEL_TOKENS
import regex as re
import ast
from datasets import Dataset, concatenate_datasets


# Function to extract labels from prompt based on model-specific tokens
def extract_labels_from_prompt(prompt, model):
    pre_token, post_token = PRE_POST_LABEL_TOKENS[model]
    # Use a regular expression to find labels between the tokens
    pattern = re.escape(pre_token) + r"(.*?)" + re.escape(post_token)
    matches = re.findall(pattern, prompt)
    # Split each match on ": " and take the second half
    labels = [match.split(": ")[1] for match in matches]

    # Extracted labels. If mistral we need to skip the first laebl too as it is the dataset label. This is due to a lack of system prompts.
    if "mistral" in model:
        return labels[1:-1]
    else:
        return labels[:-1]  # Exclude the last label extracted from the prompt


def reformat_data_full_labels(dataset):
    """Reformats the dataset to have the same format for all datasets for consistency.

    Args:
        dataset: dataset -- dataset to reformat
        dataset_name: str -- name of the dataset

    Returns:
        dataset: dataset -- reformatted dataset
    """
    return dataset.filter(lambda x: error_type_map(x) is not None)


def reformat_data_split_labels(dataset, dataset_name):
    """Reformats the dataset to have the same format for all datasets for consistency.

    Args:
        dataset: dataset -- dataset to reformat
        dataset_name: str -- name of the dataset

    Returns:
        dataset: dataset -- reformatted dataset
    """
    def duplicate_and_label(example):
        """Duplicates examples with multiple error types, assigning one label per duplicate."""
        docs = []
        summs = []
        labels = []
        
        if example['errors'] is not None:
            try:
                lst = ast.literal_eval(example['errors'])
                for label in lst:
                    docs.append(example['doc'])
                    summs.append(example['summ'])
                    labels.append(label)
            except ValueError:  # If 'errors' is not a list, e.g., it is 'correct'
                docs.append(example['doc'])
                summs.append(example['summ'])
                labels.append(example['errors'])

        return [{'doc': doc, 'summ': summ, 'error_type': label} for doc, summ, label in zip(docs, summs, labels)]

    def process_in_chunks(dataset, chunk_size=10000, map_function=duplicate_and_label):
        chunked_tables = dataset.data.to_batches(max_chunksize=chunk_size)
        processed_chunks = []
        
        for chunk in chunked_tables:
            # Convert chunk to a PyArrow table
            chunk_table = pa.Table.from_batches([chunk])
            
            # Convert the chunk table to a pandas DataFrame
            chunk_df = chunk_table.to_pandas()
            
            if map_function:
                # Rename the column before splitting lists of errors into separate examples
                chunk_df = chunk_df.rename(columns={'error_type': 'errors'})
                
                # Apply the map function and flatten the result
                flattened_rows = chunk_df.apply(lambda row: map_function(row.to_dict()), axis=1).sum()
                
                # Convert the flattened list of dictionaries to a DataFrame
                chunk_df = pd.DataFrame(flattened_rows)
            
            processed_chunks.append(chunk_df)
        
        # Combine all processed chunks back into a single DataFrame
        combined_df = pd.concat(processed_chunks, ignore_index=True)
        
        return Dataset.from_pandas(combined_df)

    if dataset_name == "Lislaam/AggreFact":
        error_types = ['correct', 'intrinsic-NP', 'intrinsic-predicate', 'extrinsic-NP', 'extrinsic-predicate']
        dataset = process_in_chunks(dataset)
        dataset = dataset.filter(lambda x: x['error_type'] in error_types)
        #dataset = dataset.filter(lambda x: len(x['doc']) < 1800)
        #dataset = dataset.map(error_type_map)

    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    return dataset


def make_binary_dataset(dataset, error_type):
    # Map dataset into error / not_error. Choose one error type only. Requires reformatted_dataset
     binary_dataset = dataset.map(lambda x: True if x != error_type else False)
     return binary_dataset


def undersampling(dataset, error_types=['correct', 'intrinsic-NP', 'intrinsic-predicate', 'extrinsic-NP', 'extrinsic-predicate'],
                    n=400):
    def sample_class(dataset, error_type, n):
        filtered = dataset.filter(lambda x: x['error_type'] == error_type)
        return filtered.shuffle(seed=42).select(range(min(n, len(filtered))))

    # Sample 400 examples from each class
    sampled_dataset = Dataset.from_dict({
        'id': [],
        'doc': [],
        'summ': [],
        'error_type': []
    })

    for error_type in error_types:
        sampled = sample_class(dataset, error_type, n)
        sampled_dataset = concatenate_datasets([sampled_dataset, sampled])

    # Shuffle the final dataset
    sampled_dataset = sampled_dataset.shuffle(seed=42)

    return sampled_dataset


def oversampling(dataset, error_types=['correct', 'intrinsic-NP', 'intrinsic-predicate', 'extrinsic-NP', 'extrinsic-predicate'], n=2330):
    def replicate_class(dataset, error_type, n):
        filtered = dataset.filter(lambda x: x['error_type'] == error_type)
        num_examples = len(filtered)
        
        if num_examples == 0:
            return filtered  # Return empty dataset if no examples
        
        # Calculate how many times to replicate the dataset
        num_repeats = n // num_examples
        num_remaining = n % num_examples
        
        # Repeat the dataset and select the needed number of examples
        replicated = concatenate_datasets([filtered] * num_repeats)
        remaining = filtered.shuffle(seed=42).select(range(num_remaining))
        
        # Concatenate the replicated examples with the additional ones needed
        return concatenate_datasets([replicated, remaining])

    # Initialize an empty dataset for oversampling
    oversampled_dataset = Dataset.from_dict({
        'id': [],
        'doc': [],
        'summ': [],
        'error_type': []
    })

    for error_type in error_types:
        oversampled = replicate_class(dataset, error_type, n)
        oversampled_dataset = concatenate_datasets([oversampled_dataset, oversampled])

    # Shuffle the final dataset
    oversampled_dataset = oversampled_dataset.shuffle(seed=42)

    return oversampled_dataset


def sample_icl_examples(train_data, num_icl_examples):
    """
    Sample ICL examples from the training data.

    Args:
        train_data: dict -- training data
        num_icl_examples: int -- number of ICL examples to sample

    Returns:
        icl_examples: dict -- ICL examples
    """
    icl_examples = train_data.select(
        random.sample(range(len(train_data)), num_icl_examples)
    )
    return icl_examples


def construct_icl_prompt_msgs(original_example, icl_examples, dataset, llm):
    """
    Construct the ICL prompt for the ICL examples.

    Args:
        original_example: str -- original example
        icl_examples: List of dictionaries -- ICL examples
        dataset: str -- dataset name
        llm: str -- LLM model name

    Returns:
        prompt: str -- ICL prompt
    """
    # Extract the prompt instruciton

    # Include the instructions for the dataset.
    if "llama" in llm:
        messages = [
            {"role": "system", "content": PROMPT_INSTRUCTIONS[dataset]},
        ]
    #else:
        # Other models don't support system messages
        #   messages = [
        #      {"role": "user", "content": PROMPT_INSTRUCTIONS[dataset]},
        #     {"role": "assistant", "content": ASSISTANT_PROMPTS[dataset]},
        #]

    # Include the ICL examples.
    for icl_example in icl_examples:
        messages = []
        messages.append(
            {
                "role": "user",
                "content": r"Document: {/doc}\nSummary: {/summ}", # Added start token /E
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": r"Error Type: {/error_type}"
            }
        )

    # Include the test context.
    messages.append(
        {"role": "user", "content": original_example},
    )

    return messages


def soft_match(pred_processed, ref_processed, multiple_references=False):
    if multiple_references:
        # Check if any ref is within pred
        return (
            1
            if any(
                [
                    re.search(r"\b" + re.escape(r) + r"\b", pred_processed, re.IGNORECASE)
                    for r in ref_processed
                ]
            )
            else 0
        )
    else:
        # Check if ref is within pred
        return (
            1
            if re.search(r"\b" + re.escape(ref_processed) + r"\b", pred_processed, re.IGNORECASE)
            else 0
        )
    

def preprocess(text, model=None, error_types=['correct', 'intrinsic', 'extrinsic', 'intrinsic-NP', 'intrinsic-predicate', 'extrinsic-NP', 'extrinsic-predicate']):
    if model!= None and 'llama' in model:
       text = text.split("\n")[-1]  # Only consider the last part
       
    try:
       text = ast.literal_eval(text) # Deals with lists of error_types in string form
    except ValueError:
       text = re.sub(r"\p{P}(?<!-)", "", text)  # Remove punctuation except -
    except AttributeError or SyntaxError: # A none or long piece of text we didn't want
       print("Error in preprocessing this:", text)
       return ''
    return text


def get_score(predictions, references):
    #processed_preds = [preprocess(pred, model) for pred in predictions]
    processed_refs = [preprocess(ref) for ref in references] # Should always be processable

    flatten = lambda lst: [item for sublist in lst for item in (sublist if isinstance(sublist, list) else [sublist])]

    total = 0
    class_errors = {'extrinsic-NP': 0, 'extrinsic-predicate': 0, 'intrinsic-NP': 0,
                    'intrinsic-predicate': 0, 'correct': 0}

    num_extrinsicnp = sum([1 for ref in flatten(processed_refs) if ref == 'extrinsic-NP']) if 'extrinsic-NP' in flatten(processed_refs) else 1
    num_extrinsicpredicate = sum([1 for ref in flatten(processed_refs) if ref == 'extrinsic-predicate']) if 'extrinsic-predicate' in flatten(processed_refs) else 1
    num_intrinsicnp = sum([1 for ref in flatten(processed_refs) if ref == 'intrinsic-NP']) if 'intrinsic-NP' in flatten(processed_refs) else 1
    num_intrinsicpredicate = sum([1 for ref in flatten(processed_refs) if ref == 'intrinsic-predicate']) if 'intrinsic-predicate' in flatten(processed_refs) else 1
    num_correct = sum([1 for ref in flatten(processed_refs) if ref == 'correct']) if 'correct' in flatten(processed_refs) else 1

    # Check if any ref is within pred
    for i in range(len(processed_refs)):
        if type(processed_refs[i])==list:
            for x in processed_refs[i]:
                print(processed_refs[i], x, predictions[i], soft_match(predictions[i], x), '/n')
                if soft_match(predictions[i], x): # Check if that ref is in the pred
                    total += 1/len(processed_refs[i])
                    class_errors[x] += 1
        else:
            print(processed_refs[i], predictions[i], soft_match(predictions[i], processed_refs[i]), '/n')
            if soft_match(predictions[i], processed_refs[i]):
                total += 1
                class_errors[processed_refs[i]] += 1

    scores = {'total': total / len(processed_refs),
              'extrinsic-NP': class_errors["extrinsic-NP"] / num_extrinsicnp if 'extrinsic-NP' in flatten(processed_refs) else None,
              'extrinsic-predicate': class_errors["extrinsic-predicate"] / num_extrinsicpredicate if 'extrinsic-predicate' in flatten(processed_refs) else None,
              'intrinsic-NP': class_errors["intrinsic-NP"] / num_intrinsicnp if 'intrinsic-NP' in flatten(processed_refs) else None,
              'intrinsic-predicate': class_errors["intrinsic-predicate"] / num_intrinsicpredicate if 'intrinsic-predicate' in flatten(processed_refs) else None,
              'correct': class_errors["correct"] / num_correct if 'correct' in flatten(processed_refs) else None}
    
    #print(processed_refs)

    return scores


def error_type_map(example):
    # Any combination of error types is mapped into the same order.
    label_map = {
        "['extrinsic-NP']" : "['extrinsic-NP']",
        "['extrinsic-predicate']" : "['extrinsic-predicate']",
        "['intrinsic-NP']" : "['intrinsic-NP']",
        "['intrinsic-predicate']" : "['intrinsic-predicate']",
        'correct' : "['correct']",
        "['correct']" : "['correct']",

        "['extrinsic-NP', 'intrinsic-NP']" : "['extrinsic-NP', 'intrinsic-NP']",
        "['intrinsic-NP', 'extrinsic-NP']" : "['extrinsic-NP', 'intrinsic-NP']",
        "['extrinsic-predicate', 'intrinsic-predicate']" : "['extrinsic-predicate', 'intrinsic-predicate']", 
        "['intrinsic-predicate', 'extrinsic-predicate']" : "['extrinsic-predicate', 'intrinsic-predicate']",
        "['extrinsic-NP', 'extrinsic-predicate']" : "['extrinsic-NP', 'extrinsic-predicate']",
        "['extrinsic-predicate', 'extrinsic-NP']" : "['extrinsic-NP', 'extrinsic-predicate']",
        "['intrinsic-predicate', 'extrinsic-NP']" : "['intrinsic-predicate', 'extrinsic-NP']",
        "['extrinsic-NP', 'intrinsic-predicate']" : "['intrinsic-predicate', 'extrinsic-NP']",
        "['extrinsic-predicate', 'intrinsic-NP']" : "['extrinsic-predicate', 'intrinsic-NP']",
        "['intrinsic-NP', 'extrinsic-predicate']" : "['extrinsic-predicate', 'intrinsic-NP']",
        "['intrinsic-NP', 'intrinsic-predicate']" : "['intrinsic-NP', 'intrinsic-predicate']",
        "['intrinsic-predicate', 'intrinsic-NP']" : "['intrinsic-NP', 'intrinsic-predicate']",

        "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP']" : "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP']",
        "['extrinsic-NP', 'intrinsic-NP', 'extrinsic-predicate']" : "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP']",
        "['intrinsic-NP', 'extrinsic-predicate', 'extrinsic-NP']" : "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP']",
        "['intrinsic-NP', 'extrinsic-NP', 'extrinsic-predicate']" : "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP']",
        "['extrinsic-predicate', 'intrinsic-NP', 'extrinsic-NP']" : "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP']",
        "['extrinsic-predicate', 'extrinsic-NP', 'intrinsic-NP']" : "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP']",
         
        "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-predicate']" : "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-predicate']",
        "['extrinsic-NP', 'intrinsic-predicate', 'extrinsic-predicate']" : "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-predicate']",
        "['intrinsic-predicate', 'extrinsic-predicate', 'extrinsic-NP']" : "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-predicate']",
        "['intrinsic-predicate', 'extrinsic-NP', 'extrinsic-predicate']" : "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-predicate']",
        "['extrinsic-predicate', 'intrinsic-predicate', 'extrinsic-NP']" : "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-predicate']",
        "['extrinsic-predicate', 'extrinsic-NP', 'intrinsic-predicate']" : "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-predicate']",

        "['extrinsic-NP', 'intrinsic-NP', 'intrinsic-predicate']" : "['extrinsic-NP', 'intrinsic-NP', 'intrinsic-predicate']",
        "['extrinsic-NP', 'intrinsic-predicate', 'intrinsic-NP']" : "['extrinsic-NP', 'intrinsic-NP', 'intrinsic-predicate']",
        "['intrinsic-predicate', 'intrinsic-NP', 'extrinsic-NP']" : "['extrinsic-NP', 'intrinsic-NP', 'intrinsic-predicate']",
        "['intrinsic-predicate', 'extrinsic-NP', 'intrinsic-NP']" : "['extrinsic-NP', 'intrinsic-NP', 'intrinsic-predicate']",
        "['intrinsic-NP', 'intrinsic-predicate', 'extrinsic-NP']" : "['extrinsic-NP', 'intrinsic-NP', 'intrinsic-predicate']",
        "['intrinsic-NP', 'extrinsic-NP', 'intrinsic-predicate']" : "['extrinsic-NP', 'intrinsic-NP', 'intrinsic-predicate']",

        "['extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']" : "['extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']",
        "['extrinsic-predicate', 'intrinsic-predicate', 'intrinsic-NP']" : "['extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']",
        "['intrinsic-predicate', 'intrinsic-NP', 'extrinsic-predicate']" : "['extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']",
        "['intrinsic-predicate', 'extrinsic-predicate', 'intrinsic-NP']" : "['extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']",
        "['intrinsic-NP', 'intrinsic-predicate', 'extrinsic-predicate']" : "['extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']",
        "['intrinsic-NP', 'extrinsic-predicate', 'intrinsic-predicate']" : "['extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']",

        "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']" : "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']",
        "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-predicate', 'intrinsic-NP']" : "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']",
        "['extrinsic-NP', 'intrinsic-predicate', 'extrinsic-predicate', 'intrinsic-NP']" : "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']",
        "['extrinsic-NP', 'intrinsic-predicate', 'intrinsic-NP', 'extrinsic-predicate']" : "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']",
        "['extrinsic-NP', 'intrinsic-predicate', 'intrinsic-NP', 'extrinsic-predicate']" : "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']",
        "['extrinsic-NP', 'intrinsic-NP', 'intrinsic-predicate', 'extrinsic-predicate']" : "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']",
        "['extrinsic-NP', 'intrinsic-NP', 'extrinsic-predicate', 'intrinsic-predicate']" : "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']",
        "['extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate', 'extrinsic-NP']" : "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']",
        "['intrinsic-NP', 'extrinsic-predicate', 'intrinsic-predicate', 'extrinsic-NP']" : "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']",
        }
    
    try:
        example['error_type'] = label_map[example['error_type']]
        return example
    
    except KeyError:
        return
