import random
from constants import (
    PROMPT_INSTRUCTIONS,
    UNCERTAINTY_DOMAINS,
    BASELINE_METRICS,
    DATASET_LABELS,
)
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import torch
from constants import PRE_POST_LABEL_TOKENS
import regex as re
import ast


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


def error_type_map(example):
    # Any combination of error types is mapped into the same order.
    label_map = {
        "['extrinsic-NP']" : "['extrinsic-NP']",
        "['extrinsic-predicate']" : "['extrinsic-predicate']",
        "['intrinsic-NP']" : "['intrinsic-NP']",
        "['intrinsic-predicate']" : "['intrinsic-predicate']",
        "correct" : "correct",
        "['correct']" : "correct",

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
    
    example['error_type'] = label_map[example['error_type']]

    return example


def reformat_data(dataset, dataset_name):
    """Reformats the dataset to have the same format for all datasets for consitency.

    Args:
        dataset: dataset -- dataset to reformat
        dataset_name: str -- name of the dataset

    Returns:
        dataset: dataset -- reformatted dataset
    """

    def output_map(label):
        if label == '[]':
            return ['null']
        try:
            # Deals with lists of error_types in string form, removing []
            label = ast.literal_eval(label) # It is a list of strings now 
        except ValueError:
            label = [label]
        return label   

    if dataset_name == "Lislaam/AggreFact":
        error_types = ['correct', 'intrinsic', 'extrinsic', 'intrinsic-NP', 
                'intrinsic-predicate', 'extrinsic-NP','extrinsic-predicate']

        dataset = dataset.filter(lambda x: all(x in error_types for x in output_map(x['error_type'])))
        #dataset = dataset.filter(lambda x: len(x['doc']) < 1000)
        dataset = dataset.map(error_type_map)

    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    return dataset


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


def get_score(predictions, references, model):
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
                print(processed_refs[i], x, predictions[i], soft_match(predictions[i], x))
                #import pdb; pdb.set_trace()
                if soft_match(predictions[i], x): # Check if that ref is in the pred
                    total += 1/len(processed_refs[i])
                    class_errors[x] += 1
        else:
            print(processed_refs[i], predictions[i], soft_match(predictions[i], processed_refs[i]))
            #import pdb; pdb.set_trace()
            if soft_match(predictions[i], processed_refs[i]):
                total += 1
                class_errors[processed_refs[i]] += 1

    # zipped = zip(processed_preds, processed_refs)

    # for pred, ref in zipped:
    #     if type(ref)==list:
    #         for i in ref:
    #             if i in pred:
    #                 total += 1/len(ref)
    #                 class_errors[i] += 1
    #     else:
    #         if ref in pred:
    #             total += 1
    #             class_errors[ref] += 1

    scores = {'total': total / len(processed_refs),
              'extrinsic-NP': class_errors["extrinsic-NP"] / num_extrinsicnp if 'extrinsic-NP' in flatten(processed_refs) else None,
              'extrinsic-predicate': class_errors["extrinsic-predicate"] / num_extrinsicpredicate if 'extrinsic-predicate' in flatten(processed_refs) else None,
              'intrinsic-NP': class_errors["intrinsic-NP"] / num_intrinsicnp if 'intrinsic-NP' in flatten(processed_refs) else None,
              'intrinsic-predicate': class_errors["intrinsic-predicate"] / num_intrinsicpredicate if 'intrinsic-predicate' in flatten(processed_refs) else None,
              'correct': class_errors["correct"] / num_correct if 'correct' in flatten(processed_refs) else None}
    
    #print(processed_refs)

    return scores


def compute_accuracy(results):
    """
    Calculate the accuracy of model predictions.

    Args:
        results: List of dictionaries containing 'posterior_variance' and 'true_label'.

    Returns:
        accuracy: Accuracy of the model predictions.
    """
    accuracies = {}

    true_labels = [result["true_label"] for result in results.values()]
    predictions = [result["llm_prediction"] for result in results.values()]

    correcntess = [
        1 if true_label == prediction else 0
        for true_label, prediction in zip(true_labels, predictions)
    ]

    for metric in UNCERTAINTY_DOMAINS:

        accuracy = np.mean(correcntess)

        accuracies[metric] = accuracy

    for metric in BASELINE_METRICS:

        accuracy = np.mean(correcntess)

        accuracies[metric] = accuracy

    return accuracies


def compute_prr(results):
    """
    Calculate the Prediction Rejection Area Ratio (PRR) for model predictions.

    Args:
        results: List of dictionaries containing 'posterior_variance' and 'true_label'.

    Returns:
        prr_score: PRR score for the predictions.
    """

    prr_scores = {}

    true_labels = [result["true_label"] for result in results.values()]
    predictions = [result["llm_prediction"] for result in results.values()]

    labels = [
        1 if true_label == prediction else 0
        for true_label, prediction in zip(true_labels, predictions)
    ]

    flipped_labels = [1 - label for label in labels]

    for metric in UNCERTAINTY_DOMAINS:
        uncertainties = [
            result[metric][f"{metric}_neg_log_likelihood"]
            for result in results.values()
        ]

        base_error = np.mean(flipped_labels)

        if base_error == 1:
            return 0

        total_data = len(labels)
        sorted_indices = np.argsort(uncertainties)
        sorted_errors = np.array(flipped_labels)[sorted_indices]

        cumulative_errors = np.cumsum(sorted_errors) / total_data
        percentages = np.arange(total_data) / total_data

        auc_uncertainty = metrics.auc(percentages, cumulative_errors)
        random_errors = np.linspace(base_error, 0, total_data)
        auc_random = metrics.auc(percentages, random_errors)

        num_misclassifications = np.sum(flipped_labels)
        oracle_percentages = np.linspace(
            0, num_misclassifications / total_data, num=num_misclassifications
        )
        oracle_errors = np.linspace(base_error, 0, num=num_misclassifications)
        auc_oracle = metrics.auc(oracle_percentages, oracle_errors)

        prr = (
            (auc_uncertainty - auc_random) / (auc_oracle - auc_random)
            if (auc_oracle - auc_random) != 0
            else 0
        )

        prr_scores[metric] = prr

    # Now do the same for the baseline metrics
    for metric in BASELINE_METRICS:
        uncertainties = [result[metric][metric] for result in results.values()]

        base_error = np.mean(flipped_labels)

        if base_error == 1:
            return 0

        total_data = len(labels)
        sorted_indices = np.argsort(uncertainties)
        sorted_errors = np.array(flipped_labels)[sorted_indices]

        cumulative_errors = np.cumsum(sorted_errors) / total_data
        percentages = np.arange(total_data) / total_data

        auc_uncertainty = metrics.auc(percentages, cumulative_errors)
        random_errors = np.linspace(base_error, 0, total_data)
        auc_random = metrics.auc(percentages, random_errors)

        num_misclassifications = np.sum(flipped_labels)
        oracle_percentages = np.linspace(
            0, num_misclassifications / total_data, num=num_misclassifications
        )
        oracle_errors = np.linspace(base_error, 0, num=num_misclassifications)
        auc_oracle = metrics.auc(oracle_percentages, oracle_errors)

        prr = (
            (auc_uncertainty - auc_random) / (auc_oracle - auc_random)
            if (auc_oracle - auc_random) != 0
            else 0
        )

        prr_scores[metric] = prr

    return prr_scores


def compute_auprc(results):
    """
    Calculate the Area Under the Precision-Recall Curve (AUPRC) for model predictions.

    Args:
        results: List of dictionaries containing 'posterior_variance' and 'true_label'.

    Returns:
        auprc_score: AUPRC score for the predictions.
    """
    auprc_scores = {}

    true_labels = [result["true_label"] for result in results.values()]
    predictions = [result["llm_prediction"] for result in results.values()]

    correctness_labels = [
        1 if true_label == prediction else 0
        for true_label, prediction in zip(true_labels, predictions)
    ]

    flipped_labels = [1 - label for label in correctness_labels]

    for metric in UNCERTAINTY_DOMAINS:
        uncertainties = [
            result[metric][f"{metric}_neg_log_likelihood"]
            for result in results.values()
        ]

        if np.mean(flipped_labels) == 0:
            return 0

        precision, recall, _ = metrics.precision_recall_curve(
            flipped_labels, uncertainties
        )
        auprc_score = metrics.auc(recall, precision)

        auprc_scores[metric] = auprc_score

    # Now do the same for the baseline metrics
    for metric in BASELINE_METRICS:
        uncertainties = [result[metric][metric] for result in results.values()]

        if np.mean(flipped_labels) == 0:
            return 0

        precision, recall, _ = metrics.precision_recall_curve(
            flipped_labels, uncertainties
        )
        auprc_score = metrics.auc(recall, precision)

        auprc_scores[metric] = auprc_score

    return auprc_scores


def compute_auroc(labels, metric_values):
    """
    Calculate the Area Under the ROC curve (AUROC) for model predictions.

    Args:
        results: List of dictionaries containing 'posterior_variance' and 'true_label'.

    Returns:
        auroc_score: AUROC score for the predictions.
    """

    fpr, tpr, _ = metrics.roc_curve(labels, metric_values)
    auroc_score = metrics.auc(fpr, tpr)

    return auroc_score


@torch.no_grad()
def find_ordered_label_positions(
    input_ids, tokenized_labels, ignore_tokens, num_icl_examples
):
    """
    Find the ordered positions of label tokens in the input IDs and capture the labels matched along with their token sequences.

    Args:
        input_ids (Tensor): Input IDs.
        tokenized_labels (List[Tensor]): Tokenized labels.
        num_icl_examples (int): Number of in-context learning examples.

    Returns:
        List[Tuple[int, int]]: Positions of the label tokens.
        List[List[int]]: Actual token sequences of the matched labels.
    """
    start_ignore_token, end_ignore_token = ignore_tokens

    positions = []
    matched_labels_tokens = []
    i = 0
    while i < len(input_ids):
        found = False
        for idx, label_tokens in enumerate(tokenized_labels):
            window_size = len(label_tokens)
            if i + window_size <= len(input_ids) and torch.equal(
                input_ids[i : i + window_size], label_tokens
            ):
                positions.append(
                    (
                        i + len(start_ignore_token),
                        i + window_size - 1 - len(end_ignore_token),
                    )
                )
                matched_labels_tokens.append(
                    label_tokens[
                        len(start_ignore_token) : -len(end_ignore_token)
                    ].tolist()
                )  # Convert tensor to list for easier handling
                i += window_size  # Move index past this label
                found = True
                break
        if not found:
            i += 1  # Only increment if no match was found to prevent skipping elements

    assert (
        len(positions) == num_icl_examples + 1
    ), "Mismatch between expected and found label positions"
    return positions, matched_labels_tokens


def find_answer_label_positions(input_ids, tokenized_labels, ignore_tokens):
    """
    Find the ordered positions of label tokens in the input IDs and capture the labels matched along with their token sequences.

    Args:
        input_ids (Tensor): Input IDs.
        tokenized_labels (List[Tensor]): Tokenized labels.

    Returns:
        List[Tuple[int, int]]: Positions of the label tokens.
        List[List[int]]: Actual token sequences of the matched labels.
    """
    start_ignore_token, end_ignore_token = ignore_tokens

    positions = []
    matched_labels_tokens = []
    i = 0
    while i < len(input_ids):
        found = False
        for idx, label_tokens in enumerate(tokenized_labels):
            window_size = len(label_tokens)
            if i + window_size <= len(input_ids) and torch.equal(
                input_ids[i : i + window_size], label_tokens
            ):
                positions.append(
                    (
                        i + len(start_ignore_token),
                        i + window_size - 1 - len(end_ignore_token),
                    )
                )
                matched_labels_tokens.append(
                    label_tokens[
                        len(start_ignore_token) : -len(end_ignore_token)
                    ].tolist()
                )  # Convert tensor to list for easier handling
                i += window_size  # Move index past this label
                found = True
                break
        if not found:
            i += 1  # Only increment if no match was found to prevent skipping elements

    # We only want the last one here as we are only interested in the answer label.
    answer_posiiton = positions[-1]
    answer_label_tokens = torch.tensor(matched_labels_tokens[-1])

    return answer_posiiton, answer_label_tokens


@torch.no_grad()
def extract_logits(logits, label_positions):
    """
    Extract logits for the given label positions.

    Args:
        logits (Tensor): Logits output from the model.
        label_positions (List[Tuple[int, int]]): Positions of the label tokens.

    Returns:
        List[Tensor]: Extracted logits for the label positions.
    """
    extracted_logits = []
    for start, end in label_positions:
        extracted_logits.append(logits[start - 1 : end, :])

    return extracted_logits


@torch.no_grad()
def pad_logits(logits, max_length):
    """
    Pad the list of logits to ensure they all have the same length in the token_in_label dimension.

    Args:
        logits (List[torch.Tensor]): List of logits with varying lengths.
        max_length (int): The length to which all logits should be padded.
        vocab_size (int): The size of the vocabulary.

    Returns:
        torch.Tensor: A tensor of shape (label_position, max_length, vocab_size) with padded logits.
    """
    padded_logits = []

    # Extract the vocabular size from the first logit tensor
    vocab_size = logits[0].shape[-1]

    for logit in logits:
        if logit.shape[0] < max_length:
            padding = torch.zeros(max_length - logit.shape[0], vocab_size).to(
                logit.device
            )
            padded_logit = torch.cat([logit, padding], dim=0)
        else:
            padded_logit = logit
        padded_logits.append(padded_logit)
    return torch.stack(padded_logits)


def get_max_token_in_label_length(tokenized_labels):
    return max([len(label) for label in tokenized_labels])


def create_plots(overall_results_dict, model_name, dataset_name, save_dir, shots):
    metrics = ["accuracy", "prr", "auprc", "auroc"]
    # Assuming UNCERTAINTY_DOMAINS and BASELINE_METRICS are imported

    # Create a subplot for each uncertainty metric with one row per domain and four columns for metrics
    fig, axs = plt.subplots(
        nrows=len(UNCERTAINTY_DOMAINS),
        ncols=len(metrics),
        figsize=(
            20,
            10 + 2 * len(UNCERTAINTY_DOMAINS),
        ),  # Adjust height based on the number of uncertainty domains
        sharex=True,
        sharey=False,  # Each subplot can independently manage y-axis scales
    )
    fig.suptitle(
        f"{model_name} on {dataset_name}", fontsize=16, fontweight="bold", y=0.99
    )  # Adjust super title position

    # Initialize a dictionary to track the minimum and maximum y-values across columns for alignment
    column_y_limits = {i: (float("inf"), float("-inf")) for i in range(len(metrics))}

    for i, uncertainty_domain in enumerate(UNCERTAINTY_DOMAINS):
        for j, metric in enumerate(metrics):
            ax = axs[i, j]
            # Plot the main metric for the current uncertainty domain
            main_values = [
                overall_results_dict[shot][metric][uncertainty_domain] for shot in shots
            ]
            ax.plot(
                shots, main_values, marker="o", label=f"{metric} ({uncertainty_domain})"
            )

            # Overlay each baseline metric on each subplot=
            for baseline_metric in BASELINE_METRICS:
                baseline_values = [
                    overall_results_dict[shot][metric][baseline_metric]
                    for shot in shots
                ]
                ax.plot(
                    shots,
                    baseline_values,
                    marker="x",
                    linestyle="--",
                    label=f"Baseline ({baseline_metric})",
                )

            # Combine values to update y-limits
            combined_values = main_values + baseline_values
            current_min, current_max = min(combined_values), max(combined_values)
            if current_min < column_y_limits[j][0]:
                column_y_limits[j] = (current_min, column_y_limits[j][1])
            if current_max > column_y_limits[j][1]:
                column_y_limits[j] = (column_y_limits[j][0], current_max)

            # Setting titles and axis labels
            if i == 0:
                ax.set_title(metric, fontweight="bold")
            if j == 0:
                ax.set_ylabel(uncertainty_domain, fontweight="bold")
            ax.set_xticks(shots)
            ax.grid(True)

            # Add legend to the first subplot of each row for clarity
            if j == 0:
                ax.legend()

    # Apply consistent y-limits with padding and enable y-axis labels for each column
    padding_factor = 0.01  # 1% padding to the range
    for row in axs:
        for idx, ax in enumerate(row):
            min_val, max_val = column_y_limits[idx]
            range_val = max_val - min_val
            padded_min = min_val - padding_factor * range_val
            padded_max = max_val + padding_factor * range_val
            ax.set_ylim(padded_min, padded_max)
            ax.set_yticks(ax.get_yticks())  # Ensure y-ticks are shown
            ax.grid(True)  # Enable grid

    # Set a common x-axis label
    fig.text(0.5, 0.00, "Number of Shots", ha="center", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"uncertainty_plots.pdf"))

    plt.savefig(os.path.join(save_dir, f"uncertainty_plots.pdf"))
