from argparse import ArgumentParser
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
from peft import LoraConfig #, get_peft_model
from datasets import load_dataset, dataset_dict, DatasetDict
from tqdm import tqdm
from torch.utils.data import DataLoader

OUTPUT_DIR = "/scratch/local/ssd/fpinto/llm_hallucinations/fine_tuning" #"/homes/53/fpinto/llm_hallucinations/saved_models" 

LABEL_CONVERSIONS = {
                    "correct": '0',
                    "incorrect": '1',
                    }


"""                    # "intrinsic-NP": '1',
                    # "intrinsic-predicate": '2',
                    # "extrinsic-NP": '3',
                    # "extrinsic-predicate": '4',
                    # ==========================================
                    # 5: "['extrinsic-NP', 'intrinsic-NP']",
                    # 6: "['extrinsic-NP', 'extrinsic-predicate']",
                    # 7: "['intrinsic-predicate', 'extrinsic-NP']",
                    # 8: "['extrinsic-predicate', 'intrinsic-NP']",
                    # 9: "['extrinsic-predicate', 'intrinsic-predicate']",
                    # 10: "['intrinsic-NP', 'intrinsic-predicate']",
                    # 11: "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP']",
                    # 13: "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-predicate']",
                    # 14: "['extrinsic-NP', 'intrinsic-NP', 'intrinsic-predicate']",
                    # 15: "['extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']",
                    # 16: "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']" """

REVERSE_LABEL_CONVERSIONS = {v: k for k, v in LABEL_CONVERSIONS.items()}

LABEL_MAP = { # Make all the labels consistent
    "['extrinsic-NP']" : "extrinsic-NP",
    "['extrinsic-predicate']" : "extrinsic-predicate",
    "['intrinsic-NP']" : "intrinsic-NP",
    "['intrinsic-predicate']" : "intrinsic-predicate",
    "correct" : "correct",
    "['correct']" : "correct",
    }


def reverse_labels(x):
    try:
        return REVERSE_LABEL_CONVERSIONS[x]
    except KeyError:
        return x
    

def plot_training_loss(log_history, output_dir):
    df = pd.DataFrame(log_history)
    if "loss" in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df["step"], df["loss"], label="Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "training_loss.png"))
        plt.show()
    else:
        print("No loss information found in log history.")


def plot_training_and_validation_loss(log_history, output_dir):
    df = pd.DataFrame(log_history)
    if "loss" in df.columns:
        plt.figure(figsize=(10, 5))
        import pdb; pdb.set_trace()
        plt.plot(df["step"], df["train_loss"], label="Training Loss")
        plt.plot(df["step"], df["eval_loss"], label="Validation Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Loss Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "training_and_validation_loss.png"))
        plt.show()
    else:
        print("No loss information found in log history.")


def get_sft_score(preds, refs, binary=False):
    # Predictions are numbers corresponding to an error type.
    processed_refs = [LABEL_CONVERSIONS[ref] for ref in refs] # Convert labels to numbers (same conversion as predictions)

    if binary:
        class_errors = {'correct': 0, 'incorrect': 0}

        num_correct = sum([1 for ref in refs if ref == 'correct']) if 'correct' in refs else 1
        num_incorrect = sum([1 for ref in refs if ref == 'incorrect']) if 'incorrect' in refs else 1

        total = 0 # Overall accuracy
        for i in range(len(preds)):
            if preds[i] == processed_refs[i]:
                total += 1
                class_errors[refs[i]] += 1

        scores = {'total': total / len(refs),
                'correct': class_errors["correct"] / num_correct if 'correct' in refs else None,
                'incorrect': class_errors["incorrect"] / num_incorrect if 'incorrect' in refs else None}

    else:
        class_errors = {'extrinsic-NP': 0, 'extrinsic-predicate': 0, 'intrinsic-NP': 0,
                    'intrinsic-predicate': 0, 'correct': 0}
        
        # Count the number of each error type in the references
        num_extrinsicnp = sum([1 for ref in refs if ref == 'extrinsic-NP']) if 'extrinsic-NP' in refs else 1
        num_extrinsicpredicate = sum([1 for ref in refs if ref == 'extrinsic-predicate']) if 'extrinsic-predicate' in refs else 1
        num_intrinsicnp = sum([1 for ref in refs if ref == 'intrinsic-NP']) if 'intrinsic-NP' in refs else 1
        num_intrinsicpredicate = sum([1 for ref in refs if ref == 'intrinsic-predicate']) if 'intrinsic-predicate' in refs else 1
        num_correct = sum([1 for ref in refs if ref == 'correct']) if 'correct' in refs else 1

        total = 0 # Overall accuracy
        for i in range(len(preds)):
            if preds[i] == processed_refs[i]:
                total += 1
                class_errors[refs[i]] += 1

        scores = {'total': total / len(refs),
                'extrinsic-NP': class_errors["extrinsic-NP"] / num_extrinsicnp if 'extrinsic-NP' in refs else None,
                'extrinsic-predicate': class_errors["extrinsic-predicate"] / num_extrinsicpredicate if 'extrinsic-predicate' in refs else None,
                'intrinsic-NP': class_errors["intrinsic-NP"] / num_intrinsicnp if 'intrinsic-NP' in refs else None,
                'intrinsic-predicate': class_errors["intrinsic-predicate"] / num_intrinsicpredicate if 'intrinsic-predicate' in refs else None,
                'correct': class_errors["correct"] / num_correct if 'correct' in refs else None}
    
    return scores


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--llm",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="The language model to use for data generation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="The batch size for data generation.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Lislaam/AggreFact",
        help="The dataset to use for data generation.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Number of validation steps with no improvement after which training will be stopped.",
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.0,
        help="Minimum improvement to be considered as an improvement for early stopping.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=6,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default=None,
        help="'oversampling' or 'undersampling'.",
    )
    return parser.parse_args()

