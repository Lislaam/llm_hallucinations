from argparse import ArgumentParser
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


OUTPUT_DIR = "/scratch/local/ssd/fpinto/llm_hallucinations/fine_tuning"

LABEL_CONVERSIONS = {
                    'extrinsic' : '0',
                    'intrinsic' : '1',
                    # 0:'0',
                    # 1:'1',
                    # 2:'2',
                    # 3:'3',
                    # 4:'4',
                    # "correct": 'C',
                    # "intrinsic-NP": 'IN',
                    # "intrinsic-predicate": 'IP',
                    # "extrinsic-NP": 'EN',
                    # "extrinsic-predicate": 'EP',
}

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
    


def split_into_facts(summary):
    # Split the summary into sentence-level facts, use your custom logic if needed
    facts = summary.split('. ')  # Example: Split by period to get atomic facts
    return facts
    

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
