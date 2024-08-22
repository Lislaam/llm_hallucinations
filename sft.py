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

OUTPUT_DIR = "/scratch/local/ssd/fpinto/llm_hallucinations/fine_tuning"

LABEL_CONVERSIONS = {
                    "correct": 0,
                    "intrinsic-NP": 1,
                    "intrinsic-predicate": 2,
                    "extrinsic-NP": 3,
                    "extrinsic-predicate": 4,
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
                    # 16: "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']"
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
        default=4,
        help="The batch size for data generation.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Lislaam/AggreFact",
        help="The dataset to use for data generation.",
    )
    return parser.parse_args()
