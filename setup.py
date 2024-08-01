from argparse import ArgumentParser
import torch
import random
import os
import numpy as np
import json
from accelerate import Accelerator
from openicl import (
    DatasetReader,
    PPLInferencer,
    RandomRetriever,
    AccEvaluator,
    VotekRetriever,
    TopkRetriever,
)
from datasets import load_dataset

from constants import DATASET_PROMPTS, TEST_SPLIT, DATASET_LABELS
from utils import reformat_data


def parse_args():
    """
    Parse command line arguments.

    Returns:
        args: argparse.Namespace -- parsed command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--num_icl_examples",
        type=int,
        nargs="+",
        default=[0, 1, 2, 4, 8, 16, 32],
        help="Number of ICL examples",
    )
    parser.add_argument(
        "--dataset", type=str, default="Lislaam/AggreFact", help="Name of dataset"
    )
    parser.add_argument(
        "--llms",
        type=str,
        nargs="+",
        default=[
            "mistralai/Mistral-7B-Instruct-v0.3",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "google/gemma-1.1-7b-it",
        ],
        help="Name of LLM",
        choices=[
            "mistralai/Mistral-7B-Instruct-v0.3",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "google/gemma-1.1-7b-it",
        ],
    )
    parser.add_argument(
        "--llm_device", type=str, default="cuda:0", help="Device to run the code on"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for the dataloader"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument(
        "--retriever",
        type=str,
        default="random",
        help="Retriever to use for ice selection.",
    )
    """parser.add_argument(
        "--scratch_dir",
        type=str,
        default="/scratch/local/ssd/tomlamb/icl_uncertainty/",
        help="Scratch directory for storing larger files.",
    )"""
    parser.add_argument(
        "--verbalised_labels",
        action="store_true",
        help="Whether to use verbalised labels. Defaults to not.",
    )
    parser.add_argument(
        "--focus_addition",
        action="store_true",
        help="Whether to use additional instructions within prompts for what model should focus on defaults to not.",
    )
    parser.add_argument(
        "--prohibit_addition",
        action="store_true",
        help="Whether to use additional instructions within prompts for what the model should not do on defaults to not.",
    )

    return parser.parse_args()


def add_ic_token_and_remove_sos_token(prompt, llm):
    # Convert messsages to string using chat template.
    if "mistral" in llm:
        # Rplace the <s> at the start of the prompt with </E>
        prompt = "</E>" + prompt[len("<s>") :]
    elif "llama" in llm:
        prompt = "</E>" + prompt[len("<|begin_of_text|>") :]
    elif "gemma" in llm:
        prompt = "</E>" + prompt[len("<bos>") :]

    return prompt


def select_retriever(
    retriever_name,
    data,
    num_ice,
    index_split,
    test_split,
    ice_separator,
    ice_eos_token,
    accelerator=None,
):
    if retriever_name == "random":
        return RandomRetriever(
            data,
            ice_num=num_ice,
            index_split=index_split,
            test_split=test_split,
            ice_separator=ice_separator,
            ice_eos_token=ice_eos_token,
            accelerator=accelerator,
        )
    elif retriever_name == "votek":
        return VotekRetriever(
            data,
            ice_num=num_ice,
            index_split=index_split,
            test_split=test_split,
            ice_separator=ice_separator,
            ice_eos_token=ice_eos_token,
            accelerator=accelerator,
        )
    elif retriever_name == "topk":
        return TopkRetriever(
            data,
            ice_num=num_ice,
            index_split=index_split,
            test_split=test_split,
            ice_separator=ice_separator,
            ice_eos_token=ice_eos_token,
            accelerator=accelerator,
        )
    else:
        raise ValueError(f"Retriever {retriever_name} not found.")
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)