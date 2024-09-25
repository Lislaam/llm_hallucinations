from argparse import ArgumentParser
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics import CalibrationError
from scipy.stats import entropy
from sklearn.metrics import brier_score_loss
from utils import *

OUTPUT_DIR = "/scratch/local/ssd/fpinto/llm_hallucinations/fine_tuning"

LABEL_CONVERSIONS = {
                      'extrinsic': '0',
                      'intrinsic': '1',
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
#                     "correct": '0',
#                     #"incorrect": '1',
#                     "intrinsic-NP": '1',
#                     "intrinsic-predicate": '2',
#                     "extrinsic-NP": '3',
#                     "extrinsic-predicate": '4',
#                     # ==========================================
#                     # 5: "['extrinsic-NP', 'intrinsic-NP']",
#                     # 6: "['extrinsic-NP', 'extrinsic-predicate']",
#                     # 7: "['intrinsic-predicate', 'extrinsic-NP']",
#                     # 8: "['extrinsic-predicate', 'intrinsic-NP']",
#                     # 9: "['extrinsic-predicate', 'intrinsic-predicate']",
#                     # 10: "['intrinsic-NP', 'intrinsic-predicate']",
#                     # 11: "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP']",
#                     # 13: "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-predicate']",
#                     # 14: "['extrinsic-NP', 'intrinsic-NP', 'intrinsic-predicate']",
#                     # 15: "['extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']",
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


def reverse_labels(x):
    try:
        return REVERSE_LABEL_CONVERSIONS[x]
    except KeyError:
        return x


def compute_shannon_entropy(pred_probs):
    """
    Compute the Shannon entropy of the predicted probabilities.
    
    Args:
        pred_probs (np.array): Array of shape [num_classes] representing
                               the predicted probabilities for each class.
                               
    Returns:
        entropy_value (float): Shannon entropy value.
    """
    pred_probs = np.array(pred_probs)
    pred_probs = pred_probs / np.sum(pred_probs)  # Ensure probabilities sum to 1
    return entropy(pred_probs)

def get_average_entropy(pred_probs_list):
    """
    Compute the average Shannon entropy for multiple samples.
    
    Args:
        pred_probs_list (list of lists): List of predicted probabilities for each sample.
                                         Each inner list contains the predicted probabilities for a sample.
                                         
    Returns:
        avg_entropy (float): The average entropy across all samples.
    """
    # Calculate entropy for each sample
    entropies = [compute_shannon_entropy(pred_probs) for pred_probs in pred_probs_list]
    
    # Calculate and return average entropy
    avg_entropy = np.mean(entropies)
    return avg_entropy


def get_ece(pred_probs, labels):
    """
    Compute the Expected Calibration Error (ECE).
    
    Args:
        pred_probs (torch.Tensor): Tensor of predicted probabilities of shape [num_samples, num_classes].
        labels (torch.Tensor): Tensor of true labels of shape [num_samples].
    
    Returns:
        ece_value (float): Expected Calibration Error.
    """
    # Convert to torch tensors if not already
    pred_probs = torch.tensor(pred_probs)
    labels = torch.tensor(labels)

    # Instantiate CalibrationError metric (use 'max_prob' strategy for ECE)
    ece_metric = CalibrationError(n_bins=10, norm='l1')
    
    # Calculate ECE
    ece_value = ece_metric(pred_probs, labels)
    return ece_value.item()


def get_brier(pred_probs, labels):
    """
    Compute the Brier Score, which is used to measure the accuracy of probabilistic predictions.
    
    Args:
        pred_probs (list): List of predicted probabilities for the positive class.
        labels (list): List of true binary labels.
    
    Returns:
        brier_score (float): Brier Score value.
    """
    # Select the probability of the positive class
    pos_class_probs = [prob[1] for prob in pred_probs]  # Assuming binary classification
    brier_score = brier_score_loss(labels, pos_class_probs)
    return brier_score


def get_score(preds, refs):
    # Predictions are numbers corresponding to an error type.
    processed_refs = [LABEL_CONVERSIONS[ref] for ref in refs] # Convert labels to numbers (same conversion as predictions)

    class_errors = {'extrinsic': 0, 'intrinsic': 0}

    num_correct = sum([1 for ref in refs if ref == 'extrinsic']) if 'extrinsic' in refs else 1
    num_incorrect = sum([1 for ref in refs if ref == 'intrinsic']) if 'intrinsic' in refs else 1

    total = 0 # Overall accuracy
    for i in range(len(preds)):
        if soft_match(preds[i], processed_refs[i]):
            total += 1
            class_errors[refs[i]] += 1

    scores = {'total': total / len(refs),
            'extrinsic accuracy': class_errors["extrinsic"] / num_correct if 'extrinsic' in refs else None,
            'intrinsic accuracy': class_errors["intrinsic"] / num_incorrect if 'intrinsic' in refs else None}
    
    return scores


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
    parser.add_argument(
        "--train",
        type=bool,
        default=True,
        help="Whether to fine-tune or not",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default='',
        help="Output directory",
    )
    return parser.parse_args()

