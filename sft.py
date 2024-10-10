import os
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from utils import soft_match
from sklearn.metrics import f1_score, log_loss

OUTPUT_DIR = "/scratch/local/ssd/fpinto/llm_hallucinations/fine_tuning"

OUTPUT_DIR2 = "/scratch/local/ssd/fpinto/llm_hallucinations/fine_tuning2"

LABEL_CONVERSIONS2 = {
                    # '1' : '1',
                    # '2' : '2',
                    # '3' : '3',
                    # '4' : '4',
                    #   'extrinsic': '0',
                    #   'intrinsic': '1',
                    "correct": '0',
                    "extrinsic-NP": '1',
                    "extrinsic-predicate": '2',
                    "intrinsic-NP": '3',
                    "intrinsic-predicate": '4',
}

LABEL_CONVERSIONS = {
                    #      'correct': '0',
                    #  'incorrect': '1',
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


def f1_score_binary(y_true, y_pred):
    processed_refs = [LABEL_CONVERSIONS[ref] for ref in y_true] # Convert labels to numbers (same conversion as predictions)
    trues = []
    preds = []
    for true, pred in zip(processed_refs, y_pred):
        if soft_match(pred, true) == 1:
            trues.append(true)
            preds.append('0') if true == '0' else preds.append('1')

        else:
            trues.append(true)
            preds.append('1') if true == '0' else preds.append('0')

    return f1_score(trues, preds, average='macro')


def f1_score2(y_true, y_pred):
    processed_refs = [LABEL_CONVERSIONS2[ref] for ref in y_true] # Convert labels to numbers (same conversion as predictions)
    trues = []
    preds = []
    for true, pred in zip(processed_refs, y_pred):
        if soft_match(pred, true) == 1:
            trues.append(true)
            preds.append('0') if true == '0' else preds.append('1')

        else:
            trues.append(true)
            preds.append('1') if true == '0' else preds.append('0')

    return f1_score(trues, preds, average='macro')


def get_score2(preds, refs):
    # Predictions are numbers corresponding to an error type.
    processed_refs = [LABEL_CONVERSIONS2[ref] for ref in refs] # Convert labels to numbers (same conversion as predictions)

    class_errors = {'correct': 0, 'extrinsic-NP': 0, 'extrinsic-predicate': 0, 'intrinsic-NP': 0, 'intrinsic-predicate': 0}

    num_correct = sum([1 for ref in refs if ref == 'correct']) if 'correct' in refs else 1
    num_extrinsic_NP = sum([1 for ref in refs if ref == 'extrinsic-NP']) if 'extrinsic-NP' in refs else 1
    num_extrinsic_predicate = sum([1 for ref in refs if ref == 'extrinsic-predicate']) if 'extrinsic-predicate' in refs else 1
    num_intrinsic_NP = sum([1 for ref in refs if ref == 'intrinsic-NP']) if 'intrinsic-NP' in refs else 1
    num_intrinsic_predicate = sum([1 for ref in refs if ref == 'intrinsic-predicate']) if 'intrinsic-predicate' in refs else 1

    total = 0 # Overall accuracy
    for i in range(len(preds)):
        if soft_match(preds[i], processed_refs[i]):
            total += 1
            class_errors[refs[i]] += 1

    scores = {'total': total / len(refs),
            'correct': class_errors["correct"] / num_correct if 'correct' in refs else None,
            'extrinsic-NP': class_errors["extrinsic-NP"] / num_extrinsic_NP if 'extrinsic-NP' in refs else None,
            'extrinsic-predicate': class_errors["extrinsic-predicate"] / num_extrinsic_predicate if 'extrinsic-predicate' in refs else None,
            'intrinsic-NP': class_errors["intrinsic-NP"] / num_intrinsic_NP if 'intrinsic-NP' in refs else None,
            'intrinsic-predicate': class_errors["intrinsic-predicate"] / num_intrinsic_predicate if 'intrinsic-predicate' in refs else None,
            }
    
    return scores


def get_extrinsic_intrinsic_score(preds, refs):
    # Predictions are numbers corresponding to an error type.
    processed_refs = [LABEL_CONVERSIONS2[ref] for ref in refs] # Convert labels to numbers (same conversion as predictions)

    class_errors = {'extrinsic': 0, 'intrinsic': 0}

    num_correct = sum([1 for ref in refs if ref == 'extrinsic']) if 'extrinsic' in refs else 1
    num_incorrect = sum([1 for ref in refs if ref == 'intrinsic']) if 'intrinsic' in refs else 1

    total = 0 # Overall accuracy
    for i in range(len(preds)):
        if soft_match(preds[i], processed_refs[i]):
            total += 1
            class_errors[refs[i]] += 1

    scores = {'total': total / len(refs),
            'extrinsic': class_errors["extrinsic"] / num_correct if 'extrinsic' in refs else None,
            'intrinsic': class_errors["intrinsic"] / num_incorrect if 'intrinsic' in refs else None}
    
    return scores


def get_score(preds, refs):
    # Predictions are numbers corresponding to an error type.
    processed_refs = [LABEL_CONVERSIONS[ref] for ref in refs] # Convert labels to numbers (same conversion as predictions)

    class_errors = {'correct': 0, 'incorrect': 0}

    num_correct = sum([1 for ref in refs if ref == 'correct']) if 'correct' in refs else 1
    num_incorrect = sum([1 for ref in refs if ref == 'incorrect']) if 'incorrect' in refs else 1

    total = 0 # Overall accuracy
    for i in range(len(preds)):
        if soft_match(preds[i], processed_refs[i]):
            total += 1
            class_errors[refs[i]] += 1

    scores = {'total': total / len(refs),
            'correct': class_errors["correct"] / num_correct if 'correct' in refs else None,
            'incorrect': class_errors["incorrect"] / num_incorrect if 'incorrect' in refs else None}
    
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
        "--do_fine_tune",
        type=str,
        default='y',
        help="Whether to fine-tune or not",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default='',
        help="Output directory",
    )
    return parser.parse_args()

