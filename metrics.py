import numpy as np
import torch.nn.functional as F


def get_entropy(pred_probs):
    """pred_probs is a list of probability distributions (e.g., softmax outputs)."""
    pred_probs = np.clip(pred_probs, 1e-12, 1.0)
    
    # Calculate entropy for each set of probabilities
    entropy = -np.sum(pred_probs * np.log(pred_probs), axis=1)
    return entropy


def get_ece(pred_probs, preds, refs, num_bins=10):
    """Calculates Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    accuracies = np.zeros(num_bins)
    confidences = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for i, probs in enumerate(pred_probs):
        confidence = np.max(probs)
        correct = preds[i] == refs[i]
        
        bin_index = np.digitize(confidence, bin_boundaries) - 1
        accuracies[bin_index] += correct
        confidences[bin_index] += confidence
        bin_sizes[bin_index] += 1

    ece = 0
    for i in range(num_bins):
        if bin_sizes[i] > 0:
            accuracies[i] /= bin_sizes[i]
            confidences[i] /= bin_sizes[i]
            ece += (bin_sizes[i] / len(preds)) * abs(accuracies[i] - confidences[i])
    return ece


def get_brier_score(pred_probs, refs):
    """Brier score for binary classification."""
    true_labels = [1 if ref == 'extrinsic' else 0 for ref in refs]
    pred_extrinsic_probs = [probs[1] for probs in pred_probs]  # Assuming 'extrinsic' is class 1

    brier_scores = [(true_label - pred_prob) ** 2 for true_label, pred_prob in zip(true_labels, pred_extrinsic_probs)]
    return np.mean(brier_scores)


def get_msp(pred_probs):
    """Returns the maximum softmax probability for each prediction."""
    max_probs = [np.max(probs) for probs in pred_probs]
    avg_msp = np.mean(max_probs)  # Average MSP across predictions
    return avg_msp


def get_variance(ensemble_preds):
    """ensemble_preds is a list of multiple predictions from an ensemble or dropout sampling."""
    variances = np.var(ensemble_preds, axis=0)
    avg_variance = np.mean(variances)  # Average variance across predictions
    return avg_variance


def mutual_information(pred_probs, mc_samples):
    """pred_probs: Mean prediction probabilities (from multiple MC samples).
       mc_samples: List of multiple predictions from an ensemble or MC dropout."""
    mean_entropy = np.mean([entropy(probs) for probs in mc_samples])
    expected_entropy = entropy(pred_probs)
    return expected_entropy - mean_entropy


def get_nll(pred_probs, refs):
    """Negative Log-Likelihood for true labels."""
    true_labels = [1 if ref == 'extrinsic' else 0 for ref in refs]
    log_likelihoods = [-np.log(pred_probs[i][true_label]) for i, true_label in enumerate(true_labels)]
    return np.mean(log_likelihoods)