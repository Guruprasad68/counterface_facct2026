"""
Find the similarity threshold for a face recognition system at a target FMR (False Match Rate).

This script takes positive (genuine) and negative (impostor) similarity scores and computes
the threshold using ROC curve analysis.
"""

import argparse
import json
import numpy as np
from sklearn.metrics import roc_curve, auc


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def extract_scores(score_dict):
    """Extract numeric scores from a dictionary, filtering out errors."""
    scores = []
    for key, value in score_dict.items():
        if isinstance(value, (int, float)):
            scores.append(value)
    return scores


def compute_threshold_at_fmr(pos_scores, neg_scores, target_fmr=0.001):
    """
    Compute the similarity threshold at a target False Match Rate (FMR/FPR).
    
    Args:
        pos_scores: List of genuine (same identity) similarity scores
        neg_scores: List of impostor (different identity) similarity scores
        target_fmr: Target false match rate (default 0.001 = 0.1%)
    
    Returns:
        threshold: Similarity threshold
        actual_fmr: Actual FMR at this threshold
    """
    y_scores = np.array(pos_scores + neg_scores)
    y_labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
    
    fpr, _, thresholds = roc_curve(y_labels, y_scores)
    
    fmr_diff = np.abs(fpr - target_fmr)
    best_idx = np.argmin(fmr_diff)
    
    return thresholds[best_idx], fpr[best_idx]


def compute_auc(pos_scores, neg_scores):
    """Compute Area Under ROC Curve."""
    y_scores = np.array(pos_scores + neg_scores)
    y_labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
    
    fpr, tpr, _ = roc_curve(y_labels, y_scores)
    return auc(fpr, tpr)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Find FR threshold at target FMR'
    )
    parser.add_argument(
        '--positive_scores', type=str, required=True,
        help='Path to JSON file with positive (genuine) similarity scores'
    )
    parser.add_argument(
        '--negative_scores', type=str, required=True,
        help='Path to JSON file with negative (impostor) similarity scores'
    )
    parser.add_argument(
        '--target_fmr', type=float, required=True,
        help='Target False Match Rate (e.g., 0.001 for 0.1%%)'
    )
    parser.add_argument(
        '--output_json', type=str, default=None,
        help='Path to save threshold results as JSON'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    pos_dict = load_json(args.positive_scores)
    neg_dict = load_json(args.negative_scores)
    
    pos_scores = extract_scores(pos_dict)
    neg_scores = extract_scores(neg_dict)
    
    print(f"Loaded {len(pos_scores)} positive scores and {len(neg_scores)} negative scores")
    
    results = {
        'num_positive': len(pos_scores),
        'num_negative': len(neg_scores),
        'auc': compute_auc(pos_scores, neg_scores)
    }
    
    print(f"AUC: {results['auc']:.4f}")
    
    threshold, actual_fmr = compute_threshold_at_fmr(
        pos_scores, neg_scores, args.target_fmr
    )
    print(f"\nTarget FMR: {args.target_fmr}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Actual FMR: {actual_fmr:.6f}")
    
    results[f'threshold_at_fmr_{args.target_fmr}'] = {
        'threshold': float(threshold),
        'actual_fmr': float(actual_fmr),
    }

    if args.output_json:
        save_json(results, args.output_json)
        print(f"\nResults saved to {args.output_json}")


if __name__ == '__main__':
    main()
