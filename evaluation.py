"""
Evaluation Utilities
====================
Computes metrics, generates confusion matrices, saves results.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import config


def compute_metrics(y_true, y_pred, model_name="model"):
    """
    Compute comprehensive classification metrics.
    
    Returns:
        dict with overall and per-class metrics
    """
    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro"),
        "macro_recall": recall_score(y_true, y_pred, average="macro"),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted"),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }

    # Per-class metrics
    per_class_p = precision_score(y_true, y_pred, average=None)
    per_class_r = recall_score(y_true, y_pred, average=None)
    per_class_f = f1_score(y_true, y_pred, average=None)

    for i, label_name in enumerate(config.LABEL_NAMES):
        results[f"{label_name}_precision"] = per_class_p[i]
        results[f"{label_name}_recall"] = per_class_r[i]
        results[f"{label_name}_f1"] = per_class_f[i]

    return results


def print_classification_report(y_true, y_pred, model_name="Model"):
    """Print a formatted classification report."""
    print(f"\n{'='*60}")
    print(f"Classification Report: {model_name}")
    print(f"{'='*60}")
    print(classification_report(
        y_true, y_pred,
        target_names=config.LABEL_NAMES,
        digits=4,
    ))


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot and optionally save a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.LABEL_NAMES,
        yticklabels=config.LABEL_NAMES,
        ax=axes[0],
    )
    axes[0].set_title(f"{model_name} — Counts")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # Normalized
    sns.heatmap(
        cm_normalized, annot=True, fmt=".3f", cmap="Blues",
        xticklabels=config.LABEL_NAMES,
        yticklabels=config.LABEL_NAMES,
        ax=axes[1],
    )
    axes[1].set_title(f"{model_name} — Normalized")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Confusion matrix saved to: {save_path}")

    plt.show()
    plt.close(fig)

    return cm


def save_metrics(metrics_dict, filepath):
    """Save metrics dict to JSON."""
    # Convert numpy types to native Python for JSON serialization
    clean = {}
    for k, v in metrics_dict.items():
        if isinstance(v, (np.floating, np.float32, np.float64)):
            clean[k] = float(v)
        elif isinstance(v, (np.integer, np.int32, np.int64)):
            clean[k] = int(v)
        else:
            clean[k] = v

    with open(filepath, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"  Metrics saved to: {filepath}")


def save_confusion_matrix_csv(y_true, y_pred, filepath):
    """Save confusion matrix as CSV."""
    cm = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(cm, index=config.LABEL_NAMES, columns=config.LABEL_NAMES)
    df.index.name = "True"
    df.columns.name = "Predicted"
    df.to_csv(filepath)
    print(f"  Confusion matrix CSV saved to: {filepath}")


def build_comparison_table(all_metrics):
    """
    Build a comparison table across all models.
    
    Args:
        all_metrics: list of dicts (one per model from compute_metrics)
    
    Returns:
        DataFrame with models as rows and metrics as columns
    """
    rows = []
    for m in all_metrics:
        row = {
            "Model": m["model"],
            "Accuracy": m["accuracy"],
            "Macro P": m["macro_precision"],
            "Macro R": m["macro_recall"],
            "Macro F1": m["macro_f1"],
            "Entail F1": m["entailment_f1"],
            "Neutral F1": m["neutral_f1"],
            "Contra F1": m["contradiction_f1"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # Round for display
    numeric_cols = df.columns[1:]
    df[numeric_cols] = df[numeric_cols].apply(lambda x: x.round(4))
    return df


def save_all_results(y_true, y_pred, model_name, model_key):
    """
    One-call function to compute, display, and save all evaluation artifacts.
    
    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        model_name: display name (e.g., "TF-IDF + Logistic Regression")
        model_key: short key for filenames (e.g., "tfidf_lr")
    
    Returns:
        metrics dict
    """
    config.create_dirs()

    print(f"\n{'#'*60}")
    print(f"# Evaluating: {model_name}")
    print(f"{'#'*60}")

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, model_name)

    # Print report
    print_classification_report(y_true, y_pred, model_name)

    # Save metrics JSON
    save_metrics(metrics, os.path.join(config.RESULTS_DIR, f"{model_key}_metrics.json"))

    # Save confusion matrix CSV
    save_confusion_matrix_csv(
        y_true, y_pred,
        os.path.join(config.RESULTS_DIR, f"{model_key}_confusion_matrix.csv"),
    )

    # Plot and save confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        model_name=model_name,
        save_path=os.path.join(config.FIGURES_DIR, f"{model_key}_confusion_matrix.png"),
    )

    return metrics
