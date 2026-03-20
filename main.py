"""
Main Pipeline Runner
====================
Runs the full experiment pipeline:
  1. Load and preprocess SNLI
  2. Train TF-IDF + Logistic Regression baseline
  3. Train TF-IDF + Linear SVM baseline
  4. Fine-tune BERT
  5. Evaluate all models and produce comparison table

Usage:
    python main.py              # run everything
    python main.py --baselines  # baselines only (no BERT)
    python main.py --bert       # BERT only (assumes data is loaded)
"""

import argparse
import os
import pandas as pd
import numpy as np
import joblib

import config
from data_utils import load_snli, extract_texts_and_labels, build_tfidf_features, tokenize_for_bert
from baselines import train_logistic_regression, train_svm, save_baseline_model
from bert_model import get_tokenizer, train_bert, predict_bert, save_bert_model
from evaluation import save_all_results, build_comparison_table


def run_baselines(X_train, y_train, X_test, y_test):
    """Train and evaluate both TF-IDF baselines."""
    all_metrics = []

    # ── Logistic Regression ─────────────────────────────────────────
    lr_model, lr_time = train_logistic_regression(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_metrics = save_all_results(y_test, lr_preds, "TF-IDF + Logistic Regression", "tfidf_lr")
    lr_metrics["train_time_s"] = lr_time
    save_baseline_model(lr_model, "tfidf_lr")
    all_metrics.append(lr_metrics)

    # ── Linear SVM ──────────────────────────────────────────────────
    svm_model, svm_time = train_svm(X_train, y_train)
    svm_preds = svm_model.predict(X_test)
    svm_metrics = save_all_results(y_test, svm_preds, "TF-IDF + Linear SVM", "tfidf_svm")
    svm_metrics["train_time_s"] = svm_time
    save_baseline_model(svm_model, "tfidf_svm")
    all_metrics.append(svm_metrics)

    return all_metrics


def run_bert(dataset):
    """Fine-tune and evaluate BERT."""
    tokenizer = get_tokenizer()

    print("\nTokenizing dataset for BERT...")
    train_tok = tokenize_for_bert(dataset["train"], tokenizer)
    val_tok = tokenize_for_bert(dataset["validation"], tokenizer)
    test_tok = tokenize_for_bert(dataset["test"], tokenizer)

    # Train
    trainer, model, bert_time = train_bert(train_tok, val_tok)

    # Predict on test set
    bert_preds = predict_bert(trainer, test_tok)
    y_test = np.array(dataset["test"]["label"])

    # Evaluate
    bert_metrics = save_all_results(y_test, bert_preds, "BERT (bert-base-uncased)", "bert")
    bert_metrics["train_time_s"] = bert_time

    # Save model
    save_bert_model(trainer)

    return [bert_metrics]


def main():
    parser = argparse.ArgumentParser(description="SNLI Contradiction Detection Pipeline")
    parser.add_argument("--baselines", action="store_true", help="Run baselines only")
    parser.add_argument("--bert", action="store_true", help="Run BERT only")
    args = parser.parse_args()

    run_all = not args.baselines and not args.bert
    config.create_dirs()

    # ── Load Data ───────────────────────────────────────────────────
    dataset = load_snli()
    all_metrics = []

    # ── Baselines ───────────────────────────────────────────────────
    if run_all or args.baselines:
        train_p, train_h, y_train = extract_texts_and_labels(dataset["train"])
        val_p, val_h, y_val = extract_texts_and_labels(dataset["validation"])
        test_p, test_h, y_test = extract_texts_and_labels(dataset["test"])

        X_train, X_val, X_test, vectorizer = build_tfidf_features(
            train_p, train_h, val_p, val_h, test_p, test_h
        )

        # Save vectorizer for reproducibility
        joblib.dump(vectorizer, os.path.join(config.MODELS_DIR, "tfidf_vectorizer.joblib"))

        baseline_metrics = run_baselines(X_train, y_train, X_test, y_test)
        all_metrics.extend(baseline_metrics)

    # ── BERT ────────────────────────────────────────────────────────
    if run_all or args.bert:
        bert_metrics = run_bert(dataset)
        all_metrics.extend(bert_metrics)

    # ── Comparison Table ────────────────────────────────────────────
    if len(all_metrics) > 1:
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        comparison_df = build_comparison_table(all_metrics)
        print(comparison_df.to_string(index=False))

        # Save comparison table
        csv_path = os.path.join(config.RESULTS_DIR, "model_comparison.csv")
        comparison_df.to_csv(csv_path, index=False)
        print(f"\nComparison table saved to: {csv_path}")

        # Also save as LaTeX for the paper
        latex_path = os.path.join(config.RESULTS_DIR, "model_comparison.tex")
        comparison_df.to_latex(latex_path, index=False, float_format="%.4f")
        print(f"LaTeX table saved to: {latex_path}")

    print("\n✓ Pipeline complete! All results saved to Google Drive.")
    print(f"  Results: {config.RESULTS_DIR}")
    print(f"  Figures: {config.FIGURES_DIR}")
    print(f"  Models:  {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
