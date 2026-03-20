"""
TF-IDF Baseline Models
======================
Logistic Regression and Linear SVM baselines using
TF-IDF features with interaction features.
"""

import time
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import config


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression classifier.
    
    Returns:
        fitted model, training time in seconds
    """
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        max_iter=config.LR_MAX_ITER,
        C=config.LR_C,
        solver=config.LR_SOLVER,
        multi_class="multinomial",
        random_state=config.SEED,
        n_jobs=-1,
        verbose=1,
    )

    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  Training time: {elapsed:.1f}s")

    return model, elapsed


def train_svm(X_train, y_train):
    """
    Train a Linear SVM classifier.
    
    Returns:
        fitted model, training time in seconds
    """
    print("\nTraining Linear SVM...")
    model = LinearSVC(
        max_iter=config.SVM_MAX_ITER,
        C=config.SVM_C,
        random_state=config.SEED,
        verbose=1,
    )

    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  Training time: {elapsed:.1f}s")

    return model, elapsed


def save_baseline_model(model, model_key):
    """Save a sklearn model to Google Drive."""
    config.create_dirs()
    filepath = os.path.join(config.MODELS_DIR, f"{model_key}.joblib")
    joblib.dump(model, filepath)
    print(f"  Model saved to: {filepath}")


def load_baseline_model(model_key):
    """Load a sklearn model from Google Drive."""
    filepath = os.path.join(config.MODELS_DIR, f"{model_key}.joblib")
    model = joblib.load(filepath)
    print(f"  Model loaded from: {filepath}")
    return model
