"""
Configuration for SNLI Contradiction Detection Project
=======================================================
All hyperparameters, paths, and constants in one place.
"""

import os

# ── Google Drive Paths ──────────────────────────────────────────────
DRIVE_BASE = "/content/drive/MyDrive/NLP_Project"
RESULTS_DIR = os.path.join(DRIVE_BASE, "results")
MODELS_DIR = os.path.join(DRIVE_BASE, "models")
FIGURES_DIR = os.path.join(DRIVE_BASE, "figures")

# ── Dataset ─────────────────────────────────────────────────────────
DATASET_NAME = "stanfordnlp/snli"
LABEL_NAMES = ["entailment", "neutral", "contradiction"]
NUM_LABELS = 3
FILTER_LABEL = -1  # SNLI uses -1 for examples with no gold label

# ── TF-IDF Settings ────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE = (1, 2)  # unigrams + bigrams
TFIDF_SUBLINEAR_TF = True   # apply log normalization

# ── Logistic Regression ────────────────────────────────────────────
LR_MAX_ITER = 1000
LR_C = 1.0
LR_SOLVER = "lbfgs"

# ── Linear SVM ──────────────────────────────────────────────────────
SVM_MAX_ITER = 1000
SVM_C = 1.0

# ── BERT Settings ───────────────────────────────────────────────────
BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LENGTH = 128
BERT_BATCH_SIZE_TRAIN = 32
BERT_BATCH_SIZE_EVAL = 64
BERT_LEARNING_RATE = 2e-5
BERT_EPOCHS = 3
BERT_WARMUP_STEPS = 500
BERT_WEIGHT_DECAY = 0.01
BERT_FP16 = True  # mixed precision for faster training on T4

# ── General ─────────────────────────────────────────────────────────
SEED = 42


def create_dirs():
    """Create output directories if they don't exist."""
    for d in [RESULTS_DIR, MODELS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)
    print(f"Output directories ready under: {DRIVE_BASE}")
