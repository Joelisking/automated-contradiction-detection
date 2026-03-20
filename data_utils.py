"""
Data Loading and Preprocessing for SNLI
========================================
Loads SNLI via HuggingFace datasets, filters bad labels,
and provides utilities for both TF-IDF and BERT pipelines.
"""

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import numpy as np

import config


def load_snli():
    """
    Load SNLI dataset and filter out examples with no gold label.
    
    Returns:
        dict with 'train', 'validation', 'test' splits, each a HF Dataset.
    """
    print("Loading SNLI dataset...")
    dataset = load_dataset(config.DATASET_NAME)

    # Filter out examples where label == -1 (no annotator agreement)
    for split in ["train", "validation", "test"]:
        original_size = len(dataset[split])
        dataset[split] = dataset[split].filter(
            lambda x: x["label"] != config.FILTER_LABEL
        )
        filtered_size = len(dataset[split])
        removed = original_size - filtered_size
        print(f"  {split}: {original_size} → {filtered_size} ({removed} removed)")

    return dataset


def extract_texts_and_labels(dataset_split):
    """
    Extract premise, hypothesis, and labels from a dataset split.
    
    Returns:
        premises: list of str
        hypotheses: list of str
        labels: numpy array of int
    """
    premises = list(dataset_split["premise"])
    hypotheses = list(dataset_split["hypothesis"])
    labels = np.array(dataset_split["label"])
    return premises, hypotheses, labels


def build_tfidf_features(train_premises, train_hypotheses,
                         val_premises, val_hypotheses,
                         test_premises, test_hypotheses):
    """
    Build TF-IDF features with interaction features for sentence pairs.
    
    Feature vector for each pair = [P; H; |P - H|; P ⊙ H]
    where P = TF-IDF(premise), H = TF-IDF(hypothesis).
    
    Returns:
        X_train, X_val, X_test: scipy sparse matrices
        vectorizer: fitted TfidfVectorizer
    """
    print("Building TF-IDF features with interaction features...")

    # Fit TF-IDF on all training text (both premises and hypotheses)
    vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        sublinear_tf=config.TFIDF_SUBLINEAR_TF,
        dtype=np.float32,
    )

    all_train_text = train_premises + train_hypotheses
    vectorizer.fit(all_train_text)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")

    def _make_features(premises, hypotheses, split_name):
        """Create interaction features for one split."""
        P = vectorizer.transform(premises)
        H = vectorizer.transform(hypotheses)

        # Interaction features
        diff = csr_matrix(np.abs(P - H))  # |premise - hypothesis|
        prod = P.multiply(H)               # element-wise product

        # Concatenate: [P; H; |P-H|; P⊙H]
        X = hstack([P, H, diff, prod], format="csr")
        print(f"  {split_name} feature matrix: {X.shape}")
        return X

    X_train = _make_features(train_premises, train_hypotheses, "Train")
    X_val = _make_features(val_premises, val_hypotheses, "Validation")
    X_test = _make_features(test_premises, test_hypotheses, "Test")

    return X_train, X_val, X_test, vectorizer


def tokenize_for_bert(dataset, tokenizer):
    """
    Tokenize premise-hypothesis pairs for BERT.
    
    Args:
        dataset: HuggingFace dataset (single split)
        tokenizer: BERT tokenizer
    
    Returns:
        Tokenized dataset ready for Trainer
    """
    def _tokenize(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=config.BERT_MAX_LENGTH,
            padding="max_length",
        )

    tokenized = dataset.map(_tokenize, batched=True, batch_size=1000)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized
