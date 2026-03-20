"""
BERT Fine-Tuning for NLI
=========================
Fine-tunes bert-base-uncased on SNLI using HuggingFace Trainer.
"""

import os
import time
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score
import torch

import config


def get_tokenizer():
    """Load BERT tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    return tokenizer


def get_model():
    """Load BERT model for sequence classification (3 classes)."""
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BERT_MODEL_NAME,
        num_labels=config.NUM_LABELS,
    )
    return model


def _compute_metrics_for_trainer(eval_pred):
    """Metric computation function for HuggingFace Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}


def get_training_args(output_dir=None):
    """
    Build TrainingArguments optimized for free Colab (T4 GPU).
    """
    if output_dir is None:
        output_dir = os.path.join(config.MODELS_DIR, "bert_checkpoints")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.BERT_EPOCHS,
        per_device_train_batch_size=config.BERT_BATCH_SIZE_TRAIN,
        per_device_eval_batch_size=config.BERT_BATCH_SIZE_EVAL,
        learning_rate=config.BERT_LEARNING_RATE,
        weight_decay=config.BERT_WEIGHT_DECAY,
        warmup_steps=config.BERT_WARMUP_STEPS,
        fp16=config.BERT_FP16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=500,
        save_total_limit=2,  # save disk space
        seed=config.SEED,
        report_to="none",  # disable wandb etc.
    )
    return args


def train_bert(train_dataset, val_dataset):
    """
    Fine-tune BERT on tokenized SNLI data.
    
    Args:
        train_dataset: tokenized HF dataset (train split)
        val_dataset: tokenized HF dataset (validation split)
    
    Returns:
        trainer: fitted Trainer object
        model: best model
        elapsed: training time in seconds
    """
    print("\nFine-tuning BERT...")
    print(f"  Model: {config.BERT_MODEL_NAME}")
    print(f"  Epochs: {config.BERT_EPOCHS}")
    print(f"  Batch size: {config.BERT_BATCH_SIZE_TRAIN}")
    print(f"  Learning rate: {config.BERT_LEARNING_RATE}")
    print(f"  FP16: {config.BERT_FP16}")

    model = get_model()
    training_args = get_training_args()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_metrics_for_trainer,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"  Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return trainer, model, elapsed


def predict_bert(trainer, dataset):
    """
    Get predictions from trained BERT model.
    
    Returns:
        predictions: numpy array of predicted labels
    """
    output = trainer.predict(dataset)
    predictions = np.argmax(output.predictions, axis=-1)
    return predictions


def save_bert_model(trainer):
    """Save the best BERT model to Google Drive."""
    config.create_dirs()
    save_path = os.path.join(config.MODELS_DIR, "bert_best")
    trainer.save_model(save_path)
    print(f"  BERT model saved to: {save_path}")
