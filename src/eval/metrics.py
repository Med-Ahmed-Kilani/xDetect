"""Evaluation metric helpers."""
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    proba: Optional[np.ndarray] = None,
) -> dict:
    """Compute accuracy, F1, and (if proba given) AUROC."""
    result = {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    if proba is not None and len(np.unique(labels)) > 1:
        result["auroc"] = float(roc_auc_score(labels, proba))
    else:
        result["auroc"] = None
    return result


def compute_per_generator(
    labels: np.ndarray,
    preds: np.ndarray,
    generators: np.ndarray,
    proba: Optional[np.ndarray] = None,
) -> dict[str, dict]:
    """Compute metrics broken down by generator name."""
    result: dict[str, dict] = {}
    unique = np.unique(generators)
    for gen in unique:
        mask = generators == gen
        g_labels = labels[mask]
        g_preds = preds[mask]
        g_proba = proba[mask] if proba is not None else None
        result[str(gen)] = compute_metrics(g_labels, g_preds, g_proba)
    return result
