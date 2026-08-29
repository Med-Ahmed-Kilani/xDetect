"""Evaluation metric helpers."""
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

_METRIC_KEYS = ("accuracy", "f1", "auroc")


def aggregate_seed_metrics(per_seed: dict[int, dict]) -> dict:
    """
    Collapse ``{seed: {accuracy, f1, auroc, n, ...}}`` into a single dict of
    ``{metric: {"mean": .., "std": .., "values": [..]}}`` plus ``n`` and the
    list of ``seeds`` that contributed.

    ``std`` is the sample standard deviation (ddof=1) across seeds, or 0.0 when
    fewer than two seeds have a value for that metric. ``None`` values (e.g.
    AUROC on a single-class split) are dropped before aggregating.
    """
    seeds = sorted(per_seed)
    agg: dict = {"seeds": seeds}
    for key in _METRIC_KEYS:
        values = [
            per_seed[s][key]
            for s in seeds
            if per_seed[s].get(key) is not None
        ]
        if values:
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        else:
            mean, std = None, None
        agg[key] = {"mean": mean, "std": std, "values": values}
    ns = [per_seed[s].get("n") for s in seeds if per_seed[s].get("n") is not None]
    agg["n"] = ns[0] if ns else None
    return agg


def mean_of(entry) -> Optional[float]:
    """
    Return the scalar mean of a metric entry, accepting either a bare number
    (legacy single-seed reports) or an aggregated ``{"mean": .., "std": ..}``
    dict (multi-seed reports). Returns ``None`` if unavailable.
    """
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get("mean")
    return entry


def fmt_mean_std(entry, places: int = 4) -> str:
    """
    Format a metric entry as ``mean ± std``. Accepts a bare number (renders
    with no ± term) or an aggregated ``{"mean": .., "std": ..}`` dict.
    """
    if entry is None:
        return "N/A"
    if isinstance(entry, dict):
        mean, std = entry.get("mean"), entry.get("std")
        if mean is None:
            return "N/A"
        if std is None:
            return f"{mean:.{places}f}"
        return f"{mean:.{places}f} ± {std:.{places}f}"
    return f"{entry:.{places}f}"


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
