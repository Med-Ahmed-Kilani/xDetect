"""
Evaluation harness.

Runs both baselines on test_en (in-distribution) and test_de (zero-shot),
with metrics broken down per generator. Produces a deterministic JSON +
human-readable summary in /reports/.

Usage:
    python -m src.eval.harness
    python -m src.eval.harness --force-rescore
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.eval.metrics import compute_metrics, compute_per_generator

logger = logging.getLogger(__name__)

SUSPICIOUS_ACCURACY_LO = 0.52   # near-chance — flag for review
SUSPICIOUS_ACCURACY_HI = 0.995  # near-perfect — flag for review
DIVERGENCE_ACC_MIN  = 0.80      # high accuracy threshold for divergence check
DIVERGENCE_AUROC_MAX = 0.75     # mediocre AUROC threshold for divergence check


def _flag_suspicious(metrics: dict, split: str, model: str) -> list[str]:
    warnings: list[str] = []
    overall = metrics.get("overall", {})
    acc   = overall.get("accuracy")
    auroc = overall.get("auroc")

    if acc is None:
        return warnings

    if acc <= SUSPICIOUS_ACCURACY_LO:
        warnings.append(
            f"WARNING: {model} on {split} accuracy={acc:.3f} is near chance "
            f"(≤{SUSPICIOUS_ACCURACY_LO}). Manual review recommended."
        )
    if acc >= SUSPICIOUS_ACCURACY_HI:
        warnings.append(
            f"WARNING: {model} on {split} accuracy={acc:.3f} is suspiciously "
            f"near-perfect (≥{SUSPICIOUS_ACCURACY_HI}). Check for data leakage."
        )
    if auroc is not None and acc >= DIVERGENCE_ACC_MIN and auroc <= DIVERGENCE_AUROC_MAX:
        warnings.append(
            f"WARNING: {model} on {split}: high accuracy ({acc:.3f}) but mediocre "
            f"AUROC ({auroc:.3f}) — possible majority-class collapse under class "
            f"imbalance, not genuine discrimination. Check confusion matrix."
        )
    return warnings


def _score_statistical(df: pd.DataFrame, force: bool) -> pd.DataFrame:
    cache = resolve_path("reports/_stat_baseline_cache.parquet")
    if cache.exists() and not force:
        logger.info("Using cached statistical baseline scores.")
        cached = pd.read_parquet(cache)
        if set(["score_nll", "proba_machine", "pred_label"]).issubset(cached.columns):
            return cached

    from src.baselines.statistical_baseline import StatisticalBaseline
    baseline = StatisticalBaseline()
    texts = df["text"].tolist()
    nlls = baseline.score(texts)
    proba = baseline.predict_proba(texts)
    df = df.copy()
    df["score_nll"] = nlls
    df["proba_machine"] = proba
    df["pred_label"] = (proba >= baseline.threshold).astype(int)
    df.to_parquet(cache, index=False)
    return df


def _score_supervised(df: pd.DataFrame, ckpt: Path, force: bool) -> pd.DataFrame:
    cache = resolve_path("reports/_sup_baseline_cache.parquet")
    if cache.exists() and not force:
        logger.info("Using cached supervised baseline scores.")
        cached = pd.read_parquet(cache)
        if set(["proba_machine_sup", "pred_label_sup"]).issubset(cached.columns):
            return cached

    from src.baselines.supervised_baseline import SupervisedBaseline
    baseline = SupervisedBaseline()
    baseline.load(ckpt)
    texts = df["text"].tolist()
    proba = baseline.predict_proba(texts)
    df = df.copy()
    df["proba_machine_sup"] = proba
    df["pred_label_sup"] = (proba >= 0.5).astype(int)
    df.to_parquet(cache, index=False)
    return df


def _eval_split(
    df: pd.DataFrame,
    pred_col: str,
    proba_col: str,
    split_name: str,
    model_name: str,
) -> dict[str, Any]:
    labels = df["label"].to_numpy()
    preds = df[pred_col].to_numpy()
    proba = df[proba_col].to_numpy()
    generators = df["generator"].to_numpy()

    overall = compute_metrics(labels, preds, proba)
    per_gen = compute_per_generator(labels, preds, generators, proba)

    return {
        "split": split_name,
        "model": model_name,
        "overall": overall,
        "per_generator": per_gen,
    }


def run(force_rescore: bool = False) -> dict[str, Any]:
    """
    Run the full evaluation and return the structured report dict.
    Running twice on the same model/data produces identical output.
    """
    cfg_ds = load_config("datasets")
    cfg_m = load_config("models")
    report_dir = resolve_path(cfg_m["eval"]["report_dir"])
    report_dir.mkdir(exist_ok=True)

    test_en_path = resolve_path(cfg_ds["processed"]["test_en"])
    test_de_path = resolve_path(cfg_ds["processed"]["test_de"])
    ckpt = resolve_path(cfg_m["supervised_baseline"]["checkpoint_dir"])

    test_en = pd.read_parquet(test_en_path)
    test_de = pd.read_parquet(test_de_path)
    all_test = pd.concat([test_en, test_de], ignore_index=True)

    # --- Statistical baseline ---
    logger.info("Scoring with statistical baseline …")
    all_scored_stat = _score_statistical(all_test, force=force_rescore)
    en_stat = all_scored_stat[all_scored_stat["language"] == "en"]
    de_stat = all_scored_stat[all_scored_stat["language"] == "de"]

    # --- Supervised baseline ---
    logger.info("Scoring with supervised baseline …")
    all_scored_sup = _score_supervised(all_test, ckpt, force=force_rescore)
    en_sup = all_scored_sup[all_scored_sup["language"] == "en"]
    de_sup = all_scored_sup[all_scored_sup["language"] == "de"]

    results: list[dict] = [
        _eval_split(en_stat, "pred_label", "proba_machine", "test_en", "statistical"),
        _eval_split(de_stat, "pred_label", "proba_machine", "test_de", "statistical"),
        _eval_split(en_sup, "pred_label_sup", "proba_machine_sup", "test_en", "supervised"),
        _eval_split(de_sup, "pred_label_sup", "proba_machine_sup", "test_de", "supervised"),
    ]

    warnings: list[str] = []
    for r in results:
        warnings.extend(_flag_suspicious(r, r["split"], r["model"]))

    report: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "results": results,
        "warnings": warnings,
    }

    # Write JSON report
    json_path = report_dir / "eval_results.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("JSON report → %s", json_path)

    # Write human-readable summary
    summary_path = report_dir / "eval_summary.txt"
    _write_summary(report, summary_path)
    logger.info("Summary → %s", summary_path)

    return report


def _write_summary(report: dict, path: Path) -> None:
    lines = [
        "=" * 70,
        "Month 1 Evaluation Results",
        f"Generated: {report['generated_at']}",
        "=" * 70,
        "",
    ]
    for r in report["results"]:
        o = r["overall"]
        auroc = f"{o['auroc']:.4f}" if o["auroc"] is not None else "N/A"
        lines += [
            f"Model: {r['model']:20s}  Split: {r['split']}",
            f"  n={o['n']}  accuracy={o['accuracy']:.4f}  "
            f"F1={o['f1']:.4f}  AUROC={auroc}",
            "",
        ]
        for gen, m in r["per_generator"].items():
            a_auroc = f"{m['auroc']:.4f}" if m.get("auroc") else "N/A"
            lines.append(
                f"    generator={gen:<20s}  n={m['n']:5d}  "
                f"acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  AUROC={a_auroc}"
            )
        lines.append("")

    if report["warnings"]:
        lines += ["WARNINGS:", ""]
        for w in report["warnings"]:
            lines.append(f"  ! {w}")
        lines.append("")

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-rescore", action="store_true")
    args = parser.parse_args()
    result = run(force_rescore=args.force_rescore)
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))
