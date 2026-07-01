"""
Month 2: train all three candidate multilingual backbones on the pooled
EN+DE+AR training set and evaluate each on all three language test sets.

Backbone configs come from configs/models.yaml (mbert, xlmr_base,
mdeberta_v3_base). Each backbone produces its own checkpoint and a per-language
metrics JSON.

mDeBERTa-v3-base: fp16 is explicitly disabled — the model card states fp16
training is not supported; the config enforces fp16=false for that entry.
"""
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path
from src.baselines.supervised_baseline import SupervisedBaseline

logger = logging.getLogger(__name__)

BACKBONE_KEYS = ["mbert", "xlmr_base", "mdeberta_v3_base"]


def train_all(force: bool = False,
              only: str | None = None) -> dict[str, Path]:
    """
    Fine-tune each backbone on the pooled training set.

    Returns a dict mapping backbone_key → checkpoint Path.
    Skips a backbone if its checkpoint already exists and force=False.

    only: if given, train only that backbone key and skip the others.
    """
    if only is not None and only not in BACKBONE_KEYS:
        raise ValueError(f"--only '{only}' is not a valid backbone key. "
                         f"Choose from: {BACKBONE_KEYS}")

    cfg_m = load_config("models")
    train_path = resolve_path(
        load_config("datasets")["processed"]["train_pooled"]
    )

    checkpoints: dict[str, Path] = {}
    for key in BACKBONE_KEYS:
        if only is not None and key != only:
            logger.info("Skipping backbone %s (--only %s).", key, only)
            continue
        cfg = cfg_m[key]
        ckpt = resolve_path(cfg["checkpoint_dir"])
        checkpoints[key] = ckpt

        if ckpt.exists() and not force:
            logger.info("Checkpoint for %s already exists — skipping training.", key)
            continue

        assert not cfg.get("fp16", False) or key != "mdeberta_v3_base", (
            "fp16 must be false for mdeberta_v3_base — check configs/models.yaml"
        )

        logger.info("=== Training backbone: %s (%s) ===", key, cfg["model_id"])
        baseline = SupervisedBaseline(cfg=cfg)
        baseline.train(train_path)
        logger.info("Checkpoint saved: %s", ckpt)

    return checkpoints


def evaluate_all(languages: list[str] | None = None,
                 only: str | None = None) -> dict[str, dict[str, dict]]:
    """
    Evaluate backbones against all language test sets.

    Returns nested dict: {backbone_key: {lang: metrics_dict}}.

    only: if given, evaluate only that backbone key and skip the others.
    """
    cfg_m  = load_config("models")
    cfg_ds = load_config("datasets")
    if languages is None:
        languages = cfg_ds["multitude_v3"]["languages"]

    processed_cfg = cfg_ds["processed"]

    results: dict[str, dict[str, dict]] = {}
    for key in BACKBONE_KEYS:
        if only is not None and key != only:
            continue
        cfg = cfg_m[key]
        ckpt = resolve_path(cfg["checkpoint_dir"])
        logger.info("Loading backbone %s from %s …", key, ckpt)

        baseline = SupervisedBaseline(cfg=cfg)
        baseline.load(ckpt)

        results[key] = {}
        for lang in languages:
            test_path = resolve_path(
                processed_cfg["test_template"].replace("{lang}", lang)
            )
            logger.info("  Evaluating %s on test_%s …", key, lang)
            metrics = baseline.evaluate(test_path)
            results[key][lang] = metrics
            logger.info("    accuracy=%.4f  F1=%.4f  AUROC=%.4f",
                        metrics["accuracy"], metrics["f1"], metrics["auroc"])

    return results


def run(force: bool = False, only: str | None = None) -> Path:
    """
    Train backbones, evaluate, and write the comparison JSON.

    force: re-train even if a checkpoint already exists.
    only:  train and evaluate only this backbone key, skip the others.

    Returns path to the saved JSON report.
    """
    train_all(force=force, only=only)
    results = evaluate_all(only=only)

    report_dir = resolve_path(load_config("models")["eval"]["report_dir"])
    report_dir.mkdir(exist_ok=True)
    out = report_dir / "backbone_comparison.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Backbone comparison saved to %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = run()
    print(f"Report: {path}")
