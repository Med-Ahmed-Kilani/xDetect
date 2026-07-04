"""
Month 3 Step 6: the three-condition scalability experiment runner.

Runs zero-shot / few-shot-adapter / full-retrain conditions across the
configured target languages (and, for few-shot, data sizes), and writes
the combined results to reports/scalability_results.json.

--lang splits the 12 GPU runs across Kaggle sessions: running with
--lang de only touches the "de" entries in each condition and merges them
into any existing report rather than clobbering the other language's results.
"""
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from src.adapters.adapter_trainer import AdapterTrainer
from src.adapters.full_retrain import FullRetrainer
from src.config import load_config, resolve_path
from src.eval.harness import evaluate_checkpoint

logger = logging.getLogger(__name__)


def _adapters_cfg() -> dict:
    return load_config("adapters")


def _train_test_paths(cfg_ds: dict, lang: str) -> tuple[Path, Path]:
    train_path = resolve_path(cfg_ds["processed"]["train_template"].replace("{lang}", lang))
    test_path = resolve_path(cfg_ds["processed"]["test_template"].replace("{lang}", lang))
    return train_path, test_path


def _clear_if_force(path: Path, force: bool) -> None:
    if force and path.exists():
        shutil.rmtree(path)


def run_zero_shot(languages: list[str]) -> dict[str, dict]:
    """Evaluate the frozen backbone directly — no adapter, no new training."""
    scal_cfg = _adapters_cfg()["scalability"]
    cfg_ds = load_config("datasets")
    backbone_path = resolve_path(scal_cfg["backbone_checkpoint"])

    results: dict[str, dict] = {}
    for lang in languages:
        _, test_path = _train_test_paths(cfg_ds, lang)
        logger.info("=== Zero-shot: evaluating backbone on test_%s ===", lang)
        results[lang] = evaluate_checkpoint(
            checkpoint_path=backbone_path, test_parquet=test_path, is_adapter=False,
        )
    return results


def run_few_shot_adapters(languages: list[str], data_sizes: list, force: bool = False) -> dict[str, dict]:
    """Train + evaluate one LoRA adapter per (language, data_size) pair."""
    scal_cfg = _adapters_cfg()["scalability"]
    cfg_ds = load_config("datasets")
    backbone_path = resolve_path(scal_cfg["backbone_checkpoint"])
    ckpt_template = scal_cfg["adapter_checkpoint_template"]

    results: dict[str, dict] = {}
    for lang in languages:
        results[lang] = {}
        train_path, test_path = _train_test_paths(cfg_ds, lang)
        for n in data_sizes:
            out_dir = resolve_path(ckpt_template.replace("{lang}", lang).replace("{n}", str(n)))
            _clear_if_force(out_dir, force)

            logger.info("=== Few-shot adapter: lang=%s n=%s ===", lang, n)
            trainer = AdapterTrainer(
                backbone_path=backbone_path, lang=lang, n_samples=n, output_dir=out_dir,
            )
            trainer.train(train_path)
            results[lang][str(n)] = evaluate_checkpoint(
                checkpoint_path=out_dir, test_parquet=test_path,
                is_adapter=True, backbone_path=backbone_path,
            )
    return results


def run_full_retrain(languages: list[str], force: bool = False) -> dict[str, dict]:
    """Fine-tune the entire model (warm-started from the backbone) per language."""
    scal_cfg = _adapters_cfg()["scalability"]
    cfg_ds = load_config("datasets")
    backbone_path = resolve_path(scal_cfg["backbone_checkpoint"])
    ckpt_template = scal_cfg["full_retrain_checkpoint_template"]

    results: dict[str, dict] = {}
    for lang in languages:
        train_path, test_path = _train_test_paths(cfg_ds, lang)
        out_dir = resolve_path(ckpt_template.replace("{lang}", lang))
        _clear_if_force(out_dir, force)

        logger.info("=== Full retrain: lang=%s ===", lang)
        retrainer = FullRetrainer(backbone_path=backbone_path, lang=lang, output_dir=out_dir)
        retrainer.train(train_path)
        results[lang] = evaluate_checkpoint(checkpoint_path=out_dir, test_parquet=test_path,
                                            is_adapter=False)
    return results


def run(force: bool = False, lang: Optional[str] = None) -> Path:
    """
    Runs all three conditions for the given language (or all target languages
    if lang is None), and writes/merges reports/scalability_results.json.
    """
    scal_cfg = _adapters_cfg()["scalability"]
    target_languages = scal_cfg["target_languages"]
    if lang is not None and lang not in target_languages:
        raise ValueError(f"--lang '{lang}' is not a valid target language. "
                         f"Choose from: {target_languages}")
    languages = [lang] if lang is not None else target_languages

    new_results = {
        "zero_shot": run_zero_shot(languages),
        "few_shot_adapter": run_few_shot_adapters(languages, scal_cfg["data_sizes"], force=force),
        "full_retrain": run_full_retrain(languages, force=force),
    }

    report_path = resolve_path(scal_cfg["report_path"])
    report_path.parent.mkdir(exist_ok=True)

    results = new_results
    if report_path.exists():
        with open(report_path) as f:
            existing = json.load(f)
        for condition, lang_results in new_results.items():
            existing.setdefault(condition, {}).update(lang_results)
        results = existing

    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Scalability results saved to %s", report_path)
    return report_path
