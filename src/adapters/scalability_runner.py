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
from src.eval.metrics import aggregate_seed_metrics

logger = logging.getLogger(__name__)


def _adapters_cfg() -> dict:
    return load_config("adapters")


def _scal_seeds() -> list[int]:
    """
    Seeds every condition is run with. Sourced from scalability.seeds, falling
    back to adapter_training.seeds / a single legacy seed for older configs.
    """
    cfg = _adapters_cfg()
    scal = cfg["scalability"]
    if "seeds" in scal:
        return list(scal["seeds"])
    at = cfg.get("adapter_training", {})
    if "seeds" in at:
        return list(at["seeds"])
    return [at.get("seed", 42)]


def _backbone_for_seed(backbone_path: Path, seed: int) -> Path:
    """
    Use the matching per-seed backbone checkpoint (<backbone>/seed_<N>/) when it
    exists, else fall back to the backbone dir itself (legacy single-seed
    backbone).
    """
    cand = backbone_path / f"seed_{seed}"
    return cand if cand.exists() else backbone_path


def _aggregate_condition(per_seed: dict[int, dict]) -> dict:
    """
    Collapse ``{seed: {"overall": {...}, "per_generator": {...}}}`` (one
    evaluate_checkpoint result per seed) into

        {"overall": {"accuracy"|"f1"|"auroc": {"mean", "std", "values"},
                     "n": int, "seeds": [...]},
         "per_seed": {seed: full_metrics_dict}}
    """
    overall = aggregate_seed_metrics({s: m["overall"] for s, m in per_seed.items()})
    return {"overall": overall,
            "per_seed": {str(s): m for s, m in per_seed.items()}}


def _train_test_paths(cfg_ds: dict, lang: str) -> tuple[Path, Path]:
    train_path = resolve_path(cfg_ds["processed"]["train_template"].replace("{lang}", lang))
    test_path = resolve_path(cfg_ds["processed"]["test_template"].replace("{lang}", lang))
    return train_path, test_path


def _clear_if_force(path: Path, force: bool) -> None:
    if force and path.exists():
        shutil.rmtree(path)


def run_zero_shot(languages: list[str]) -> dict[str, dict]:
    """
    Evaluate the frozen backbone directly — no adapter, no new training.
    Run once per seed against the matching per-seed backbone checkpoint and
    aggregate as mean ± std.
    """
    scal_cfg = _adapters_cfg()["scalability"]
    cfg_ds = load_config("datasets")
    backbone_path = resolve_path(scal_cfg["backbone_checkpoint"])
    seeds = _scal_seeds()

    results: dict[str, dict] = {}
    for lang in languages:
        _, test_path = _train_test_paths(cfg_ds, lang)
        per_seed: dict[int, dict] = {}
        for seed in seeds:
            bb = _backbone_for_seed(backbone_path, seed)
            logger.info("=== Zero-shot: backbone seed=%d on test_%s (%s) ===",
                        seed, lang, bb)
            per_seed[seed] = evaluate_checkpoint(
                checkpoint_path=bb, test_parquet=test_path, is_adapter=False,
            )
        results[lang] = _aggregate_condition(per_seed)
    return results


def run_few_shot_adapters(languages: list[str], data_sizes: list, force: bool = False) -> dict[str, dict]:
    """
    Train + evaluate one LoRA adapter per (language, data_size, seed) triple,
    then aggregate across seeds as mean ± std. Each seed trains into
    <adapter_ckpt>/seed_<N>/ so a partially completed run resumes seed-by-seed.
    """
    scal_cfg = _adapters_cfg()["scalability"]
    cfg_ds = load_config("datasets")
    backbone_path = resolve_path(scal_cfg["backbone_checkpoint"])
    ckpt_template = scal_cfg["adapter_checkpoint_template"]
    seeds = _scal_seeds()

    results: dict[str, dict] = {}
    for lang in languages:
        results[lang] = {}
        train_path, test_path = _train_test_paths(cfg_ds, lang)
        for n in data_sizes:
            out_dir = resolve_path(ckpt_template.replace("{lang}", lang).replace("{n}", str(n)))
            _clear_if_force(out_dir, force)

            per_seed: dict[int, dict] = {}
            for seed in seeds:
                bb = _backbone_for_seed(backbone_path, seed)
                seed_dir = out_dir / f"seed_{seed}"
                logger.info("=== Few-shot adapter: lang=%s n=%s seed=%d ===", lang, n, seed)
                trainer = AdapterTrainer(
                    backbone_path=bb, lang=lang, n_samples=n, output_dir=seed_dir,
                    seed=seed,
                )
                trainer.train(train_path)
                per_seed[seed] = evaluate_checkpoint(
                    checkpoint_path=seed_dir, test_parquet=test_path,
                    is_adapter=True, backbone_path=bb,
                )
            results[lang][str(n)] = _aggregate_condition(per_seed)
    return results


def run_full_retrain(languages: list[str], force: bool = False) -> dict[str, dict]:
    """
    Fine-tune the entire model (warm-started from the backbone) per language,
    once per seed into <full_retrain_ckpt>/seed_<N>/, and aggregate across
    seeds as mean ± std.
    """
    scal_cfg = _adapters_cfg()["scalability"]
    cfg_ds = load_config("datasets")
    backbone_path = resolve_path(scal_cfg["backbone_checkpoint"])
    ckpt_template = scal_cfg["full_retrain_checkpoint_template"]
    seeds = _scal_seeds()

    results: dict[str, dict] = {}
    for lang in languages:
        train_path, test_path = _train_test_paths(cfg_ds, lang)
        out_dir = resolve_path(ckpt_template.replace("{lang}", lang))
        _clear_if_force(out_dir, force)

        per_seed: dict[int, dict] = {}
        for seed in seeds:
            bb = _backbone_for_seed(backbone_path, seed)
            seed_dir = out_dir / f"seed_{seed}"
            logger.info("=== Full retrain: lang=%s seed=%d ===", lang, seed)
            retrainer = FullRetrainer(
                backbone_path=bb, lang=lang, output_dir=seed_dir, seed=seed,
            )
            retrainer.train(train_path, seed=seed, output_dir=seed_dir)
            per_seed[seed] = evaluate_checkpoint(
                checkpoint_path=seed_dir, test_parquet=test_path, is_adapter=False,
            )
        results[lang] = _aggregate_condition(per_seed)
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
