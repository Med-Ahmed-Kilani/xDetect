"""
Smoke-test the full pipeline using HC3 only (no MultiTuDe needed).

Splits HC3 80/20 into a temporary train/test set, runs preprocessing,
trains the supervised baseline, and runs the eval harness.

Usage:
    python test_hc3_only.py              # full run
    python test_hc3_only.py --skip-train # skip RoBERTa fine-tune
    python test_hc3_only.py --sample 500 # use only 500 rows (fast CPU test)
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("hc3_smoke")

PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip RoBERTa fine-tuning (assumes checkpoint exists)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Use only N rows total (useful for quick CPU tests)")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Step 1: Load HC3
    # ------------------------------------------------------------------ #
    logger.info("=== Step 1: Loading HC3 ===")
    from src.acquisition.hc3 import load as load_hc3
    df = load_hc3()

    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42).reset_index(drop=True)
        logger.info("Sampled %d rows for quick test.", len(df))

    # ------------------------------------------------------------------ #
    # Step 2: Unify schema (same as pipeline.py does for HC3)
    # ------------------------------------------------------------------ #
    logger.info("=== Step 2: Applying unified schema ===")
    from src.preprocessing.pipeline import _unify_hc3, _clean_df, deduplicate

    df = _unify_hc3(df)
    df = _clean_df(df, "hc3")
    df = deduplicate(df)
    logger.info("After cleaning: %d rows", len(df))

    # ------------------------------------------------------------------ #
    # Step 3: 80/20 train/test split (stratified by label)
    # ------------------------------------------------------------------ #
    logger.info("=== Step 3: Splitting 80/20 ===")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    logger.info("Train: %d rows  |  Test: %d rows", len(train_df), len(test_df))

    # Write to the standard processed paths so the rest of the pipeline
    # can read them without modification
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DIR / "train_en.parquet"
    test_path  = PROCESSED_DIR / "test_en.parquet"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path,  index=False)
    logger.info("Wrote %s and %s", train_path, test_path)

    # ------------------------------------------------------------------ #
    # Step 4: Train supervised baseline (optional)
    # ------------------------------------------------------------------ #
    if not args.skip_train:
        logger.info("=== Step 4: Training supervised baseline ===")
        from src.baselines.supervised_baseline import SupervisedBaseline
        baseline = SupervisedBaseline()
        baseline.run_and_save(
            train_parquet=train_path,
            test_parquet=test_path,
        )
    else:
        logger.info("Skipping training (--skip-train).")

    # ------------------------------------------------------------------ #
    # Step 5: Eval harness (English only — no German test set here)
    # ------------------------------------------------------------------ #
    logger.info("=== Step 5: Running eval harness (EN only) ===")
    from src.eval.harness import _eval_split
    from src.baselines.statistical_baseline import StatisticalBaseline

    stat = StatisticalBaseline()
    scored = stat.run_on_dataset(test_path)

    results = []
    results.append(_eval_split(scored, "pred_label", "proba_machine", "test_en_hc3", "statistical"))

    if not args.skip_train:
        from src.baselines.supervised_baseline import SupervisedBaseline
        from src.config import load_config, resolve_path
        sup = SupervisedBaseline()
        sup.load()
        texts = test_df["text"].tolist()
        import numpy as np
        proba = sup.predict_proba(texts)
        scored_sup = test_df.copy()
        scored_sup["proba_machine_sup"] = proba
        scored_sup["pred_label_sup"] = (proba >= 0.5).astype(int)
        results.append(_eval_split(scored_sup, "pred_label_sup", "proba_machine_sup",
                                   "test_en_hc3", "supervised"))

    # ------------------------------------------------------------------ #
    # Step 6: Print results
    # ------------------------------------------------------------------ #
    logger.info("=== Results ===")
    REPORT_DIR.mkdir(exist_ok=True)
    report_path = REPORT_DIR / "hc3_smoke_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    for r in results:
        o = r["overall"]
        auroc = f"{o['auroc']:.4f}" if o["auroc"] else "N/A"
        print(f"\nModel: {r['model']}  |  Split: {r['split']}")
        print(f"  n={o['n']}  accuracy={o['accuracy']:.4f}  F1={o['f1']:.4f}  AUROC={auroc}")
        print("  Per generator:")
        for gen, m in r["per_generator"].items():
            print(f"    {gen:<12s}  n={m['n']:5d}  acc={m['accuracy']:.4f}  F1={m['f1']:.4f}")

    print(f"\nFull results saved to {report_path}")


if __name__ == "__main__":
    main()
