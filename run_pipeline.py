"""
Top-level pipeline runner — Month 1 (MultiTuDe v3, English baseline).

Steps:
  1. Cache MultiTuDe v3 splits from the local CSV
  2. Preprocess → data/processed/
  3. Train supervised baseline (RoBERTa-base, English only)
  4. Run evaluation harness (EN in-distribution + DE zero-shot)
  5. Generate Month 1 findings report

Note: HC3 acquisition exists in src/acquisition/hc3.py but is NOT called
here — it is reserved for an optional Month 4 robustness check.

Usage:
    python run_pipeline.py [--force] [--skip-acquisition] [--skip-train]
"""
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Month 1 pipeline runner")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all steps even if outputs are already cached")
    parser.add_argument("--skip-acquisition", action="store_true",
                        help="Skip CSV filtering (assume data/raw/multitude_v3/ parquets exist)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip supervised baseline training (assume checkpoint exists)")
    args = parser.parse_args()

    # Step 1 — Filter and cache MultiTuDe v3 splits
    if not args.skip_acquisition:
        logger.info("=== Step 1: Caching MultiTuDe v3 splits ===")
        from src.acquisition.multitude import load as load_multitude
        load_multitude(force=args.force)
    else:
        logger.info("Skipping acquisition.")

    # Step 2 — Preprocess
    logger.info("=== Step 2: Preprocessing ===")
    from src.preprocessing.pipeline import run as preprocess
    preprocess(force=args.force)

    # Step 3 — Train supervised baseline
    if not args.skip_train:
        logger.info("=== Step 3: Training supervised baseline ===")
        from src.config import resolve_path
        from src.baselines.supervised_baseline import SupervisedBaseline
        baseline = SupervisedBaseline()
        baseline.run_and_save(
            train_parquet=resolve_path("data/processed/train_en.parquet"),
            test_parquet=resolve_path("data/processed/test_en.parquet"),
        )
    else:
        logger.info("Skipping supervised baseline training.")

    # Step 4 — Evaluation harness (statistical + supervised, EN + DE)
    logger.info("=== Step 4: Running evaluation harness ===")
    from src.eval.harness import run as run_eval
    run_eval(force_rescore=args.force)

    # Step 5 — Month 1 findings report
    logger.info("=== Step 5: Generating Month 1 findings report ===")
    from src.eval.report import generate
    report_path = generate()
    logger.info("Done. Month 1 findings: %s", report_path)


if __name__ == "__main__":
    main()
