"""
Top-level pipeline runner for Month 1.

Runs all steps in order:
  1. Download MultiTuDe
  2. Download HC3
  3. Preprocess
  4. Train supervised baseline
  5. Run evaluation harness
  6. Generate Month 1 findings report

Usage:
    python run_pipeline.py [--force] [--skip-download] [--skip-train]
"""
import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Month 1 pipeline runner")
    parser.add_argument("--force", action="store_true",
                        help="Re-download and re-run all steps even if cached")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data acquisition (assume raw files exist)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip supervised baseline training (assume checkpoint exists)")
    args = parser.parse_args()

    # Step 1 & 2 — Data acquisition
    if not args.skip_download:
        logger.info("=== Step 1: Downloading MultiTuDe ===")
        from src.acquisition.multitude import download as dl_multitude
        dl_multitude(force=args.force)

        logger.info("=== Step 2: Downloading HC3 ===")
        from src.acquisition.hc3 import download as dl_hc3
        dl_hc3(force=args.force)
    else:
        logger.info("Skipping data acquisition.")

    # Step 3 — Preprocessing
    logger.info("=== Step 3: Preprocessing ===")
    from src.preprocessing.pipeline import run as preprocess
    preprocess(force=args.force)

    # Step 4 — Statistical baseline (no training needed, but scored at eval time)
    logger.info("=== Step 6: Statistical baseline — scored in eval harness ===")

    # Step 5 — Supervised baseline training
    if not args.skip_train:
        logger.info("=== Step 7: Training supervised baseline ===")
        from src.config import resolve_path
        from src.baselines.supervised_baseline import SupervisedBaseline
        baseline = SupervisedBaseline()
        baseline.run_and_save(
            train_parquet=resolve_path("data/processed/train_en.parquet"),
            test_parquet=resolve_path("data/processed/test_en.parquet"),
        )
    else:
        logger.info("Skipping supervised baseline training.")

    # Step 8 — Evaluation harness
    logger.info("=== Step 8: Running evaluation harness ===")
    from src.eval.harness import run as run_eval
    run_eval(force_rescore=args.force)

    # Step 9 — Month 1 findings report
    logger.info("=== Step 9: Generating Month 1 findings report ===")
    from src.eval.report import generate
    report_path = generate()
    logger.info("Done. Month 1 findings: %s", report_path)


if __name__ == "__main__":
    main()
