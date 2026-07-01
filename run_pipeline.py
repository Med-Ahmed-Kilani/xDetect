"""
Top-level pipeline runner — supports Month 1 (EN baseline) and Month 2
(pooled EN+DE+AR backbone comparison).

Steps (Month 1 mode — default):
  1. Cache MultiTuDe v3 splits from the local CSV
  2. Preprocess → data/processed/ (English only by default)
  3. Train supervised baseline (RoBERTa-base, English only)
  4. Run evaluation harness (EN in-distribution + DE zero-shot)
  5. Generate Month 1 findings report

Additional steps (Month 2 mode — --backbone):
  6. Train mBERT, XLM-R-base, mDeBERTa-v3-base on pooled EN+DE+AR
  7. Evaluate all three backbones on all three language test sets
  8. Generate backbone comparison report with winner declaration

Note: HC3 acquisition exists in src/acquisition/hc3.py but is NOT called
here — it is reserved for an optional Month 4 robustness check.

Usage:
    python run_pipeline.py [--force] [--skip-acquisition] [--skip-train]
    python run_pipeline.py --backbone [--force] [--skip-acquisition]
    python run_pipeline.py --backbone --only mdeberta_v3_base [--force]
    python run_pipeline.py --backbone --skip-training
"""
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="xDetect pipeline runner")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all steps even if outputs are already cached")
    parser.add_argument("--skip-acquisition", action="store_true",
                        help="Skip CSV filtering (assume raw parquets exist)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (assume checkpoints exist)")
    parser.add_argument("--backbone", action="store_true",
                        help="Run Month 2 backbone comparison (mBERT/XLM-R/mDeBERTa)")
    parser.add_argument("--only", metavar="MODEL_KEY",
                        help="(--backbone) Train and evaluate only this backbone key "
                             "(mbert | xlmr_base | mdeberta_v3_base); skip the others")
    parser.add_argument("--skip-training", action="store_true",
                        help="(--backbone) Skip all training; go straight to comparison "
                             "report generation using existing checkpoints")
    args = parser.parse_args()

    # Step 1 — Filter and cache MultiTuDe v3 splits (all configured languages)
    if not args.skip_acquisition:
        logger.info("=== Step 1: Caching MultiTuDe v3 splits ===")
        from src.acquisition.multitude import load as load_multitude
        load_multitude(force=args.force)
    else:
        logger.info("Skipping acquisition.")

    # Step 2 — Preprocess (pooled EN+DE+AR training set + per-language test sets)
    logger.info("=== Step 2: Preprocessing ===")
    from src.preprocessing.pipeline import run as preprocess
    preprocess(force=args.force)

    if not args.backbone:
        # Month 1 path: RoBERTa-base, English only --------------------------

        # Step 3 — Train supervised baseline
        if not args.skip_train:
            logger.info("=== Step 3: Training supervised baseline (EN) ===")
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

    else:
        # Month 2 path: multilingual backbone comparison ---------------------
        from src.config import resolve_path

        if args.skip_training:
            # Skip all training; go straight to report with existing checkpoints
            logger.info("Skipping backbone training (--skip-training).")
            comparison_json = resolve_path("reports/backbone_comparison.json")
        else:
            only = args.only or None
            if only:
                logger.info("=== Steps 6–7: Training + evaluating backbone '%s' only ===", only)
            else:
                logger.info("=== Steps 6–7: Backbone training + evaluation ===")
            from src.baselines.backbone_trainer import run as run_backbone
            comparison_json = run_backbone(force=args.force, only=only)

        # Step 8 — Backbone comparison report
        logger.info("=== Step 8: Generating backbone comparison report ===")
        from src.eval.backbone_report import generate as gen_backbone_report
        report_path = gen_backbone_report(comparison_json=comparison_json)
        logger.info("Done. Month 2 findings: %s", report_path)


if __name__ == "__main__":
    main()
