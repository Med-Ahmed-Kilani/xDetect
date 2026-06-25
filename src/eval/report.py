"""
Generate the Month 1 findings summary from eval_results.json.

Outputs reports/month1_findings.txt — a plain-language document stating
the EN→DE zero-shot performance gap for the thesis.
"""
import json
import logging
from pathlib import Path

from src.config import load_config, resolve_path

logger = logging.getLogger(__name__)


def _find(results: list[dict], model: str, split: str) -> dict | None:
    for r in results:
        if r["model"] == model and r["split"] == split:
            return r
    return None


def generate(eval_json: Path | None = None, out: Path | None = None) -> Path:
    if eval_json is None:
        eval_json = resolve_path("reports/eval_results.json")
    if out is None:
        out = resolve_path("reports/month1_findings.txt")

    with open(eval_json) as f:
        report = json.load(f)

    results = report["results"]

    def fmt(r: dict | None) -> str:
        if r is None:
            return "  (not found in report)"
        o = r["overall"]
        auroc = f"{o['auroc']:.4f}" if o["auroc"] is not None else "N/A"
        return (
            f"  n={o['n']}  accuracy={o['accuracy']:.4f}  "
            f"F1={o['f1']:.4f}  AUROC={auroc}"
        )

    stat_en = _find(results, "statistical", "test_en")
    stat_de = _find(results, "statistical", "test_de")
    sup_en  = _find(results, "supervised",  "test_en")
    sup_de  = _find(results, "supervised",  "test_de")

    def gap(en_r: dict | None, de_r: dict | None, metric: str) -> str:
        if en_r and de_r:
            en_v = en_r["overall"].get(metric)
            de_v = de_r["overall"].get(metric)
            if en_v is not None and de_v is not None:
                return f"{en_v - de_v:+.4f}"
        return "N/A"

    warnings_block = ""
    if report.get("warnings"):
        warnings_block = (
            "\nSUSPICIOUS RESULTS — MANUAL REVIEW RECOMMENDED\n"
            + "\n".join(f"  ! {w}" for w in report["warnings"])
            + "\n"
        )

    lines = [
        "=" * 70,
        "Month 1 Findings: EN→DE Zero-Shot Transfer",
        f"Generated from: {eval_json}",
        "=" * 70,
        "",
        "1. STATISTICAL BASELINE (GPT-2 perplexity, zero training)",
        "",
        "   English test (in-distribution):",
        fmt(stat_en),
        "",
        "   German test (zero-shot transfer):",
        fmt(stat_de),
        "",
        f"   EN→DE accuracy gap:  {gap(stat_en, stat_de, 'accuracy')}",
        f"   EN→DE F1 gap:        {gap(stat_en, stat_de, 'f1')}",
        f"   EN→DE AUROC gap:     {gap(stat_en, stat_de, 'auroc')}",
        "",
        "2. SUPERVISED BASELINE (RoBERTa-base, English-only fine-tune)",
        "",
        "   English test (in-distribution):",
        fmt(sup_en),
        "",
        "   German test (zero-shot transfer):",
        fmt(sup_de),
        "",
        f"   EN→DE accuracy gap:  {gap(sup_en, sup_de, 'accuracy')}",
        f"   EN→DE F1 gap:        {gap(sup_en, sup_de, 'f1')}",
        f"   EN→DE AUROC gap:     {gap(sup_en, sup_de, 'auroc')}",
        "",
        "3. INTERPRETATION",
        "",
        "   The EN→DE gaps above are the thesis's first empirical data point",
        "   on cross-lingual generalization.  A large gap confirms that the",
        "   English-only supervised model does NOT transfer zero-shot to German",
        "   and motivates the multilingual XLM-R + adapter architecture in",
        "   Month 3.  A small or negligible gap would be surprising and should",
        "   be investigated before proceeding.",
        "",
        "   NOTE: Training data is MultiTuDe v3 English only (news domain,",
        "   7,954 rows, 7 generators + human).  German training data exists",
        "   in v3 but is intentionally withheld until Month 3 to preserve",
        "   the zero-shot transfer experiment.",
        "",
        "   NOTE: The supervised RoBERTa baseline is a throwaway reference",
        "   point, not the final architecture.  It will be replaced by the",
        "   multilingual backbone in Month 3.",
        "",
        warnings_block,
    ]

    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    logger.info("Month 1 findings written to %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate()
    print(f"Report: {path}")
