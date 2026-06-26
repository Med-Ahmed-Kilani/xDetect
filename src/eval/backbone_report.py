"""
Month 2: backbone comparison report.

Reads backbone_comparison.json and produces a human-readable summary
that explicitly declares a winner with supporting numbers.

If results are close or ambiguous, the report says so rather than picking
arbitrarily — closeness threshold is configurable (default: 0.5pp).
"""
import json
import logging
from pathlib import Path

from src.config import load_config, resolve_path

logger = logging.getLogger(__name__)

CLOSE_THRESHOLD = 0.005   # 0.5 percentage points — below this, call it a tie


def select_winner(
    results: dict[str, dict[str, dict]],
    metric: str = "f1",
) -> tuple[str, dict[str, float], bool]:
    """
    Select the backbone with the highest mean metric across all languages.

    Returns (winner_key, per_backbone_mean_scores, is_tie).
    A tie is declared when the top-2 scores differ by less than CLOSE_THRESHOLD.
    """
    means: dict[str, float] = {}
    for backbone, lang_metrics in results.items():
        values = [m[metric] for m in lang_metrics.values() if m.get(metric) is not None]
        means[backbone] = sum(values) / len(values) if values else 0.0

    ranked = sorted(means.items(), key=lambda x: x[1], reverse=True)
    winner = ranked[0][0]
    is_tie = len(ranked) >= 2 and (ranked[0][1] - ranked[1][1]) < CLOSE_THRESHOLD
    return winner, means, is_tie


def generate(comparison_json: Path | None = None, out: Path | None = None) -> Path:
    if comparison_json is None:
        comparison_json = resolve_path("reports/backbone_comparison.json")
    if out is None:
        out = resolve_path("reports/backbone_comparison_report.txt")

    with open(comparison_json) as f:
        results = json.load(f)

    cfg_m = load_config("models")
    languages = load_config("datasets")["multitude_v3"]["languages"]
    backbone_keys = cfg_m["eval"]["backbone_names"]

    winner_f1,    means_f1,    tie_f1    = select_winner(results, "f1")
    winner_auroc, means_auroc, tie_auroc = select_winner(results, "auroc")
    winner_acc,   means_acc,   tie_acc   = select_winner(results, "accuracy")

    # Primary ranking metric is F1; AUROC used as tiebreaker
    primary_winner = winner_f1
    is_tie = tie_f1

    lines = [
        "=" * 70,
        "Month 2: Backbone Comparison Report",
        f"Generated from: {comparison_json}",
        "=" * 70,
        "",
        "PER-BACKBONE RESULTS (accuracy / F1 / AUROC)",
        "",
    ]

    for backbone in backbone_keys:
        model_id = cfg_m.get(backbone, {}).get("model_id", backbone)
        lines.append(f"  {backbone} ({model_id})")
        lang_data = results.get(backbone, {})
        f1_vals, acc_vals, auroc_vals = [], [], []
        for lang in languages:
            m = lang_data.get(lang, {})
            acc   = m.get("accuracy")
            f1    = m.get("f1")
            auroc = m.get("auroc")
            if acc   is not None: acc_vals.append(acc)
            if f1    is not None: f1_vals.append(f1)
            if auroc is not None: auroc_vals.append(auroc)
            auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"
            lines.append(
                f"    test_{lang:<4s}  acc={acc:.4f}  F1={f1:.4f}  AUROC={auroc_str}"
                if acc is not None else f"    test_{lang}: (missing)"
            )
        mean_acc   = sum(acc_vals)   / len(acc_vals)   if acc_vals   else 0.0
        mean_f1    = sum(f1_vals)    / len(f1_vals)    if f1_vals    else 0.0
        mean_auroc = sum(auroc_vals) / len(auroc_vals) if auroc_vals else 0.0
        lines.append(
            f"    MEAN          acc={mean_acc:.4f}  F1={mean_f1:.4f}  AUROC={mean_auroc:.4f}"
        )
        lines.append("")

    lines += ["=" * 70, "WINNER DECLARATION", "=" * 70, ""]

    if is_tie:
        top_two = sorted(means_f1.items(), key=lambda x: x[1], reverse=True)[:2]
        lines += [
            f"  INCONCLUSIVE: {top_two[0][0]} and {top_two[1][0]} are within "
            f"{CLOSE_THRESHOLD*100:.1f}pp of each other on mean F1 "
            f"({top_two[0][1]:.4f} vs {top_two[1][1]:.4f}).",
            f"  AUROC tiebreaker: {winner_auroc} leads on mean AUROC.",
            f"  Recommended selection: {winner_auroc} — but difference is marginal;",
            f"  verify on a held-out set before committing to Month 3.",
            "",
        ]
    else:
        model_id = cfg_m.get(primary_winner, {}).get("model_id", primary_winner)
        lines += [
            f"  WINNER: {primary_winner} ({model_id})",
            f"  Mean F1={means_f1[primary_winner]:.4f}  "
            f"Mean AUROC={means_auroc.get(primary_winner, 0):.4f}  "
            f"Mean accuracy={means_acc.get(primary_winner, 0):.4f}",
            "",
            f"  {primary_winner} achieves the highest mean F1 across English, "
            + ", ".join(languages[1:]) + " test sets",
            f"  and is selected as the Month 3 adapter backbone.",
            "",
        ]

    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    logger.info("Backbone comparison report written to %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate()
    print(f"Report: {path}")
