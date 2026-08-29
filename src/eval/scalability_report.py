"""
Month 3: scalability experiment report.

Reads scalability_results.json (zero_shot / few_shot_adapter / full_retrain
conditions) and produces a human-readable report with a condition × language
× metric table, per-language data-efficiency curves, the crossover point
(how many samples an adapter needs before it beats zero-shot), and a
plain-language winner declaration per language.
"""
import json
import logging
from pathlib import Path
from typing import Optional

from src.config import resolve_path
from src.eval.metrics import fmt_mean_std, mean_of

logger = logging.getLogger(__name__)

CLOSE_THRESHOLD = 0.005  # 0.5 percentage points — below this, call it a tie


def _sort_key(n):
    return (1, 0) if n == "full" else (0, int(n))


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"


def _seed_count(*conditions: dict) -> "int | str":
    """Best-effort count of seeds behind the aggregated numbers."""
    def walk(node):
        if isinstance(node, dict):
            if isinstance(node.get("seeds"), list):
                return len(node["seeds"])
            if isinstance(node.get("values"), list) and "mean" in node:
                return len(node["values"])
            for v in node.values():
                found = walk(v)
                if found is not None:
                    return found
        return None

    for cond in conditions:
        found = walk(cond)
        if found is not None:
            return found
    return "1 (legacy single-seed report)"


def compute_data_efficiency_curve(few_shot_lang: dict, metric: str = "auroc") -> list[tuple]:
    """
    Returns [(n_samples, metric_value), ...] ordered smallest → "full".

    Each ``overall`` metric may be a bare number (legacy single-seed) or an
    aggregated ``{"mean": .., "std": ..}`` dict — the curve carries the mean.
    """
    sizes = sorted(few_shot_lang.keys(), key=_sort_key)
    return [(n, mean_of(few_shot_lang[n]["overall"].get(metric))) for n in sizes]


def compute_crossover(curve: list[tuple], zero_shot_value: Optional[float]) -> Optional[object]:
    """
    Returns the smallest n_samples at which the adapter's metric first
    exceeds zero_shot_value, or None if it never does (or zero_shot_value
    is unavailable).
    """
    if zero_shot_value is None:
        return None
    for n, value in curve:
        if value is not None and value > zero_shot_value:
            return n
    return None


def declare_winner(zero_shot_auroc: Optional[float], full_retrain_auroc: Optional[float],
                   best_adapter_auroc: Optional[float]) -> tuple[str, dict]:
    """Picks the condition with the highest AUROC; declares a tie within CLOSE_THRESHOLD."""
    candidates = {
        "zero_shot": zero_shot_auroc,
        "few_shot_adapter": best_adapter_auroc,
        "full_retrain": full_retrain_auroc,
    }
    candidates = {k: v for k, v in candidates.items() if v is not None}
    if not candidates:
        return "inconclusive (no AUROC data)", candidates

    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    if len(ranked) >= 2 and (ranked[0][1] - ranked[1][1]) < CLOSE_THRESHOLD:
        return f"tie between {ranked[0][0]} and {ranked[1][0]}", candidates
    return ranked[0][0], candidates


def generate(results_json: Optional[Path] = None, out: Optional[Path] = None) -> Path:
    if results_json is None:
        results_json = resolve_path("reports/scalability_results.json")
    if out is None:
        out = resolve_path("reports/scalability_report.txt")

    with open(results_json) as f:
        results = json.load(f)

    zero_shot = results.get("zero_shot", {})
    few_shot = results.get("few_shot_adapter", {})
    full_retrain = results.get("full_retrain", {})
    languages = sorted(set(zero_shot) | set(few_shot) | set(full_retrain))
    n_seeds = _seed_count(zero_shot, few_shot, full_retrain)

    lines = [
        "=" * 70,
        "Month 3: Scalability Experiment Report",
        f"Generated from: {results_json}",
        f"Seeds per condition: {n_seeds}  (values shown as mean ± std across seeds)",
        "=" * 70,
        "",
        "CONDITION x LANGUAGE x METRIC TABLE (accuracy / F1 / AUROC)",
        "",
    ]

    for lang in languages:
        lines.append(f"  Language: {lang}")
        zs = zero_shot.get(lang, {}).get("overall", {})
        lines.append(
            f"    zero_shot         acc={fmt_mean_std(zs.get('accuracy'))}  "
            f"F1={fmt_mean_std(zs.get('f1'))}  AUROC={fmt_mean_std(zs.get('auroc'))}"
        )
        for n in sorted(few_shot.get(lang, {}).keys(), key=_sort_key):
            m = few_shot[lang][n]["overall"]
            lines.append(
                f"    adapter n={str(n):<6} acc={fmt_mean_std(m.get('accuracy'))}  "
                f"F1={fmt_mean_std(m.get('f1'))}  AUROC={fmt_mean_std(m.get('auroc'))}"
            )
        fr = full_retrain.get(lang, {}).get("overall", {})
        lines.append(
            f"    full_retrain      acc={fmt_mean_std(fr.get('accuracy'))}  "
            f"F1={fmt_mean_std(fr.get('f1'))}  AUROC={fmt_mean_std(fr.get('auroc'))}"
        )
        lines.append("")

    lines += ["=" * 70, "DATA-EFFICIENCY CURVES (n_samples -> AUROC) + CROSSOVER POINT",
             "=" * 70, ""]

    winners: dict[str, tuple[str, dict]] = {}
    for lang in languages:
        curve = compute_data_efficiency_curve(few_shot.get(lang, {}))
        zs_auroc = mean_of(zero_shot.get(lang, {}).get("overall", {}).get("auroc"))
        crossover = compute_crossover(curve, zs_auroc)

        lines.append(f"  Language: {lang}")
        for n, v in curve:
            lines.append(f"    n={str(n):<6} AUROC={_fmt(v)}")
        if crossover is not None:
            lines.append(
                f"    Crossover: adapter exceeds zero-shot AUROC ({_fmt(zs_auroc)}) at n={crossover}"
            )
        else:
            lines.append(
                f"    Crossover: adapter never exceeds zero-shot AUROC ({_fmt(zs_auroc)}) "
                f"within the data sizes run"
            )
        lines.append("")

        best_adapter = max((v for _, v in curve if v is not None), default=None)
        fr_auroc = mean_of(full_retrain.get(lang, {}).get("overall", {}).get("auroc"))
        winners[lang] = declare_winner(zs_auroc, fr_auroc, best_adapter)

    lines += ["=" * 70, "WINNER DECLARATION (by AUROC)", "=" * 70, ""]
    for lang, (winner, scores) in winners.items():
        score_str = ", ".join(f"{k}={_fmt(v)}" for k, v in scores.items())
        lines.append(f"  {lang}: {winner}  ({score_str})")
    lines.append("")

    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    logger.info("Scalability report written to %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate()
    print(f"Report: {path}")
