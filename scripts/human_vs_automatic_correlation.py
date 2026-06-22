#!/usr/bin/env python3
"""
Correlate expert (human) judgements with automatic metric scores at the
summary level: for every human-rated (paper_id, model) item, pair the mean
expert rating per dimension against each automatic metric, then compute
Spearman's rho and Kendall's tau across all items.

System-level correlation is intentionally omitted: only 4 models were rated by
humans, so n=4 -> not meaningful. Item-level (n = rated papers x models) is the
field standard (cf. UniEval/QAFactEval human-correlation reporting).

Inputs (in the hashed run dir, copy from prod):
- human_evaluations/evaluation_*.json   (expert ratings)
- detailed_scores_per_paper.json        (per-(model, paper_id) metric scores)

Outputs (same run dir):
- human_vs_automatic_spearman.csv   (metrics x human dimensions, rho)
- human_vs_automatic_correlation.tex
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, kendalltau

# --- config (mirrors human_evaluation_analysis.py) ---
EVAL_FILES = [f"evaluation_{i}.json" for i in range(1, 9)]
DIMENSIONS = ["coherence", "fluency", "relevance", "consistency"]

# storage-key -> label (mirrors generate_metric_correlations.py)
METRICS = [
    ("rouge1", "ROUGE-1"),
    ("rouge2", "ROUGE-2"),
    ("rougeL", "ROUGE-L"),
    ("meteor", "METEOR"),
    ("bleu", "BLEU"),
    ("bert_roberta-large_f1", "RoBERTa"),
    ("bert_microsoft/deberta-xlarge-mnli_f1", "DeBERTa"),
    ("sentence_transformer", "all-mpnet-base-v2"),
    ("alignscore", "AlignScore"),
    ("summac", "SummaC"),
    ("factcc", "FactCC"),
    ("minicheck_ft5", "MiniCheck-FT5"),
    ("minicheck_7b", "MiniCheck-7B"),
]

SCRIPT_DIR = Path(__file__).resolve().parent
GH_HASH = "1362b291718b57188a7909f08de26da760a0b9346d52111c97671d97d713af38"
BASE_DIR = SCRIPT_DIR.parent / "Output" / "llm_summarization_benchmark" / GH_HASH
EVAL_DIR = SCRIPT_DIR.parent / "Output" / "scripts" / "human_evaluation_data"
PER_PAPER_JSON = BASE_DIR / "detailed_scores_per_paper.json"
JUDGE_JSON = SCRIPT_DIR.parent / "Output" / "scripts" / "llm_judge_scores.json"

# LLM-judge dimensions appear as extra "metrics"; the diagonal (judge dim vs
# same human dim) is the headline agreement, off-diagonal shows discriminance.
JUDGE_METRICS = [(f"judge_{d}", f"Judge-{d.capitalize()}") for d in DIMENSIONS]


def load_judge_scores(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """(paper_id, model) -> {judge_<dim>: score}. Empty if no judge run yet."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out: dict[tuple[str, str], dict[str, float]] = {}
    for combined, ratings in raw.items():
        paper_id, model = combined.split("|||")
        out[(paper_id, model)] = {
            f"judge_{d}": float(ratings[d]) for d in DIMENSIONS if d in ratings
        }
    return out


def load_human_ratings(eval_dir: Path) -> dict[tuple[str, str], dict[str, list[float]]]:
    """(paper_id, model) -> {dimension: [rating from each rater who scored it]}."""
    ratings: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {d: [] for d in DIMENSIONS}
    )
    missing = [fp for fp in EVAL_FILES if not (eval_dir / fp).exists()]
    if missing:
        sys.exit(f"Missing eval files in {eval_dir}: {missing}")
    for fp in EVAL_FILES:
        path = eval_dir / fp
        with open(path) as f:
            data = json.load(f)
        for a in data["assessments"]:
            key = (a["paper_id"], a["model"])
            for dim in DIMENSIONS:
                if dim in a["ratings"]:
                    ratings[key][dim].append(float(a["ratings"][dim]))
    return ratings


def load_automatic_scores(per_paper_json: Path) -> dict[tuple[str, str], dict[str, float]]:
    """(paper_id, model) -> {metric_key: score}. Uses max() per paper to match
    the per-model aggregation in save_scores_per_model()."""
    with open(per_paper_json) as f:
        data = json.load(f)
    out: dict[tuple[str, str], dict[str, float]] = {}
    for model, papers in data.items():
        for paper in papers:
            key = (paper["id"], model)
            scores = paper.get("scores", {})
            out[key] = {
                m: float(max(scores[m])) for m, _ in METRICS if scores.get(m)
            }
    return out


def correlate(human, automatic, metrics=METRICS):
    """For each (metric, dimension) build paired item-level vectors and correlate.
    Returns {metric_key: {dimension: (rho, tau, p_rho, n)}}."""
    results: dict[str, dict[str, tuple]] = {}
    for metric_key, _ in metrics:
        results[metric_key] = {}
        for dim in DIMENSIONS:
            h_vals, a_vals = [], []
            for key, dims in human.items():
                if key not in automatic or metric_key not in automatic[key]:
                    continue
                if not dims[dim]:
                    continue
                h_vals.append(float(np.mean(dims[dim])))  # mean across raters
                a_vals.append(automatic[key][metric_key])
            # need >=3 points and variance in both vectors (else rho undefined)
            if len(h_vals) < 3 or len(set(h_vals)) < 2 or len(set(a_vals)) < 2:
                results[metric_key][dim] = (np.nan, np.nan, np.nan, len(h_vals))
                continue
            rho, p_rho = spearmanr(a_vals, h_vals)
            tau, _ = kendalltau(a_vals, h_vals)
            results[metric_key][dim] = (rho, tau, p_rho, len(h_vals))
    return results


def write_csv(results, path: Path, metrics=METRICS):
    header = ["Metric"] + [d.capitalize() for d in DIMENSIONS]
    lines = [",".join(header)]
    for metric_key, label in metrics:
        cells = [label]
        for dim in DIMENSIONS:
            rho, _, _, _ = results[metric_key][dim]
            cells.append("" if np.isnan(rho) else f"{rho:.3f}")
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {path}")


def write_latex(results, path: Path, metrics=METRICS):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Summary-level correlation (Spearman's $\rho$) between automatic "
        r"metrics and expert ratings per dimension. $^{*}p<0.05$.}",
        r"\label{tab:human_vs_automatic}",
        r"\begin{tabular}{l" + "c" * len(DIMENSIONS) + "}",
        r"\toprule",
        "Metric & " + " & ".join(d.capitalize() for d in DIMENSIONS) + r" \\",
        r"\midrule",
    ]
    for metric_key, label in metrics:
        cells = []
        for dim in DIMENSIONS:
            rho, _, p, _ = results[metric_key][dim]
            if np.isnan(rho):
                cells.append("---")
            else:
                star = "^{*}" if p < 0.05 else ""
                cells.append(f"${rho:.3f}{star}$")
        lines.append(f"{label} & {' & '.join(cells)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {path}")


def within_model_breakdown(human, automatic, metrics):
    """Spearman per model (controls for between-model spread). Prints a compact
    table: rows = metrics, one rho per dimension, restricted to each model's
    rated items."""
    models = sorted({m for _, m in human})
    print("\n--- Within-model Spearman rho (per model, controls for "
          "between-model spread) ---")
    for model in models:
        sub_human = {k: v for k, v in human.items() if k[1] == model}
        sub_auto = {k: v for k, v in automatic.items() if k[1] == model}
        res = correlate(sub_human, sub_auto, metrics)
        n = len(sub_human)
        print(f"\n  {model}  (n={n}):")
        for metric_key, label in metrics:
            cells = []
            for dim in DIMENSIONS:
                rho, _, p, _ = res[metric_key][dim]
                cells.append("  n/a " if np.isnan(rho)
                             else f"{rho:+.2f}{'*' if p < 0.05 else ' '}")
            print(f"    {label:>18s}: " +
                  "  ".join(f"{d[:4]}={c}" for d, c in zip(DIMENSIONS, cells)))


def demo():
    """ponytail: self-check — perfect monotonic pairing must yield rho==1."""
    human = {("p1", "m"): {"consistency": [4.0], "coherence": [1.0],
                            "fluency": [], "relevance": []},
             ("p2", "m"): {"consistency": [2.0], "coherence": [2.0],
                           "fluency": [], "relevance": []},
             ("p3", "m"): {"consistency": [5.0], "coherence": [3.0],
                           "fluency": [], "relevance": []}}
    automatic = {("p1", "m"): {"alignscore": 0.8},
                 ("p2", "m"): {"alignscore": 0.4},
                 ("p3", "m"): {"alignscore": 0.95}}
    res = correlate(human, automatic)
    rho, _, _, n = res["alignscore"]["consistency"]
    assert n == 3 and abs(rho - 1.0) < 1e-9, res["alignscore"]["consistency"]
    # dimension with no human ratings -> NaN, n=0
    assert res["alignscore"]["fluency"][3] == 0
    print("selfcheck OK")


def main():
    if "--selfcheck" in sys.argv:
        demo()
        return
    if not PER_PAPER_JSON.exists():
        sys.exit(f"Missing {PER_PAPER_JSON} — copy detailed_scores_per_paper.json "
                 f"and human_evaluations/ from the prod run dir.")

    human = load_human_ratings(EVAL_DIR)
    automatic = load_automatic_scores(PER_PAPER_JSON)

    # merge LLM-judge scores as extra "metrics" if a judge run exists
    judge = load_judge_scores(JUDGE_JSON)
    metrics = list(METRICS)
    if judge:
        for key, dims in judge.items():
            automatic.setdefault(key, {}).update(dims)
        metrics += JUDGE_METRICS
    print(f"Loaded {len(human)} human-rated items, "
          f"{len(automatic)} automatic-scored items, "
          f"judge scores: {len(judge)}.")

    results = correlate(human, automatic, metrics)

    print("\n--- Spearman rho (Kendall tau), item-level ---")
    for metric_key, label in metrics:
        print(f"\n  {label}:")
        for dim in DIMENSIONS:
            rho, tau, p, n = results[metric_key][dim]
            if np.isnan(rho):
                print(f"    {dim:>12s}: n/a (n={n})")
            else:
                sig = "*" if p < 0.05 else ""
                print(f"    {dim:>12s}: rho={rho:+.3f} tau={tau:+.3f} "
                      f"p={p:.4f}{sig} n={n}")

    within_model_breakdown(human, automatic, metrics)

    write_csv(results, BASE_DIR / "human_vs_automatic_spearman.csv", metrics)
    write_latex(results, BASE_DIR / "human_vs_automatic_correlation.tex", metrics)
    print("\nDone.")


if __name__ == "__main__":
    main()
