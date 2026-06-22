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


def load_evaluators(eval_dir: Path) -> list[dict[tuple[str, str], dict[str, float]]]:
    """Per-evaluator ratings (kept separate, for inter-expert agreement)."""
    evaluators = []
    for fp in EVAL_FILES:
        with open(eval_dir / fp) as f:
            data = json.load(f)
        evaluators.append({(a["paper_id"], a["model"]): a["ratings"]
                           for a in data["assessments"]})
    return evaluators


def inter_expert_rho(evaluators) -> dict[str, float]:
    """Mean pairwise Spearman between experts, per dimension — the agreement
    ceiling the judge is compared against."""
    from itertools import combinations
    out = {}
    for dim in DIMENSIONS:
        rhos = []
        for a, b in combinations(range(len(evaluators)), 2):
            shared = [k for k in evaluators[a]
                      if k in evaluators[b]
                      and dim in evaluators[a][k] and dim in evaluators[b][k]]
            if len(shared) < 3:
                continue
            va = [evaluators[a][k][dim] for k in shared]
            vb = [evaluators[b][k][dim] for k in shared]
            if len(set(va)) < 2 or len(set(vb)) < 2:
                continue
            rhos.append(spearmanr(va, vb)[0])
        out[dim] = float(np.mean(rhos)) if rhos else np.nan
    return out


def judge_summary(results) -> dict | None:
    """Judge-vs-expert agreement: per-dimension diagonal (judge dim vs same
    human dim) and the off-diagonal mean (discriminant check). None if no judge."""
    if f"judge_{DIMENSIONS[0]}" not in results:
        return None
    diagonal = {d: results[f"judge_{d}"][d][0] for d in DIMENSIONS}
    off = []
    for jd in DIMENSIONS:
        for hd in DIMENSIONS:
            if jd != hd and not np.isnan(results[f"judge_{jd}"][hd][0]):
                off.append(results[f"judge_{jd}"][hd][0])
    valid = [v for v in diagonal.values() if not np.isnan(v)]
    return {
        "diagonal": diagonal,
        "diagonal_mean": float(np.mean(valid)) if valid else np.nan,
        "off_diagonal_mean": float(np.mean(off)) if off else np.nan,
    }


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


def _cell_color(rho: float, p: float) -> str:
    """Green gradient for significant positive rho; grey for n.s./missing."""
    if np.isnan(rho) or p >= 0.05:
        return "#eceff1"
    v = max(0.0, min(1.0, rho))
    # interpolate pale -> deep teal
    r = int(224 - v * (224 - 13))
    g = int(242 - v * (242 - 115))
    b = int(241 - v * (241 - 110))
    return f"rgb({r},{g},{b})"


def write_html(results, metrics, inter_expert, jsummary, n_items, path: Path):
    """Self-contained explainer + heatmap report for colleagues."""
    def fmt(cell):
        rho, _, p, _ = cell
        if np.isnan(rho):
            return "&mdash;", "#eceff1", ""
        weight = "700" if p < 0.05 else "400"
        fg = "#fff" if (p < 0.05 and rho > 0.55) else "#222"
        return (f'<span style="color:{fg};font-weight:{weight}">{rho:+.2f}</span>',
                _cell_color(rho, p), weight)

    # heatmap rows
    body_rows = []
    for key, label in metrics:
        is_judge = key.startswith("judge_")
        cells = "".join(
            f'<td style="background:{c}">{txt}</td>'
            for txt, c, _ in (fmt(results[key][d]) for d in DIMENSIONS)
        )
        cls = ' class="judge"' if is_judge else ""
        body_rows.append(f"<tr{cls}><th>{label}</th>{cells}</tr>")

    ceil = "".join(
        f"<td>{'&mdash;' if np.isnan(inter_expert[d]) else f'{inter_expert[d]:.2f}'}</td>"
        for d in DIMENSIONS
    )

    # judge summary cards
    judge_block = "<p><em>No LLM-judge scores yet &mdash; run "
    judge_block += "<code>llm_judge_eval.py</code>, then regenerate.</em></p>"
    if jsummary:
        diag = jsummary["diagonal_mean"]
        off = jsummary["off_diagonal_mean"]
        ie = np.nanmean(list(inter_expert.values()))
        judge_block = f"""
        <div class="cards">
          <div class="card"><div class="big">{diag:+.2f}</div>
            <div>Judge&ndash;expert agreement<br><small>mean of matched
            dimensions (the diagonal)</small></div></div>
          <div class="card"><div class="big">{ie:+.2f}</div>
            <div>Inter-expert ceiling<br><small>mean pairwise human&ndash;human
            &rho; &mdash; the judge cannot be expected to beat this</small></div></div>
          <div class="card"><div class="big">{off:+.2f}</div>
            <div>Off-diagonal mean<br><small>judge dim vs <em>other</em> human
            dims; lower than the diagonal = dimension-specific</small></div></div>
        </div>
        <p><b>Read it like this:</b> if the diagonal (judge rates the dimension it
        was asked about) is close to the inter-expert ceiling <em>and</em> clearly
        higher than the off-diagonal, the judge is tracking each construct the way
        experts do &mdash; something the bulk metrics below do not achieve.</p>"""

    header_cells = "".join(f"<th>{d.capitalize()}</th>" for d in DIMENSIONS)
    html = f"""<title>Human vs. automatic agreement &mdash; summarization benchmark</title>
<style>
  :root {{ --ink:#1a2b32; --muted:#5b7079; --line:#d8e0e3; }}
  body {{ font:16px/1.6 -apple-system,Segoe UI,Roboto,sans-serif; color:var(--ink);
         max-width:920px; margin:0 auto; padding:2.5rem 1.25rem 5rem; }}
  h1 {{ font-size:1.7rem; line-height:1.25; margin:0 0 .25rem; }}
  h2 {{ font-size:1.2rem; margin:2.4rem 0 .6rem; border-bottom:2px solid var(--line);
        padding-bottom:.3rem; }}
  .sub {{ color:var(--muted); margin:0 0 1.5rem; }}
  .flow {{ display:flex; gap:.5rem; flex-wrap:wrap; align-items:stretch; margin:1rem 0; }}
  .flow .step {{ flex:1 1 160px; background:#f3f7f8; border:1px solid var(--line);
        border-radius:10px; padding:.8rem .9rem; }}
  .flow .step b {{ display:block; font-size:.92rem; }}
  .flow .step small {{ color:var(--muted); }}
  table {{ border-collapse:collapse; width:100%; margin:1rem 0; font-variant-numeric:tabular-nums; }}
  th,td {{ border:1px solid var(--line); padding:.42rem .55rem; text-align:center; }}
  tbody th {{ text-align:left; font-weight:600; background:#fafcfc; }}
  thead th {{ background:#1a2b32; color:#fff; }}
  tr.judge th, tr.judge td {{ border-top:2px solid #1a2b32; }}
  tr.judge th::after {{ content:" \\2190 judge"; color:var(--muted); font-weight:400; font-size:.8rem; }}
  .ceil th {{ background:#fff4e0; }} .ceil td {{ background:#fff4e0; font-weight:600; }}
  .cards {{ display:flex; gap:.75rem; flex-wrap:wrap; margin:1rem 0; }}
  .card {{ flex:1 1 200px; border:1px solid var(--line); border-radius:10px;
        padding:1rem; background:#f3f7f8; }}
  .card .big {{ font-size:2rem; font-weight:700; color:#0d7370; }}
  .note {{ background:#fff8f0; border-left:4px solid #e0a458; padding:.8rem 1rem;
        border-radius:0 8px 8px 0; margin:1rem 0; }}
  code {{ background:#eef3f4; padding:.1rem .35rem; border-radius:4px; font-size:.9em; }}
  small {{ font-size:.82rem; }}
  .scroll {{ overflow-x:auto; }}
</style>

<h1>Do automatic metrics agree with human experts?</h1>
<p class="sub">Summary-level validation of {len(METRICS)} automatic metrics &mdash;
and an LLM-as-a-judge &mdash; against expert ratings. n&nbsp;=&nbsp;{n_items} rated
(paper, model) items.</p>

<h2>1. What this answers</h2>
<p>This report tests whether the reference-based metrics used in the benchmark
reflect human judgement, and whether an LLM-as-a-judge can stand in for human
evaluation. It measures, for every metric, how strongly it correlates with
expert ratings on the same summaries &mdash; and validates the LLM judge the same
way before using it to extend evaluation to systems that were not manually
rated.</p>

<h2>2. The data &amp; the pipeline</h2>
<div class="flow">
  <div class="step"><b>8 experts</b><small>rated 4 systems &times; 20 papers on a
    1&ndash;5 scale</small></div>
  <div class="step"><b>4 dimensions</b><small>coherence, fluency, relevance,
    consistency</small></div>
  <div class="step"><b>{n_items} items</b><small>(paper, model) pairs with an
    expert rating</small></div>
  <div class="step"><b>Spearman &amp; Kendall</b><small>metric score vs mean expert
    rating, per dimension</small></div>
</div>
<p>For each rated item we pair the <b>mean expert rating</b> (averaged over the
experts who saw that item) with each automatic metric score for the same
summary, then compute rank correlation (Spearman's &rho; and Kendall's &tau;)
across all items. Where a metric produces several values per summary we take the
per-summary maximum, matching how the benchmark aggregates elsewhere.</p>
<p>We use <b>summary-level</b> correlation (one data point per rated summary),
not system-level (one point per model): with only 4 rated systems, a
system-level correlation has n=4 and is uninformative. We additionally compute a
<b>within-system</b> breakdown &mdash; the same correlation restricted to each
model's 20 summaries &mdash; to separate genuine metric&ndash;quality agreement
from agreement that merely reflects the large quality gap <em>between</em>
systems. The <b>inter-expert ceiling</b> (mean pairwise Spearman between experts
who rated shared items) bounds how high any automatic measure could plausibly
correlate, since experts do not perfectly agree either.</p>

<h2>3. Correlation with expert judgement (Spearman &rho;)</h2>
<p>Cells are coloured by strength; <b>bold</b> = significant (p&nbsp;&lt;&nbsp;0.05),
grey = not significant. The orange row is the human&ndash;human agreement ceiling.</p>
<div class="scroll"><table>
  <thead><tr><th>Metric</th>{header_cells}</tr></thead>
  <tbody>
    {''.join(body_rows)}
    <tr class="ceil"><th>Inter-expert ceiling</th>{ceil}</tr>
  </tbody>
</table></div>

<h2>4. LLM-as-a-judge vs. experts</h2>
<p>The judge is given the <b>same rubric and the same evidence</b> the experts
saw (title, abstract, reference highlights, and the summary) and returns a 1&ndash;5
score per dimension via a constrained tool call at temperature 0, so its output
is schema-valid and reproducible. It is first <b>validated</b> against the experts
on the {n_items} jointly-covered items (the diagonal below), then applied to all
63 systems on the same 20 papers &mdash; a human-aligned, multi-dimensional
evaluation at a scale manual annotation cannot reach.</p>
<p>Two quantities summarise the validation: the <b>diagonal</b> (judge's score for
a dimension vs. the experts' score for that <em>same</em> dimension) measures
agreement; the <b>off-diagonal</b> (judge's score for a dimension vs. experts'
<em>other</em> dimensions) tests whether the judge separates the four constructs
rather than emitting one global impression.</p>
{judge_block}

<h2>5. How to interpret the results</h2>
<div class="note">
<p><b>Strength.</b> A high, significant &rho; means the metric ranks summaries in
the same order experts do. Most metrics here reach &rho; up to ~0.85, so they
reliably separate stronger from weaker systems.</p>
<p><b>Dimension-specificity.</b> Compare a metric's four cells: if they are all
similar, the metric reflects overall quality rather than the specific construct
(coherence vs. consistency, etc.). Across these metrics the four columns move
together &mdash; agreement is not dimension-specific.</p>
<p><b>Between- vs. within-system.</b> The high pooled correlations are largely
driven by the quality gap between systems; within a single system (the
within-system breakdown, printed to console) the correlations fall toward zero.
A metric is therefore a strong proxy for comparing systems but a weak one for
ranking individual summaries of similar quality.</p>
<p><b>Non-significant cells</b> (grey) indicate no measurable agreement &mdash;
e.g. FactCC shows no significant correlation with experts on any dimension.</p>
<p><b>Judge validity</b> is read from section 4: the diagonal relative to the
inter-expert ceiling, and the diagonal relative to the off-diagonal.</p>
</div>
<p class="sub"><small>Generated by the analysis script
<code>human_vs_automatic_correlation.py</code> from the expert ratings and
automatic metric scores. Re-run after the judge completes to populate
section 4.</small></p>
"""
    path.write_text(html)
    print(f"Wrote: {path}")


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

    # n of the analysis (rated items that also have automatic scores)
    n_items = sum(1 for k in human if k in automatic)
    inter_expert = inter_expert_rho(load_evaluators(EVAL_DIR))
    jsummary = judge_summary(results)

    print("\n--- Inter-expert agreement ceiling (mean pairwise Spearman) ---")
    for d in DIMENSIONS:
        print(f"    {d:>12s}: rho={inter_expert[d]:+.3f}")

    if jsummary:
        print("\n--- Judge vs. expert agreement ---")
        for d in DIMENSIONS:
            print(f"    {d:>12s}: judge rho={jsummary['diagonal'][d]:+.3f}  "
                  f"(expert ceiling {inter_expert[d]:+.3f})")
        print(f"    {'diagonal mean':>12s}: {jsummary['diagonal_mean']:+.3f}")
        print(f"    {'off-diag mean':>12s}: {jsummary['off_diagonal_mean']:+.3f} "
              f"(lower = more dimension-specific)")
    else:
        print("\n(no LLM-judge scores yet — run llm_judge_eval.py to populate "
              "judge rows)")

    write_csv(results, BASE_DIR / "human_vs_automatic_spearman.csv", metrics)
    write_latex(results, BASE_DIR / "human_vs_automatic_correlation.tex", metrics)
    write_html(results, metrics, inter_expert, jsummary, n_items,
               BASE_DIR / "human_vs_automatic_report.html")
    print("\nDone.")


if __name__ == "__main__":
    main()
