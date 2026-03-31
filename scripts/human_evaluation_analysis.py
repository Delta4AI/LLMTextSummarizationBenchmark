import json
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy.stats import mannwhitneyu

# config
EVAL_FILES = [
    "evaluation_PPE_after80_2026-03-24.json",
    "evaluation_KKI_final.json",
]
DIMENSIONS = ["coherence", "fluency", "relevance", "consistency"]
MODEL_LABELS = {
    "huggingface_google/bigbird-pegasus-large-pubmed": "BigBird-Pegasus",
    "huggingface_csebuetnlp/mT5_multilingual_XLSum": "mT5-XLSum",
    "ollama_mistral-small3.2:24b": "Mistral-Small-3.2 (24B)",
    "mistral_mistral-small-2506": "Mistral-Small-2506",
}

script_dir = Path(__file__).resolve().parent
GH_HASH = "1362b291718b57188a7909f08de26da760a0b9346d52111c97671d97d713af38"
base_dir = script_dir.parent / "Output" / "llm_summarization_benchmark" / GH_HASH
eval_dir = base_dir / "human_evaluations"
output_dir = base_dir

# load assessments per evaluator, keyed by (paper_id, model)
def load_evaluator_data(filepaths):
    evaluators = []
    for fp in filepaths:
        with open(eval_dir / fp, "r") as f:
            data = json.load(f)
        keyed = {}
        for a in data["assessments"]:
            key = (a["paper_id"], a["model"])
            keyed[key] = a["ratings"]
        evaluators.append(keyed)
    return evaluators

def merge_all_assessments(evaluators):
    all_assessments = []
    for ev in evaluators:
        for (paper_id, model), ratings in ev.items():
            all_assessments.append({"paper_id": paper_id, "model": model, "ratings": ratings})
    return all_assessments

def aggregate_ratings(assessments):
    ratings = {m: {d: [] for d in DIMENSIONS} for m in MODEL_LABELS}
    for a in assessments:
        model = a["model"]
        if model not in MODEL_LABELS:
            continue
        for dim in DIMENSIONS:
            if dim in a["ratings"]:
                ratings[model][dim].append(a["ratings"][dim])
    return ratings

# krippendorff_alpha calculation (to analyze inter-rater agreement)
def krippendorff_alpha(evaluators, dimension):
    all_keys = set()
    for ev in evaluators:
        all_keys.update(ev.keys())

    items = []
    for key in sorted(all_keys):
        raters_with_val = [(i, ev[key][dimension])
                           for i, ev in enumerate(evaluators)
                           if key in ev and dimension in ev[key]]
        if len(raters_with_val) >= 2:
            items.append(raters_with_val)

    if not items:
        return None

    do_sum = 0.0
    do_pairs = 0
    for item_ratings in items:
        vals = [v for _, v in item_ratings]
        for a, b in combinations(vals, 2):
            do_sum += (a - b) ** 2
            do_pairs += 1

    if do_pairs == 0:
        return None

    do = do_sum / do_pairs

    all_vals = [v for item_ratings in items for _, v in item_ratings]
    de_sum = sum((a - b) ** 2 for a, b in combinations(all_vals, 2))
    de_pairs = len(all_vals) * (len(all_vals) - 1) / 2
    de = de_sum / de_pairs if de_pairs else 0

    if de == 0:
        return 1.0

    return 1.0 - (do / de)

def krippendorff_alpha_overall(evaluators):
    evaluators_flat = []
    for ev in evaluators:
        flat = {}
        for key, rats in ev.items():
            for dim in DIMENSIONS:
                if dim in rats:
                    flat[(*key, dim)] = rats[dim]
        evaluators_flat.append(flat)

    all_keys = set()
    for ef in evaluators_flat:
        all_keys.update(ef.keys())

    items_flat = []
    for key in sorted(all_keys):
        raters_with_val = [(i, ef[key]) for i, ef in enumerate(evaluators_flat) if key in ef]
        if len(raters_with_val) >= 2:
            items_flat.append(raters_with_val)

    if not items_flat:
        return None

    do_sum = 0.0
    do_pairs = 0
    for item_ratings in items_flat:
        vals = [v for _, v in item_ratings]
        for a, b in combinations(vals, 2):
            do_sum += (a - b) ** 2
            do_pairs += 1
    do = do_sum / do_pairs if do_pairs else 0

    all_vals = [v for item_ratings in items_flat for _, v in item_ratings]
    de_sum = sum((a - b) ** 2 for a, b in combinations(all_vals, 2))
    de_pairs = len(all_vals) * (len(all_vals) - 1) / 2
    de = de_sum / de_pairs if de_pairs else 0

    return 1.0 - (do / de) if de != 0 else 1.0

# pairwise Mann-Whitney U tests with Bonferroni correction
def pairwise_mann_whitney(ratings):
    models = list(MODEL_LABELS.keys())
    n_comparisons = len(list(combinations(models, 2)))

    print("\n--- Pairwise Mann-Whitney U Tests (Bonferroni-corrected) ---")
    for dim in DIMENSIONS:
        print(f"\n  {dim.capitalize()}:")
        for m1, m2 in combinations(models, 2):
            l1 = MODEL_LABELS[m1]
            l2 = MODEL_LABELS[m2]
            vals1 = ratings[m1][dim]
            vals2 = ratings[m2][dim]

            stat, p_raw = mannwhitneyu(vals1, vals2, alternative="two-sided")
            p_corr = min(p_raw * n_comparisons, 1.0)  # Bonferroni

            sig = ""
            if p_corr < 0.001:
                sig = "***"
            elif p_corr < 0.01:
                sig = "**"
            elif p_corr < 0.05:
                sig = "*"
            else:
                sig = "n.s."

            print(f"    {l1:>25s} vs {l2:<25s}  U={stat:8.1f}  p={p_corr:.4f} {sig}")

# TEMP -> LaTeX table generation for publication
def generate_latex_table(ratings, evaluators, output_path):
    models = list(MODEL_LABELS.keys())

    overall_alpha = krippendorff_alpha_overall(evaluators)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Human evaluation results (mean $\pm$ SEM, $n$=40 per model). "
                 r"Krippendorff's $\alpha$ measures inter-rater agreement.}")
    lines.append(r"\label{tab:human_eval}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Coherence & Fluency & Relevance & Consistency \\")
    lines.append(r"\midrule")

    for m in models:
        label = MODEL_LABELS[m]
        cells = []
        for dim in DIMENSIONS:
            vals = ratings[m][dim]
            mean = np.mean(vals)
            sem = np.std(vals, ddof=1) / np.sqrt(len(vals))
            cells.append(f"${mean:.2f} \\pm {sem:.2f}$")
        lines.append(f"{label} & {' & '.join(cells)} \\\\")

    lines.append(r"\midrule")

    alpha_cells = []
    for dim in DIMENSIONS:
        a = krippendorff_alpha(evaluators, dim)
        alpha_cells.append(f"${a:.3f}$" if a is not None else "---")
    lines.append(f"Krippendorff's $\\alpha$ & {' & '.join(alpha_cells)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    if overall_alpha is not None:
        lines.append(r"\vspace{2pt}")
        lines.append(f"\\\\\\small Overall $\\alpha = {overall_alpha:.3f}$")

    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex_str)

    print(f"\nSaved LaTeX table: {output_path}")
    print("\n--- LaTeX Table Preview ---")
    print(latex_str)



def main():
    evaluators = load_evaluator_data(EVAL_FILES)
    print(f"Loaded data from {len(evaluators)} evaluator(s).")
    for i, ev in enumerate(evaluators):
        print(f"  Evaluator {i+1}: {len(ev)} assessments")

    all_assessments = merge_all_assessments(evaluators)
    ratings = aggregate_ratings(all_assessments)

    for model, label in MODEL_LABELS.items():
        print(f"\n{label}:")
        for dim in DIMENSIONS:
            vals = ratings[model][dim]
            sem = np.std(vals, ddof=1) / np.sqrt(len(vals))
            print(f"  {dim:>12s}: mean={np.mean(vals):.2f}  sem={sem:.2f}  n={len(vals)}")

    print("\n--- Inter-Rater Agreement (Krippendorff's alpha) ---")
    for dim in DIMENSIONS:
        alpha = krippendorff_alpha(evaluators, dim)
        if alpha is not None:
            print(f"  {dim:>12s}: α = {alpha:.3f}")
        else:
            print(f"  {dim:>12s}: not enough overlapping ratings")

    overall_alpha = krippendorff_alpha_overall(evaluators)
    if overall_alpha is not None:
        print(f"\n  {'overall':>12s}: α = {overall_alpha:.3f}")

    pairwise_mann_whitney(ratings)

    # TEMP -> generate LaTeX table
    latex_path = output_dir / "human_evaluation_table.tex"
    generate_latex_table(ratings, evaluators, latex_path)

    print("\nDone.")

if __name__ == "__main__":
    main()