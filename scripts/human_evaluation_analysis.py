import json
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy.stats import mannwhitneyu, spearmanr

# config
EVAL_FILES = [
    "evaluation_1.json",
    "evaluation_2.json",
    "evaluation_3.json",
    "evaluation_4.json",
    "evaluation_5.json",
    "evaluation_6.json",
    "evaluation_7.json",
    "evaluation_8.json",
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

# pairwise Spearman correlation between evaluators (to analyze ranking agreement)
def pairwise_spearman(evaluators):
    n_eval = len(evaluators)

    print("\n--- Pairwise Spearman Correlation Between Evaluators ---")

    for dim in DIMENSIONS:
        print(f"\n  {dim.capitalize()}:")
        rho_values = []

        for i, j in combinations(range(n_eval), 2):
            # find shared (paper_id, model) keys rated by both evaluators
            shared_keys = sorted(
                k for k in evaluators[i]
                if k in evaluators[j]
                and dim in evaluators[i][k]
                and dim in evaluators[j][k]
            )

            if len(shared_keys) < 3:
                print(f"    Evaluator {i+1} vs Evaluator {j+1}: too few shared items ({len(shared_keys)})")
                continue

            vals_i = [evaluators[i][k][dim] for k in shared_keys]
            vals_j = [evaluators[j][k][dim] for k in shared_keys]

            rho, p = spearmanr(vals_i, vals_j)
            rho_values.append(rho)
            print(f"    Evaluator {i+1} vs Evaluator {j+1}: ρ = {rho:.3f}  (p={p:.4f}, n={len(shared_keys)})")

        if rho_values:
            mean_rho = np.mean(rho_values)
            print(f"    {'Mean':>30s}: ρ = {mean_rho:.3f}")

    # overall across all dimensions
    print(f"\n  Overall (all dimensions combined):")
    rho_values_all = []

    for i, j in combinations(range(n_eval), 2):
        shared_vals_i = []
        shared_vals_j = []

        for k in sorted(evaluators[i].keys()):
            if k in evaluators[j]:
                for dim in DIMENSIONS:
                    if dim in evaluators[i][k] and dim in evaluators[j][k]:
                        shared_vals_i.append(evaluators[i][k][dim])
                        shared_vals_j.append(evaluators[j][k][dim])

        if len(shared_vals_i) < 3:
            print(f"    Evaluator {i+1} vs Evaluator {j+1}: too few shared items")
            continue

        rho, p = spearmanr(shared_vals_i, shared_vals_j)
        rho_values_all.append(rho)
        print(f"    Evaluator {i+1} vs Evaluator {j+1}: ρ = {rho:.3f}  (p={p:.4f}, n={len(shared_vals_i)})")

    if rho_values_all:
        mean_rho = np.mean(rho_values_all)
        print(f"    {'Mean':>30s}: ρ = {mean_rho:.3f}")

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

    # Compute per-dimension mean Spearman
    n_eval = len(evaluators)
    spearman_per_dim = {}
    for dim in DIMENSIONS:
        rho_values = []
        for i, j in combinations(range(n_eval), 2):
            shared_keys = sorted(
                k for k in evaluators[i]
                if k in evaluators[j]
                and dim in evaluators[i][k]
                and dim in evaluators[j][k]
            )
            if len(shared_keys) >= 3:
                vals_i = [evaluators[i][k][dim] for k in shared_keys]
                vals_j = [evaluators[j][k][dim] for k in shared_keys]
                rho, _ = spearmanr(vals_i, vals_j)
                rho_values.append(rho)
        spearman_per_dim[dim] = np.mean(rho_values) if rho_values else None

    # Compute overall mean Spearman
    rho_values_all = []
    for i, j in combinations(range(n_eval), 2):
        shared_vals_i = []
        shared_vals_j = []
        for k in sorted(evaluators[i].keys()):
            if k in evaluators[j]:
                for dim in DIMENSIONS:
                    if dim in evaluators[i][k] and dim in evaluators[j][k]:
                        shared_vals_i.append(evaluators[i][k][dim])
                        shared_vals_j.append(evaluators[j][k][dim])
        if len(shared_vals_i) >= 3:
            rho, _ = spearmanr(shared_vals_i, shared_vals_j)
            rho_values_all.append(rho)
    overall_spearman = np.mean(rho_values_all) if rho_values_all else None

    # Determine n per model from data
    first_model = list(MODEL_LABELS.keys())[0]
    n_per_model = len(ratings[first_model][DIMENSIONS[0]])

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Expert assessment results (mean $\pm$ SEM, $n$="
                 f"{n_per_model}"
                 r" per model). "
                 r"Krippendorff's $\alpha$ and mean pairwise Spearman's $\rho$ measure inter-rater agreement.}")
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

    spearman_cells = []
    for dim in DIMENSIONS:
        rho = spearman_per_dim[dim]
        spearman_cells.append(f"${rho:.3f}$" if rho is not None else "---")
    lines.append(f"Spearman's $\\rho$ (mean) & {' & '.join(spearman_cells)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    footer_parts = []
    if overall_alpha is not None:
        footer_parts.append(f"Overall $\\alpha = {overall_alpha:.3f}$")
    if overall_spearman is not None:
        footer_parts.append(f"Overall $\\rho = {overall_spearman:.3f}$")
    if footer_parts:
        lines.append(r"\vspace{2pt}")
        lines.append(f"\\\\\\small {', '.join(footer_parts)}")

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

    pairwise_spearman(evaluators)

    pairwise_mann_whitney(ratings)

    # TEMP -> generate LaTeX table
    latex_path = output_dir / "human_evaluation_table.tex"
    generate_latex_table(ratings, evaluators, latex_path)

    print("\nDone.")

if __name__ == "__main__":
    main()