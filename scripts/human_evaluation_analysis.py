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

# Likert categories used by the agreement coefficients below
CATEGORIES = [1, 2, 3, 4, 5]

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


# Gwet's AC2 and Brennan-Prediger (paradox-resistant agreement coefficients)
def build_weights(categories, kind="quadratic"):
    cats = list(categories)
    q = len(cats)
    if kind == "identity":
        return [[1.0 if k == l else 0.0 for l in range(q)] for k in range(q)]
    denom = (max(cats) - min(cats)) ** 2
    return [[1.0 - ((cats[k] - cats[l]) ** 2) / denom for l in range(q)]
            for k in range(q)]


def _subject_counts(evaluators, dimension):
    """Per-subject category-count vectors for one dimension (subjects with >=2 raters)."""
    cat_index = {c: idx for idx, c in enumerate(CATEGORIES)}
    all_keys = set()
    for ev in evaluators:
        all_keys.update(ev.keys())

    subjects = []
    for key in sorted(all_keys):
        counts = [0] * len(CATEGORIES)
        n_raters = 0
        for ev in evaluators:
            if key in ev and dimension in ev[key]:
                counts[cat_index[ev[key][dimension]]] += 1
                n_raters += 1
        if n_raters >= 2:
            subjects.append(counts)
    return subjects


def _subject_counts_overall(evaluators):
    cat_index = {c: idx for idx, c in enumerate(CATEGORIES)}
    flat_evs = []
    for ev in evaluators:
        flat = {}
        for key, rats in ev.items():
            for dim in DIMENSIONS:
                if dim in rats:
                    flat[(*key, dim)] = rats[dim]
        flat_evs.append(flat)

    all_keys = set()
    for fe in flat_evs:
        all_keys.update(fe.keys())

    subjects = []
    for key in sorted(all_keys):
        counts = [0] * len(CATEGORIES)
        n_raters = 0
        for fe in flat_evs:
            if key in fe:
                counts[cat_index[fe[key]]] += 1
                n_raters += 1
        if n_raters >= 2:
            subjects.append(counts)
    return subjects


def agreement_coefficients(subjects, W):
    q = len(CATEGORIES)

    pa_sum, used = 0.0, 0
    for counts in subjects:
        r_i = sum(counts)
        if r_i < 2:
            continue
        weighted_pairs = 0.0
        for k in range(q):
            if counts[k] == 0:
                continue
            for l in range(q):
                if counts[l] == 0:
                    continue
                weighted_pairs += W[k][l] * counts[k] * counts[l]
        pa_sum += (weighted_pairs - r_i) / (r_i * (r_i - 1))
        used += 1
    if used == 0:
        return None
    pa = pa_sum / used

    pi = [0.0] * q
    for counts in subjects:
        r_i = sum(counts)
        if r_i < 2:
            continue
        for k in range(q):
            pi[k] += counts[k] / r_i
    pi = [p / used for p in pi]

    T_w = sum(W[k][l] for k in range(q) for l in range(q))
    sum_pi = sum(pi[k] * (1 - pi[k]) for k in range(q))

    pe_ac2 = (T_w / (q * (q - 1))) * sum_pi
    pe_bp = T_w / (q * q)

    ac2 = (pa - pe_ac2) / (1 - pe_ac2) if pe_ac2 != 1 else 1.0
    bp = (pa - pe_bp) / (1 - pe_bp) if pe_bp != 1 else 1.0
    return {"pa": pa, "pe_ac2": pe_ac2, "pe_bp": pe_bp,
            "ac2": ac2, "bp": bp, "n": used}


def gwet_ac2(evaluators, dimension, W):
    res = agreement_coefficients(_subject_counts(evaluators, dimension), W)
    return res["ac2"] if res else None


def brennan_prediger(evaluators, dimension, W):
    res = agreement_coefficients(_subject_counts(evaluators, dimension), W)
    return res["bp"] if res else None

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

    # Inter-rater agreement: Gwet's AC2 (primary) with Brennan-Prediger as a
    # paradox-resistant cross-check and the observed weighted agreement (pa).
    print("\n--- Inter-Rater Agreement (Gwet's AC2, quadratic weights) ---")
    print("  AC2 = primary coefficient; BP = Brennan-Prediger cross-check; "
          "pa = observed agreement")
    print(f"  {'dimension':>12s}  {'AC2':>7s}  {'BP':>7s}  {'pa':>7s}")

    W = build_weights(CATEGORIES, kind="quadratic")
    for dim in DIMENSIONS:
        res = agreement_coefficients(_subject_counts(evaluators, dim), W)
        if res is None:
            print(f"  {dim:>12s}: insufficient data")
            continue
        print(f"  {dim:>12s}  {res['ac2']:>7.3f}  {res['bp']:>7.3f}  {res['pa']:>7.3f}")

    res_all = agreement_coefficients(_subject_counts_overall(evaluators), W)
    if res_all is not None:
        print(f"  {'overall':>12s}  {res_all['ac2']:>7.3f}  "
              f"{res_all['bp']:>7.3f}  {res_all['pa']:>7.3f}")

    pairwise_spearman(evaluators)

    pairwise_mann_whitney(ratings)

    print("\nDone.")


if __name__ == "__main__":
    main()