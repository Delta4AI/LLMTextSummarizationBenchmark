#!/usr/bin/env python3
"""
Generate Games–Howell p-value matrices for:
- categories (model groups)
- families

Input:
- detailed_scores_per_model.json  (per-model lists of per-paper metric scores)

Outputs:
- games_howell_pvalues_matrix_categories.csv
- games_howell_pvalues_matrix_families.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pingouin as pg

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

RUN_DIR = (
    REPO_ROOT
    / "Output"
    / "llm_summarization_benchmark"
    / "1362b291718b57188a7909f08de26da760a0b9346d52111c97671d97d713af38"
)

INPUT_JSON = RUN_DIR / "detailed_scores_per_model.json"

OUTPUT_CATEGORIES_CSV = RUN_DIR / "games_howell_pvalues_matrix_categories.csv"
OUTPUT_FAMILIES_CSV = RUN_DIR / "games_howell_pvalues_matrix_families.csv"



EXCLUDED_MODELS = {"mistral_mistral-medium-2508"}

METRICS = {
    "lexical": ["rouge1", "rouge2", "rougeL", "meteor", "bleu"],
    "semantic": ["bert_roberta-large_f1", "bert_microsoft/deberta-xlarge-mnli_f1", "sentence_transformer"],
    "factual": ["alignscore", "summac", "minicheck_ft5", "minicheck_7b"],
}

WEIGHTS = {"lexical": 1 / 3, "semantic": 1 / 3, "factual": 1 / 3}

MODEL_GROUPS = {
    "traditional_models": [
        "local:frequency", "local:textrank"
    ],
    "general_purpose_edms": [
        "huggingface_facebook/bart-base",
        "huggingface_google-t5/t5-base", "huggingface_google-t5/t5-large",
        "huggingface_google/pegasus-large"
    ],
    "domain_specific_edms": [
        "huggingface_facebook/bart-large-cnn",
        "huggingface_google/pegasus-xsum",
        "huggingface_google/pegasus-cnn_dailymail",
        "huggingface_google/pegasus-pubmed",
        "huggingface_google/bigbird-pegasus-large-pubmed",
        "huggingface_csebuetnlp/mT5_multilingual_XLSum",
        "huggingface_AlgorithmicResearchGroup/led_large_16384_arxiv_summarization"
    ],
    "general_purpose_slms": [
        "ollama_gemma3:270M", "ollama_gemma3:1b", "ollama_gemma3:4b",
        "ollama_PetrosStav/gemma3-tools:4b",
        "ollama_granite3.3:2b", "ollama_granite3.3:8b",
        "ollama_granite4:tiny-h", "ollama_granite4:small-h",
        "ollama_granite4:micro", "ollama_granite4:micro-h",
        "ollama_llama3.1:8b", "ollama_llama3.2:1b", "ollama_llama3.2:3b",
        "ollama_mistral:7b", "ollama_phi3:3.8b",
        "openai_gpt-4o-mini", "openai_gpt-4.1-mini",
        "huggingface:chat_swiss-ai/Apertus-8B-Instruct-2509"
    ],
    "general_purpose_llms": [
        "ollama_gemma3:12b", "ollama_mistral-nemo:12b",
        "ollama_mistral-small3.2:24b", "mistral_mistral-small-2506",
        "mistral_mistral-medium-2505", "mistral_mistral-large-2411",
        "ollama_phi4:14b",
        "openai_gpt-3.5-turbo", "openai_gpt-4o",
        "openai_gpt-4.1", "anthropic_claude-3-5-haiku-20241022"
    ],
    "reasoning_slms": [
        "ollama_deepseek-r1:1.5b", "ollama_deepseek-r1:7b",
        "ollama_deepseek-r1:8b", "ollama_qwen3:4b", "ollama_qwen3:8b"
    ],
    "reasoning_llms": [
        "ollama_deepseek-r1:14b", "ollama_gpt-oss:20b",
        "openai_gpt-5-nano-2025-08-07", "openai_gpt-5-mini-2025-08-07",
        "openai_gpt-5-2025-08-07", "anthropic_claude-sonnet-4-20250514",
        "anthropic_claude-opus-4-20250514",
        "anthropic_claude-opus-4-1-20250805",
        "mistral_magistral-medium-2509"
    ],
    "domain_specific_slms": [
        "huggingface:completion_microsoft/biogpt",
        "ollama_medllama2:7b",
        "huggingface:chat_aaditya/OpenBioLLM-Llama3-8B",
        "huggingface:conversational_BioMistral/BioMistral-7B",
        "huggingface:chat_Uni-SMART/SciLitLLM1.5-7B"
    ],
    "domain_specific_llms": [
        "huggingface:chat_Uni-SMART/SciLitLLM1.5-14B"
    ],
}

MODEL_FAMILIES = {
    "Apertus": [
        "huggingface:chat_swiss-ai/Apertus-8B-Instruct-2509",
    ],
    "BART": [
        "huggingface_facebook/bart-base",
        "huggingface_facebook/bart-large-cnn",
    ],
    "Claude": [
        "anthropic_claude-3-5-haiku-20241022",
        "anthropic_claude-sonnet-4-20250514",
        "anthropic_claude-opus-4-20250514",
        "anthropic_claude-opus-4-1-20250805",
    ],
    "DeepSeek": [
        "ollama_deepseek-r1:1.5b",
        "ollama_deepseek-r1:7b",
        "ollama_deepseek-r1:8b",
        "ollama_deepseek-r1:14b",
    ],
    "Gemma": [
        "ollama_gemma3:270M",
        "ollama_gemma3:1b",
        "ollama_gemma3:4b",
        "ollama_gemma3:12b",
        "ollama_PetrosStav/gemma3-tools:4b",
    ],
    "GPT": [
        "openai_gpt-3.5-turbo",
        "openai_gpt-4o",
        "openai_gpt-4o-mini",
        "openai_gpt-4.1",
        "openai_gpt-4.1-mini",
        "ollama_gpt-oss:20b",
        "openai_gpt-5-nano-2025-08-07",
        "openai_gpt-5-mini-2025-08-07",
        "openai_gpt-5-2025-08-07",
        "huggingface:completion_microsoft/biogpt",
    ],
    "Granite": [
        "ollama_granite3.3:2b",
        "ollama_granite3.3:8b",
        "ollama_granite4:tiny-h",
        "ollama_granite4:small-h",
        "ollama_granite4:micro",
        "ollama_granite4:micro-h",
    ],
    "LED": [
        "huggingface_AlgorithmicResearchGroup/led_large_16384_arxiv_summarization",
    ],
    "Llama": [
        "ollama_llama3.1:8b",
        "ollama_llama3.2:1b",
        "ollama_llama3.2:3b",
        "ollama_medllama2:7b",
        "huggingface:chat_aaditya/OpenBioLLM-Llama3-8B",
    ],
    "Mistral": [
        "ollama_mistral:7b",
        "ollama_mistral-nemo:12b",
        "ollama_mistral-small3.2:24b",
        "mistral_mistral-small-2506",
        "mistral_mistral-medium-2505",
        "mistral_mistral-large-2411",
        "mistral_magistral-medium-2509",
        "huggingface:conversational_BioMistral/BioMistral-7B",
    ],
    "Pegasus": [
        "huggingface_google/pegasus-xsum",
        "huggingface_google/pegasus-cnn_dailymail",
        "huggingface_google/pegasus-large",
        "huggingface_google/pegasus-pubmed",
        "huggingface_google/bigbird-pegasus-large-pubmed",
    ],
    "Phi": [
        "ollama_phi3:3.8b",
        "ollama_phi4:14b",
    ],
    "Qwen": [
        "huggingface:chat_Uni-SMART/SciLitLLM1.5-7B",
        "huggingface:chat_Uni-SMART/SciLitLLM1.5-14B",
        "ollama_qwen3:4b",
        "ollama_qwen3:8b",
    ],
    "T5": [
        "huggingface_google-t5/t5-base",
        "huggingface_google-t5/t5-large",
        "huggingface_csebuetnlp/mT5_multilingual_XLSum",
    ],
    "Word-frequency": [
        "local:frequency",
        "local:textrank",
    ],
}

# label mappings for csv files
CATEGORY_DISPLAY_NAMES = {
    "traditional_models": "Traditional Models",
    "general_purpose_edms": "General-purpose EDMs",
    "domain_specific_edms": "Domain-specific EDMs",
    "general_purpose_slms": "General-purpose SLMs",
    "general_purpose_llms": "General-purpose LLMs",
    "reasoning_slms": "Reasoning-oriented SLMs",
    "reasoning_llms": "Reasoning-oriented LLMs",
    "domain_specific_slms": "Domain-specific SLMs",
    "domain_specific_llms": "Domain-specific LLMs",
}
FAMILY_DISPLAY_NAMES = {
    "Apertus": "Apertus",
    "BART": "BART",
    "Claude": "Claude",
    "DeepSeek": "DeepSeek",
    "Gemma": "Gemma",
    "GPT": "GPT",
    "Granite": "Granite",
    "LED": "LED",
    "Llama": "Llama",
    "Mistral": "Mistral",
    "Pegasus": "Pegasus",
    "Phi": "Phi",
    "Qwen": "Qwen",
    "T5": "T5",
    "Word-frequency": "Word-frequency",
}

def pad_to_length(arr, target_len: int, pad_value: float = 0.0) -> np.ndarray:
    arr = list(arr)
    if len(arr) < target_len:
        arr += [pad_value] * (target_len - len(arr))
    return np.array(arr, dtype=float)

def compute_final_score_per_model(data: dict) -> Dict[str, np.ndarray]:
    metric_mean_scores: Dict[str, np.ndarray] = {}
    aggregate_scores: Dict[str, Dict[str, np.ndarray]] = {}
    required = METRICS["lexical"] + METRICS["semantic"] + METRICS["factual"]

    all_lexical = []
    all_semantic = []
    all_factual = []

    for model, metrics in data.items():
        if model in EXCLUDED_MODELS:
            continue

        missing = [m for m in required if m not in metrics]
        if missing:
            raise KeyError(
                f"Model '{model}' is missing metrics: {missing}. "
                f"Check input JSON keys."
            )

        lengths = [len(metrics[m]) for m in required]
        max_len = max(lengths)

        rouge1 = pad_to_length(metrics["rouge1"], max_len)
        rouge2 = pad_to_length(metrics["rouge2"], max_len)
        rougeL = pad_to_length(metrics["rougeL"], max_len)
        meteor = pad_to_length(metrics["meteor"], max_len)
        bleu = pad_to_length(metrics["bleu"], max_len)

        roberta = pad_to_length(metrics["bert_roberta-large_f1"], max_len)
        deberta = pad_to_length(metrics["bert_microsoft/deberta-xlarge-mnli_f1"], max_len)
        sent_tf = pad_to_length(metrics["sentence_transformer"], max_len)

        align = pad_to_length(metrics["alignscore"], max_len)
        summac = pad_to_length(metrics["summac"], max_len)
        minicheck_ft5 = pad_to_length(metrics["minicheck_ft5"], max_len)
        minicheck_7b = pad_to_length(metrics["minicheck_7b"], max_len)

        avg_lexical = (rouge1 + rouge2 + rougeL + meteor + bleu) / 5.0
        avg_semantic = (roberta + deberta + sent_tf) / 3.0
        avg_factual = (align + summac + minicheck_ft5 + minicheck_7b) / 4.0

        aggregate_scores[model] = {
            "lexical": avg_lexical,
            "semantic": avg_semantic,
            "factual": avg_factual,
        }

        all_lexical.append(avg_lexical)
        all_semantic.append(avg_semantic)
        all_factual.append(avg_factual)

    all_lexical = np.concatenate(all_lexical)
    all_semantic = np.concatenate(all_semantic)
    all_factual = np.concatenate(all_factual)

    lexical_mean = np.mean(all_lexical)
    lexical_std = np.std(all_lexical)
    semantic_mean = np.mean(all_semantic)
    semantic_std = np.std(all_semantic)
    factual_mean = np.mean(all_factual)
    factual_std = np.std(all_factual)

    for model, scores in aggregate_scores.items():
        avg_lexical = scores["lexical"]
        avg_semantic = scores["semantic"]
        avg_factual = scores["factual"]

        z_lexical = (avg_lexical - lexical_mean) / lexical_std
        z_semantic = (avg_semantic - semantic_mean) / semantic_std
        z_factual = (avg_factual - factual_mean) / factual_std

        final_score = (
            WEIGHTS["lexical"] * z_lexical
            + WEIGHTS["semantic"] * z_semantic
            + WEIGHTS["factual"] * z_factual
        )

        metric_mean_scores[model] = final_score

    return metric_mean_scores

def pool_scores(
    metric_mean_scores: Dict[str, np.ndarray],
    definitions: Dict[str, List[str]],
    label: str,
) -> Dict[str, np.ndarray]:
    pooled: Dict[str, np.ndarray] = {}

    for name, models in definitions.items():
        values: List[float] = []
        for m in models:
            if m not in metric_mean_scores:
                print(f"Warning: model '{m}' not found in metric_mean_scores ({label} '{name}')")
                continue
            values.extend(metric_mean_scores[m].tolist())
        pooled[name] = np.array(values, dtype=float)

    return pooled

def games_howell_pvalue_matrix(scores: Dict[str, np.ndarray], factor_name: str) -> pd.DataFrame:
    names = list(scores.keys())

    long_df = pd.DataFrame(
        {
            factor_name: np.concatenate([[n] * len(scores[n]) for n in names]),
            "score": np.concatenate([scores[n] for n in names]),
        }
    )

    gh = pg.pairwise_gameshowell(dv="score", between=factor_name, data=long_df)

    pmat = pd.DataFrame(np.ones((len(names), len(names))), index=names, columns=names)

    for _, row in gh.iterrows():
        a = row["A"]
        b = row["B"]
        p = float(row["pval"])
        pmat.loc[a, b] = p
        pmat.loc[b, a] = p

    return pmat

def format_pvalue_for_csv(p: float) -> str:
    if p == 1.0:
        return "1"
    if p == 0.0:
        return "0"

    if abs(p) < 1e-4:
        return f"{p:.2E}".replace("e", "E")

    s = f"{p:.9f}".rstrip("0").rstrip(".")
    return s

def prettify_and_write_matrix(
    pmat: pd.DataFrame,
    display_name_map: Dict[str, str],
    top_left_label: str,
    output_csv: Path,
) -> None:
    # rename rows & columns
    pretty_index = [display_name_map.get(x, x) for x in pmat.index.tolist()]
    pretty_cols = [display_name_map.get(x, x) for x in pmat.columns.tolist()]

    pmat2 = pmat.copy()
    pmat2.index = pretty_index
    pmat2.columns = pretty_cols

    formatted = pmat2.applymap(lambda v: format_pvalue_for_csv(float(v)))
    formatted.index.name = top_left_label
    formatted.to_csv(output_csv)


def main() -> None:
    if not RUN_DIR.exists():
        raise FileNotFoundError(f"Run directory not found: {RUN_DIR.resolve()}")

    if not INPUT_JSON.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_JSON.resolve()}\n"
            f"Expected it inside: {RUN_DIR.resolve()}"
        )

    with INPUT_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metric_mean_scores = compute_final_score_per_model(data)

    group_scores = pool_scores(metric_mean_scores, MODEL_GROUPS, label="category")
    family_scores = pool_scores(metric_mean_scores, MODEL_FAMILIES, label="family")

    print("\nNumber of pooled scores per category:")
    for k, v in group_scores.items():
        print(f"  {k}: {len(v)}")

    print("\nNumber of pooled scores per family:")
    for k, v in family_scores.items():
        print(f"  {k}: {len(v)}")

    categories_pmat = games_howell_pvalue_matrix(group_scores, factor_name="category")
    families_pmat = games_howell_pvalue_matrix(family_scores, factor_name="family")

    prettify_and_write_matrix(
        categories_pmat,
        display_name_map=CATEGORY_DISPLAY_NAMES,
        top_left_label="Categories",
        output_csv=OUTPUT_CATEGORIES_CSV,
    )

    prettify_and_write_matrix(
        families_pmat,
        display_name_map=FAMILY_DISPLAY_NAMES,
        top_left_label="Families",
        output_csv=OUTPUT_FAMILIES_CSV,
    )

    print(f"\nWrote: {OUTPUT_CATEGORIES_CSV.resolve()}")
    print(f"Wrote: {OUTPUT_FAMILIES_CSV.resolve()}")


if __name__ == "__main__":
    main()