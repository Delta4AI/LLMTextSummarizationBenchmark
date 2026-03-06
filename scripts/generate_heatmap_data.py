from __future__ import annotations

from pathlib import Path
import pandas as pd

EXCLUDED_MODELS = {"mistral_mistral-medium-2508"}

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

# label mappings for csv file
MODEL_GROUP_LABELS = {
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

def _invert_mapping(mapping: dict[str, list[str]], value_name: str) -> dict[str, str]:
    inv: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}

    for label, models in mapping.items():
        for m in models:
            if m in inv and inv[m] != label:
                duplicates.setdefault(m, []).extend([inv[m], label])
            inv[m] = label

    if duplicates:
        msg = "\n".join(f" - {m}: {sorted(set(lbls))}" for m, lbls in duplicates.items())
        raise ValueError(f"{value_name} duplicates detected (model in multiple buckets):\n{msg}")

    return inv

def build_heatmap_data_from_comparison_csv(
    comparison_csv_path: Path,
    out_xlsx_path: Path,
) -> None:
    comparison_csv_path = Path(comparison_csv_path)
    out_xlsx_path = Path(out_xlsx_path)

    df = pd.read_csv(comparison_csv_path)

    if "method" not in df.columns:
        raise ValueError("comparison_report.csv must contain a 'method' column (model id).")

    df = df[~df["method"].isin(EXCLUDED_MODELS)].copy()

    # exclude certain metric categories that also include "mean"
    EXCLUDED_MEAN_COLUMNS = {
        "exec_time_mean",
        "length_mean",
        "roberta_precision_mean",
        "roberta_recall_mean",
        "deberta_precision_mean",
        "deberta_recall_mean",
    }

    mean_cols = [
        c for c in df.columns
        if c.endswith("_mean") and c not in EXCLUDED_MEAN_COLUMNS
    ]
    if not mean_cols:
        raise ValueError("No '*_mean' columns found in comparison_report.csv.")
    
    df_out = df[["method"] + mean_cols].copy()
    df_out = df_out.rename(columns={"method": "model_name"})

    model_to_family = _invert_mapping(MODEL_FAMILIES, "MODEL_FAMILIES")

    internal_group_to_models = MODEL_GROUPS
    model_to_group_internal = _invert_mapping(internal_group_to_models, "MODEL_GROUPS")
    model_to_group = {
        model: MODEL_GROUP_LABELS.get(group_key, group_key)
        for model, group_key in model_to_group_internal.items()
    }

    df_out["model_family"] = df_out["model_name"].map(model_to_family).fillna("Unknown")
    df_out["model_group"] = df_out["model_name"].map(model_to_group).fillna("Unknown")

    df_out = df_out[["model_name"] + mean_cols + ["model_family", "model_group"]]

    unknown = df_out[(df_out["model_family"] == "Unknown") | (df_out["model_group"] == "Unknown")]["model_name"].tolist()
    if unknown:
        print("WARNING: Unmapped models (family/group = Unknown):")
        for m in unknown:
            print(" -", m)

    df_out.to_excel(out_xlsx_path, index=False)
    print(f"Wrote {out_xlsx_path}")

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent

    RUN_DIR = (
        REPO_ROOT
        / "Output"
        / "llm_summarization_benchmark"
        / "1362b291718b57188a7909f08de26da760a0b9346d52111c97671d97d713af38"
    )

    build_heatmap_data_from_comparison_csv(
        comparison_csv_path=RUN_DIR / "comparison_report.csv",
        out_xlsx_path=RUN_DIR / "heatmap_data.xlsx",
    )