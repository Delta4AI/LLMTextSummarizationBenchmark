#!/usr/bin/env python3
"""
Generate Spearman correlation matrix between metric mean scores (per model)

Input:
- detailed_scores_per_model.json  (per-model lists of per-paper metric scores)

Output:
- metric_correlation_spearman.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

RUN_DIR = (
    REPO_ROOT
    / "Output"
    / "llm_summarization_benchmark"
    / "1362b291718b57188a7909f08de26da760a0b9346d52111c97671d97d713af38"
)

INPUT_JSON = RUN_DIR / "detailed_scores_per_model.json"
OUTPUT_CSV = RUN_DIR / "metric_correlation_spearman.csv"

EXCLUDED_MODELS = {"mistral_mistral-medium-2508"}

# label mappings for csv files
METRICS: List[Tuple[str, str]] = [
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

def _format_corr_value(v: float) -> str:
    if np.isnan(v):
        return ""
    if v == 1.0:
        return "1"
    s = f"{v:.9f}".rstrip("0").rstrip(".")
    return s

def main() -> None:
    if not RUN_DIR.exists():
        raise FileNotFoundError(f"Run directory not found: {RUN_DIR.resolve()}")

    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_JSON.resolve()}")

    with INPUT_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metric_keys = [k for k, _ in METRICS]
    metric_labels = [lbl for _, lbl in METRICS]

    rows: List[Dict[str, float]] = []

    for model, metrics in data.items():
        if model in EXCLUDED_MODELS:
          continue
        row: Dict[str, float] = {"model": model}

        missing = [m for m in metric_keys if m not in metrics]
        if missing:
            raise KeyError(
                f"Model '{model}' is missing metrics: {missing}. "
                f"Check input JSON keys."
            )

        for m in metric_keys:
            values = np.asarray(metrics[m], dtype=float)
            values = values[~np.isnan(values)]
            row[m] = float(np.mean(values)) if values.size else np.nan

        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")

    df = df[metric_keys]

    print(f"Loaded {df.shape[0]} models with {df.shape[1]} metrics.")
    if df.isna().any().any():
        nan_counts = df.isna().sum().to_dict()
        bad = {k: v for k, v in nan_counts.items() if v > 0}
        if bad:
            print("Warning: some per-model metric means are NaN (empty after NaN filtering):")
            for k, v in bad.items():
                print(f"  {k}: {v} models")

    spearman_corr = df.corr(method="spearman")

    spearman_corr.index = metric_labels
    spearman_corr.columns = metric_labels
    spearman_corr.index.name = "Metrics"

    formatted = spearman_corr.copy().astype(object)
    for i in range(formatted.shape[0]):
        for j in range(formatted.shape[1]):
            formatted.iat[i, j] = _format_corr_value(float(spearman_corr.iat[i, j]))

    formatted.to_csv(OUTPUT_CSV)
    print(f"Wrote: {OUTPUT_CSV.resolve()}")

if __name__ == "__main__":
    main()