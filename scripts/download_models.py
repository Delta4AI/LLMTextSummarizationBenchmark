
#!/usr/bin/env python3
"""download_models.py — Parse benchmark.py and download ollama/huggingface models."""

import re
import subprocess
import sys
import os
from pathlib import Path

BENCHMARK_FILE = "../src/llm_summarization_benchmark/benchmark.py"

ADD_PATTERN = re.compile(
    r'benchmark\.add\(\s*"([^"]+)"\s*,\s*"([^"]+)"'
)


def parse_models(filepath: str) -> tuple[list[str], list[str]]:
    """Return (ollama_models, huggingface_models) from benchmark.py."""
    ollama = []
    huggingface = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            m = ADD_PATTERN.search(line)
            if not m:
                continue
            provider, model = m.group(1), m.group(2)
            if provider == "ollama":
                ollama.append(model)
            elif provider.startswith("huggingface"):
                huggingface.append(model)

    return ollama, huggingface


def print_model_list(ollama: list[str], huggingface: list[str]) -> None:
    total = len(ollama) + len(huggingface)
    print(f"\n{'='*60}")
    print(f"  Found {total} model(s) to download")
    print(f"{'='*60}")

    if ollama:
        dest = os.environ.get("OLLAMA_MODELS", "(default)")
        print(f"\n  Ollama ({len(ollama)})  →  {dest}")
        for m in ollama:
            print(f"    • {m}")

    if huggingface:
        dest = os.environ.get("HF_HOME", "(default)")
        print(f"\n  HuggingFace ({len(huggingface)})  →  {dest}")
        for m in huggingface:
            print(f"    • {m}")

    print()


def confirm(prompt: str = "Proceed with download? [y/N] ") -> bool:
    return input(prompt).strip().lower() in ("y", "yes")


def download_ollama(models: list[str]) -> None:
    for model in models:
        print(f"\n>>> ollama pull {model}")
        result = subprocess.run(["ollama", "pull", model])
        if result.returncode != 0:
            print(f"  ⚠  Failed to pull {model} (exit {result.returncode})")


def download_metric_models() -> None:
    """Download models required by evaluation metrics (SummaC, FactCC)."""
    print("\n>>> Downloading SummaC model (vitc) ...")
    try:
        from summac.model_summac import SummaCZS
        SummaCZS(model_name="vitc", granularity="sentence", device="cpu")
        print("  ✓  SummaC model cached")
    except Exception as e:
        print(f"  ⚠  Failed to download SummaC model: {e}")

    print("\n>>> Downloading FactCC model (manueldeprada/FactCC) ...")
    try:
        from transformers import BertForSequenceClassification, BertTokenizer
        BertTokenizer.from_pretrained("manueldeprada/FactCC")
        BertForSequenceClassification.from_pretrained("manueldeprada/FactCC")
        print("  ✓  FactCC model cached")
    except Exception as e:
        print(f"  ⚠  Failed to download FactCC model: {e}")

    print("\n>>> Downloading MiniCheck-FT5 model (lytang/MiniCheck-Flan-T5-Large) ...")
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        AutoTokenizer.from_pretrained("lytang/MiniCheck-Flan-T5-Large")
        AutoModelForSeq2SeqLM.from_pretrained("lytang/MiniCheck-Flan-T5-Large")
        print("  ✓  MiniCheck-FT5 model cached")
    except Exception as e:
        print(f"  ⚠  Failed to download MiniCheck-FT5 model: {e}")

    print("\n>>> Downloading MiniCheck-7B model (bespokelabs/Bespoke-MiniCheck-7B) ...")
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id="bespokelabs/Bespoke-MiniCheck-7B")
        print(f"  ✓  MiniCheck-7B model cached at {path}")
    except Exception as e:
        print(f"  ⚠  Failed to download MiniCheck-7B model: {e}")


def download_huggingface(models: list[str]) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  huggingface_hub not installed — installing via uv ...")
        subprocess.run(["uv", "add", "huggingface_hub"], check=True)
        from huggingface_hub import snapshot_download

    for model in models:
        print(f"\n>>> huggingface_hub.snapshot_download({model!r})")
        try:
            path = snapshot_download(repo_id=model)
            print(f"  ✓  Downloaded to {path}")
        except Exception as e:
            print(f"  ⚠  Failed to download {model}: {e}")


def main() -> None:
    if not Path(BENCHMARK_FILE).exists():
        sys.exit(f"Error: {BENCHMARK_FILE} not found in current directory.")

    ollama, huggingface = parse_models(BENCHMARK_FILE)

    if not ollama and not huggingface:
        sys.exit("No ollama or huggingface models found in benchmark.py.")

    print_model_list(ollama, huggingface)

    if not confirm():
        sys.exit("Aborted.")

    if ollama:
        print(f"\n{'─'*60}")
        print("  Downloading Ollama models ...")
        print(f"{'─'*60}")
        download_ollama(ollama)

    if huggingface:
        print(f"\n{'─'*60}")
        print("  Downloading HuggingFace models ...")
        print(f"{'─'*60}")
        download_huggingface(huggingface)

    print(f"\n{'─'*60}")
    print("  Downloading metric models (SummaC, FactCC, MiniCheck) ...")
    print(f"{'─'*60}")
    download_metric_models()

    print(f"\n{'='*60}")
    print("  Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()