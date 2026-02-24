
#!/usr/bin/env python3
"""download_models.py — Parse benchmark.py and download ollama/huggingface models.

Pre-flight checks detect already-cached models so only missing ones are
downloaded.
"""

import re
import subprocess
import sys
import os
from pathlib import Path

BENCHMARK_FILE = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "llm_summarization_benchmark"
    / "benchmark.py"
)

ADD_PATTERN = re.compile(
    r'benchmark\.add\(\s*"([^"]+)"\s*,\s*"([^"]+)"'
)

# Metric models that aren't discovered from benchmark.py
METRIC_OLLAMA_MODELS = ["bespoke-minicheck"]
METRIC_HF_MODELS = [
    "lytang/MiniCheck-Flan-T5-Large",
    "manueldeprada/FactCC",
]


def parse_models(filepath: str | Path) -> tuple[list[str], list[str]]:
    """Return (ollama_models, huggingface_models) from benchmark.py."""
    ollama: list[str] = []
    huggingface: list[str] = []

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


def get_installed_ollama_models() -> set[str]:
    """Return base names of models already pulled in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set()

    if result.returncode != 0:
        return set()

    names: set[str] = set()
    for line in result.stdout.splitlines()[1:]:  # skip header row
        parts = line.split()
        if not parts:
            continue
        # NAME column is e.g. "bespoke-minicheck:latest"
        name = parts[0].split(":")[0]
        names.add(name)
    return names


def get_cached_hf_repos() -> set[str]:
    """Return repo IDs already present in the HuggingFace cache."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return set()

    try:
        cache_info = scan_cache_dir()
    except Exception:
        return set()

    return {repo.repo_id for repo in cache_info.repos}


def print_model_list(
    ollama: list[str],
    huggingface: list[str],
    metric_ollama: list[str],
    metric_hf: list[str],
    installed_ollama: set[str],
    cached_hf: set[str],
) -> None:
    """Display all models with [cached] / [missing] status."""
    total = len(ollama) + len(huggingface) + len(metric_ollama) + len(metric_hf) + 1  # +1 for SummaC

    def status(name: str, cached_set: set[str]) -> str:
        return "[cached]" if name in cached_set else "[missing]"

    missing = 0
    for m in ollama:
        if m.split(":")[0] not in installed_ollama:
            missing += 1
    for m in huggingface:
        if m not in cached_hf:
            missing += 1
    for m in metric_ollama:
        if m.split(":")[0] not in installed_ollama:
            missing += 1
    for m in metric_hf:
        if m not in cached_hf:
            missing += 1
    missing += 1  # SummaC always counts as "needs check"

    cached_count = total - missing

    print(f"\n{'='*60}")
    print(f"  {total} model(s) total — {cached_count} cached, {missing} to download")
    print(f"{'='*60}")

    if ollama:
        dest = os.environ.get("OLLAMA_MODELS", "(default)")
        print(f"\n  Ollama LLMs ({len(ollama)})  →  {dest}")
        for m in ollama:
            base = m.split(":")[0]
            tag = status(base, installed_ollama)
            print(f"    • {m}  {tag}")

    if huggingface:
        dest = os.environ.get("HF_HOME", "(default)")
        print(f"\n  HuggingFace LLMs ({len(huggingface)})  →  {dest}")
        for m in huggingface:
            tag = status(m, cached_hf)
            print(f"    • {m}  {tag}")

    # Metric models section
    metric_total = len(metric_ollama) + len(metric_hf) + 1  # +1 SummaC
    print(f"\n  Metric models ({metric_total})")
    for m in metric_ollama:
        tag = status(m.split(":")[0], installed_ollama)
        print(f"    • {m} (Ollama)  {tag}")
    for m in metric_hf:
        tag = status(m, cached_hf)
        print(f"    • {m} (HuggingFace)  {tag}")
    print("    • SummaC/vitc (custom)  [always checked]")

    print()


def confirm(prompt: str = "Proceed with download? [y/N] ") -> bool:
    return input(prompt).strip().lower() in ("y", "yes")


def download_ollama(models: list[str]) -> None:
    for model in models:
        print(f"\n>>> ollama pull {model}")
        result = subprocess.run(["ollama", "pull", model])
        if result.returncode != 0:
            print(f"  ⚠  Failed to pull {model} (exit {result.returncode})")


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


def download_metric_models(
    installed_ollama: set[str],
    cached_hf: set[str],
) -> None:
    """Download metric models (FactCC, MiniCheck), skipping cached ones."""
    # FactCC
    if "manueldeprada/FactCC" in cached_hf:
        print("\n  ✓  FactCC (manueldeprada/FactCC) already cached — skipping")
    else:
        print("\n>>> Downloading FactCC model (manueldeprada/FactCC) ...")
        try:
            from transformers import BertForSequenceClassification, BertTokenizer
            BertTokenizer.from_pretrained("manueldeprada/FactCC")
            BertForSequenceClassification.from_pretrained("manueldeprada/FactCC")
            print("  ✓  FactCC model cached")
        except Exception as e:
            print(f"  ⚠  Failed to download FactCC model: {e}")

    # MiniCheck-FT5
    if "lytang/MiniCheck-Flan-T5-Large" in cached_hf:
        print("\n  ✓  MiniCheck-FT5 (lytang/MiniCheck-Flan-T5-Large) already cached — skipping")
    else:
        print("\n>>> Downloading MiniCheck-FT5 model (lytang/MiniCheck-Flan-T5-Large) ...")
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            AutoTokenizer.from_pretrained("lytang/MiniCheck-Flan-T5-Large")
            AutoModelForSeq2SeqLM.from_pretrained("lytang/MiniCheck-Flan-T5-Large")
            print("  ✓  MiniCheck-FT5 model cached")
        except Exception as e:
            print(f"  ⚠  Failed to download MiniCheck-FT5 model: {e}")

    # MiniCheck-7B via Ollama
    if "bespoke-minicheck" in installed_ollama:
        print("\n  ✓  MiniCheck-7B (bespoke-minicheck) already cached — skipping")
    else:
        print("\n>>> Downloading MiniCheck-7B model via Ollama (bespoke-minicheck) ...")
        result = subprocess.run(["ollama", "pull", "bespoke-minicheck"])
        if result.returncode == 0:
            print("  ✓  MiniCheck-7B (Ollama) model cached")
        else:
            print(f"  ⚠  Failed to pull bespoke-minicheck (exit {result.returncode})")


def main() -> None:
    if not Path(BENCHMARK_FILE).exists():
        sys.exit(f"Error: {BENCHMARK_FILE} not found in current directory.")

    ollama, huggingface = parse_models(BENCHMARK_FILE)

    if not ollama and not huggingface:
        sys.exit("No ollama or huggingface models found in benchmark.py.")

    # Merge metric models (deduplicate)
    metric_ollama = [m for m in METRIC_OLLAMA_MODELS if m not in ollama]
    metric_hf = [m for m in METRIC_HF_MODELS if m not in huggingface]

    # Pre-flight: detect what's already cached
    print("  Checking installed Ollama models ...")
    installed_ollama = get_installed_ollama_models()
    print("  Checking HuggingFace cache ...")
    cached_hf = get_cached_hf_repos()

    # Compute missing sets
    missing_ollama = [m for m in ollama if m.split(":")[0] not in installed_ollama]
    missing_hf = [m for m in huggingface if m not in cached_hf]
    missing_metric_ollama = [m for m in metric_ollama if m.split(":")[0] not in installed_ollama]
    missing_metric_hf = [m for m in metric_hf if m not in cached_hf]

    # Show full listing with status
    print_model_list(
        ollama, huggingface,
        metric_ollama, metric_hf,
        installed_ollama, cached_hf,
    )

    has_missing = (
        missing_ollama or missing_hf or missing_metric_ollama or missing_metric_hf
    )
    # SummaC always needs a check, so we always have "something to do"
    # unless everything else is cached too — still prompt since SummaC init
    # is effectively a no-op when cached.
    if not has_missing:
        print("  All models are cached! (SummaC will still be verified.)")
        if not confirm("Run SummaC verification? [y/N] "):
            print("  Nothing to do. Exiting.")
            return

    else:
        if not confirm():
            sys.exit("Aborted.")

    if missing_ollama:
        print(f"\n{'─'*60}")
        print("  Downloading Ollama models ...")
        print(f"{'─'*60}")
        download_ollama(missing_ollama)

    if missing_hf:
        print(f"\n{'─'*60}")
        print("  Downloading HuggingFace models ...")
        print(f"{'─'*60}")
        download_huggingface(missing_hf)

    if missing_metric_ollama or missing_metric_hf:
        print(f"\n{'─'*60}")
        print("  Downloading metric models (FactCC, MiniCheck) ...")
        print(f"{'─'*60}")
        download_metric_models(installed_ollama, cached_hf)

    # SummaC uses a custom init path — always verify (fast when cached)
    print(f"\n{'─'*60}")
    print("  Verifying SummaC model (vitc) ...")
    print(f"{'─'*60}")
    try:
        from summac.model_summac import SummaCZS
        SummaCZS(model_name="vitc", granularity="sentence", device="cpu")
        print("  ✓  SummaC model cached")
    except Exception as e:
        print(f"  ⚠  Failed to download SummaC model: {e}")

    print(f"\n{'='*60}")
    print("  Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
