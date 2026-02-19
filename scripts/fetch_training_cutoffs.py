#!/usr/bin/env python3
"""
fetch_training_cutoffs.py — Collect LLM training-data cutoff dates for every
model referenced in benchmark.py.

Strategy (in priority order):
  1. Curated JSON file       – user-editable; survives re-runs.
  2. Community dataset       – github.com/HaoooWang/llm-knowledge-cutoff-dates
                                High-quality, crowd-sourced cutoff dates with
                                citations to official docs / papers.
  3. HuggingFace model-card  – scrape README.md for cutoff / date patterns.
  4. Provider-level defaults  – new OpenAI / Anthropic / Mistral models
                                automatically get the right docs URL.
  5. Mark as unknown         – so the user can fill the gap in the JSON.

On first run the curated JSON is seeded with a built-in knowledge base.
On subsequent runs the existing JSON is loaded, new models are appended,
and nothing already in the file is overwritten — so manual edits stick.

Usage:
    python scripts/fetch_training_cutoffs.py [--output FILE] [--benchmark FILE]
                                             [--skip-community]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# 1.  Parse models from benchmark.py
# ──────────────────────────────────────────────────────────────────────

ADD_PATTERN = re.compile(
    r'benchmark\.add\(\s*"([^"]+)"\s*(?:,\s*"([^"]+)")?'
)


def parse_models(filepath: str) -> list[dict]:
    """Return list of {platform, model} from benchmark.py (uncommented lines only)."""
    models = []
    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            m = ADD_PATTERN.search(stripped)
            if not m:
                continue
            provider_raw = m.group(1)
            model_name = m.group(2) or ""
            platform = provider_raw.split(":")[0]
            if platform == "local" or not model_name:
                continue
            models.append({"platform": provider_raw, "model": model_name})
    return models


# ──────────────────────────────────────────────────────────────────────
# 2.  Seed data — written to the JSON on first run only
# ──────────────────────────────────────────────────────────────────────

SEED_DATA: list[dict] = [
    # ── OpenAI ────────────────────────────────────────────────────────
    {"platform": "openai", "model": "gpt-3.5-turbo",
     "training_cutoff": "2021-09", "source": "confirmed",
     "evidence_urls": ["https://platform.openai.com/docs/models/gpt-3.5-turbo"],
     "notes": "Alias for gpt-3.5-turbo-0125. Knowledge cutoff Sep 2021."},
    {"platform": "openai", "model": "gpt-4o",
     "training_cutoff": "2023-10", "source": "confirmed",
     "evidence_urls": ["https://platform.openai.com/docs/models/gpt-4o"],
     "notes": "gpt-4o-2024-08-06. Knowledge cutoff Oct 2023."},
    {"platform": "openai", "model": "gpt-4o-mini",
     "training_cutoff": "2023-10", "source": "confirmed",
     "evidence_urls": ["https://platform.openai.com/docs/models/gpt-4o-mini"],
     "notes": "gpt-4o-mini-2024-07-18. Knowledge cutoff Oct 2023."},
    {"platform": "openai", "model": "gpt-4.1",
     "training_cutoff": "2024-06", "source": "confirmed",
     "evidence_urls": ["https://platform.openai.com/docs/models/gpt-4.1"],
     "notes": "gpt-4.1-2025-04-14. Knowledge cutoff Jun 2024."},
    {"platform": "openai", "model": "gpt-4.1-mini",
     "training_cutoff": "2024-06", "source": "confirmed",
     "evidence_urls": ["https://platform.openai.com/docs/models/gpt-4.1-mini"],
     "notes": "gpt-4.1-mini-2025-04-14. Knowledge cutoff Jun 2024."},
    {"platform": "openai", "model": "gpt-5-nano-2025-08-07",
     "training_cutoff": None, "source": "unknown",
     "evidence_urls": ["https://platform.openai.com/docs/models"],
     "notes": "Released Aug 2025. Check OpenAI docs for confirmed cutoff."},
    {"platform": "openai", "model": "gpt-5-mini-2025-08-07",
     "training_cutoff": None, "source": "unknown",
     "evidence_urls": ["https://platform.openai.com/docs/models"],
     "notes": "Released Aug 2025. Check OpenAI docs for confirmed cutoff."},
    {"platform": "openai", "model": "gpt-5-2025-08-07",
     "training_cutoff": None, "source": "unknown",
     "evidence_urls": ["https://platform.openai.com/docs/models"],
     "notes": "Released Aug 2025. Check OpenAI docs for confirmed cutoff."},

    # ── Anthropic ─────────────────────────────────────────────────────
    {"platform": "anthropic", "model": "claude-3-5-haiku-20241022",
     "training_cutoff": "2024-04", "source": "confirmed",
     "evidence_urls": ["https://docs.anthropic.com/en/docs/about-claude/models"],
     "notes": "Early Apr 2024 training data cutoff."},
    {"platform": "anthropic", "model": "claude-sonnet-4-20250514",
     "training_cutoff": "2025-03", "source": "confirmed",
     "evidence_urls": ["https://docs.anthropic.com/en/docs/about-claude/models"],
     "notes": "Early Mar 2025 training data cutoff."},
    {"platform": "anthropic", "model": "claude-opus-4-20250514",
     "training_cutoff": "2025-03", "source": "confirmed",
     "evidence_urls": ["https://docs.anthropic.com/en/docs/about-claude/models"],
     "notes": "Early Mar 2025 training data cutoff."},
    {"platform": "anthropic", "model": "claude-opus-4-1-20250805",
     "training_cutoff": None, "source": "unknown",
     "evidence_urls": ["https://docs.anthropic.com/en/docs/about-claude/models"],
     "notes": "Released Aug 2025. Check Anthropic docs for confirmed cutoff."},

    # ── Mistral (API) ────────────────────────────────────────────────
    {"platform": "mistral", "model": "mistral-large-2411",
     "training_cutoff": None, "source": "unknown",
     "evidence_urls": ["https://docs.mistral.ai/getting-started/models/premier/"],
     "notes": "Mistral Large 24.11. Cutoff not publicly documented."},
    {"platform": "mistral", "model": "mistral-medium-2505",
     "training_cutoff": None, "source": "unknown",
     "evidence_urls": ["https://docs.mistral.ai/getting-started/models/premier/"],
     "notes": "Mistral Medium 3 (May 2025). Cutoff not publicly documented."},
    {"platform": "mistral", "model": "mistral-medium-2508",
     "training_cutoff": None, "source": "unknown",
     "evidence_urls": ["https://docs.mistral.ai/getting-started/models/premier/"],
     "notes": "Mistral Medium 3 update (Aug 2025). Cutoff not publicly documented."},
    {"platform": "mistral", "model": "mistral-small-2506",
     "training_cutoff": None, "source": "unknown",
     "evidence_urls": ["https://docs.mistral.ai/getting-started/models/premier/"],
     "notes": "Mistral Small 3.2 (Jun 2025). Cutoff not publicly documented."},
    {"platform": "mistral", "model": "magistral-medium-2509",
     "training_cutoff": None, "source": "unknown",
     "evidence_urls": ["https://docs.mistral.ai/getting-started/models/premier/"],
     "notes": "Magistral Medium (Sep 2025). Cutoff not publicly documented."},

    # ── Ollama / open-source ──────────────────────────────────────────
    # DeepSeek-R1
    *[{"platform": "ollama", "model": f"deepseek-r1:{sz}",
       "training_cutoff": None, "source": "unknown",
       "evidence_urls": ["https://arxiv.org/abs/2501.12948",
                         "https://huggingface.co/deepseek-ai/DeepSeek-R1"],
       "notes": "DeepSeek-R1 paper (Jan 2025) does not state an explicit cutoff. "
                "Built on DeepSeek-V3 (pre-trained through ~late 2024)."}
      for sz in ("1.5b", "7b", "8b", "14b")],
    # Gemma 3
    *[{"platform": "ollama", "model": f"gemma3:{sz}",
       "training_cutoff": "2024-08", "source": "confirmed",
       "evidence_urls": ["https://ai.google.dev/gemma/docs/core/model_card_3"],
       "notes": "Knowledge cutoff August 2024 per Gemma 3 model card."}
      for sz in ("270M", "1b", "4b", "12b")],
    {"platform": "ollama", "model": "PetrosStav/gemma3-tools:4b",
     "training_cutoff": "2024-08", "source": "confirmed",
     "evidence_urls": ["https://ai.google.dev/gemma/docs/core/model_card_3",
                       "https://huggingface.co/PetrosStav/gemma3-4b-it-tools"],
     "notes": "Fine-tuned Gemma 3 4B. Base model knowledge cutoff August 2024 per Gemma 3 model card."},
    # Granite 3.3
    *[{"platform": "ollama", "model": f"granite3.3:{sz}",
       "training_cutoff": "2024-04", "source": "estimated",
       "evidence_urls": ["https://github.com/orgs/ibm-granite/discussions/18",
                         "https://huggingface.co/ibm-granite/granite-3.3-8b-instruct",
                         "https://www.ibm.com/granite"],
       "notes": "Estimated April 2024. Granite 3.3 README attributes its dataset to "
                "ibm-granite/granite-3.0-language-models; additional training uses synthetic "
                "data from open-source LLMs (no new world-knowledge). Granite 3.0 cutoff "
                "confirmed April 2024 by IBM maintainer (GitHub discussion). "
                "No explicit cutoff published for 3.3 itself."}
      for sz in ("2b", "8b")],
    # Granite 4
    *[{"platform": "ollama", "model": f"granite4:{sz}",
       "training_cutoff": None, "source": "unknown",
       "evidence_urls": ["https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models",
                         "https://www.ibm.com/granite"],
       "notes": "IBM Granite 4. Released Oct 2025. Cutoff not publicly documented."}
      for sz in ("micro", "micro-h", "tiny-h", "small-h")],
    # LLaMA 3.1
    {"platform": "ollama", "model": "llama3.1:8b",
     "training_cutoff": "2023-12", "source": "confirmed",
     "evidence_urls": ["https://ai.meta.com/blog/meta-llama-3-1/",
                       "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md"],
     "notes": "Knowledge cutoff December 2023."},
    # LLaMA 3.2
    *[{"platform": "ollama", "model": f"llama3.2:{sz}",
       "training_cutoff": "2023-12", "source": "confirmed",
       "evidence_urls": ["https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/",
                         "https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md"],
       "notes": "Knowledge cutoff December 2023."}
      for sz in ("1b", "3b")],
    # MedLLaMA 2
    {"platform": "ollama", "model": "medllama2:7b",
     "training_cutoff": "2022-09", "source": "confirmed",
     "evidence_urls": ["https://huggingface.co/meta-llama/Llama-2-7b",
                       "https://arxiv.org/abs/2307.09288"],
     "notes": "Based on LLaMA 2 (Sep 2022 pre-training cutoff), fine-tuned on medical data."},
    # Mistral 7B
    {"platform": "ollama", "model": "mistral:7b",
     "training_cutoff": None, "source": "unknown",
     "evidence_urls": ["https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3",
                       "https://arxiv.org/abs/2310.06825"],
     "notes": "Mistral 7B v0.3. No official cutoff published."},
    # Mistral Nemo
    {"platform": "ollama", "model": "mistral-nemo:12b",
     "training_cutoff": None, "source": "unknown",
     "evidence_urls": ["https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407",
                       "https://mistral.ai/news/mistral-nemo"],
     "notes": "Mistral Nemo 12B. No official cutoff published."},
    # Mistral Small 3.2
    {"platform": "ollama", "model": "mistral-small3.2:24b",
     "training_cutoff": None, "source": "unknown",
     "evidence_urls": ["https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506"],
     "notes": "Mistral Small 3.2. No official cutoff published."},
    # Phi-3
    {"platform": "ollama", "model": "phi3:3.8b",
     "training_cutoff": "2023-10", "source": "confirmed",
     "evidence_urls": ["https://huggingface.co/microsoft/Phi-3-mini-4k-instruct",
                       "https://arxiv.org/abs/2404.14219"],
     "notes": "Cutoff date Oct 2023 per model card."},
    # Phi-4
    {"platform": "ollama", "model": "phi4:14b",
     "training_cutoff": "2024-06", "source": "confirmed",
     "evidence_urls": ["https://huggingface.co/microsoft/phi-4",
                       "https://arxiv.org/abs/2412.08905"],
     "notes": "Cutoff date Jun 2024 per model card."},
    # Qwen 3
    *[{"platform": "ollama", "model": f"qwen3:{sz}",
       "training_cutoff": None, "source": "unknown",
       "evidence_urls": ["https://huggingface.co/Qwen/Qwen3-8B",
                         "https://qwenlm.github.io/blog/qwen3/"],
       "notes": "Alibaba Qwen 3. No official cutoff published."}
      for sz in ("4b", "8b")],
    # GPT-OSS
    {"platform": "ollama", "model": "gpt-oss:20b",
     "training_cutoff": None, "source": "unknown",
     "evidence_urls": ["https://ollama.com/library/gpt-oss"],
     "notes": "GPT-OSS 20B. Limited public documentation."},

    # ── HuggingFace (pre-LLM summarisation models) ───────────────────
    {"platform": "huggingface", "model": "facebook/bart-large-cnn",
     "training_cutoff": "2019", "source": "estimated",
     "evidence_urls": ["https://huggingface.co/facebook/bart-large-cnn",
                       "https://arxiv.org/abs/1910.13461"],
     "notes": "Estimated. BART paper Oct 2019; pre-trained on books + news corpus "
              "collected through ~2019. Fine-tuned on CNN/DailyMail (articles up to ~2015)."},
    {"platform": "huggingface", "model": "facebook/bart-base",
     "training_cutoff": "2019", "source": "estimated",
     "evidence_urls": ["https://huggingface.co/facebook/bart-base",
                       "https://arxiv.org/abs/1910.13461"],
     "notes": "Estimated. Same pre-training corpus as BART-Large (~2019)."},
    {"platform": "huggingface", "model": "google-t5/t5-base",
     "training_cutoff": "2019-04", "source": "estimated",
     "evidence_urls": ["https://huggingface.co/google-t5/t5-base",
                       "https://arxiv.org/abs/1910.10683"],
     "notes": "Estimated. T5 pre-trained on C4, a cleaned Common Crawl snapshot from Apr 2019."},
    {"platform": "huggingface", "model": "google-t5/t5-large",
     "training_cutoff": "2019-04", "source": "estimated",
     "evidence_urls": ["https://huggingface.co/google-t5/t5-large",
                       "https://arxiv.org/abs/1910.10683"],
     "notes": "Estimated. Same pre-training corpus as T5-Base (C4, Apr 2019)."},
    {"platform": "huggingface", "model": "csebuetnlp/mT5_multilingual_XLSum",
     "training_cutoff": "2020", "source": "estimated",
     "evidence_urls": ["https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum",
                       "https://arxiv.org/abs/2106.13822",
                       "https://arxiv.org/abs/2010.11934"],
     "notes": "Estimated. mT5 pre-trained on mC4 (multilingual Common Crawl, ~2020). "
              "XL-Sum fine-tuning data collected from BBC through ~2020."},
    {"platform": "huggingface", "model": "google/pegasus-xsum",
     "training_cutoff": "2020", "source": "estimated",
     "evidence_urls": ["https://huggingface.co/google/pegasus-xsum",
                       "https://arxiv.org/abs/1912.08777"],
     "notes": "Estimated. PEGASUS pre-trained on C4 + HugeNews (web-crawled news, ~2019-2020)."},
    {"platform": "huggingface", "model": "google/pegasus-large",
     "training_cutoff": "2020", "source": "estimated",
     "evidence_urls": ["https://huggingface.co/google/pegasus-large",
                       "https://arxiv.org/abs/1912.08777"],
     "notes": "Estimated. Same pre-training corpus as other PEGASUS variants (~2019-2020)."},
    {"platform": "huggingface", "model": "google/pegasus-cnn_dailymail",
     "training_cutoff": "2020", "source": "estimated",
     "evidence_urls": ["https://huggingface.co/google/pegasus-cnn_dailymail",
                       "https://arxiv.org/abs/1912.08777"],
     "notes": "Estimated. PEGASUS pre-trained ~2019-2020, fine-tuned on CNN/DailyMail."},
    {"platform": "huggingface", "model": "AlgorithmicResearchGroup/led_large_16384_arxiv_summarization",
     "training_cutoff": "2020", "source": "estimated",
     "evidence_urls": ["https://huggingface.co/AlgorithmicResearchGroup/led_large_16384_arxiv_summarization",
                       "https://arxiv.org/abs/2004.05150"],
     "notes": "Estimated. Longformer Encoder-Decoder, initialised from BART and published "
              "Apr 2020. Fine-tuned on arXiv papers available at that time."},
    {"platform": "huggingface", "model": "google/pegasus-pubmed",
     "training_cutoff": "2020", "source": "estimated",
     "evidence_urls": ["https://huggingface.co/google/pegasus-pubmed",
                       "https://arxiv.org/abs/1912.08777"],
     "notes": "Estimated. PEGASUS pre-trained ~2019-2020, fine-tuned on PubMed."},
    {"platform": "huggingface", "model": "google/bigbird-pegasus-large-pubmed",
     "training_cutoff": "2020", "source": "estimated",
     "evidence_urls": ["https://huggingface.co/google/bigbird-pegasus-large-pubmed",
                       "https://arxiv.org/abs/2007.14062"],
     "notes": "Estimated. BigBird paper Jul 2020; PEGASUS backbone pre-trained ~2019-2020, "
              "fine-tuned on PubMed."},

    # ── HuggingFace (LLM / chat models) ──────────────────────────────
    {"platform": "huggingface:completion", "model": "microsoft/biogpt",
     "training_cutoff": "2021", "source": "estimated",
     "evidence_urls": ["https://huggingface.co/microsoft/biogpt",
                       "https://arxiv.org/abs/2210.10341"],
     "notes": "Estimated. BioGPT paper (Oct 2022) states pre-training on 15M PubMed "
              "abstracts up to 2021."},
    {"platform": "huggingface:chat", "model": "swiss-ai/Apertus-8B-Instruct-2509",
     "training_cutoff": "2024-03", "source": "confirmed",
     "evidence_urls": ["https://github.com/swiss-ai/apertus-tech-report/blob/main/Apertus_Tech_Report.pdf",
                       "https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509"],
     "notes": "Knowledge cutoff March 2024 per Apertus Tech Report (system prompt section)."},
    {"platform": "huggingface:chat", "model": "Uni-SMART/SciLitLLM1.5-7B",
     "training_cutoff": "2024", "source": "estimated",
     "evidence_urls": ["https://arxiv.org/abs/2408.15545",
                       "https://github.com/QwenLM/Qwen2.5/issues/525",
                       "https://huggingface.co/Qwen/Qwen2.5-7B",
                       "https://huggingface.co/Uni-SMART/SciLitLLM1.5-7B"],
     "notes": "Estimated ~early-to-mid 2024, inherited from Qwen2.5-Base (released Sep 2024, "
              "training data through ~early-mid 2024). CPT corpora (RedPajama v1 ~early 2023, "
              "scientific textbooks/papers) and SFT data (SciRIFF) predate the base model cutoff. "
              "SciLitLLM paper does not state an explicit cutoff."},
    {"platform": "huggingface:chat", "model": "Uni-SMART/SciLitLLM1.5-14B",
     "training_cutoff": "2024", "source": "estimated",
     "evidence_urls": ["https://arxiv.org/abs/2408.15545",
                       "https://github.com/QwenLM/Qwen2.5/issues/525",
                       "https://huggingface.co/Qwen/Qwen2.5-14B",
                       "https://huggingface.co/Uni-SMART/SciLitLLM1.5-14B"],
     "notes": "Estimated ~early-to-mid 2024, inherited from Qwen2.5-Base (released Sep 2024, "
              "training data through ~early-mid 2024). CPT corpora (RedPajama v1 ~early 2023, "
              "scientific textbooks/papers) and SFT data (SciRIFF) predate the base model cutoff. "
              "SciLitLLM paper does not state an explicit cutoff."},
    {"platform": "huggingface:chat", "model": "aaditya/OpenBioLLM-Llama3-8B",
     "training_cutoff": "2023-03", "source": "confirmed",
     "evidence_urls": ["https://huggingface.co/aaditya/OpenBioLLM-Llama3-8B"],
     "notes": "Based on LLaMA 3 8B (knowledge cutoff Mar 2023), fine-tuned on medical data."},
    {"platform": "huggingface:conversational", "model": "BioMistral/BioMistral-7B",
     "training_cutoff": "2023", "source": "estimated",
     "evidence_urls": ["https://arxiv.org/abs/2402.10373",
                       "https://github.com/BioMistral/BioMistral",
                       "https://huggingface.co/BioMistral/BioMistral-7B"],
     "notes": "Estimated ~mid-to-late 2023. Base model Mistral 7B Instruct v0.1 "
              "(released Sep 2023, general knowledge ~mid-2023). CPT on PMC Open Access "
              "Subset (~1.47M documents, likely downloaded Q3-Q4 2023). Biomedical domain "
              "knowledge may be slightly more recent than general knowledge. "
              "Neither the BioMistral paper nor Mistral specify exact cutoff dates."},
]


# ──────────────────────────────────────────────────────────────────────
# 2b. Community dataset — github.com/HaoooWang/llm-knowledge-cutoff-dates
# ──────────────────────────────────────────────────────────────────────

COMMUNITY_DATASET_URL = (
    "https://raw.githubusercontent.com/HaoooWang/llm-knowledge-cutoff-dates"
    "/main/README.md"
)

# Maps *lowercased* model names from the community README → benchmark
# (platform, model) pairs.  Only models that appear in our benchmark need
# an entry here.  When the community README adds a model we care about,
# just extend this dict.
COMMUNITY_ALIASES: dict[str, list[tuple[str, str]]] = {
    # ── OpenAI ────────────────────────────────────────────────────────
    "gpt-3.5*":                     [("openai", "gpt-3.5-turbo")],
    "gpt-4o (2024-05-13)":          [("openai", "gpt-4o")],
    "gpt-4o (2024-08-06)":          [("openai", "gpt-4o")],
    "gpt-4o mini (2024-07-18)":     [("openai", "gpt-4o-mini")],
    "gpt-4.1":                      [("openai", "gpt-4.1")],
    "gpt-4.1-mini":                 [("openai", "gpt-4.1-mini")],
    "gpt-5":                        [("openai", "gpt-5-2025-08-07")],
    "gpt-5 mini":                   [("openai", "gpt-5-mini-2025-08-07")],
    "gpt-5 nano":                   [("openai", "gpt-5-nano-2025-08-07")],
    "gpt-oss":                      [("ollama", "gpt-oss:20b")],
    # ── Anthropic ─────────────────────────────────────────────────────
    "claude 3.5 haiku":             [("anthropic", "claude-3-5-haiku-20241022")],
    "claude 4 sonnet":              [("anthropic", "claude-sonnet-4-20250514")],
    "claude 4 opus":                [("anthropic", "claude-opus-4-20250514")],
    "claude 4.1 opus":              [("anthropic", "claude-opus-4-1-20250805")],
    # ── Meta ──────────────────────────────────────────────────────────
    "llama-2-7b,13b,70b":           [("ollama", "medllama2:7b")],
    "llama-3-7b":                   [("huggingface:chat", "aaditya/OpenBioLLM-Llama3-8B")],
    "llama-3.1-8b":                 [("ollama", "llama3.1:8b")],
    "llama-3.2-1b":                 [("ollama", "llama3.2:1b")],
    "llama-3.2-3b":                 [("ollama", "llama3.2:3b")],
    # ── DeepSeek ──────────────────────────────────────────────────────
    "deepseek-r1":                  [("ollama", f"deepseek-r1:{sz}")
                                     for sz in ("1.5b", "7b", "8b", "14b")],
    # ── Microsoft ─────────────────────────────────────────────────────
    "phi-3-*":                      [("ollama", "phi3:3.8b")],
    # ── Qwen ──────────────────────────────────────────────────────────
    "qwen3":                        [("ollama", "qwen3:4b"),
                                     ("ollama", "qwen3:8b")],
}


def _parse_community_tables(text: str) -> list[dict]:
    """Parse every markdown table row from the community README.

    Returns a list of dicts:
        name, company, cutoff_raw, source_url, section
    """
    records: list[dict] = []
    lines = text.splitlines()
    current_section = ""
    in_table = False
    col_indices: dict[str, int] = {}          # column role → cell index

    for line in lines:
        stripped = line.strip()

        # Track top-level section headers (# OpenAI, # Google, …)
        if stripped.startswith("# ") and not stripped.startswith("## "):
            current_section = stripped.lstrip("# ").strip()
            in_table = False
            col_indices = {}
            continue

        if not stripped.startswith("|"):
            # A non-table, non-blockquote line ends the current table
            if in_table and stripped and not stripped.startswith(">"):
                in_table = False
                col_indices = {}
            continue

        cells = [c.strip() for c in stripped.strip("|").split("|")]

        # Skip separator rows  ( |---| ... )
        if all(re.fullmatch(r"[-:\s]+", c) for c in cells if c):
            continue

        # Detect header row
        if not in_table:
            lower = [c.lower() for c in cells]
            if any("model" in c for c in lower):
                for i, c in enumerate(lower):
                    if "model" in c:
                        col_indices["name"] = i
                    elif "company" in c:
                        col_indices["company"] = i
                    elif "training data cut-off" in c:
                        col_indices["cutoff"] = i
                    elif "cut-off" in c or "cut off" in c:
                        col_indices.setdefault("cutoff", i)
                    elif "source" in c:
                        col_indices["source"] = i
                in_table = (
                    col_indices.get("name") is not None
                    and col_indices.get("cutoff") is not None
                )
                continue

        # Data row
        if not in_table:
            continue
        max_idx = max(col_indices.values(), default=0)
        if len(cells) <= max_idx:
            continue

        name = cells[col_indices["name"]]
        company = cells[col_indices.get("company", 1)] if "company" in col_indices else ""
        cutoff_raw = cells[col_indices["cutoff"]]
        source_cell = cells[col_indices["source"]] if "source" in col_indices else ""

        # Extract URL from markdown link [text](url)
        url_m = re.search(r"\[.*?\]\((.*?)\)", source_cell)
        source_url = url_m.group(1) if url_m else ""

        records.append({
            "name": name.strip(),
            "company": company.strip(),
            "cutoff_raw": cutoff_raw.strip(),
            "source_url": source_url.strip(),
            "section": current_section,
        })

    return records


def _normalise_community_cutoff(raw: str) -> str | None:
    """Normalise a cutoff string from the community dataset → YYYY-MM or YYYY."""
    raw = raw.strip()
    if not raw or raw.lower() in ("unknown", "tbd", "-", ""):
        return None

    # "Pretraining 2022.09, Finetuning 2023.07" → take the pretraining date
    m = re.search(r"[Pp]retraining\s+(\d{4}\.\d{2})", raw)
    if m:
        raw = m.group(1)

    # "early 2023", "end of 2023", "late 2024"
    m = re.match(r"(?:early|end\s+of|late|mid)\s+(\d{4})", raw, re.I)
    if m:
        return m.group(1)

    # "2024.06.01" or "2024.06"
    m = re.match(r"(\d{4})\.(\d{2})(?:\.\d{2})?", raw)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    # Bare year "2023"
    m = re.match(r"^(\d{4})$", raw)
    if m:
        return m.group(1)

    return None


async def _fetch_community_data(client: httpx.AsyncClient) -> dict[str, dict]:
    """Fetch the community README and return a lookup keyed by 'platform|model'.

    Each value is ``{"training_cutoff": ..., "source_url": ...,
    "community_name": ...}``.
    """
    try:
        resp = await client.get(COMMUNITY_DATASET_URL, timeout=20,
                                follow_redirects=True)
        resp.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        log.warning("Failed to fetch community dataset: %s", exc)
        return {}

    records = _parse_community_tables(resp.text)
    log.info("Parsed %d rows from community dataset", len(records))

    result: dict[str, dict] = {}
    matched = 0
    for rec in records:
        name_lower = rec["name"].lower().strip()
        cutoff = _normalise_community_cutoff(rec["cutoff_raw"])
        targets = COMMUNITY_ALIASES.get(name_lower, [])
        if not targets:
            continue
        for platform, model in targets:
            key = f"{platform}|{model}"
            result[key] = {
                "training_cutoff": cutoff,
                "source_url": rec["source_url"],
                "community_name": rec["name"],
            }
            if cutoff:
                matched += 1

    log.info("Matched %d community rows to benchmark models (%d with dates)",
             len(result), matched)
    return result


# ──────────────────────────────────────────────────────────────────────
# 3.  Provider-level fallbacks for NEW models not in the JSON yet
# ──────────────────────────────────────────────────────────────────────

PROVIDER_DEFAULTS: dict[str, dict] = {
    "openai": {
        "evidence_urls": ["https://platform.openai.com/docs/models"],
        "notes_template": "New OpenAI model. Check the models page for training data cutoff.",
    },
    "anthropic": {
        "evidence_urls": ["https://docs.anthropic.com/en/docs/about-claude/models"],
        "notes_template": "New Anthropic model. Check the models page for training data cutoff.",
    },
    "mistral": {
        "evidence_urls": ["https://docs.mistral.ai/getting-started/models/premier/"],
        "notes_template": "New Mistral model. Check the models page for training data cutoff.",
    },
    "ollama": {
        "evidence_urls": [],
        "notes_template": "Ollama model. Check the upstream model card for cutoff info.",
    },
    "huggingface": {
        "evidence_urls": [],   # filled dynamically with HF link
        "notes_template": "HuggingFace model. Check the model card for cutoff info.",
    },
}


def make_default_record(platform: str, model: str) -> dict:
    """Create a scaffold record for a model not yet in the JSON."""
    base_platform = platform.split(":")[0]
    defaults = PROVIDER_DEFAULTS.get(base_platform, {})
    evidence = list(defaults.get("evidence_urls", []))
    notes = defaults.get("notes_template", "No information available.")

    # For HF models, auto-generate a link to the model card
    if base_platform == "huggingface":
        evidence.append(f"https://huggingface.co/{model}")
    # For Ollama, link to the library page if it looks like a plain name
    elif base_platform == "ollama" and "/" not in model.split(":")[0]:
        lib_name = model.split(":")[0]
        evidence.append(f"https://ollama.com/library/{lib_name}")

    return {
        "platform": platform,
        "model": model,
        "training_cutoff": None,
        "source": "unknown",
        "release_date": None,
        "release_date_source": None,
        "evidence_urls": evidence,
        "notes": notes,
    }


# ──────────────────────────────────────────────────────────────────────
# 4.  HuggingFace model-card scraping
# ──────────────────────────────────────────────────────────────────────

CUTOFF_PATTERNS = [
    re.compile(
        r"(?:cutoff|cut[\s-]?off|knowledge.*?cut|training.*?data.*?(?:up\s+to|through|until))"
        r"[^.\n]{0,60}?"
        r"(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{4}\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:cutoff|cut[\s-]?off|knowledge.*?cut|training.*?data.*?(?:up\s+to|through|until))"
        r"[^.\n]{0,60}?"
        r"(\b\d{4}[-/]\d{2}(?:[-/]\d{2})?\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:cutoff|cut[\s-]?off)[^.\n]{0,30}?"
        r"(\b(?:early|late|mid)?\s*\d{4}\b)",
        re.IGNORECASE,
    ),
]

MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}


def normalise_date(raw: str) -> str:
    """Best-effort normalisation of a date string to YYYY-MM or YYYY."""
    raw = raw.strip()
    m = re.match(
        r"(January|February|March|April|May|June|July|August|September"
        r"|October|November|December)\s+(\d{4})", raw, re.I)
    if m:
        return f"{m.group(2)}-{MONTH_MAP[m.group(1).lower()]}"
    m = re.match(r"(\d{4})[-/](\d{2})", raw)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.match(r"(\d{4})", raw)
    if m:
        return m.group(1)
    return raw


# Ollama model base name → HuggingFace repo for README scraping
OLLAMA_TO_HF: dict[str, str] = {
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "gemma3": "google/gemma-3-4b-it",
    "granite3.3": "ibm-granite/granite-3.3-8b-instruct",
    "granite4": "ibm-granite/granite-4.0-tiny",
    "llama3.1": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.2": "meta-llama/Llama-3.2-3B-Instruct",
    "medllama2": "meta-llama/Llama-2-7b",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "mistral-small3.2": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "phi4": "microsoft/phi-4",
    "qwen3": "Qwen/Qwen3-8B",
    "gpt-oss": "nomic-ai/gpt-oss-20b",
}


def hf_repo_for(platform: str, model: str) -> str | None:
    """Determine the HuggingFace repo ID to fetch a README from."""
    if platform.startswith("huggingface"):
        return model
    if platform == "ollama":
        base = model.split(":")[0]
        return OLLAMA_TO_HF.get(base)
    return None


async def fetch_hf_readme(client: httpx.AsyncClient, repo_id: str) -> str | None:
    url = f"https://huggingface.co/{repo_id}/raw/main/README.md"
    try:
        resp = await client.get(url, timeout=15, follow_redirects=True)
        if resp.status_code == 200:
            return resp.text
    except (httpx.RequestError, httpx.TimeoutException):
        pass
    return None


def extract_cutoff_from_readme(text: str) -> str | None:
    for pat in CUTOFF_PATTERNS:
        m = pat.search(text)
        if m:
            return normalise_date(m.group(1))
    return None


# ──────────────────────────────────────────────────────────────────────
# 4b. Release-date helpers
# ──────────────────────────────────────────────────────────────────────

# Fallback for API-only models whose names don't encode a date.
KNOWN_RELEASE_DATES: dict[str, str] = {
    # OpenAI — API-only models without date-encoded names
    "openai|gpt-3.5-turbo":     "2023-03-01",   # gpt-3.5-turbo API launch
    "openai|gpt-4o":            "2024-05-13",   # GPT-4o announcement
    "openai|gpt-4o-mini":       "2024-07-18",   # GPT-4o-mini launch
    "openai|gpt-4.1":           "2025-04-14",
    "openai|gpt-4.1-mini":      "2025-04-14",
    # IBM Granite 4 — gated HF repo, date from announcement
    **{f"ollama|granite4:{sz}": "2025-10-02"
       for sz in ("micro", "micro-h", "tiny-h", "small-h")},
    # GPT-OSS — gated HF repo
    "ollama|gpt-oss:20b":       "2025-08-05",
    # PetrosStav/gemma3-tools — no HF API match for Ollama fork
    "ollama|PetrosStav/gemma3-tools:4b": "2025-04",   # ~10 months ago per ollama.com
}


def _extract_release_date_from_name(model: str) -> str | None:
    """Try to extract a release date encoded in the model name.

    Handles three conventions:
      - YYYYMMDD suffix:  claude-opus-4-20250514  → 2025-05-14
      - YYYY-MM-DD infix: gpt-5-nano-2025-08-07   → 2025-08-07
      - YYMM suffix:      mistral-large-2411       → 2024-11
    """
    base = model.split(":")[0]

    # YYYYMMDD at end
    m = re.search(r"(\d{4})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$", base)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # YYYY-MM-DD embedded
    m = re.search(r"(20[2-3]\d)-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])", base)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # YYMM at end  (only 2020–2029 range to avoid false positives)
    m = re.search(r"-(2[0-9])(0[1-9]|1[0-2])$", base)
    if m:
        return f"20{m.group(1)}-{m.group(2)}"

    return None


async def _fetch_hf_created_at(
    client: httpx.AsyncClient, repo_id: str,
) -> str | None:
    """Return the HuggingFace repo creation date as YYYY-MM-DD (or None)."""
    url = f"https://huggingface.co/api/models/{repo_id}"
    try:
        resp = await client.get(url, timeout=15, follow_redirects=True)
        if resp.status_code == 200:
            created = resp.json().get("createdAt", "")
            if created and len(created) >= 10:
                return created[:10]           # "2024-12-11T16:06:38.000Z" → "2024-12-11"
    except (httpx.RequestError, httpx.TimeoutException, ValueError):
        pass
    return None


# ──────────────────────────────────────────────────────────────────────
# 5.  HTML report
# ──────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model Training Cutoff Dates</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --yellow: #d29922; --red: #f85149;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); padding: 2rem; line-height: 1.5;
  }
  h1 { font-size: 1.6rem; margin-bottom: .3rem; }
  .subtitle { color: var(--muted); margin-bottom: 1.5rem; font-size: .95rem; }
  .stats {
    display: flex; gap: 1.5rem; margin-bottom: 1.5rem; flex-wrap: wrap;
  }
  .stat {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: .8rem 1.2rem; min-width: 140px;
  }
  .stat .num { font-size: 1.8rem; font-weight: 700; }
  .stat .label { color: var(--muted); font-size: .82rem; text-transform: uppercase; letter-spacing: .04em; }
  .controls {
    display: flex; gap: .8rem; margin-bottom: 1rem; flex-wrap: wrap; align-items: center;
  }
  .controls input, .controls select {
    background: var(--surface); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: .45rem .7rem; font-size: .9rem; outline: none;
  }
  .controls input:focus, .controls select:focus { border-color: var(--accent); }
  .controls input { width: 260px; }
  table {
    width: 100%; border-collapse: collapse; background: var(--surface);
    border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
    font-size: .88rem;
  }
  th {
    background: #1c2128; text-align: left; padding: .55rem .7rem;
    border-bottom: 1px solid var(--border); color: var(--muted);
    font-weight: 600; font-size: .8rem; text-transform: uppercase;
    letter-spacing: .04em; cursor: pointer; user-select: none;
    white-space: nowrap;
  }
  th:hover { color: var(--text); }
  th .arrow { margin-left: .3rem; font-size: .7rem; }
  td {
    padding: .45rem .7rem; border-bottom: 1px solid var(--border);
    vertical-align: top;
  }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #1c2128; }

  /* ── Shared badge base ── */
  .badge {
    display: inline-block; padding: .1rem .45rem; border-radius: 10px;
    font-size: .7rem; font-weight: 600; vertical-align: middle;
  }
  .badge-confirmed  { background: #0d2818; color: var(--green); }
  .badge-community  { background: #0d1f2d; color: var(--accent); }
  .badge-estimated  { background: #1a1800; color: var(--yellow); }
  .badge-model-card { background: #1a1800; color: var(--yellow); }
  .badge-unknown    { background: #2a1215; color: var(--red); }

  /* ── Inline source sub-badges (smaller, sits beside the date) ── */
  .src-tag {
    display: inline-block; padding: .05rem .4rem; border-radius: 8px;
    font-size: .65rem; font-weight: 600; letter-spacing: .02em;
    vertical-align: middle; margin-left: .35rem;
  }
  .src-tag-confirmed  { background: #0d2818; color: var(--green); }
  .src-tag-community  { background: #0d1f2d; color: var(--accent); }
  .src-tag-estimated  { background: #1a1800; color: var(--yellow); }
  .src-tag-model-card { background: #1a1800; color: var(--yellow); }
  .src-tag-unknown    { background: #2a1215; color: var(--red); }
  .src-tag-neutral    { background: #1c2128; color: var(--muted); border: 1px solid var(--border); }

  /* ── Date cells ── */
  .date-cell { white-space: nowrap; }
  .cutoff-confirmed { color: var(--green); font-weight: 600; font-variant-numeric: tabular-nums; }
  .cutoff-estimated { color: var(--yellow); font-weight: 600; font-variant-numeric: tabular-nums; }
  .cutoff-unknown   { color: var(--muted); font-style: italic; }
  .release-date     { font-variant-numeric: tabular-nums; }
  .release-date-unknown { color: var(--muted); font-style: italic; }

  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .links { display: flex; flex-direction: column; gap: .15rem; }
  .links a {
    font-size: .8rem; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; max-width: 340px; display: inline-block;
  }
  .notes { color: var(--muted); font-size: .8rem; max-width: 340px; }
  .platform-tag {
    display: inline-block; padding: .1rem .45rem; border-radius: 4px;
    font-size: .75rem; font-weight: 600; background: #1c2128;
    border: 1px solid var(--border);
  }
</style>
</head>
<body>

<h1>Model Training Cutoff Dates</h1>
<p class="subtitle">Generated by <code>scripts/fetch_training_cutoffs.py</code> · {{TIMESTAMP}}</p>

<div class="stats">
  <div class="stat"><div class="num">{{TOTAL}}</div><div class="label">Total Models</div></div>
  <div class="stat"><div class="num" style="color:var(--green)">{{CONFIRMED}}</div><div class="label">Confirmed</div></div>
  <div class="stat"><div class="num" style="color:var(--accent)">{{COMMUNITY}}</div><div class="label">Community</div></div>
  <div class="stat"><div class="num" style="color:var(--yellow)">{{ESTIMATED}}</div><div class="label">Estimated</div></div>
  <div class="stat"><div class="num" style="color:var(--red)">{{UNKNOWN}}</div><div class="label">Unknown</div></div>
</div>

<div class="controls">
  <input type="text" id="search" placeholder="Filter models…">
  <select id="platformFilter">
    <option value="">All platforms</option>
    {{PLATFORM_OPTIONS}}
  </select>
  <select id="statusFilter">
    <option value="">All statuses</option>
    <option value="confirmed">✅ Confirmed</option>
    <option value="community">🔵 Community</option>
    <option value="estimated">🔶 Estimated</option>
    <option value="unknown">❓ Unknown</option>
  </select>
</div>

<table>
<thead>
<tr>
  <th data-col="0">Platform <span class="arrow"></span></th>
  <th data-col="1">Model <span class="arrow"></span></th>
  <th data-col="2">Released <span class="arrow"></span></th>
  <th data-col="3">Training Cutoff <span class="arrow"></span></th>
  <th>Evidence</th>
  <th>Notes</th>
</tr>
</thead>
<tbody id="tbody">
{{ROWS}}
</tbody>
</table>

<script>
const rows = [...document.querySelectorAll('#tbody tr')];
const search = document.getElementById('search');
const platformFilter = document.getElementById('platformFilter');
const statusFilter = document.getElementById('statusFilter');

function applyFilters() {
  const q = search.value.toLowerCase();
  const plat = platformFilter.value;
  const stat = statusFilter.value;
  rows.forEach(r => {
    const text = r.textContent.toLowerCase();
    const rPlat = r.dataset.platform;
    const rStat = r.dataset.status;
    const show = text.includes(q)
      && (!plat || rPlat === plat)
      && (!stat || rStat === stat);
    r.style.display = show ? '' : 'none';
  });
}
search.addEventListener('input', applyFilters);
platformFilter.addEventListener('change', applyFilters);
statusFilter.addEventListener('change', applyFilters);

let sortCol = -1, sortAsc = true;
document.querySelectorAll('th[data-col]').forEach(th => {
  th.addEventListener('click', () => {
    const col = +th.dataset.col;
    if (sortCol === col) sortAsc = !sortAsc; else { sortCol = col; sortAsc = true; }
    document.querySelectorAll('th .arrow').forEach(a => a.textContent = '');
    th.querySelector('.arrow').textContent = sortAsc ? '▲' : '▼';
    const tbody = document.getElementById('tbody');
    rows.sort((a, b) => {
      const av = a.children[col].dataset.sort || a.children[col].textContent;
      const bv = b.children[col].dataset.sort || b.children[col].textContent;
      return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
    });
    rows.forEach(r => tbody.appendChild(r));
  });
});
</script>
</body>
</html>
"""


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def generate_html(results: list[dict], path: Path) -> None:
    from datetime import datetime, timezone

    confirmed = sum(1 for r in results if r["source"] == "confirmed")
    community = sum(1 for r in results if r["source"] == "community_dataset")
    estimated = sum(1 for r in results if r["source"] in ("estimated", "model_card"))
    unknown = len(results) - confirmed - community - estimated
    platforms = sorted({r["platform"].split(":")[0] for r in results})
    platform_opts = "\n".join(f'    <option value="{_esc(p)}">{_esc(p)}</option>' for p in platforms)

    # Mapping: cutoff source key → (css class suffix, display label)
    _CUTOFF_SRC_MAP = {
        "confirmed":        ("confirmed",  "confirmed"),
        "community_dataset":("community",  "community"),
        "estimated":        ("estimated",  "estimated"),
        "model_card":       ("model-card", "model card"),
    }
    # Mapping: release-date source key → display label
    _RD_SRC_LABELS = {
        "huggingface_api": "hf api",
        "model_name":      "model name",
        "manual":          "manual",
    }

    row_fragments: list[str] = []
    for r in results:
        plat_base = r["platform"].split(":")[0]
        src = r["source"]
        is_estimated = src in ("estimated", "model_card")
        is_community = src == "community_dataset"
        has_cutoff = bool(r["training_cutoff"])

        if is_community:
            status = "community"
        elif is_estimated:
            status = "estimated"
        elif has_cutoff:
            status = "confirmed"
        else:
            status = "unknown"

        # ── Released cell (date + source sub-badge) ───────────────────
        rd = r.get("release_date")
        rds = r.get("release_date_source") or ""
        rds_label = _RD_SRC_LABELS.get(rds, rds.replace("_", " "))
        release_sort = rd or "9999"

        if rd:
            rds_tag = (f' <span class="src-tag src-tag-neutral">{_esc(rds_label)}</span>'
                       if rds_label else "")
            release_html = (f'<span class="date-cell">'
                            f'<span class="release-date">{_esc(rd)}</span>'
                            f'{rds_tag}</span>')
        else:
            release_html = '<span class="release-date-unknown">—</span>'

        # ── Training Cutoff cell (date + source sub-badge) ────────────
        cutoff_sort = r["training_cutoff"] or "9999"
        css_suffix, src_label = _CUTOFF_SRC_MAP.get(src, ("unknown", "unknown"))

        if has_cutoff and is_estimated:
            date_span = f'<span class="cutoff-estimated">~{_esc(r["training_cutoff"])}</span>'
        elif has_cutoff:
            date_span = f'<span class="cutoff-confirmed">{_esc(r["training_cutoff"])}</span>'
        else:
            date_span = '<span class="cutoff-unknown">unknown</span>'

        src_tag = f' <span class="src-tag src-tag-{css_suffix}">{_esc(src_label)}</span>'
        cutoff_html = f'<span class="date-cell">{date_span}{src_tag}</span>'

        # ── Evidence & Notes ──────────────────────────────────────────
        links_html = '<div class="links">' + "".join(
            f'<a href="{_esc(u)}" target="_blank" rel="noopener">{_esc(u)}</a>'
            for u in r.get("evidence_urls", [])
        ) + "</div>" if r.get("evidence_urls") else "–"

        notes_html = f'<span class="notes">{_esc(r["notes"])}</span>' if r.get("notes") else "–"

        row_fragments.append(
            f'<tr data-platform="{_esc(plat_base)}" data-status="{status}">'
            f'<td data-sort="{_esc(plat_base)}"><span class="platform-tag">{_esc(r["platform"])}</span></td>'
            f'<td data-sort="{_esc(r["model"])}">{_esc(r["model"])}</td>'
            f'<td data-sort="{_esc(release_sort)}">{release_html}</td>'
            f'<td data-sort="{_esc(cutoff_sort)}">{cutoff_html}</td>'
            f"<td>{links_html}</td>"
            f"<td>{notes_html}</td>"
            f"</tr>"
        )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = (
        HTML_TEMPLATE
        .replace("{{TIMESTAMP}}", now)
        .replace("{{TOTAL}}", str(len(results)))
        .replace("{{CONFIRMED}}", str(confirmed))
        .replace("{{COMMUNITY}}", str(community))
        .replace("{{ESTIMATED}}", str(estimated))
        .replace("{{UNKNOWN}}", str(unknown))
        .replace("{{PLATFORM_OPTIONS}}", platform_opts)
        .replace("{{ROWS}}", "\n".join(row_fragments))
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ──────────────────────────────────────────────────────────────────────
# 6.  Main pipeline
# ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        sys.exit(f"Error: {benchmark_path} not found.")

    output_path = Path(args.output)
    html_path = output_path.with_suffix(".html")

    # ── Parse benchmark models ────────────────────────────────────────
    models = parse_models(str(benchmark_path))
    log.info("Parsed %d models from %s", len(models), benchmark_path)

    # ── Load existing JSON (if any) ───────────────────────────────────
    existing: dict[str, dict] = {}   # key = "platform|model"
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for entry in json.load(f):
                key = f"{entry['platform']}|{entry['model']}"
                existing[key] = entry
        log.info("Loaded %d existing entries from %s", len(existing), output_path)
    else:
        # First run → seed from built-in data
        log.info("No existing JSON found — seeding from built-in knowledge base")
        for entry in SEED_DATA:
            key = f"{entry['platform']}|{entry['model']}"
            existing[key] = dict(entry)  # copy

    # ── Merge: ensure every benchmark model has an entry ──────────────
    new_count = 0
    for m in models:
        key = f"{m['platform']}|{m['model']}"
        if key not in existing:
            existing[key] = make_default_record(m["platform"], m["model"])
            new_count += 1
            log.info("  + new model: %s / %s", m["platform"], m["model"])
    if new_count:
        log.info("Added %d new model(s) from benchmark.py", new_count)

    # Build ordered result list (benchmark order, then any extras from JSON)
    seen_keys: set[str] = set()
    results: list[dict] = []
    for m in models:
        key = f"{m['platform']}|{m['model']}"
        if key not in seen_keys:
            results.append(existing[key])
            seen_keys.add(key)
    # Append entries in the JSON that are no longer in benchmark (keep them)
    for key, entry in existing.items():
        if key not in seen_keys:
            results.append(entry)
            seen_keys.add(key)

    # ── Community dataset + HuggingFace scraping ────────────────────
    async with httpx.AsyncClient() as client:
        # Step 1: community dataset (highest-priority automatic source)
        if not args.skip_community:
            community_data = await _fetch_community_data(client)
            community_applied = 0
            for idx, r in enumerate(results):
                key = f"{r['platform']}|{r['model']}"
                if key not in community_data:
                    continue
                cdata = community_data[key]
                # Never overwrite manually curated entries
                if r["source"] in ("confirmed", "estimated"):
                    continue
                if cdata["training_cutoff"]:
                    results[idx]["training_cutoff"] = cdata["training_cutoff"]
                    results[idx]["source"] = "community_dataset"
                    repo_url = "https://github.com/HaoooWang/llm-knowledge-cutoff-dates"
                    if (cdata["source_url"]
                            and cdata["source_url"] not in results[idx]["evidence_urls"]):
                        results[idx]["evidence_urls"].insert(0, cdata["source_url"])
                    if repo_url not in results[idx]["evidence_urls"]:
                        results[idx]["evidence_urls"].append(repo_url)
                    results[idx]["notes"] = (
                        f"From community dataset (HaoooWang/llm-knowledge-cutoff-dates). "
                        f"Original entry: \"{cdata['community_name']}\"."
                    )
                    community_applied += 1
                    log.info("  ✓ community: %s/%s → %s",
                             r["platform"], r["model"], cdata["training_cutoff"])
            if community_applied:
                log.info("Applied %d cutoff(s) from community dataset",
                         community_applied)
        else:
            log.info("Skipping community dataset (--skip-community)")

        # Step 2: HuggingFace README scraping for remaining unknowns
        hf_queue: list[tuple[int, str]] = []
        for idx, r in enumerate(results):
            if not r["training_cutoff"] and r["source"] == "unknown":
                repo = hf_repo_for(r["platform"], r["model"])
                if repo:
                    hf_queue.append((idx, repo))

        if hf_queue:
            unique_repos: dict[str, list[int]] = {}
            for idx, repo in hf_queue:
                unique_repos.setdefault(repo, []).append(idx)

            log.info("Fetching %d HuggingFace READMEs for cutoff extraction ...",
                     len(unique_repos))
            sem = asyncio.Semaphore(10)

            async def _fetch(repo: str) -> tuple[str, str | None]:
                async with sem:
                    return repo, await fetch_hf_readme(client, repo)

            for coro in asyncio.as_completed([_fetch(r) for r in unique_repos]):
                repo, readme = await coro
                if readme:
                    cutoff = extract_cutoff_from_readme(readme)
                    if cutoff:
                        log.info("  ✓ Found cutoff %s in %s", cutoff, repo)
                        for idx in unique_repos[repo]:
                            results[idx]["training_cutoff"] = cutoff
                            results[idx]["source"] = "model_card"
                            hf_url = f"https://huggingface.co/{repo}"
                            if hf_url not in results[idx]["evidence_urls"]:
                                results[idx]["evidence_urls"].append(hf_url)

        # Step 3: Populate release dates ───────────────────────────────
        # Ensure fields exist on all records (handles pre-existing JSON)
        for r in results:
            r.setdefault("release_date", None)
            r.setdefault("release_date_source", None)

        # 3a. Known manual dates (API-only models without date-encoded names)
        for r in results:
            if r["release_date"]:
                continue
            key = f"{r['platform']}|{r['model']}"
            if key in KNOWN_RELEASE_DATES:
                r["release_date"] = KNOWN_RELEASE_DATES[key]
                r["release_date_source"] = "manual"

        # 3b. Extract from model name
        for r in results:
            if r["release_date"]:
                continue
            date = _extract_release_date_from_name(r["model"])
            if date:
                r["release_date"] = date
                r["release_date_source"] = "model_name"

        # 3c. Fetch from HuggingFace API for the rest
        rd_queue: list[tuple[int, str]] = []
        for idx, r in enumerate(results):
            if r["release_date"]:
                continue
            repo = hf_repo_for(r["platform"], r["model"])
            if repo:
                rd_queue.append((idx, repo))

        if rd_queue:
            rd_repos: dict[str, list[int]] = {}
            for idx, repo in rd_queue:
                rd_repos.setdefault(repo, []).append(idx)

            log.info("Fetching %d HuggingFace release dates ...", len(rd_repos))
            sem_rd = asyncio.Semaphore(10)

            async def _fetch_rd(repo: str) -> tuple[str, str | None]:
                async with sem_rd:
                    return repo, await _fetch_hf_created_at(client, repo)

            for coro in asyncio.as_completed(
                [_fetch_rd(r) for r in rd_repos]
            ):
                repo, created = await coro
                if created:
                    log.info("  ✓ %s created %s", repo, created)
                    for idx in rd_repos[repo]:
                        results[idx]["release_date"] = created
                        results[idx]["release_date_source"] = "huggingface_api"

        rd_count = sum(1 for r in results if r["release_date"])
        log.info("Release dates: %d / %d populated", rd_count, len(results))

    # ── Write outputs ─────────────────────────────────────────────────
    confirmed = sum(1 for r in results if r["source"] == "confirmed")
    community_ct = sum(1 for r in results if r["source"] == "community_dataset")
    estimated = sum(1 for r in results if r["source"] in ("estimated", "model_card"))
    unknown = len(results) - confirmed - community_ct - estimated
    log.info("Results: %d confirmed, %d community, %d estimated, %d unknown (total %d)",
             confirmed, community_ct, estimated, unknown, len(results))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Saved → %s", output_path)

    generate_html(results, html_path)
    log.info("Saved → %s", html_path)

    # ── Terminal summary ──────────────────────────────────────────────
    print(f"\n{'Platform':<35} {'Model':<50} {'Cutoff':<12} {'Released':<14} {'Source'}")
    print("─" * 125)
    for r in results:
        cutoff = r["training_cutoff"] or "❓ unknown"
        if r["source"] == "estimated":
            cutoff = f"~{cutoff}"
        released = r.get("release_date") or "—"
        print(f"{r['platform']:<35} {r['model']:<50} {cutoff:<12} {released:<14} {r['source']}")


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--benchmark", "-b",
        default="src/llm_summarization_benchmark/benchmark.py",
        help="Path to benchmark.py (default: src/llm_summarization_benchmark/benchmark.py)",
    )
    p.add_argument(
        "--output", "-o",
        default="Resources/model_training_cutoffs.json",
        help="Output JSON path (default: Resources/model_training_cutoffs.json)",
    )
    p.add_argument(
        "--skip-community", action="store_true",
        help="Skip fetching the community cutoff dataset from GitHub.",
    )
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(cli()))
