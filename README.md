# Scientific Paper Summarization Benchmark

A comprehensive benchmarking framework for evaluating text summarization methods on scientific papers.
It systematically compares **62 summarization approaches** — from classical extractive algorithms (TextRank, word-frequency)
through encoder–decoder models (BART, T5, PEGASUS) to modern large language models (GPT-4.1, Claude 4, Mistral, LLaMA, DeepSeek-R1, and more) —
against expert-written reference summaries using a multi-metric evaluation suite.

The benchmark is designed for reproducibility: every LLM response is cached, model training-data cutoff dates are
tracked automatically, and results are exported as interactive HTML reports with per-model breakdowns.

### Key features

- **Multi-provider LLM support** — OpenAI, Anthropic, Mistral, HuggingFace, and Ollama via a unified API layer
- **Rich evaluation metrics** — ROUGE-1/2/L, BERTScore, METEOR, BLEU, semantic similarity (MPNet), and factual consistency (AlignScore)
- **Length-constrained generation** — configurable min/max word counts with compliance tracking
- **Training cutoff tracking** — automatically collects model knowledge cutoff dates from the
  [community LLM knowledge-cutoff dataset](https://github.com/HaoooWang/llm-knowledge-cutoff-dates),
  HuggingFace model cards, and provider documentation
- **Publication date enrichment** — fetches paper publication dates from [CrossRef](https://www.crossref.org/) to enable temporal analyses of model knowledge vs. paper novelty
- **Interactive visualisations** — per-metric box plots, radar charts, rank heatmaps, correlation matrices, and sortable HTML tables

## Quick Start

This project uses [uv](https://github.com/astral-sh/uv) for package management.

1. Clone this repository
2. Install dependencies
    ```bash
    uv sync
    uv run spacy download en_core_web_sm
    ```
3. Install AlignScore-large
    ```bash
    mkdir -p Output/llm_summarization_benchmark
    cd Output/llm_summarization_benchmark
    wget https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt
    cd ../..
    ```
    > **Note:** If wget fails, download AlignScore-large manually from
    > [github.com/yuh-zha/AlignScore](https://github.com/yuh-zha/AlignScore) (download provided in section "Checkpoints")
    > and place it in `Output/llm_summarization_benchmark/`.
4. Copy `Resources/example.env` to `Resources/.env` and fill in your API keys
5. Run
    ```bash
    uv run benchmark
    ```

### CLI options

```
uv run benchmark [--clear] [--test N]
                 [--reset-all-metrics]
                 [--reset-metrics-for-models MODEL [MODEL ...]]
                 [--reset-metric-types TYPE [TYPE ...]]
                 [--gold-standard-data FILE [FILE ...]]
```

| Flag | Description |
|------|-------------|
| `--clear` | Delete all existing results (responses + metrics) and re-run from scratch |
| `--test N` | Run on the first *N* publications only (useful for quick sanity checks) |
| `--reset-all-metrics` | Recompute all metrics while keeping cached LLM responses |
| `--reset-metrics-for-models` | Recompute metrics for specific models only (space-separated) |
| `--reset-metric-types` | Recompute only specific metric types (default: all) |
| `--gold-standard-data` | Override the gold-standard data file(s) |

### Configuration

LLM generation parameters (temperature, token limits, system prompt, etc.) are configured in
`src/llm_apis/config.py`. The default word-count constraints are **15–100 words**.

### Resume / re-run without benchmarking

The following files must be in place to load previous results:
- `Output/llm_summarization_benchmark/benchmark.pkl`
- `Output/llm_apis/cache.json`

Run `uv run benchmark` again — already-processed models will be skipped and only visualisations will be regenerated.

---

## Workflow

![Workflow](Resources/text_summarization_benchmark_diagram.svg)

> **Note:** The diagram shows an earlier snapshot of the project.
> Actual word limits are 15–100 (not 15–50), the input file is `text_summarization_goldstandard_data.json`,
> and the evaluation suite now includes BLEU, MPNet semantic similarity, and AlignScore in addition to the metrics shown.

---

## Gold-Standard Data

Document store in `Resources/text_summarization_goldstandard_data.json`, containing ID, title, abstract and reference summaries.
1–N reference summaries can be provided per paper.
Multiple reference summaries improve evaluation robustness and reduce single-annotator bias.

```json
[
  {
    "title": "Paper Title",
    "abstract": "Paper abstract text...",
    "id": "paper_001",
    "summaries": [
      "This paper analyzes ..",
      "The paper investigates .. "
    ]
  }
]
```

### Reference summary sources

- Highlight sections of Elsevier and Cell papers, manually extracted and joined by ". ".

---

## Summarization Methods

### local:textrank
1. Tokenises sentences ([nltk](https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.sent_tokenize))
2. Creates TF-IDF vectors for sentence representation ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html))
3. Calculates cosine similarities between TF-IDF vectors (sklearn)
4. Builds similarity graph with cosine similarities as edge weights (networkx)
5. Applies PageRank to rank sentences by importance ([networkx](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html))
6. Selects highest-scoring sentences within word count limits while preserving original order

> **Warning:** Results may be misleading when gold-standard summaries are (partial) copies from the source document rather than abstractive.

### local:frequency
1. Calculates word frequency scores
2. Ranks sentences by average word frequency (excluding stopwords via [nltk](https://www.nltk.org/))
3. Selects highest-scoring sentences (in original order) within word count limits

### LLM providers

- **Anthropic** — Claude 3.5 Haiku, Claude Sonnet 4, Claude Opus 4, Claude Opus 4.1
- **OpenAI** — GPT-3.5 Turbo, GPT-4o / 4o-mini, GPT-4.1 / 4.1-mini, GPT-5 / 5-mini / 5-nano
- **Mistral** — Mistral Large, Mistral Medium, Mistral Small, Magistral Medium
- **HuggingFace** — BART, T5, PEGASUS, BigBird-PEGASUS, LED, mT5, BioGPT, BioMistral, OpenBioLLM, SciLitLLM, Apertus
- **Ollama** — DeepSeek-R1, Gemma 3, Granite 3.3/4, LLaMA 3.1/3.2, MedLLaMA 2, Mistral 7B / Nemo / Small 3.2, Phi-3/4, Qwen 3, GPT-OSS

---

## Evaluation Metrics

Each generated summary is evaluated against all available gold-standard reference summaries.
For each metric, mean / min / max / std and individual scores are computed.

### ROUGE
Set of metrics for evaluating summary quality by comparing n-gram overlap with reference summaries.
[wiki](https://en.wikipedia.org/wiki/ROUGE_(metric)) | [package](https://github.com/google-research/google-research/tree/master/rouge) | [paper](https://aclanthology.org/W04-1013.pdf)
- **ROUGE-1** — unigram overlap
- **ROUGE-2** — bigram overlap
- **ROUGE-L** — longest common subsequence (sentence-level structure similarity)

### BERTScore
Semantic similarity using contextual embeddings.
[paper](https://arxiv.org/abs/1904.09675) | [package](https://github.com/Tiiiger/bert_score)
- `roberta-large` — default model ([paper](https://arxiv.org/abs/1907.11692) | [model](https://huggingface.co/FacebookAI/roberta-large))
- `microsoft/deberta-xlarge-mnli` — proposed as improved model ([paper](https://arxiv.org/abs/2006.03654) | [model](https://huggingface.co/microsoft/deberta-xlarge-mnli))

### METEOR
Matches words through exact matches, stemming, and synonyms; considers word order.
[paper](https://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf)

### BLEU
N-gram overlap with brevity penalty.
[paper](https://www.aclweb.org/anthology/P02-1040.pdf) | [function](https://www.nltk.org/api/nltk.translate.bleu_score.html#nltk.translate.bleu_score.sentence_bleu)

### Semantic Similarity (all-mpnet-base-v2)
Compares the generated summary directly against the **source document** (rather than reference summaries) using sentence-transformer embeddings.
[model](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

### AlignScore
Factual consistency evaluation — measures whether the generated summary is entailed by the abstract.
[paper](https://arxiv.org/abs/2305.16739) | [modified repository](https://github.com/MNikley/AlignScore)

---

## Training Cutoff Dates

Model training-data cutoff dates are tracked in [`Resources/model_training_cutoffs.json`](Resources/model_training_cutoffs.json)
and rendered as an interactive HTML report at [`Resources/model_training_cutoffs.html`](Resources/model_training_cutoffs.html).

To regenerate after editing the JSON or adding new models to the benchmark:
```bash
python scripts/fetch_training_cutoffs.py
```

The script automatically merges data from:
- A curated seed knowledge base (built-in)
- The [community LLM knowledge-cutoff dataset](https://github.com/HaoooWang/llm-knowledge-cutoff-dates)
- HuggingFace model cards (scraped for cutoff patterns)
- Provider documentation (OpenAI, Anthropic, Mistral)

Manual edits to the JSON are preserved across re-runs.

---

## Utility Scripts

| Script | Description |
|--------|-------------|
| `scripts/fetch_training_cutoffs.py` | Collects model training-data cutoff dates from multiple sources and generates a JSON + interactive HTML report |
| `scripts/add_publication_dates.py` | Enriches the gold-standard dataset with publication dates fetched from the CrossRef API using DOIs |
| `scripts/download_models.py` | Parses `benchmark.py` and pre-downloads all required Ollama and HuggingFace models |

---

## Acknowledgements & Data Sources

| Resource | Usage | Link |
|----------|-------|------|
| **LLM Knowledge Cutoff Dates** (HaoooWang) | Primary source for model training-data cutoff dates, crowd-sourced with citations to official docs and papers | [github.com/HaoooWang/llm-knowledge-cutoff-dates](https://github.com/HaoooWang/llm-knowledge-cutoff-dates) |
| **CrossRef API** | Publication date retrieval for gold-standard papers via DOI lookup | [crossref.org](https://www.crossref.org/) |
| **HuggingFace Hub** | Model hosting, model cards used for cutoff date extraction, and inference for encoder–decoder models | [huggingface.co](https://huggingface.co/) |
| **Ollama** | Local inference runtime for open-weight LLMs (LLaMA, Gemma, DeepSeek, Phi, Qwen, etc.) | [ollama.com](https://ollama.com/) |
| **AlignScore** | Factual consistency evaluation metric | [Zha et al. 2023](https://arxiv.org/abs/2305.16739) · [modified repo](https://github.com/MNikley/AlignScore) |
| **BERTScore** | Semantic similarity evaluation using contextual embeddings | [Zhang et al. 2020](https://arxiv.org/abs/1904.09675) |
| **ROUGE** | N-gram overlap metrics for summarization evaluation | [Lin 2004](https://aclanthology.org/W04-1013.pdf) |

---

## Reference

**A Systematic evaluation and benchmarking of text summarization methods for biomedical literature: From word-frequency methods to language models**
Baumgärtel F, Bono E, Fillinger L, Galou L, Kęska-Izworska K, Walter S, Andorfer P, Kratochwill K, Perco P, Ley M
bioRxiv 2026, [doi.org/10.64898/2026.01.09.697335](https://doi.org/10.64898/2026.01.09.697335)
