# Scientific Paper Summarization Benchmark

A comprehensive benchmarking framework for evaluating text summarization methods on scientific papers.
It systematically compares **60+ summarization approaches** — from classical extractive algorithms (TextRank, word-frequency)
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
- **Interactive visualisations** — per-metric box plots, radar charts, and sortable HTML tables

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
    > **Note:** In case wget fails, you can download AlignScore-large manually from:
    > https://github.com/yuh-zha/AlignScore (download provided in section "Checkpoints")
    > and place it in `Output/llm_summarization_benchmark/`.
4. Copy `Resources/example.env` to `Resources/.env` and adjust
5. Run
    ```bash
    uv run benchmark
    ```

> **Hint:** Individual LLM config parameters are stored in `/src/llm_apis/config.py`

### Run the visualization without benchmarking

The following files must be in place in order to load previous results:    
- `Output/llm_summarization_benchmark/benchmark.pkl`
- `Output/llm_apis/cache.json`

Afterwards, simply run the benchmark again - processed results will be skipped.

---

## Workflow
![Workflow](Resources/text_summarization_benchmark_diagram.svg)

---

## text_summarization_goldstandard_data.json
Document store in `Resources` folder, containing ID, title, abstract and reference summaries.
1-N reference summaries can be provided per paper. 
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

### **local:textrank**
1. Tokenizes sentences ([nltk](https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.sent_tokenize))
2. Creates TF-IDF vectors for sentence representation ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer))
3. Calculates cosine similarities between TF-IDF vectors (sklearn)
4. Builds similarity graph with cosine similarities as edge weights (networkx)
5. Applies PageRank to rank sentences by importance ([networkx](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html))
6. Selects highest-scoring sentences within word count limits while preserving original order
> **WARNING**: Results might be misleading when gold-standard summaries are (partial) copies from the source document, rather than being abstractive

### **local:frequency**
1. Calculates word frequency scores
2. ranks sentences by avg. word frequency (excluding stopwords ([nltk](https://www.nltk.org/))
3. selects highest-scoring sentences (in original order) within word count limits

### LLM Providers
- **Anthropic**, **Mistral**, **OpenAI**, **HuggingFace**, **Ollama**

---

## Evaluation Metrics
> Each generated summary is evaluated against all available gold-standard reference summaries of a document using a number of metrics as listed below. For each metric, mean/min/max/std and individual counts are computed.

### Rouge
Set of metrics for evaluating summary quality by comparing to reference summaries. [wiki](https://en.wikipedia.org/wiki/ROUGE_(metric)) | [package](https://github.com/google-research/google-research/tree/master/rouge) | [publication](https://aclanthology.org/W04-1013.pdf)
- **ROUGE-N**: N-gram co-occurrence statistics between system and reference summaries.
  - **ROUGE-1**: Overlap of unigrams (individual words)
  - **ROUGE-2**: Overlap of bigrams (word pairs)
- **ROUGE-L**: Longest Common Subsequence (LCS) based statistics that capture sentence-level structure similarity by awarding credit only to in-sequence word matches.

### Bert
Semantic similarity using BERT embeddings. [paper](https://arxiv.org/abs/1904.09675) | [package](https://github.com/Tiiiger/bert_score)
- `roberta-large`: Default model [paper](https://arxiv.org/abs/1907.11692) | [model](https://huggingface.co/FacebookAI/roberta-large)
- `microsoft/deberta-xlarge-mnli`: Proposed as "better model" [paper](https://arxiv.org/abs/2006.03654) | [model](https://huggingface.co/microsoft/deberta-xlarge-mnli))

### Meteor
Matches words through exact matches, stemming, synonyms, and considers word order. Claims to outperform BLEU. [paper](https://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf) | [function]()

### BLEU
N-gram overlaps with brevity penalty. [paper](https://www.aclweb.org/anthology/P02-1040.pdf) | [function](https://www.nltk.org/api/nltk.translate.bleu_score.html#nltk.translate.bleu_score.sentence_bleu)

### all-mpnet-base-v2
Semantic similarity using sentence transformers. Compares generated summary directly against the source document 
(rather than reference summaries like other metrics). [model](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

### AlignScore
Factual consistency evaluation using the abstract. [paper](https://arxiv.org/abs/2305.16739) | [modified repository](https://github.com/MNikley/AlignScore)


---

## Models with Unknown Training Cutoffs

The following models have no publicly documented training-data cutoff date.
If you can confirm a date, please update `Resources/model_training_cutoffs.json`
and re-run `scripts/fetch_training_cutoffs.py` to regenerate the HTML report.

| Platform | Model | Model card / docs |
|----------|-------|-------------------|
| `ollama` | `granite4:{micro,micro-h,tiny-h,small-h}` | [IBM Granite](https://www.ibm.com/granite) |
| `ollama` | `mistral:7b` | [HuggingFace](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |
| `ollama` | `mistral-nemo:12b` | [HuggingFace](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) |
| `ollama` | `mistral-small3.2:24b` | [HuggingFace](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506) |
| `ollama` | `qwen3:{4b,8b}` | [HuggingFace](https://huggingface.co/Qwen/Qwen3-8B) · [Blog](https://qwenlm.github.io/blog/qwen3/) |
| `mistral` | `mistral-large-2411` | [Mistral docs](https://docs.mistral.ai/getting-started/models/premier/) |
| `mistral` | `mistral-medium-2505` | [Mistral docs](https://docs.mistral.ai/getting-started/models/premier/) |
| `mistral` | `mistral-medium-2508` | [Mistral docs](https://docs.mistral.ai/getting-started/models/premier/) |
| `mistral` | `mistral-small-2506` | [Mistral docs](https://docs.mistral.ai/getting-started/models/premier/) |
| `mistral` | `magistral-medium-2509` | [Mistral docs](https://docs.mistral.ai/getting-started/models/premier/) |

> **Note on estimated cutoffs:** Some models have cutoff dates marked as *estimated* rather than
> confirmed. These are inferred from the training pipeline rather than stated explicitly by the
> provider. For example, **Granite 3.3** is estimated at ~April 2024 because its
> [README](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct) attributes its dataset to
> `ibm-granite/granite-3.0-language-models` and its additional training uses only synthetic data
> (no new world-knowledge); Granite 3.0's cutoff was
> [confirmed as April 2024](https://github.com/orgs/ibm-granite/discussions/18) by an IBM
> maintainer, but no explicit cutoff has been published for 3.3 itself.

---

## Utility Scripts

| Script | Description |
|--------|-------------|
| `scripts/fetch_training_cutoffs.py` | Collects model training-data cutoff dates from multiple sources (community dataset, HuggingFace model cards, provider docs) and generates a JSON + interactive HTML report. |
| `scripts/add_publication_dates.py` | Enriches the gold-standard dataset with publication dates fetched from the CrossRef API using DOIs. |
| `scripts/download_models.py` | Parses `benchmark.py` and pre-downloads all required Ollama and HuggingFace models. |

---

## Acknowledgements & Data Sources

This project builds on and gratefully acknowledges the following external resources:

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


