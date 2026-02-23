# Scripts

Auxiliary scripts for dataset preparation, model management, and data-leakage
analysis.  They live outside the main benchmark pipeline and are run
independently as needed.

All scripts should be invoked from the **repository root** using `uv run`:

```bash
uv run python scripts/<script>.py [OPTIONS]
```

API keys are read from `Resources/.env` (see `Resources/example.env`).

---

## Overview

| Script | Purpose |
|--------|---------|
| [`add_publication_dates.py`](#add_publication_datespy) | Enrich the gold-standard dataset with publication dates from CrossRef |
| [`fetch_training_cutoffs.py`](#fetch_training_cutoffspy) | Collect training-data cutoff dates for every benchmarked model |
| [`check_data_leakage_cutoffs.py`](#check_data_leakage_cutoffspy) | Compare cutoff dates with article dates to quantify overlap risk |
| [`check_data_leakage_completion.py`](#check_data_leakage_completionpy) | Probe models for memorization via abstract completion |
| [`download_models.py`](#download_modelspy) | Pre-download Ollama models referenced in the benchmark |
| [`human_evaluation_server.py`](#human_evaluation_serverpy) | Web server for human evaluation of model-generated summaries |

### Typical execution order

```text
1. add_publication_dates.py      # one-time: add dates to the dataset
2. fetch_training_cutoffs.py     # one-time: gather model cutoff dates
3. download_models.py            # one-time: pull Ollama models locally
4. check_data_leakage_cutoffs.py # quick overlap analysis (no API calls)
5. check_data_leakage_completion.py  # deep memorization probe (API calls)
```

---

## add_publication_dates.py

Extends `Resources/text_summarization_goldstandard_data.json` with publication
dates fetched from the [CrossRef API](https://www.crossref.org/) using each
entry's DOI.

Adds three fields per entry: `publication_date`, `publication_date_source`, and
`publication_date_raw`.  The script is **resumable** — entries that already
have a date are skipped, and progress is saved periodically.

```bash
uv run python scripts/add_publication_dates.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input, -i` | `Resources/text_summarization_goldstandard_data.json` | Input JSON file |
| `--output, -o` | *(overwrite input)* | Output JSON file |
| `--concurrency, -c` | `10` | Max concurrent requests |
| `--save-every, -s` | `50` | Save progress every *N* entries |
| `--email, -e` | — | Email for [CrossRef polite pool](https://www.crossref.org/documentation/retrieve-metadata/rest-api/tips-for-using-the-crossref-rest-api/#pick-a-good-user-agent) (faster rate limits) |

---

## fetch_training_cutoffs.py

Collects LLM training-data cutoff dates for every model referenced in
`benchmark.py`.  Sources are tried in priority order:

1. **Curated JSON file** — user-editable, survives re-runs
2. **Community dataset** — [llm-knowledge-cutoff-dates](https://github.com/HaoooWang/llm-knowledge-cutoff-dates)
3. **HuggingFace model cards** — pattern-matched from README metadata
4. **Provider-level defaults** — OpenAI / Anthropic / Mistral docs
5. **Mark as unknown** — for manual completion later

On first run the curated JSON is seeded with a built-in knowledge base.
Subsequent runs append new models without overwriting manual edits.

**Output:** `Resources/model_training_cutoffs.json`

```bash
uv run python scripts/fetch_training_cutoffs.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--benchmark, -b` | `src/llm_summarization_benchmark/benchmark.py` | Path to benchmark.py |
| `--output, -o` | `Resources/model_training_cutoffs.json` | Output JSON path |
| `--skip-community` | — | Skip fetching the community cutoff dataset from GitHub |

---

## check_data_leakage_cutoffs.py

Compares model training-cutoff dates with benchmark article publication dates
to identify potential overlap — articles published before a model's cutoff
could theoretically be in its training data.  No API calls are made; this is a
fast, purely local analysis.

**Outputs:**
- `Output/scripts/data_leakage_cutoff_check.json`
- `Output/scripts/data_leakage_cutoff_check.html` — interactive dashboard

```bash
uv run python scripts/check_data_leakage_cutoffs.py
```

No options. Reads from `Resources/model_training_cutoffs.json` and the
gold-standard dataset.

---

## check_data_leakage_completion.py

Active memorization probe.  For each at-risk model (overlap > 21%), prompts
with the first sentence of every abstract and asks the model to continue.
The continuation is compared to the actual abstract using ROUGE and
longest-common-substring metrics.  Articles are split into *pre-cutoff*
(could be memorized) and *post-cutoff* (safe) groups; a significant gap in
scores indicates memorization.

Statistical analysis uses a one-sided Mann-Whitney U test with rank-biserial
effect size and bootstrapped 95% confidence intervals.

Results are cached per (model, article) pair so re-runs are instant.

**Outputs:**
- `Output/scripts/data_leakage_completion_probe.json`
- `Output/scripts/data_leakage_completion_probe.html` — interactive report

**Requires:** `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` in `Resources/.env` (and/or
a running Ollama instance).

### Synchronous mode (default)

Queries models concurrently via the standard APIs with per-platform rate
limiting.

```bash
uv run python scripts/check_data_leakage_completion.py
```

### Batch mode

Uses the OpenAI and Anthropic **Batch APIs** for cheaper (50% off),
high-volume processing.  The workflow is split into three steps:

```bash
# 1. Submit — builds requests, shows a plan, asks for confirmation,
#    then submits batches.  Batch IDs are saved for later retrieval.
uv run python scripts/check_data_leakage_completion.py --batch submit

# 2. Status — polls the APIs and shows the current state of each batch.
#    Re-run periodically until all batches show "completed" / "ended".
uv run python scripts/check_data_leakage_completion.py --batch status

# 3. Retrieve — downloads results from completed batches into the
#    local cache.  Incomplete batches are skipped (re-run later).
uv run python scripts/check_data_leakage_completion.py --batch retrieve

# 4. Generate reports — a normal run reads everything from cache.
uv run python scripts/check_data_leakage_completion.py
```

Batch IDs and metadata are persisted in
`Output/scripts/data_leakage_completion_batches.json`.

> **Note:** Ollama models have no batch API and are skipped in batch mode.
> Run without `--batch` to process them synchronously.

| Option | Description |
|--------|-------------|
| `--batch submit` | Submit API batch requests (OpenAI & Anthropic) |
| `--batch status` | Check the status of pending batches |
| `--batch retrieve` | Download completed results into the local cache |

---

## download_models.py

Parses `benchmark.py` and pre-downloads all referenced Ollama models via
`ollama pull`.  Run this before the benchmark to avoid download delays during
evaluation.

```bash
uv run python scripts/download_models.py
```

No options.  Must be run from the `scripts/` directory (or adjust the
relative path to `benchmark.py` inside the script).

---

## human_evaluation_server.py

Web server for collecting human quality assessments of model-generated
summaries.  Complements the automated metrics (ROUGE, BERTScore, AlignScore,
etc.) with human judgement — the gold standard for summarization evaluation.

### Evaluation framework

**Detailed mode** (default) uses the four dimensions from the
[SummEval](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00373/100686/SummEval-Re-evaluating-Summarization-Evaluation)
framework (Fabbri et al., 2021, *Transactions of the Association for
Computational Linguistics*), each rated on a 1–5 Likert scale:

| Dimension | Question posed to the reviewer |
|-----------|-------------------------------|
| **Coherence** | Is the summary well-structured and logically organized? |
| **Consistency** | Is the summary factually consistent with the source? No hallucinations? |
| **Fluency** | Is the summary grammatical, well-written, and readable? |
| **Relevance** | Does the summary capture the important information from the source? |

These same four dimensions are used by
[G-Eval](https://arxiv.org/abs/2303.16634) (Liu et al., 2023) and have become
the *de facto* standard for summarization evaluation in NLP research.

**Simple mode** (`--simple-ratings`) uses a single "Acceptability" score
(1–5), analogous to the "Overall Responsiveness" metric from DUC/TAC
evaluations.

**Side-by-side mode** (`--side-by-side`) shows all 4 model summaries
simultaneously for each paper, labeled A/B/C/D (blinded, randomised per
reviewer).  The reviewer **ranks** them from 1 (best) to 4 (worst).  This
yields only **20 ranking tasks** per reviewer instead of 80 independent
ratings, reducing fatigue and eliminating carryover bias.  Results can be
analysed with Kendall's W (inter-rater agreement) or Bradley-Terry models.

### How it works

1. The server loads the gold-standard dataset and `detailed_scores_per_paper.json`,
   then selects **one paper per journal category** (top *N* by article count)
   that has valid generated summaries from all specified models.
2. Each reviewer registers with a name and receives a unique **token URL**
   (e.g. `http://localhost:9987/evaluate?token=r_a1b2c3d4e5f6`).
3. **All reviewers evaluate the exact same papers × models** — only the
   presentation order is shuffled per reviewer (seeded by token) to reduce
   ordering bias.  Model identities are hidden (blinded evaluation).
4. Progress is saved to a **per-reviewer JSON file** in the data directory
   after every submission, so reviewers can close the browser and resume later.
5. With the default of 20 papers and 4 models this yields **80 assessments
   per reviewer** (detailed/simple) or **20 ranking tasks** (side-by-side).

### Usage

```bash
uv run python scripts/human_evaluation_server.py MODEL1 MODEL2 MODEL3 MODEL4 [OPTIONS]
```

Exactly four model names are required (as they appear in the results file).

#### Examples

```bash
# Detailed mode (SummEval 4 dimensions)
uv run python scripts/human_evaluation_server.py \
    openai_gpt-4o anthropic_claude-opus-4-20250514 \
    local:textrank ollama_gemma3:270M

# Simple mode (single acceptability score)
uv run python scripts/human_evaluation_server.py \
    openai_gpt-4o local:textrank local:frequency ollama_gemma3:1b \
    --simple-ratings

# Side-by-side ranking mode (20 tasks instead of 80)
uv run python scripts/human_evaluation_server.py \
    openai_gpt-4o anthropic_claude-opus-4-20250514 \
    local:textrank ollama_gemma3:270M \
    --side-by-side
```

| Option | Default | Description |
|--------|---------|-------------|
| `--port` | `9987` | Server port |
| `--simple-ratings` | — | Use single acceptability score instead of SummEval 4 dimensions |
| `--side-by-side` | — | Rank all 4 summaries per paper instead of rating individually |
| `--results-file` | `Output/…/detailed_scores_per_paper.json` | Path to per-paper results |
| `--goldstandard` | `Resources/text_summarization_goldstandard_data.json` | Path to gold-standard dataset |
| `--data-dir` | `Output/scripts/human_evaluation_data` | Directory for reviewer JSON files |
| `--num-papers` | `20` | Number of papers to select (one per journal category) |

### Output

Reviewer data is stored as one JSON file per reviewer in the data directory:

```
Output/scripts/human_evaluation_data/
├── reviewer_r_a1b2c3d4e5f6.json
├── reviewer_r_b7d2e4f8a9c0.json
└── ...
```

Each file contains the reviewer's assignments (shuffled order), rating mode,
and all submitted assessments with timestamps.

### References

- Fabbri, A. R., Kryściński, W., McCann, B., Xiong, C., Socher, R., & Radev, D. (2021).
  SummEval: Re-evaluating Summarization Evaluation. *Transactions of the Association
  for Computational Linguistics*, 9, 391–409.
  https://doi.org/10.1162/tacl_a_00373

- Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023).
  G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.
  *Proceedings of EMNLP 2023*.
  https://arxiv.org/abs/2303.16634
