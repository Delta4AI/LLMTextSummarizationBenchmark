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
