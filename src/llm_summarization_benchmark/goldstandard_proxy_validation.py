"""
Gold-Standard Proxy Validation
==============================

Validate that the author-written highlights shipped with each publication are a
sound *reference proxy* for summarization evaluation.

The benchmark (``benchmark.py``) scores model-generated summaries against the
publication source using a battery of metrics defined in ``metrics.py``. This
script turns that battery on the gold standard itself: it treats the author
highlights (``summaries[0]``) as the "generated summary" and the publication
source (``Paper.full_text`` = title + abstract) as the reference/source — the
exact source the benchmark's factual metrics consume.

If the highlights are a valid proxy, the factual-consistency metrics (AlignScore,
SummaC, MiniCheck) and content-overlap metrics (ROUGE / BERTScore / MPNet) should
score the (highlights, source) pairs highly. This script only produces the raw
per-pair scores and descriptive statistics; interpretation thresholds are left to
the analyst.

Every benchmark metric is computed EXCEPT FactCC:
    ROUGE-1/2/L, METEOR, BLEU, BERTScore (RoBERTa + DeBERTa), MPNet cosine,
    AlignScore, SummaC, MiniCheck-FT5 (HuggingFace), MiniCheck-7B (Ollama).

The metric implementations are reused verbatim from ``metrics.py`` — nothing is
re-implemented here.

Usage (run on the workstation where the models/checkpoints/Ollama live, with the
same setup ``uv run benchmark`` requires — installed deps, AlignScore checkpoint,
pre-downloaded metric models, and a running Ollama with ``bespoke-minicheck``):

    uv run proxy-validation
    uv run proxy-validation --test 25 --only rouge alignscore

(equivalently: ``uv run python -m llm_summarization_benchmark.goldstandard_proxy_validation``)
"""
from __future__ import annotations

import os

# Mirror benchmark.py: keep HuggingFace offline/quiet by default on the
# workstation (models are pre-downloaded via scripts/download_models.py).
# Allow the caller to override by exporting these before launching.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import time
from argparse import ArgumentParser
from collections.abc import Callable
from pathlib import Path

import nltk
import pandas as pd

from utilities import get_project_root, get_logger, setup_logging

OUT_DIR = get_project_root() / "Output" / "llm_summarization_benchmark" / "goldstandard_proxy_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DATA_FILE = get_project_root() / "Resources" / "text_summarization_goldstandard_data.json"

setup_logging(OUT_DIR / "goldstandard_proxy_validation.log")
logger = get_logger(__name__)

# Imported after logging is configured so any model-load chatter is captured.
from llm_summarization_benchmark.metrics import (  # noqa: E402
    get_rouge_scores,
    get_meteor_scores,
    get_bleu_scores,
    get_bert_scores,
    get_sentence_transformer_similarity,
    get_alignscore_scores,
    get_summac_scores,
    get_minicheck_scores,
    get_minicheck_ollama_scores,
    empty_cuda_cache,
    cleanup_metrics_cache,
)
from data_models import Paper, InterferenceRunContainer  # noqa: E402


# Per-pair score keys written by each metric into ``paper.scores`` (see metrics.py),
# mapped to friendly CSV column labels. FactCC is intentionally excluded.
SCORE_KEY_LABELS: dict[str, str] = {
    "rouge1": "rouge1",
    "rouge2": "rouge2",
    "rougeL": "rougeL",
    "meteor": "meteor",
    "bleu": "bleu",
    "bert_roberta-large_precision": "bertscore_roberta_precision",
    "bert_roberta-large_recall": "bertscore_roberta_recall",
    "bert_roberta-large_f1": "bertscore_roberta_f1",
    "bert_microsoft/deberta-xlarge-mnli_precision": "bertscore_deberta_precision",
    "bert_microsoft/deberta-xlarge-mnli_recall": "bertscore_deberta_recall",
    "bert_microsoft/deberta-xlarge-mnli_f1": "bertscore_deberta_f1",
    "sentence_transformer": "mpnet_cosine",
    "alignscore": "alignscore",
    "summac": "summac",
    "minicheck_ft5": "minicheck_ft5",
    "minicheck_7b": "minicheck_7b",
}


def load_highlight_papers(data_file: Path, limit: int | None = None) -> list[Paper]:
    """Load the gold standard and build one :class:`Paper` per publication.

    The author highlight (``summaries[0]``) is stored as the sole summary so the
    metric helpers can treat it as the candidate, while ``Paper.full_text``
    (title + abstract) serves as the reference/source for every metric.
    """
    with open(data_file, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    papers: list[Paper] = []
    for i, item in enumerate(data):
        for key in ("title", "abstract", "id", "summaries"):
            if key not in item:
                raise ValueError(f"Publication {i} missing required field '{key}'")

        summaries = item["summaries"]
        if not isinstance(summaries, list) or not summaries or not isinstance(summaries[0], str):
            raise ValueError(f"Publication {i} has no usable author highlight in 'summaries[0]'")

        highlight = summaries[0].strip()
        if not highlight:
            raise ValueError(f"Publication {i} has an empty author highlight")

        papers.append(
            Paper(
                title=item["title"],
                abstract=item["abstract"],
                id=item["id"],
                summaries=[highlight],
            )
        )

        if limit and len(papers) >= limit:
            break

    logger.info(f"Loaded {len(papers)} (highlight, source) pairs from {data_file}")
    return papers


def build_metric_runners(
    generated: list[str],
    references_multi: list[list[str]],
    sources: list[str],
    irc: InterferenceRunContainer,
) -> dict[str, Callable[[], object]]:
    """Map metric name -> zero-arg callable that reuses the metrics.py helpers.

    ``references_multi`` (list[list[str]]) is used by the n-gram / embedding-pair
    metrics that accept multiple references; ``sources`` (list[str]) is used by
    the factual-consistency metrics. In both cases the reference is the
    publication source — exactly what the benchmark's factual metrics consume.
    """
    return {
        "rouge": lambda: get_rouge_scores(generated, references_multi, irc),
        "meteor": lambda: get_meteor_scores(generated, references_multi, irc),
        "bleu": lambda: get_bleu_scores(generated, references_multi, irc),
        "bertscore_roberta": lambda: get_bert_scores(generated, references_multi, "roberta-large", irc),
        "bertscore_deberta": lambda: get_bert_scores(
            generated, references_multi, "microsoft/deberta-xlarge-mnli", irc
        ),
        "mpnet": lambda: get_sentence_transformer_similarity(
            generated, sources, "all-mpnet-base-v2", irc
        ),
        "alignscore": lambda: get_alignscore_scores(generated, sources, irc),
        "summac": lambda: get_summac_scores(generated, sources, irc),
        "minicheck_ft5": lambda: get_minicheck_scores(generated, sources, irc, model_name="flan-t5-large"),
        "minicheck_7b": lambda: get_minicheck_ollama_scores(generated, sources, irc),
    }


def run_metrics(runners: dict[str, Callable[[], object]]) -> list[str]:
    """Execute each metric, logging clear per-metric progress and timing.

    Per-pair scores are written into ``paper.scores`` as a side effect (the
    metric helpers handle that). A failure in one metric is logged and skipped
    so the remaining metrics still run. Returns the list of failed metric names.
    """
    failed: list[str] = []
    total = len(runners)

    for idx, (name, runner) in enumerate(runners.items(), start=1):
        logger.info(f"[{idx}/{total}] Computing metric: {name} ...")
        start = time.time()
        try:
            aggregate = runner()
            elapsed = time.time() - start
            logger.info(f"[{idx}/{total}] Finished {name} in {elapsed:.1f}s — aggregate: {aggregate}")
        except Exception as e:  # noqa: BLE001 — one bad metric must not abort the rest
            elapsed = time.time() - start
            logger.error(f"[{idx}/{total}] Metric {name} FAILED after {elapsed:.1f}s: {e}")
            failed.append(name)
        finally:
            empty_cuda_cache(sync=True)

    return failed


def _scalar(values: list) -> float | None:
    """Collapse a paper's per-metric score list (one entry per reference) to a scalar.

    Each pair has a single reference, so the list holds one value; ``max`` keeps
    it robust if a metric ever appends more than one.
    """
    return float(max(values)) if values else None


def build_per_pair_dataframe(papers: list[Paper]) -> pd.DataFrame:
    """Assemble the per-pair score table (one row per publication)."""
    present_keys = [k for k in SCORE_KEY_LABELS if any(k in p.scores for p in papers)]

    rows = []
    for paper in papers:
        row = {
            "id": paper.id,
            "title": paper.title,
            "highlight_words": len(paper.summaries[0].split()),
            "source_words": len(paper.full_text.split()),
        }
        for key in present_keys:
            row[SCORE_KEY_LABELS[key]] = _scalar(paper.scores.get(key, []))
        rows.append(row)

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-metric summary table: count/mean/std/min/quartiles/max (+ missing count)."""
    metric_cols = [c for c in df.columns if c not in ("id", "title", "highlight_words", "source_words")]

    described = df[metric_cols].describe().T  # count, mean, std, min, 25%, 50%, 75%, max
    described["n_missing"] = [int(df[c].isna().sum()) for c in metric_cols]

    return described


def main() -> None:
    parser = ArgumentParser(
        description="Validate author highlights as a reference proxy by scoring "
        "(highlights, source) pairs with the benchmark metrics (except FactCC)."
    )
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE,
                        help="Gold standard JSON file (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR,
                        help="Directory for CSV outputs (default: %(default)s)")
    parser.add_argument("--only", nargs="+", default=None, metavar="METRIC",
                        help="Run only these metrics (names: rouge meteor bleu bertscore_roberta "
                             "bertscore_deberta mpnet alignscore summac minicheck_ft5 minicheck_7b)")
    parser.add_argument("--skip", nargs="+", default=[], metavar="METRIC",
                        help="Skip these metrics by name")
    parser.add_argument("--test", type=int, default=None, metavar="N",
                        help="Only process the first N pairs (smoke test)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # MiniCheck decomposes summaries into sentences via nltk; ROUGE/METEOR need wordnet.
    for pkg in ("punkt", "punkt_tab", "wordnet"):
        nltk.download(pkg, quiet=True)

    papers = load_highlight_papers(args.data_file, limit=args.test)
    if not papers:
        logger.error("No (highlight, source) pairs loaded — nothing to validate.")
        raise SystemExit(1)

    irc = InterferenceRunContainer(
        platform="goldstandard_proxy",
        model_name="author_highlights",
        method_name="goldstandard_proxy_author_highlights",
        papers=papers,
    )

    generated = [p.summaries[0] for p in papers]
    references_multi = [[p.full_text] for p in papers]
    sources = [p.full_text for p in papers]

    runners = build_metric_runners(generated, references_multi, sources, irc)

    selected = list(args.only) if args.only else list(runners.keys())
    unknown = [m for m in selected + args.skip if m not in runners]
    if unknown:
        raise SystemExit(f"Unknown metric name(s): {unknown}. Valid: {list(runners.keys())}")
    runners = {name: runners[name] for name in selected if name not in args.skip}

    logger.info(
        f"Validating {len(papers)} (highlight, source) pairs with metrics: {list(runners.keys())}"
    )

    start = time.time()
    failed = run_metrics(runners)
    cleanup_metrics_cache()
    logger.info(f"All metrics finished in {time.time() - start:.1f}s")
    if failed:
        logger.warning(f"The following metrics failed and are absent from the report: {failed}")

    df = build_per_pair_dataframe(papers)
    per_pair_path = args.output_dir / "goldstandard_proxy_scores_per_pair.csv"
    df.to_csv(per_pair_path, index=False)
    logger.info(f"Per-pair scores ({len(df)} rows) written to {per_pair_path}")

    summary = summarize(df)
    summary_path = args.output_dir / "goldstandard_proxy_summary.csv"
    summary.to_csv(summary_path)
    logger.info(f"Summary table written to {summary_path}")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    logger.info(
        f"\n===== Gold-standard proxy validation summary (n={len(df)}) =====\n{summary.to_string()}\n"
    )


if __name__ == "__main__":
    main()
