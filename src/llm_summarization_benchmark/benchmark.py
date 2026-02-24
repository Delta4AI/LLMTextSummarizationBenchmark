"""
Scientific Paper Summarization Benchmarking Tool
==============================================

This script provides a comprehensive framework for benchmarking different
text summarization methods on scientific publications using multiple reference summaries.

Usage:
    python benchmark.py --data-file papers.json --output-dir results --min-words 15 --max-words 35

Requirements:
    uv add transformers rouge-score bert-score nltk scikit-learn
    uv add matplotlib seaborn pandas numpy tqdm torch
"""
from collections import defaultdict
from copy import deepcopy
import os

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import hashlib
import json
import pickle
from typing import Any
from dataclasses import asdict
from pathlib import Path
import time
from argparse import ArgumentParser
import shutil

import pandas as pd
import nltk

from utilities import (get_project_root, get_logger, setup_logging, extract_response, find_truncated,
                       normalize_llm_response)

OUT_DIR = get_project_root() / "Output" / "llm_summarization_benchmark"
OUT_DIR.mkdir(exist_ok=True, parents=True)

GOLD_STANDARD_DATA: list[Path] = [
    get_project_root() / "Resources" / "text_summarization_goldstandard_data.json"
]

setup_logging(OUT_DIR / "benchmark.log")
logger = get_logger(__name__)

from llm_apis import (
    BatchCache, BatchStatus, RefusalError, NoContentError, UnknownResponse,
    OllamaSummaryClient, MistralSummaryClient, AnthropicSummaryClient, OpenAISummaryClient,
    HuggingFacePipelineSummaryClient, HuggingFaceCompletionModelSummaryClient, HuggingFaceChatModelSummaryClient,
    HuggingFaceConversationalModelSummaryClient, TextRankSummarizer, FrequencySummarizer,
    SUMMARY_MIN_WORDS, SUMMARY_MAX_WORDS, TOKEN_SIZE_SAMPLE_TEXT
)

from llm_summarization_benchmark.metrics import (get_length_scores, get_meteor_scores, ROUGE_TYPES,
                                                 get_rouge_scores,
                                                 get_bert_scores, get_bleu_scores,
                                                 get_sentence_transformer_similarity,
                                                 get_alignscore_scores, get_summac_scores,
                                                 get_factcc_scores, get_minicheck_scores,
                                                 cleanup_metrics_cache, empty_cuda_cache,
                                                 METRIC_TYPES)
from llm_summarization_benchmark.visualization import SummarizationVisualizer
from data_models import RunStatus, Paper, EvaluationResult, InterferenceRunContainer


class SummarizationResult:
    def __init__(self, file_name: str | Path, papers_hash: str):
        self.fn = file_name
        self.data = {}
        self.papers_hash = papers_hash

    def load(self):
        try:
            import sys
            sys.modules['__main__'].EvaluationResult = EvaluationResult
            sys.modules['__main__'].Paper = Paper
            sys.modules['__main__'].defaultdict = defaultdict

            with open(self.fn, "rb") as f:
                self.data = pickle.load(f)
            logger.info(f"Loaded benchmark results from {self.fn}")
        except FileNotFoundError:
            logger.info("No previous benchmark results found.")

    def save(self):
        with open(self.fn, "wb") as f:
            pickle.dump(self.data, f)
        logger.info(f"Saved benchmark results to {self.fn}")

    def get(self):
        return self.data.get(self.papers_hash, {})

    def exists(self, method_name: str) -> bool:
        return self.data.get(self.papers_hash, {}).get(method_name, None) is not None

    def same_size_as(self, method_name: str, input_set_size: int) -> bool:
        result = self.data.get(self.papers_hash, {}).get(method_name)
        if result is None:
            return False
        stored_count = result.input_paper_count if result.input_paper_count is not None else len(result.full_responses)
        return stored_count == input_set_size

    def add(self, method_name: str, result: EvaluationResult):
        if not self.data.get(self.papers_hash):
            self.data[self.papers_hash] = {}

        self.data[self.papers_hash][method_name] = result


class SummarizationBenchmark:
    """Main benchmarking framework with length constraint tracking."""

    def __init__(self, reset_metric_types: list[str], reset_metrics_for_models: list[str]):

        self.output_dir = OUT_DIR

        self.db_path = self.output_dir / "benchmark.pkl"

        self.output_dir.mkdir(exist_ok=True)
        self.hashed_and_dated_output_dir = None

        self.min_words = SUMMARY_MIN_WORDS
        self.max_words = SUMMARY_MAX_WORDS

        self.results = None
        self.force_refresh = False

        self.papers = []
        self.models = []

        self.run_status: dict[str, RunStatus] = {}

        self.visualizer = SummarizationVisualizer(
            benchmark_ref=self
        )

        self.rouge_types = ROUGE_TYPES
        nltk.download('wordnet', quiet=True)

        self._load_api_clients()
        logger.info(f"Benchmark initialized with {len(self.api_clients)} API clients. "
                    f"Length constraints: {self.min_words}-{self.max_words}")

        self.reset_metric_types = reset_metric_types
        self.reset_metrics_for_models = {k: True for k in reset_metrics_for_models}

    def _load_api_clients(self):
        self.api_clients = {}

        for _key, _class in [
            ("ollama", OllamaSummaryClient),
            ("openai", OpenAISummaryClient),
            ("anthropic", AnthropicSummaryClient),
            ("mistral", MistralSummaryClient),
            ("huggingface", HuggingFacePipelineSummaryClient),
            ("huggingface:completion", HuggingFaceCompletionModelSummaryClient),
            ("huggingface:chat", HuggingFaceChatModelSummaryClient),
            ("huggingface:conversational", HuggingFaceConversationalModelSummaryClient),
            ("local:textrank", TextRankSummarizer),
            ("local:frequency", FrequencySummarizer),
        ]:
            try:
                self.api_clients[_key] = _class()
            except Exception as e:
                logger.warning(f"Failed to load {_key} API client: {e}")

    def load_papers(self, gold_standard_data: list[str] | None = None, test: int = None):
        for file in gold_standard_data:
            file_path = Path(__file__).parents[3] / file
            if not Path(file_path).exists():
                logger.error(f"Data file not found: {file_path}")
                continue

            try:
                self.append_papers(json_file_path=file_path, test=test)
            except ValueError as e:
                logger.error(f"Failed to load papers: {e}")

        logger.info(f"Loaded {len(self.papers)} papers for benchmarking. "
                    f"Average number of reference summaries per paper: {self._get_avg_summaries(self.papers):.2f}")

        self._calculate_papers_hash()
        self._init_out_dir()

    def _calculate_papers_hash(self):
        papers_data = []

        # sort papers by keys and summaries to ensure order doesnt affect hash
        for paper in self.papers:
            paper_dict = asdict(paper)
            paper_dict.pop("scores", None)  # to be in line with server results, we remove the scores dict
            paper_dict['summaries'] = sorted(paper_dict['summaries'])
            papers_data.append(paper_dict)

        json_str = json.dumps(papers_data, sort_keys=True)
        hash_obj = hashlib.sha256(json_str.encode("utf-8"))
        self.papers_hash = hash_obj.hexdigest()

        logger.info(f"Papers hash: {self.papers_hash}")

    def _init_out_dir(self):
        self.hashed_and_dated_output_dir = self.output_dir / self.papers_hash
        self.hashed_and_dated_output_dir.mkdir(parents=True, exist_ok=True)

    def clear(self):
        """Clear existing benchmark results."""
        if self.hashed_and_dated_output_dir.exists():
            logger.info(f"Clearing existing benchmark results in {self.hashed_and_dated_output_dir}")
            shutil.rmtree(self.hashed_and_dated_output_dir)

        if self.db_path.exists():
            logger.info(f"Clearing existing benchmark results in {self.db_path}")
            self.db_path.unlink()

        self.hashed_and_dated_output_dir.mkdir(parents=True, exist_ok=True)

    def append_papers(self, json_file_path: str | Path, test: int):
        """Load papers from JSON file."""
        try:
            with open(json_file_path, mode="r", encoding="utf-8") as f:
                data = json.load(f)
                papers = []

                for i, item in enumerate(data):

                    if not all(key in item for key in ['title', 'abstract', 'id', 'summaries']):
                        raise ValueError(f"Paper {i} missing required fields (title, abstract, id, summaries)")

                    if not isinstance(item['summaries'], list) or len(item['summaries']) == 0:
                        raise ValueError(f"Paper {i} has invalid summaries field (must be non-empty list)")

                    paper = Paper(
                        title=item["title"],
                        abstract=item["abstract"],
                        id=item["id"],
                        summaries=item["summaries"]
                    )
                    papers.append(paper)

                    if test and i + 1 == test:
                        break

                logger.info(f"Successfully loaded {len(papers)} papers from {json_file_path}. "
                            f"Average number of reference summaries per paper: {self._get_avg_summaries(papers):.2f}")

                self.papers.extend(papers)

        except Exception as e:
            raise ValueError(f"Failed to load papers from {json_file_path}: {e}") from e

    @staticmethod
    def _get_avg_summaries(papers: list[Paper]):
        total_summaries = sum(len(paper.summaries) for paper in papers)
        return total_summaries / len(papers) if papers else 0

    def load_results(self):
        self.results = SummarizationResult(
            file_name=self.db_path,
            papers_hash=self.papers_hash,
        )
        self.results.load()

    def add(self, platform: str, model_name: str | None = None, model_param_overrides: dict[str, Any] | None = None,
            tokenizer_param_overrides: dict[str, Any] | None = None, clear_api_cache: bool = False,
            clear_metrics: bool = False, batch: bool = False):

        if self.force_refresh:
            clear_api_cache = True
            clear_metrics = True

        if f"{platform}_{model_name}" in self.reset_metrics_for_models:
            clear_metrics = True

        self._clear_cache(platform=platform, model_name=model_name,
                          clear_api_cache=clear_api_cache, clear_metrics=clear_metrics)

        self.models.append((platform, model_name, model_param_overrides, tokenizer_param_overrides, batch))
        self.run_status[f"{platform}_{model_name}" if model_name else platform] = RunStatus.NOT_STARTED

    def get_status(self):
        run_status_counts = defaultdict(int)

        logger.info(f"{'method_name':<80}{'status':<12}")

        for k, v in self.run_status.items():
            logger.info(f"{k:<80}{v.value:<12}")
            run_status_counts[v.value] += 1

        for k, v in run_status_counts.items():
            logger.info(f"{k:<12}{v:<6}")

    def run_retry_loop(self, max_retries: int = 5):
        retries = 0
        while any(k == RunStatus.FAILED for k in self.run_status.values()):
            if retries >= max_retries:
                logger.info(f"Aborting retry loop after {max_retries} retries")
                break

            retries += 1
            wait_time = min(10 * (2 ** (retries - 1)), 300)
            logger.info(f"Retry attempt {retries}/{max_retries}, waiting {wait_time} seconds ..")
            time.sleep(wait_time)
            self.run()
            self.get_status()

    def _clear_cache(self, platform: str, model_name: str,
                     clear_api_cache: bool = False, clear_metrics: bool = False) -> None:
        method_name = f"{platform}_{model_name}" if model_name else platform

        # TODO: remove hardcoded unclean at some point
        metric_type_map = {
            "rouge_scores": ["rouge1", "rouge2", "rougeL"],
            "roberta_scores": ["bert_roberta-large_precision", "bert_roberta-large_recall",
                               "bert_roberta-large_f1"],
            "deberta_scores": ["bert_microsoft/deberta-xlarge-mnli_precision",
                               "bert_microsoft/deberta-xlarge-mnli_recall",
                               "bert_microsoft/deberta-xlarge-mnli_f1"],
            "meteor_scores": ["meteor"],
            "bleu_scores": ["bleu"],
            "mpnet_content_coverage_scores": ["sentence_transformer"],
            "alignscore_scores": ["alignscore"],
            "summac_scores": ["summac"],
            "factcc_scores": ["factcc"]
        }

        if clear_api_cache:
            self.api_clients[platform].clean_cache(method_name=method_name)
            logger.info(f"Cleared API cache for {method_name}")

        if clear_metrics:
            try:
                if (self.results and self.papers_hash in self.results.data and method_name in self.results.data[
                    self.papers_hash]):
                    if self.reset_metric_types == METRIC_TYPES:
                        del self.results.data[self.papers_hash][method_name]
                        self.results.save()
                        logger.info(f"Cleared all metric results for {method_name}")
                    else:
                        for metric in self.reset_metric_types:
                            existing_result = self.results.data[self.papers_hash][method_name]
                            setattr(existing_result, metric, {})
                            for paper in existing_result.full_paper_details:
                                for k in metric_type_map.get(metric, []):
                                    if k in paper.scores:
                                        del paper.scores[k]
                                    else:
                                        logger.warning(f"Could not clear {metric} for {method_name} as it is not "
                                                       f"existent yet")
                            logger.info(f"Cleared {metric} metric results for {method_name}")
                        self.results.save()


            except Exception as exc:
                logger.warning(f"Failed to clear results database for {method_name}: {exc}")

    def run(self, reset_metrics: bool = False):
        logger.info(f"Running benchmark for {len(self.models)} models ..")

        for idx, (platform, model_name, model_param_overrides, tokenizer_param_overrides,
                  batch) in enumerate(self.models):
            logger.info(f"Running model {idx+1}/{len(self.models)}: {platform} {model_name}")
            _method_name = f"{platform}_{model_name}" if model_name else platform

            irc = InterferenceRunContainer(
                platform=platform,
                model_name=model_name,
                method_name = _method_name,
                model_param_overrides=model_param_overrides,
                tokenizer_param_overrides=tokenizer_param_overrides,
                papers=deepcopy(self.papers),
            )

            skip_inference = self.results.exists(irc.method_name) and self.results.same_size_as(irc.method_name,
                                                                                                len(self.papers))

            needs_metric_recalc = _method_name in self.reset_metrics_for_models

            # Check if any requested metric types are missing on the existing result
            # Uses __dict__ lookup to bypass __getattr__ fallback which returns placeholder values
            needs_missing_metrics = False
            if skip_inference and self.reset_metric_types != METRIC_TYPES:
                existing = self.results.data.get(self.papers_hash, {}).get(irc.method_name)
                if existing is not None:
                    for metric_type in self.reset_metric_types:
                        val = existing.__dict__.get(metric_type)
                        if val is None or not val:
                            needs_missing_metrics = True
                            break

            if skip_inference and not reset_metrics and not needs_metric_recalc and not needs_missing_metrics:
                logger.info(f"Skipping interference and metrics for existing method: {irc.method_name}")
                self.run_status[_method_name] = RunStatus.SKIPPED
                continue

            try:
                if skip_inference and (reset_metrics or needs_metric_recalc or needs_missing_metrics):
                    logger.info(f"Recalculating metrics for existing method: {irc.method_name}")
                    result = self._recalculate_metrics_from_cache(irc=irc)
                elif batch:
                    result = self._check_batch(irc=irc)
                else:
                    logger.info(f"Running interference and calculating metrics for method: {irc.method_name}")
                    result = self._run_interference_and_calculate_metrics(irc=irc)
                    if result is None:
                        self.run_status[_method_name] = RunStatus.NO_RESULTS
                        continue
            except Exception as e:
                logger.error(f"Failed to run interference and calculating metrics for method: {irc.method_name}: {e}")
                logger.error("Re-run the benchmark to retry. Pipeline will continue now.")
                self.run_status[_method_name] = RunStatus.FAILED
                continue

            if result:
                self.results.add(method_name=irc.method_name, result=result)
                self.results.save()
                self.run_status[_method_name] = RunStatus.OK

            self._cleanup(run_params=irc)

            cleanup_metrics_cache()

    def _recalculate_metrics_from_cache(self, irc: InterferenceRunContainer) -> EvaluationResult:
        existing = self.results.data[self.papers_hash][irc.method_name]
        generated_summaries = [extract_response(r) for r in existing.full_responses]
        reference_summaries = [p.summaries for p in irc.papers]

        return self._get_evaluation_result(
            irc=irc,
            generated_summaries=generated_summaries,
            reference_summaries=reference_summaries,
            existing_data=existing
        )

    def _run_interference_and_calculate_metrics(self, irc: InterferenceRunContainer) -> EvaluationResult | None:
        try:
            self._run_batched_interference(run_params=irc)
        except NotImplementedError:
            try:
                self._run_sequential_interference(run_params=irc)
            except Exception as e:
                logger.error(f"Interference failed for {irc.method_name}: {e}")
                return None

        original_count = len(irc.papers)
        irc.papers = [p for p in irc.papers if p.extracted_response is not None]
        deleted_count = original_count - len(irc.papers)

        if deleted_count > 0:
            logger.warning(f"Skipping {deleted_count} papers with no response for {irc.method_name}")

        generated_summaries = [p.extracted_response for p in irc.papers]
        reference_summaries = [p.summaries for p in irc.papers]

        return self._get_evaluation_result(
            irc=irc,
            generated_summaries=generated_summaries,
            reference_summaries=reference_summaries,
            existing_data=None,
            input_paper_count=original_count
        )

    def _needs_recalc(self, metric_attr: str, existing_data: EvaluationResult | None) -> bool:
        if existing_data is None:
            empty_cuda_cache(sync=True)
            return True

        if metric_attr in self.reset_metric_types:
            empty_cuda_cache(sync=True)
            return True

        return False

    def _get_evaluation_result(
            self, irc: InterferenceRunContainer,
            generated_summaries: list[str],
            reference_summaries: list[list[str]],
            existing_data: EvaluationResult | None,
            input_paper_count: int | None = None
    ) -> EvaluationResult:

        if existing_data:
            _execution_times = existing_data.execution_times
            _full_responses = existing_data.full_responses
            _input_tokens = [_ for _ in existing_data.input_tokens if _ is not None]
            _output_tokens = [_ for _ in existing_data.output_tokens if _ is not None]
        else:
            _execution_times = [p.execution_time for p in irc.papers]
            _full_responses = [p.raw_response for p in irc.papers]
            _input_tokens = [p.input_tokens for p in irc.papers if p.input_tokens is not None]
            _output_tokens = [p.output_tokens for p in irc.papers if p.output_tokens is not None]

        _length_stats = get_length_scores(generated_summaries, self.min_words, self.max_words)

        if self._needs_recalc("rouge_scores", existing_data):
            _rouge_scores = get_rouge_scores(generated_summaries, reference_summaries, irc)
        else:
            _rouge_scores = existing_data.rouge_scores

        if self._needs_recalc("roberta_scores", existing_data):
            _roberta_scores = get_bert_scores(generated_summaries, reference_summaries, "roberta-large", irc)
        else:
            _roberta_scores = existing_data.roberta_scores

        if self._needs_recalc("deberta_scores", existing_data):
            _deberta_scores = get_bert_scores(generated_summaries, reference_summaries, "microsoft/deberta-xlarge-mnli",
                                              irc)
        else:
            _deberta_scores = existing_data.deberta_scores

        if self._needs_recalc("meteor_scores", existing_data):
            _meteor_scores = get_meteor_scores(generated_summaries, reference_summaries, irc)
        else:
            _meteor_scores = existing_data.meteor_scores

        if self._needs_recalc("bleu_scores", existing_data):
            _bleu_scores = get_bleu_scores(generated_summaries, reference_summaries, irc)
        else:
            _bleu_scores = existing_data.bleu_scores

        if self._needs_recalc("mpnet_content_coverage_scores", existing_data):
            _mpnet_content_coverage_scores = get_sentence_transformer_similarity(
                generated=generated_summaries,
                source_documents=[p.full_text for p in irc.papers],
                model_name="all-mpnet-base-v2",
                irc=irc
            )
        else:
            _mpnet_content_coverage_scores = existing_data.mpnet_content_coverage_scores

        if self._needs_recalc("alignscore_scores", existing_data):
            _alignscore_scores = get_alignscore_scores(
                generated=generated_summaries,
                references=[p.abstract for p in irc.papers],
                irc=irc
            )
        else:
            _alignscore_scores = existing_data.alignscore_scores

        if self._needs_recalc("summac_scores", existing_data):
            _summac_scores = get_summac_scores(
                generated=generated_summaries,
                references=[p.full_text for p in irc.papers],
                irc=irc
            )
        else:
            _summac_scores = existing_data.summac_scores

        if self._needs_recalc("factcc_scores", existing_data):
            _factcc_scores = get_factcc_scores(
                generated=generated_summaries,
                references=[p.full_text for p in irc.papers],
                irc=irc
            )
        else:
            _factcc_scores = existing_data.factcc_scores

        if self._needs_recalc("minicheck_ft5_scores", existing_data):
            _minicheck_ft5_scores = get_minicheck_scores(
                generated=generated_summaries,
                references=[p.abstract for p in irc.papers],
                irc=irc,
                model_name="flan-t5-large"
            )
        else:
            _minicheck_ft5_scores = existing_data.minicheck_ft5_scores

        if self._needs_recalc("minicheck_7b_scores", existing_data):
            _minicheck_7b_scores = get_minicheck_scores(
                generated=generated_summaries,
                references=[p.abstract for p in irc.papers],
                irc=irc,
                model_name="Bespoke-MiniCheck-7B"
            )
        else:
            _minicheck_7b_scores = existing_data.minicheck_7b_scores

        _input_paper_count = input_paper_count
        if _input_paper_count is None and existing_data is not None:
            _input_paper_count = existing_data.input_paper_count
        if _input_paper_count is None:
            _input_paper_count = len(generated_summaries)

        return EvaluationResult(
            method_name=irc.method_name,
            execution_times=_execution_times,
            full_responses=_full_responses,
            summaries=generated_summaries,
            length_stats=_length_stats,
            input_tokens=_input_tokens,
            output_tokens=_output_tokens,
            rouge_scores=_rouge_scores,
            roberta_scores=_roberta_scores,
            deberta_scores=_deberta_scores,
            meteor_scores=_meteor_scores,
            bleu_scores=_bleu_scores,
            mpnet_content_coverage_scores=_mpnet_content_coverage_scores,
            alignscore_scores=_alignscore_scores,
            summac_scores=_summac_scores,
            factcc_scores=_factcc_scores,
            minicheck_ft5_scores=_minicheck_ft5_scores,
            minicheck_7b_scores=_minicheck_7b_scores,
            full_paper_details=deepcopy(irc.papers),
            input_paper_count=_input_paper_count
        )

    def _warmup(self, run_params: InterferenceRunContainer) -> None:
        time.sleep(5)
        try:
            self.api_clients[run_params.platform].warmup(
                model_name=run_params.model_name,
                train_corpus=[_.formatted_text for _ in run_params.papers]
            )
            time.sleep(5)
        except NotImplementedError:
            pass
        except Exception as e:
            logger.error(f"Warmup failed: {e}")

    def _run_batched_interference(self, run_params: InterferenceRunContainer) -> None:
        logger.info(f"Attempting to run batched interference for {run_params.method_name} ..")
        _system_prompt = self.api_clients[run_params.platform].text_summarization_system_prompt
        _method = run_params.method_name

        papers_to_process = []
        cached_count = 0

        for idx, paper in enumerate(run_params.papers):
            cache = self.api_clients[run_params.platform].load_cache(
                method_name=_method,
                system_prompt=_system_prompt,
                user_query=paper.formatted_text
            )

            if cache:
                paper.raw_response = cache["response"]
                paper.extracted_response = extract_response(cache["response"])
                paper.execution_time = cache["execution_time"]
                paper.input_tokens = cache["input_tokens"]
                paper.output_tokens = cache["output_tokens"]
                cached_count += 1
                logger.info(f"Using cached response for paper {idx + 1}/{len(run_params.papers)} in {_method}")
            else:
                papers_to_process.append((idx, paper))

        if papers_to_process:
            self._warmup(run_params=run_params)
            time.sleep(2)

            logger.info(f"Processing {len(papers_to_process)} uncached papers in batch for {_method}")
            start_time = time.time()

            _responses = self.api_clients[run_params.platform].summarize_batch(
                texts=[paper.formatted_text for _, paper in papers_to_process],
                model_name=run_params.model_name,
                system_prompt_override=None,
                parameter_overrides=run_params.model_param_overrides,
                tokenizer_overrides=run_params.tokenizer_param_overrides,
            )

            batch_completed_time = time.time() - start_time
            individual_time = batch_completed_time / len(papers_to_process)

            # responses preserve order; different lengths of inputs/responses throw an exception so using zip is safe
            for (idx, paper), response in zip(papers_to_process, _responses):
                paper.raw_response = response["summary_text"]
                paper.extracted_response = extract_response(response["summary_text"])
                paper.execution_time = individual_time
                paper.input_tokens = None
                paper.output_tokens = None

                self.api_clients[run_params.platform].save_cache(
                    method_name=_method,
                    system_prompt=_system_prompt,
                    user_query=paper.formatted_text,
                    response=paper.raw_response,
                    execution_time=paper.execution_time,
                    input_tokens=paper.input_tokens,
                    output_tokens=paper.output_tokens,
                )

        logger.info(f"Finished batched interference for {_method}. Cached: {cached_count}, "
                    f"Processed: {len(papers_to_process)}")

    def _run_sequential_interference(self, run_params: InterferenceRunContainer) -> None:
        model_start_time = time.time()
        _system_prompt = self.api_clients[run_params.platform].text_summarization_system_prompt
        _method = run_params.method_name

        warmed_up = False

        for idx, paper in enumerate(run_params.papers):
            _idx = f"{idx + 1}/{len(run_params.papers)}"

            try:
                cache = self.api_clients[run_params.platform].load_cache(
                    method_name=_method,
                    system_prompt=_system_prompt,
                    user_query=paper.formatted_text
                )

                if cache:
                    paper.raw_response = cache["response"]
                    paper.extracted_response = extract_response(cache["response"])
                    paper.execution_time = cache["execution_time"]
                    paper.input_tokens = cache["input_tokens"]
                    paper.output_tokens = cache["output_tokens"]
                    logger.info(f"Using Cached response {_idx} for {_method}")
                    continue

                if not warmed_up:
                    self._warmup(run_params=run_params)
                    warmed_up = True

                logger.info(f"Running sequential interference {_idx} for {_method}")
                if run_params.platform in ["openai", "anthropic", "mistral"]:
                    time.sleep(2)  # ensure not to exceed 5 requests per minute

                start_time = time.time()

                try:
                    _raw_response, _input_tokens, _output_tokens = self.api_clients[run_params.platform].summarize(
                        text=paper.formatted_text,
                        model_name=run_params.model_name,
                        system_prompt_override=None,
                        parameter_overrides=run_params.model_param_overrides,
                        tokenizer_overrides=run_params.tokenizer_param_overrides,
                    )
                except RefusalError:
                    _raw_response = "INSUFFICIENT_FINDINGS"
                    _input_tokens = None
                    _output_tokens = None

                _execution_time = time.time() - start_time

                paper.raw_response = _raw_response
                paper.extracted_response = extract_response(_raw_response)
                paper.execution_time = _execution_time
                paper.input_tokens = _input_tokens
                paper.output_tokens = _output_tokens

                self.api_clients[run_params.platform].save_cache(
                    method_name=_method,
                    system_prompt=_system_prompt,
                    user_query=paper.formatted_text,
                    response=_raw_response,
                    execution_time=_execution_time,
                    input_tokens=_input_tokens,
                    output_tokens=_output_tokens,
                )

            except RefusalError:
                logger.warning(f"Interference {_idx} refused for {_method}")
            except NoContentError:
                logger.warning(f"Interference {_idx} returned no content for {_method}")
            except UnknownResponse:
                logger.error(f"Interference {_idx} returned unknown response for {_method}")
                raise
            except Exception as exc:
                logger.error(f"Interference {_idx} aborted for {_method} - {exc}")
                raise

        logger.info(f"Finished sequential interference for {_method} in {time.time() - model_start_time:.2f}s")

    def _cleanup(self, run_params: InterferenceRunContainer) -> None:
        try:
            self.api_clients[run_params.platform].cleanup(model_name=run_params.model_name)
        except NotImplementedError:
            pass

    def test_token_sizes(self):
        logger.info(f"Testing token sizes for {len(self.models)} models ..")

        out_fn = self.hashed_and_dated_output_dir / "token_sizes.csv"

        with open(out_fn, mode="w", encoding="utf-8") as f:
            f.write("Platform,Model Name,Words,Tokens,Ratio\n")
            logger.info(f"{'Platform':<12}|{'Model Name':<80}|{'Words':<8}|{'Tokens':<8}|{'Ratio':<8}")
            text = TOKEN_SIZE_SAMPLE_TEXT
            words = len(text.split())

            for platform, model_name, parameter_overrides, *unused in self.models:
                try:
                    token_size = self.api_clients[platform].test_token_size(model_name=model_name, text=text)
                    ratio = token_size / words

                    logger.info(f"{platform:<12}|{model_name:<80}|{words:<8}|{token_size:<8}|{ratio:<8.2f}")
                    f.write(f"{platform},{model_name},{words},{token_size},{ratio}\n")
                except NotImplementedError:
                    continue

        logger.info(f"Token sizes written to {out_fn}")

    def export(self):
        self.save_empty_formatted_responses()
        self.save_truncated_formatted_responses()
        self.generate_comparison_report()
        self.save_detailed_results_as_json()
        self.save_detailed_scores_per_paper()
        self.save_scores_per_model()
        self.visualizer.create_all_visualizations()

    def save_empty_formatted_responses(self):
        empty_responses = defaultdict(list)
        for model, metrics in self.results.data[self.papers_hash].items():
            for paper, resp in zip(metrics.full_paper_details, metrics.full_responses):
                if paper.extracted_response == "":
                    empty_responses[model].append(resp)

        if empty_responses:
            out_fn = self.output_dir / "debug_empty_formatted_responses.json"
            with open(out_fn, mode="w", encoding="utf-8") as f:
                json.dump(empty_responses, f, indent=4, ensure_ascii=False)
            logger.warning(f"Created {out_fn} with {sum(len(_) for _ in empty_responses.values())} empty responses")

    def save_truncated_formatted_responses(self):
        all_formatted_responses = []
        threshold = 0.85

        for metrics in self.results.data[self.papers_hash].values():
            all_formatted_responses.extend([extract_response(_) for _ in metrics.full_responses])

        truncated = find_truncated(all_formatted_responses, threshold=0.85)

        if truncated:
            out_fn = self.output_dir / "debug_truncated_formatted_responses.json"
            with open(out_fn, mode="w", encoding="utf-8") as f:
                json.dump(truncated, f, indent=4, ensure_ascii=False)
            logger.warning(f"Created {out_fn} with {len(truncated)} truncated responses "
                        f"(threshold length of {threshold}% of original responses). Check file to recognize truncated "
                        f"responses and potential upgrades to summarization_utilities.extract_response() !")

    def generate_comparison_report(self):
        """Generate comparison report with length compliance statistics."""
        comparison_data = []
        for result in self.results.data[self.papers_hash].values():
            comparison_data.append(result.as_json(detailed=False))

        df = pd.DataFrame(comparison_data)
        report_path = self.hashed_and_dated_output_dir / "comparison_report.csv"
        df.to_csv(report_path, index=False)
        logger.info(f"Comparison report saved to {report_path}")

    def save_detailed_results_as_json(self):
        """Save detailed results to JSON including length statistics."""
        detailed_results = {}

        for method_name, result in self.results.data[self.papers_hash].items():
            detailed_results[method_name] = result.as_json(detailed=True)

        results_path = self.hashed_and_dated_output_dir / "detailed_results.json"
        with open(results_path, mode='w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Detailed results saved to {results_path}")

    def save_detailed_scores_per_paper(self):
        file_path = self.hashed_and_dated_output_dir / "detailed_scores_per_paper.json"
        formatted_data = defaultdict(list)
        for method_name, result in  self.results.data[self.papers_hash].items():
            if not hasattr(result,  "full_paper_details"):
                logger.warning(f"benchmark.pkl is generated using an old EvaluationResult structure and should be "
                               f"re-calculated by deleting benchmark.pkl (recalculates all metrics). Detailed metric "
                               f"scores will not be exported.")
                return

            for paper in result.full_paper_details:
                formatted_data[method_name].append({
                "id": paper.id,
                "title": paper.title,
                "abstract": paper.abstract,
                "summaries": paper.summaries,
                "extracted_response": paper.extracted_response,
                "raw_response": paper.raw_response,
                "scores": {k: [float(vv) for vv in v] for k, v in paper.scores.items()}
            })

        with open(file_path, mode='w', encoding='utf-8') as f:
            f.write(json.dumps(formatted_data, indent=2, ensure_ascii=False))

        logger.info(f"Detailed metric scores per paper saved to {file_path}")

    def save_scores_per_model(self):
        file_path = self.hashed_and_dated_output_dir / "detailed_scores_per_model.json"
        formatted_data = defaultdict(lambda: defaultdict(list))

        for method_name, result in self.results.data[self.papers_hash].items():
            if not hasattr(result, "full_paper_details"):
                logger.warning(f"benchmark.pkl is generated using an old EvaluationResult structure and should be "
                               f"re-calculated by deleting benchmark.pkl (recalculates all metrics). Detailed metric "
                               f"scores will not be exported.")
                return

            for idx, paper in enumerate(result.full_paper_details):
                for metric, similarity_scores in paper.scores.items():
                    formatted_data[method_name][metric].append(float(max(similarity_scores)))

        with open(file_path, mode="w", encoding="utf-8") as f:
            f.write(json.dumps(formatted_data, indent=2, ensure_ascii=False))

        logger.info(f"Detailed metric scores per model saved to {file_path}")

    def apply_token_size_hotfix(self):
        changes = False
        for model_key, results in self.results.data[self.papers_hash].items():
            if len(results.input_tokens) > 0 and all(v == 0 or v is None for v in results.input_tokens):
                results.input_tokens = []
                changes = True
            if len(results.output_tokens) > 0 and all(v == 0 or v is None for v in results.output_tokens):
                results.output_tokens = []
                changes = True

            if changes:
                logger.info(f"Applying token size hotfix for model {model_key}")
                self.results.save()

            changes = False

    def _check_batch(self, irc: InterferenceRunContainer) -> EvaluationResult | None:
        if irc.platform not in ["mistral"]:
            logger.error(f"Model {irc.model_name} is not supported on platform {irc.platform} for batch processing")
            return None

        cache: BatchCache = self.api_clients[irc.platform].load_batch_cache(method_name=irc.model_name,
                                                                            papers_hash=self.papers_hash)

        if cache.status == BatchStatus.NOT_STARTED:
            logger.info(f"Starting batch job for model {irc.model_name} on platform {irc.platform} ..")
            job_id = self.api_clients[irc.platform].submit_batch(
                model_name=irc.model_name,
                system_prompt_override=None,
                parameter_overrides=None,
                prompts=[_.formatted_text for _ in self.papers],
                custom_ids=[_.id for _ in self.papers],
                papers_hash=self.papers_hash
            )
            cache.status = BatchStatus.PENDING
            cache.batch_uuid = job_id
            self.api_clients[irc.platform].save_batch_cache(method_name=irc.model_name, papers_hash=self.papers_hash,
                                                            batch_cache=cache)
            return None

        elif cache.status != BatchStatus.COMPLETED:
            logger.info(f"Checking batch job {cache.batch_uuid} ..")
            batch_data = self.api_clients[irc.platform].get_batch(job_id=cache.batch_uuid)
            _batch_job, _results, _errors = batch_data["batch_job"], batch_data["results"], batch_data["errors"]
            logger.info(f"Batch {cache.batch_uuid} on platform {irc.platform} returned status: {_batch_job.status}")

            if _batch_job.status == "SUCCESS":
                logger.info(f"Batch {cache.batch_uuid} completed successfully. Saving to cache.")
                cache.status = BatchStatus.COMPLETED
                cache.duration = _batch_job.completed_at - _batch_job.created_at
                cache.results = _results
                cache.errors = _errors
                self.api_clients[irc.platform].save_batch_cache(
                    method_name=irc.model_name, papers_hash=self.papers_hash, batch_cache=cache)
            else:
                return None

        if cache.status == BatchStatus.COMPLETED:
            self._update_papers_from_batch(irc=irc, cache=cache)
            existing = self.results.data.get(self.papers_hash, {}).get(irc.method_name)
            return self._get_evaluation_result(
                irc=irc,
                generated_summaries=[p.extracted_response for p in irc.papers],
                reference_summaries=[p.summaries for p in irc.papers],
                existing_data=existing
            )

        return None

    def _update_papers_from_batch(self, irc: InterferenceRunContainer, cache: BatchCache) -> None:
        _paper_id_to_paper = {v.id: v for v in irc.papers}
        _execution_time = cache.duration
        _exec_time_per_paper = _execution_time / len(irc.papers)

        for resp in cache.results or []:
            _paper = _paper_id_to_paper[resp["custom_id"]]
            if not _paper:
                logger.warning(f"Paper {_paper.id} not found in cache.")
                continue

            _r = resp["response"]
            if _r["status_code"] != 200:
                logger.warning(f"Paper {_paper.id} returned invalid status code: {_r['status_code']}")
                continue

            _body = _r["body"]

            _paper.input_tokens = _body["usage"]["prompt_tokens"]
            _paper.output_tokens = _body["usage"]["completion_tokens"]
            _paper.execution_time = _exec_time_per_paper

            _response = normalize_llm_response(response_body=_body)

            _paper.raw_response = _response["raw_response"]
            _paper.extracted_response = extract_response(_response["answer"])

        for resp in cache.errors or []:
            raise NotImplementedError(f"Investigate why errors are thrown and handle accordingly: {resp}")

def main():
    """Main execution function with length constraints."""
    parser = ArgumentParser(description="LLM Text Summarization Benchmark. Set HF_HUB_OFFLINE to 0 on first run.")
    parser.add_argument("--clear", action="store_true",
                        help="Clear all existing benchmark results (responses and metrics)")
    parser.add_argument("--reset-all-metrics", action="store_true",
                        help="Reset all metrics while keeping LLM responses")
    parser.add_argument("--reset-metrics-for-models", default=[], type=str, nargs="+",
                        help="Reset metrics for individual models (space separated) (e.g. ollama_gemma3:270M")
    parser.add_argument("--reset-metric-types", default=METRIC_TYPES, nargs="+",
                        help=f"Space separated list of metric types to reset (default: {' '.join(METRIC_TYPES)})")
    parser.add_argument("--gold-standard-data", default=GOLD_STANDARD_DATA, nargs="+",
                        help="Gold standard data files to load (default: %(default)s)")
    parser.add_argument("--test", type=int, default=None,
                        help="Run tests with specified number of publications (e.g. --test 10)")
    args = parser.parse_args()

    benchmark = SummarizationBenchmark(reset_metric_types=args.reset_metric_types,
                                       reset_metrics_for_models=args.reset_metrics_for_models)
    benchmark.load_papers(gold_standard_data=args.gold_standard_data, test=args.test)
    if args.clear:
        benchmark.clear()
        benchmark.force_refresh = True

    benchmark.load_results()

    benchmark.add("local:textrank")
    benchmark.add("local:frequency")

    _p14 = {"max_new_tokens": int(SUMMARY_MAX_WORDS * 1.3), "min_new_tokens": int(SUMMARY_MIN_WORDS * 1.3)}
    _p16 = {"max_new_tokens": int(SUMMARY_MAX_WORDS * 1.6), "min_new_tokens": int(SUMMARY_MIN_WORDS * 1.6)}
    _p15 = {"max_new_tokens": int(SUMMARY_MAX_WORDS * 1.5), "min_new_tokens": int(SUMMARY_MIN_WORDS * 1.5)}
    _p17 = {"max_new_tokens": int(SUMMARY_MAX_WORDS * 1.7), "min_new_tokens": int(SUMMARY_MIN_WORDS * 1.7)}

    # https://huggingface.co/models?pipeline_tag=summarization&language=en&sort=trending
    benchmark.add("huggingface", "facebook/bart-large-cnn", _p16)
    benchmark.add("huggingface", "facebook/bart-base", _p16)
    benchmark.add("huggingface", "google-t5/t5-base", _p17)
    benchmark.add("huggingface", "google-t5/t5-large", _p17)
    benchmark.add("huggingface", "csebuetnlp/mT5_multilingual_XLSum", _p16)
    benchmark.add("huggingface", "google/pegasus-xsum", _p14)
    benchmark.add("huggingface", "google/pegasus-large", _p14)
    benchmark.add("huggingface", "google/pegasus-cnn_dailymail", _p14)
    benchmark.add("huggingface", "AlgorithmicResearchGroup/led_large_16384_arxiv_summarization",
                  _p16)
    benchmark.add("huggingface", "google/pegasus-pubmed", _p14)
    benchmark.add("huggingface", "google/bigbird-pegasus-large-pubmed", _p14)

    benchmark.add("huggingface:completion", "microsoft/biogpt",
                  {"num_beams": 5, "early_stopping": True})
    benchmark.add("huggingface:chat", "swiss-ai/Apertus-8B-Instruct-2509",
                  tokenizer_param_overrides={"add_special_tokens": False, "truncation": False, }
                  )
    benchmark.add("huggingface:chat", "Uni-SMART/SciLitLLM1.5-7B", _p14)
    benchmark.add("huggingface:chat", "Uni-SMART/SciLitLLM1.5-14B", _p14)
    benchmark.add("huggingface:chat", "aaditya/OpenBioLLM-Llama3-8B", _p14)
    benchmark.add("huggingface:conversational", "BioMistral/BioMistral-7B", _p14)

    benchmark.add("ollama", "deepseek-r1:1.5b")
    benchmark.add("ollama", "deepseek-r1:7b")
    benchmark.add("ollama", "deepseek-r1:8b")
    benchmark.add("ollama", "deepseek-r1:14b")
    benchmark.add("ollama", "gemma3:270M")
    benchmark.add("ollama", "gemma3:1b")
    benchmark.add("ollama", "gemma3:4b")
    benchmark.add("ollama", "gemma3:12b")
    benchmark.add("ollama", "granite3.3:2b")
    benchmark.add("ollama", "granite3.3:8b")
    benchmark.add("ollama", "llama3.1:8b")
    benchmark.add("ollama", "llama3.2:1b")
    benchmark.add("ollama", "llama3.2:3b")
    benchmark.add("ollama", "medllama2:7b")
    benchmark.add("ollama", "mistral:7b")
    benchmark.add("ollama", "mistral-nemo:12b")
    benchmark.add("ollama", "mistral-small3.2:24b")
    benchmark.add("ollama", "PetrosStav/gemma3-tools:4b")
    benchmark.add("ollama", "phi3:3.8b")
    benchmark.add("ollama", "phi4:14b")
    benchmark.add("ollama", "qwen3:4b")
    benchmark.add("ollama", "qwen3:8b")
    # benchmark.add("ollama", "taozhiyuai/openbiollm-llama-3:8b_q8_0")  # removed from ollama.com, super bad performance
    # benchmark.add("ollama", "koesn/llama3-openbiollm-8b:q4_K_M")  # moved to HF
    # benchmark.add("ollama", "adrienbrault/biomistral-7b:Q4_K_M")  # moved to HF
    benchmark.add("ollama", "gpt-oss:20b")
    benchmark.add("ollama", "granite4:micro")
    benchmark.add("ollama", "granite4:micro-h")
    benchmark.add("ollama", "granite4:tiny-h")
    benchmark.add("ollama", "granite4:small-h")

    # https://platform.openai.com/docs/models
    # "protected" models (gpt-3o, ..) need ID verification and allows openai to freely disclose personal data ..
    # https://community.openai.com/t/openai-non-announcement-requiring-identity-card-verification-for-access-to-new-api-models-and-capabilities/1230004/32
    benchmark.add("openai", "gpt-3.5-turbo")  # gpt-3.5-turbo-0125
    benchmark.add("openai", "gpt-4.1")  # gpt-4.1-2025-04-14
    benchmark.add("openai", "gpt-4.1-mini")  # gpt-4.1-mini-2025-04-14
    benchmark.add("openai", "gpt-4o")  # gpt-4o-2024-08-06
    benchmark.add("openai", "gpt-4o-mini")  # gpt-4o-mini-2024-07-18
    benchmark.add("openai", "gpt-5-nano-2025-08-07")
    benchmark.add("openai", "gpt-5-mini-2025-08-07")
    benchmark.add("openai", "gpt-5-2025-08-07")

    # https://docs.anthropic.com/en/docs/about-claude/models/overview
    benchmark.add("anthropic", "claude-3-5-haiku-20241022")  # fastest
    benchmark.add("anthropic", "claude-sonnet-4-20250514")  # high intelligence, balanced performance
    benchmark.add("anthropic", "claude-opus-4-20250514")  # most capable
    benchmark.add("anthropic", "claude-opus-4-1-20250805")
    # benchmark.add("anthropic", "claude-haiku-4-5-20251001", batch=True)
    # benchmark.add("anthropic", "claude-sonnet-4-5-20250929", batch=True)

    # https://docs.mistral.ai/getting-started/models/models_overview/
    benchmark.add("mistral", "mistral-medium-2505")  # frontier-class multimodal model
    # benchmark.add("mistral", "magistral-medium-2507")  # frontier-class reasoning
    benchmark.add("mistral", "magistral-medium-2509", batch=True)
    benchmark.add("mistral", "mistral-large-2411")  # top-tier large model, high complexity tasks
    benchmark.add("mistral", "mistral-small-2506")
    # benchmark.add("mistral", "mistral-medium-2508", batch=True)

    # expensive
    # benchmark.add("ollama", "deepseek-r1:32b")
    # benchmark.add("ollama", "gemma3:27b")
    # benchmark.add("ollama", "llama3.3:latest")
    # benchmark.add("ollama", "PetrosStav/gemma3-tools:27b")
    # benchmark.add("ollama", "meditron:7b")

    # broken?
    # benchmark.add("ollama", "llama3-gradient:latest")
    # benchmark.add("ollama", "oscardp96/medcpt-query:latest")

    # benchmark.test_token_sizes()

    benchmark.run(reset_metrics=args.reset_all_metrics)


    if benchmark.papers_hash not in benchmark.results.data:
        logger.info("No results available. Exiting")
        exit(1)

    benchmark.get_status()
    benchmark.run_retry_loop()
    benchmark.apply_token_size_hotfix()
    benchmark.export()


if __name__ == "__main__":
    main()
