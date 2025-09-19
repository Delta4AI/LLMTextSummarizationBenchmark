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
import os

from llm_apis.exceptions import RefusalError, NoContentError, UnknownResponse

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import hashlib
import json
import pickle
from typing import  Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
import time
from argparse import ArgumentParser
import shutil

import pandas as pd
import nltk

from exploration_utilities import get_project_root, get_logger, setup_logging
from text_summarization.summarization_utilities import extract_response, get_min_max_mean_std


OUT_DIR = get_project_root() / "Output" / "text_summarization_benchmark"
OUT_DIR.mkdir(exist_ok=True, parents=True)

GOLD_STANDARD_DATA: list[str] = [
    "Resources/text_summarization_goldstandard_data.json"
]

setup_logging(OUT_DIR / "benchmark.log")
logger = get_logger(__name__)

from llm_apis.ollama_client import OllamaClient
from llm_apis.mistral_client import MistralClient
from llm_apis.anthropic_client import AnthropicClient
from llm_apis.openai_client import OpenAIClient
from llm_apis.huggingface_client import HuggingFaceClient
from llm_apis.local_client import TextRankSummarizer, FrequencySummarizer
from llm_apis.config import SUMMARY_MIN_WORDS, SUMMARY_MAX_WORDS, TOKEN_SIZE_SAMPLE_TEXT
from text_summarization.metrics import (get_length_scores, get_meteor_scores, ROUGE_TYPES, get_rouge_scores,
                                        get_bert_scores, get_bleu_scores, get_sentence_transformer_similarity,
                                        cleanup_metrics_cache)
from text_summarization.visualization import SummarizationVisualizer


@dataclass
class Paper:
    """Data class for scientific papers with gold-standard summaries."""
    title: str
    abstract: str
    id: str
    summaries: list[str]  # Gold-standard reference summaries
    formatted_text: str = field(init=False)
    full_text: str = field(init=False)
    raw_response: str | None = None
    extracted_response: str | None = None
    execution_time: float | None = None
    input_tokens: int | None = None  # N/A for huggingface pipeline
    output_tokens: int | None = None  # N/A for huggingface pipeline

    def __post_init__(self):
        self.formatted_text = f"Title: {self.title}\n\nAbstract: \n{self.abstract}"
        self.full_text = f"{self.title}\n\n{self.abstract}"


@dataclass
class InterferenceRunContainer:
    """Data class for run parameters."""
    platform: str
    model_name: str | None = None
    method_name: str | None = None
    parameter_overrides: dict[str, Any] | None = None
    papers: list[Paper] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Data class for evaluation results."""
    method_name: str
    execution_times: list[float]
    full_responses: list[str]
    summaries: list[str]
    input_tokens: list[int]
    output_tokens: list[int]
    length_stats: dict
    rouge_scores: dict[str, dict[str, float]]
    roberta_scores: dict[str, dict[str, float]]
    deberta_scores: dict[str, dict[str, float]]
    meteor_scores: dict[str, float]
    bleu_scores: dict[str, float]
    mpnet_content_coverage_scores: dict[str, float]

    def as_json(self, detailed: bool = False) -> dict[str, Any]:
        rouge = {f"{k}_{kk}": vv for k, v in self.rouge_scores.items() for kk, vv in v.items()}
        roberta = {f"roberta_{k}_{kk}": vv for k, v in self.roberta_scores.items() for kk, vv in v.items()}
        deberta = {f"deberta_{k}_{kk}": vv for k, v in self.deberta_scores.items() for kk, vv in v.items()}
        meteor = {f"meteor_{k}": v for k, v in self.meteor_scores.items()}
        bleu = {f"bleu_{k}": v for k, v in self.bleu_scores.items()}
        mpnet_content_coverage = {f"content_coverage_{k}": v for k, v in self.mpnet_content_coverage_scores.items()}
        _exec_times = get_min_max_mean_std(self.execution_times)
        exec_times = {f"exec_time_{k}": v for k, v in _exec_times.items()}
        _lengths = get_min_max_mean_std(self.length_stats["all_lengths"])
        lengths = {f"length_{k}": v for k, v in _lengths.items()}

        _variable_results = {
            "length_statistics": self.length_stats,
            "summaries": self.summaries,
            "full_responses": self.full_responses,
        } if detailed else {
            **lengths,
            'length_within_bounds_pct': self.length_stats['within_bounds_pct'],
            'length_too_short_pct': self.length_stats['too_short_pct'],
            'length_too_long_pct': self.length_stats['too_long_pct']
        }

        return {
            'method': self.method_name,
            **rouge,
            **roberta,
            **deberta,
            **meteor,
            **bleu,
            **mpnet_content_coverage,
            **exec_times,
            **_variable_results
        }


class SummarizationResult:
    def __init__(self, file_name: str | Path, papers_hash: str):
        self.fn = file_name
        self.data = {}
        self.papers_hash = papers_hash

    def load(self):
        try:
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
        return len(self.data.get(self.papers_hash, {}).get(method_name, {}).full_responses) == input_set_size

    def add(self, method_name: str, result: EvaluationResult):
        if not self.data.get(self.papers_hash):
            self.data[self.papers_hash] = {}

        self.data[self.papers_hash][method_name] = result


class SummarizationBenchmark:
    """Main benchmarking framework with length constraint tracking."""

    def __init__(self):

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

        self.visualizer = SummarizationVisualizer(
            benchmark_ref=self
        )

        self.rouge_types = ROUGE_TYPES
        nltk.download('wordnet', quiet=True)

        self._load_api_clients()
        logger.info(f"Benchmark initialized with {len(self.api_clients)} API clients. "
                    f"Length constraints: {self.min_words}-{self.max_words}")

    def _load_api_clients(self):
        self.api_clients = {}

        for _key, _class in [
            ("ollama", OllamaClient),
            ("openai", OpenAIClient),
            ("anthropic", AnthropicClient),
            ("mistral", MistralClient),
            ("huggingface", HuggingFaceClient),
            ("local:textrank", TextRankSummarizer),
            ("local:frequency", FrequencySummarizer),
        ]:
            try:
                self.api_clients[_key] = _class()
            except Exception as e:
                logger.warning(f"Failed to load {_key} API client: {e}")

    def load_papers(self, gold_standard_data: list[str] | None = None):
        for file in gold_standard_data:
            file_path = Path(__file__).parents[3] / file
            if not Path(file_path).exists():
                logger.error(f"Data file not found: {file_path}")
                continue

            try:
                self.append_papers(file_path)
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

    def append_papers(self, json_file_path: str | Path):
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

    def add(self, platform: str, model_name: str | None = None, parameter_overrides: dict[str, Any] | None = None,
            force_refresh: bool = False):
        if force_refresh or self.force_refresh:
            self._clear_cache(platform=platform, model_name=model_name)
        self.models.append((platform, model_name, parameter_overrides))

    def _clear_cache(self, platform: str, model_name: str) -> None:
        method_name = f"{platform}_{model_name}" if model_name else platform

        self.api_clients[platform].clean_cache(method_name=method_name)
        logger.info(f"Cleared API cache for {method_name}")

        try:
            if (self.results and self.papers_hash in self.results.data and method_name in self.results.data[
                self.papers_hash]):
                del self.results.data[self.papers_hash][method_name]
                self.results.save()
                logger.info(f"Cleared results database for {method_name}")
        except Exception as exc:
            logger.warning(f"Failed to clear results database for {method_name}: {exc}")

    def run(self):
        logger.info(f"Running benchmark for {len(self.models)} models ..")

        for idx, (platform, model_name, parameter_overrides) in enumerate(self.models):
            logger.info(f"Running model {idx+1}/{len(self.models)}: {platform} {model_name}")
            irc = InterferenceRunContainer(
                platform=platform,
                model_name=model_name,
                method_name = f"{platform}_{model_name}" if model_name else platform,
                parameter_overrides=parameter_overrides,
                papers=self.papers,
            )

            self._warmup(run_params=irc)

            time.sleep(1.5)

            try:
                self._run_batched_interference(run_params=irc)
            except NotImplementedError:
                try:
                    self._run_sequential_interference(run_params=irc)
                except Exception as e:
                    logger.error(f"Interference failed for {irc.method_name}: {e}")
                    continue

            original_count = len(irc.papers)
            irc.papers = [p for p in irc.papers if p.extracted_response is not None]
            deleted_count = original_count - len(irc.papers)

            if deleted_count > 0:
                logger.warning(f"Skipping {deleted_count} papers with no response for {irc.method_name}")

            generated_summaries = [p.extracted_response for p in irc.papers]
            reference_summaries = [p.summaries for p in irc.papers]

            result = EvaluationResult(
                method_name=irc.method_name,
                execution_times=[p.execution_time for p in irc.papers],
                full_responses=[p.raw_response for p in irc.papers],
                summaries=generated_summaries,
                length_stats=get_length_scores(generated_summaries, self.min_words, self.max_words),
                input_tokens=[p.input_tokens for p in irc.papers if p.input_tokens is not None],
                output_tokens=[p.output_tokens for p in irc.papers if p.output_tokens is not None],
                rouge_scores=get_rouge_scores(generated_summaries, reference_summaries),
                roberta_scores=get_bert_scores(generated_summaries, reference_summaries, "roberta-large"),
                deberta_scores=get_bert_scores(generated_summaries, reference_summaries, "microsoft/deberta-xlarge-mnli"),
                meteor_scores=get_meteor_scores(generated_summaries, reference_summaries),
                bleu_scores=get_bleu_scores(generated_summaries, reference_summaries),
                mpnet_content_coverage_scores=get_sentence_transformer_similarity(
                    generated=generated_summaries,
                    source_documents=[p.full_text for p in irc.papers],
                    model_name="all-mpnet-base-v2")
            )

            if result:
                self.results.add(method_name=irc.method_name, result=result)
                self.results.save()

            self._cleanup(run_params=irc)

            cleanup_metrics_cache()

    def _warmup(self, run_params: InterferenceRunContainer) -> None:
        try:
            self.api_clients[run_params.platform].warmup(
                model_name=run_params.model_name,
                train_corpus=[_.formatted_text for _ in run_params.papers]
            )
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
            logger.info(f"Processing {len(papers_to_process)} uncached papers in batch for {_method}")
            start_time = time.time()

            _responses = self.api_clients[run_params.platform].summarize_batch(
                texts=[paper.formatted_text for _, paper in papers_to_process],
                model_name=run_params.model_name,
                system_prompt_override=None,
                parameter_overrides=run_params.parameter_overrides
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

                logger.info(f"Running sequential interference {_idx} for {_method}")
                if run_params.platform in ["openai", "anthropic", "mistral"]:
                    time.sleep(2)  # ensure not to exceed 5 requests per minute

                start_time = time.time()

                _raw_response, _input_tokens, _output_tokens = self.api_clients[run_params.platform].summarize(
                    text=paper.formatted_text,
                    model_name=run_params.model_name,
                    system_prompt_override=None,
                    parameter_overrides=run_params.parameter_overrides
                )

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

            for platform, model_name, parameter_overrides in self.models:
                try:
                    token_size = self.api_clients[platform].test_token_size(model_name=model_name, text=text)
                    ratio = token_size / words

                    logger.info(f"{platform:<12}|{model_name:<80}|{words:<8}|{token_size:<8}|{ratio:<8.2f}")
                    f.write(f"{platform},{model_name},{words},{token_size},{ratio}\n")
                except NotImplementedError:
                    continue

        logger.info(f"Token sizes written to {out_fn}")

    def export(self):
        self.generate_comparison_report()
        self.save_detailed_results_as_json()
        self.visualizer.create_all_visualizations()
    
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


def main():
    """Main execution function with length constraints."""
    parser = ArgumentParser(description="LLM Text Summarization Benchmark. Set HF_HUB_OFFLINE to 0 on first run.")
    parser.add_argument("--clear", action="store_true", help="Clear existing benchmark results")
    parser.add_argument("--gold-standard-data", default=GOLD_STANDARD_DATA, nargs="+",
                        help="Gold standard data files to load (default: %(default)s)")
    args = parser.parse_args()

    benchmark = SummarizationBenchmark()
    benchmark.load_papers(args.gold_standard_data)
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
    # TODO: delete benchmark.pkl (or save as benchmark.pkl.old)
    # TODO: remove force_refresh=True after run is finished
    benchmark.add("huggingface", "facebook/bart-large-cnn", _p16, True)
    benchmark.add("huggingface", "facebook/bart-base", _p16, True)
    benchmark.add("huggingface", "google-t5/t5-base", _p17, True)
    benchmark.add("huggingface", "google-t5/t5-large", _p17, True)
    benchmark.add("huggingface", "csebuetnlp/mT5_multilingual_XLSum", _p16, True)
    benchmark.add("huggingface", "google/pegasus-xsum", _p14, True)
    benchmark.add("huggingface", "google/pegasus-large", _p14, True)
    benchmark.add("huggingface", "google/pegasus-cnn_dailymail", _p14, True)
    benchmark.add("huggingface", "AlgorithmicResearchGroup/led_large_16384_arxiv_summarization", _p16, True)

    benchmark.add("ollama", "deepseek-r1:1.5b")
    benchmark.add("ollama", "deepseek-r1:7b")
    benchmark.add("ollama", "deepseek-r1:8b")
    benchmark.add("ollama", "deepseek-r1:14b")
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
    benchmark.add("ollama", "taozhiyuai/openbiollm-llama-3:8b_q8_0")
    benchmark.add("ollama", "gpt-oss:20b")

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

    # https://docs.mistral.ai/getting-started/models/models_overview/
    benchmark.add("mistral", "mistral-medium-2505")  # frontier-class multimodal model
    benchmark.add("mistral", "magistral-medium-2507")  # frontier-class reasoning
    benchmark.add("mistral", "mistral-large-2411")  # top-tier large model, high complexity tasks
    benchmark.add("mistral", "mistral-small-2506")

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
    benchmark.run()
    benchmark.export()


if __name__ == "__main__":
    main()
