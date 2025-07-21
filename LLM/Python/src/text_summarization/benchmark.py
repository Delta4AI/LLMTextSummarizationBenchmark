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
import hashlib
import json
import pickle
from typing import  Any
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from argparse import ArgumentParser
import shutil

import pandas as pd
from tqdm import tqdm
import nltk

from exploration_utilities import get_project_root, get_logger, setup_logging
from text_summarization.summarization_utilities import extract_response, get_min_max_mean_std


OUT_DIR = get_project_root() / "Output" / "text_summarization_benchmark"
OUT_DIR.mkdir(exist_ok=True, parents=True)

GOLD_STANDARD_DATA: list[str] = [
    "Resources/text_summarization_goldstandard_data_AKI_CKD.json",
    "Resources/text_summarization_goldstandard_data_test.json"
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


@dataclass
class EvaluationResult:
    """Data class for evaluation results."""
    method_name: str
    execution_times: list[float]
    full_responses: list[str]
    summaries: list[str]
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

    def load_papers(self):
        for file in GOLD_STANDARD_DATA:
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

    def add(self, platform: str, model_name: str | None = None, parameter_overrides: dict[str, Any] | None = None):
        self.models.append((platform, model_name, parameter_overrides))
        
    def run(self):
        logger.info(f"Running benchmark for {len(self.models)} models ..")
        for idx, (platform, model_name, parameter_overrides) in enumerate(self.models):
            logger.info(f"Running model {idx+1}/{len(self.models)}: {platform} {model_name}")
            self._run(platform=platform, model_name=model_name, parameter_overrides=parameter_overrides)
    
    def _run(self, platform: str, model_name: str | None = None, parameter_overrides: dict[str, Any] | None = None):
        """Run external model evaluation."""
        if model_name:
            method_name = f"{platform}_{model_name}"
        else:
            method_name = platform

        if self.results.exists(method_name):
            logger.info(f"Skipping interference for existing method: {method_name}")
            return

        try:
            train_corpus = [f"Title: {paper.title}\n\nAbstract: {paper.abstract}" for paper in self.papers]
            self.api_clients[platform].warmup(model_name=model_name, train_corpus=train_corpus)
        except NotImplementedError:
            pass
        except Exception as e:
            logger.error(f"Warmup failed: {e}")

        results = []
        full_responses = []
        execution_times = []

        for paper in tqdm(self.papers, desc=f"Processing {method_name}"):
            time.sleep(1.5)

            start_time = time.time()
            try:
                formatted_publication_text = f"Title: {paper.title}\n\nAbstract: \n{paper.abstract}"

                response = self.api_clients[platform].summarize(
                    text=formatted_publication_text,
                    model_name=model_name,
                    system_prompt_override=None,
                    parameter_overrides=parameter_overrides
                )
                full_responses.append(response)
                summary = extract_response(response)

                if summary and summary.strip():
                    results.append((paper, summary))

                else:
                    logger.warning(f"Empty summary for paper {paper.id} with {method_name}")
                    return

            except Exception as e:
                logger.error(f"Error processing paper {paper.id} with {method_name}: {e}")
                return

            finally:
                execution_times.append(time.time() - start_time)

        try:
            self.api_clients[platform].cleanup(model_name=model_name)
        except NotImplementedError:
            pass

        successful_papers = [paper for paper, _ in results]
        generated_summaries = [summary for _, summary in results]

        all_references = [paper.summaries for paper in successful_papers]
        all_full_text_papers = [f"{p.title}\n\n{p.abstract}" for p in successful_papers]

        result = EvaluationResult(
            method_name=method_name,
            execution_times=execution_times,
            full_responses=full_responses,
            summaries=generated_summaries,
            length_stats=get_length_scores(generated_summaries, self.min_words, self.max_words),
            rouge_scores=get_rouge_scores(generated_summaries, all_references),
            roberta_scores=get_bert_scores(generated_summaries, all_references, "roberta-large"),
            deberta_scores=get_bert_scores(generated_summaries, all_references, "microsoft/deberta-xlarge-mnli"),
            meteor_scores=get_meteor_scores(generated_summaries, all_references),
            bleu_scores=get_bleu_scores(generated_summaries, all_references),
            mpnet_content_coverage_scores=get_sentence_transformer_similarity(
                generated_summaries, all_full_text_papers, "all-mpnet-base-v2")
        )
        cleanup_metrics_cache()

        self.results.add(method_name=method_name, result=result)
        self.results.save()

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
    parser = ArgumentParser(description="LLM Text Summarization Benchmark Utility")
    parser.add_argument("--clear", action="store_true", help="Clear existing benchmark results")
    args = parser.parse_args()

    benchmark = SummarizationBenchmark()
    benchmark.load_papers()
    if args.clear:
        benchmark.clear()

    benchmark.load_results()

    benchmark.add("local:textrank")
    benchmark.add("local:frequency")

    _p14 = {"max_new_tokens": SUMMARY_MAX_WORDS * 1.3, "min_new_tokens": SUMMARY_MIN_WORDS * 1.3}
    _p16 = {"max_new_tokens": SUMMARY_MAX_WORDS * 1.6, "min_new_tokens": SUMMARY_MIN_WORDS * 1.6}
    _p15 = {"max_new_tokens": SUMMARY_MAX_WORDS * 1.5, "min_new_tokens": SUMMARY_MIN_WORDS * 1.5}
    _p17 = {"max_new_tokens": SUMMARY_MAX_WORDS * 1.7, "min_new_tokens": SUMMARY_MIN_WORDS * 1.7}

    # https://huggingface.co/models?pipeline_tag=summarization&language=en&sort=trending
    benchmark.add("huggingface", "facebook/bart-large-cnn", _p16)
    benchmark.add("huggingface", "facebook/bart-base", _p16)
    benchmark.add("huggingface", "google-t5/t5-base", _p17)
    benchmark.add("huggingface", "google-t5/t5-large", _p17)
    benchmark.add("huggingface", "csebuetnlp/mT5_multilingual_XLSum", _p16)
    benchmark.add("huggingface", "google/pegasus-xsum", _p14)
    benchmark.add("huggingface", "google/pegasus-large", _p14)
    benchmark.add("huggingface", "google/pegasus-cnn_dailymail", _p14)
    benchmark.add("huggingface", "AlgorithmicResearchGroup/led_large_16384_arxiv_summarization", _p16)

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
    benchmark.add("ollama", "meditron:7b")
    benchmark.add("ollama", "medllama2:7b")
    benchmark.add("ollama", "mistral:7b")
    benchmark.add("ollama", "mistral-nemo:latest")
    benchmark.add("ollama", "PetrosStav/gemma3-tools:4b")
    benchmark.add("ollama", "phi3:3.8b")
    benchmark.add("ollama", "phi4:14b")
    benchmark.add("ollama", "phi4:latest")
    benchmark.add("ollama", "qwen3:4b")
    benchmark.add("ollama", "qwen3:8b")
    benchmark.add("ollama", "taozhiyuai/openbiollm-llama-3:8b_q8_0")

    # https://platform.openai.com/docs/models
    # "protected" models (gpt-3o, ..) need ID verification and allows openai to freely disclose personal data ..
    # https://community.openai.com/t/openai-non-announcement-requiring-identity-card-verification-for-access-to-new-api-models-and-capabilities/1230004/32
    benchmark.add("openai", "gpt-3.5-turbo")
    benchmark.add("openai", "gpt-4.1")
    benchmark.add("openai", "gpt-4.1-mini")
    benchmark.add("openai", "gpt-4o")
    benchmark.add("openai", "gpt-4o-mini")

    # https://docs.anthropic.com/en/docs/about-claude/models/overview
    benchmark.add("anthropic", "claude-3-5-haiku-20241022")  # fastest
    benchmark.add("anthropic", "claude-sonnet-4-20250514")  # high intelligence, balanced performance
    benchmark.add("anthropic", "claude-opus-4-20250514")  # most capable

    # https://docs.mistral.ai/getting-started/models/models_overview/
    benchmark.add("mistral", "mistral-medium-latest")  # frontier-class multimodal model
    benchmark.add("mistral", "magistral-medium-latest")  # frontier-class reasoning
    benchmark.add("mistral", "mistral-large-latest")  # top-tier large model, high complexity tasks
    benchmark.add("mistral", "mistral-small-latest")

    # expensive
    # benchmark.add("ollama", "deepseek-r1:32b")
    # benchmark.add("ollama", "gemma3:27b")
    # benchmark.add("ollama", "llama3.3:latest")
    # benchmark.add("ollama", "PetrosStav/gemma3-tools:27b")

    # broken?
    # benchmark.add("ollama", "llama3-gradient:latest")
    # benchmark.add("ollama", "oscardp96/medcpt-query:latest")

    # benchmark.test_token_sizes()
    benchmark.run()
    benchmark.export()


if __name__ == "__main__":
    main()
