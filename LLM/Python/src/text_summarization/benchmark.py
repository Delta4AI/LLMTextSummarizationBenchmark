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
import logging
import pickle
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import time

import pandas as pd
from tqdm import tqdm
import nltk
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark.log'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

from text_summarization.llm_apis.ollama_client import OllamaClient
from text_summarization.llm_apis.mistral_client import MistralClient
from text_summarization.llm_apis.anthropic_client import AnthropicClient
from text_summarization.llm_apis.openai_client import OpenAIClient
from text_summarization.llm_apis.huggingface_client import HuggingFaceClient
from text_summarization.llm_apis.local_client import TextRankSummarizer, FrequencySummarizer
from text_summarization.config import MIN_WORDS, MAX_WORDS, OUTPUT_DIR, GOLD_STANDARD_DATA
from text_summarization.utilities import extract_response, get_min_max_mean_std
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
    meteor_scores: dict[str, float]
    roberta_scores: dict[str, dict[str, float]]
    deberta_scores: dict[str, dict[str, float]]
    rouge_scores: dict[str, dict[str, float]]

    def as_json(self, detailed: bool = False) -> dict[str, Any]:
        rouge = {f"{k}_{kk}": vv for k, v in self.rouge_scores.items() for kk, vv in v.items()}
        roberta = {f"{k}_{kk}": vv for k, v in self.roberta_scores.items() for kk, vv in v.items()}
        deberta = {f"{k}_{kk}": vv for k, v in self.deberta_scores.items() for kk, vv in v.items()}
        meteor = {f"meteor_{k}": v for k, v in self.meteor_scores.items()}
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
            **exec_times,
            **_variable_results
        }


class SummarizationResult:
    def __init__(self, file_name: str | Path, papers_hash: str):
        self.fn = file_name
        self.data = {}
        self._reference_embeddings = None  # TODO: implement
        self.papers_hash = papers_hash

    @property
    def reference_embeddings(self):
        if not self._reference_embeddings:
            self._reference_embeddings = self._generate_reference_embeddings()
        return self._reference_embeddings

    def _generate_reference_embeddings(self) -> dict[str, list[float]]:
        return {}

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

        script_dir = Path(__file__).parent
        if not Path(OUTPUT_DIR).is_absolute():
            self.output_dir = script_dir / OUTPUT_DIR
        else:
            self.output_dir = Path(OUTPUT_DIR)

        self.output_dir.mkdir(exist_ok=True)
        self.hashed_and_dated_output_dir = None

        self.min_words = MIN_WORDS
        self.max_words = MAX_WORDS

        self.results = None

        self.papers = []

        self.visualizer = SummarizationVisualizer(
            benchmark_ref=self
        )

        self.rouge_types = ["rouge1", "rouge2", "rougeL"]
        self.rouge_scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=True)
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

    def calculate_length_stats(self, summaries: List[str]) -> Dict:
        """Calculate length compliance statistics for a set of summaries."""
        lengths = [len(summary.split()) for summary in summaries]

        too_short = sum(1 for length in lengths if length < self.min_words)
        too_long = sum(1 for length in lengths if length > self.max_words)
        within_bounds = len(lengths) - too_short - too_long

        return {
            'total_summaries': len(summaries),
            'too_short': too_short,
            'too_long': too_long,
            'within_bounds': within_bounds,
            'too_short_pct': too_short / len(lengths) * 100 if lengths else 0,
            'too_long_pct': too_long / len(lengths) * 100 if lengths else 0,
            'within_bounds_pct': within_bounds / len(lengths) * 100 if lengths else 0,
            **get_min_max_mean_std(lengths),
            'target_min': self.min_words,
            'target_max': self.max_words,
            'all_lengths': lengths
        }

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
            file_name=self.output_dir / "db.pkl",
            papers_hash=self.papers_hash,
        )
        self.results.load()

    def run(self, platform: str, model_name: str | None = None, parameter_overrides: dict[str, Any] | None = None):
        """Run external model evaluation."""
        if model_name:
            method_name = f"{platform}_{model_name}"
        else:
            method_name = platform

        if self.results.exists(method_name):
            logger.info(f"Skipping interference for existing method: {method_name}")
            return

        if hasattr(self.api_clients[platform], 'warmup'):
            try:
                train_corpus = [f"Title: {paper.title}\n\nAbstract: {paper.abstract}" for paper in self.papers]
                self.api_clients[platform].warmup(model_name=model_name, train_corpus=train_corpus)
            except Exception as e:
                logger.error(f"Warmup failed: {e}")

        results = []
        full_responses = []
        failed_papers = []
        execution_times = []

        for paper in tqdm(self.papers, desc=f"Processing {method_name}"):
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
                    failed_papers.append(paper.id)

            except Exception as e:
                logger.error(f"Error processing paper {paper.id} with {method_name}: {e}")
                failed_papers.append(paper.id)
                continue

            finally:
                execution_times.append(time.time() - start_time)

        if failed_papers:
            logger.error(
                f"Method {method_name} failed on {len(failed_papers)} papers: {failed_papers} - skipping evaluation")
            return

        if not results:
            logger.error(f"Method {method_name} failed on all papers - skipping evaluation")
            return

        successful_papers, generated_summaries = zip(*results)
        successful_papers = list(successful_papers)
        generated_summaries = list(generated_summaries)

        logger.info(f"Method {method_name} succeeded on {len(successful_papers)}/{len(self.papers)} papers")

        all_references = [paper.summaries for paper in successful_papers]

        result = EvaluationResult(
            method_name=method_name,
            execution_times=execution_times,
            full_responses=full_responses,
            summaries=generated_summaries,
            length_stats=self.calculate_length_stats(generated_summaries),
            meteor_scores=self.calculate_meteor_score(generated_summaries, all_references),
            roberta_scores=self.calculate_bert_score(generated_summaries, all_references, "roberta-large"),
            deberta_scores=self.calculate_bert_score(generated_summaries, all_references, "microsoft/deberta-xlarge-mnli"),
            rouge_scores=self.calculate_rouge_scores(generated_summaries, all_references)
        )

        self.results.add(method_name=method_name, result=result)
        logger.info(f"Completed evaluation of {method_name}")
        self.results.save()

    def calculate_rouge_scores(self, generated: List[str], references: List[List[str]]) -> dict[str, dict[str, float]]:
        """Calculate ROUGE scores against multiple references (max score)."""
        rouge_scores = {rouge_type: [] for rouge_type in self.rouge_types}

        for gen, ref_list in zip(generated, references):
            max_scores = dict.fromkeys(self.rouge_types, 0.0)

            for ref in ref_list:
                scores = self.rouge_scorer.score(ref, gen)
                for rouge_type in self.rouge_types:
                    max_scores[rouge_type] = max(max_scores[rouge_type], scores[rouge_type].fmeasure)

            for rouge_type in self.rouge_types:
                rouge_scores[rouge_type].append(max_scores[rouge_type])

        return {
            rouge_type: get_min_max_mean_std(rouge_scores[rouge_type])
            for rouge_type in self.rouge_types
        }

    @staticmethod
    def calculate_bert_score(generated: List[str], references: List[List[str]],
                             model: str) -> dict[str, dict[str, float]]:
        """Calculate BERTScore using best reference for each generated summary."""
        best_precision = []
        best_recall = []
        best_f1 = []

        try:
            for gen, ref_list in zip(generated, references):
                if not ref_list:
                    continue

                # Calculate BERTScore against all references for this summary
                P, R, F1 = bert_score(
                    cands=[gen] * len(ref_list),
                    refs=ref_list,
                    model_type=model,
                    lang="en",
                    verbose=False
                )

                # Take the maximum scores
                best_precision.append(P.max().item())
                best_recall.append(R.max().item())
                best_f1.append(F1.max().item())

        except Exception as e:
            logger.error(f"BERTScore calculation failed: {e}")

        return {
            'precision': get_min_max_mean_std(best_precision),
            'recall': get_min_max_mean_std(best_recall),
            'f1': get_min_max_mean_std(best_f1)
        }

    def calculate_meteor_score(self, generated: List[str], references: List[List[str]]) -> dict[str, float]:
        """Calculate METEOR score using best reference for each generated summary."""
        meteor_scores = []

        try:
            for gen, ref_list in zip(generated, references):
                if not ref_list:
                    continue

                # Pre-tokenize the generated summary
                gen_tokens = gen.split()

                # For each reference, calculate the METEOR score
                ref_scores = []
                for ref in ref_list:
                    ref_tokens = ref.split()
                    score = meteor_score([ref_tokens], gen_tokens)
                    ref_scores.append(score)

                # Take the best score for this document
                best_score = max(ref_scores) if ref_scores else 0.0
                meteor_scores.append(best_score)

        except Exception as e:
            logger.error(f"METEOR calculation failed: {str(e)}")

        return get_min_max_mean_std([])

    def export(self):
        self.hashed_and_dated_output_dir = self.output_dir / self.papers_hash
        self.hashed_and_dated_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.generate_comparison_report()
        self.save_detailed_results_as_json()
        self.create_visualizations()
    
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

    def create_visualizations(self):
        """Create all visualization plots."""
        if not self.results:
            logger.warning("No results to visualize")
            return

        self.visualizer.create_all_visualizations()


def main():
    """Main execution function with length constraints."""
    benchmark = SummarizationBenchmark()
    benchmark.load_papers()
    benchmark.load_results()

    benchmark.run("local:textrank")
    benchmark.run("local:frequency")

    # https://huggingface.co/models?pipeline_tag=summarization&language=en&sort=trending
    benchmark.run("huggingface", "facebook/bart-large-cnn")
    benchmark.run("huggingface", "facebook/bart-base")
    # benchmark.run("huggingface", "google-t5/t5-base")
    # benchmark.run("huggingface", "google-t5/t5-large")
    # benchmark.run("huggingface", "csebuetnlp/mT5_multilingual_XLSum")
    # benchmark.run("huggingface", "google/pegasus-xsum")
    # benchmark.run("huggingface", "google/pegasus-large")
    # benchmark.run("huggingface", "google/pegasus-cnn_dailymail")
    # benchmark.run("huggingface", "AlgorithmicResearchGroup/led_large_16384_arxiv_summarization")

    benchmark.run("ollama", "deepseek-r1:1.5b")
    benchmark.run("ollama", "deepseek-r1:7b")
    # benchmark.run("ollama", "deepseek-r1:8b")
    # benchmark.run("ollama", "deepseek-r1:14b")
    # benchmark.run("ollama", "gemma3:1b")
    # benchmark.run("ollama", "gemma3:4b")
    # benchmark.run("ollama", "gemma3:12b")
    # benchmark.run("ollama", "granite3.3:2b")
    # benchmark.run("ollama", "granite3.3:8b")
    # benchmark.run("ollama", "llama3.1:8b")
    # benchmark.run("ollama", "llama3.2:1b")
    # benchmark.run("ollama", "llama3.2:3b")
    # benchmark.run("ollama", "meditron:7b")
    # benchmark.run("ollama", "medllama2:7b")
    # benchmark.run("ollama", "mistral:7b")
    # benchmark.run("ollama", "mistral-nemo:latest")
    # benchmark.run("ollama", "PetrosStav/gemma3-tools:4b")
    # benchmark.run("ollama", "phi3:3.8b")
    # benchmark.run("ollama", "phi4:14b")
    # benchmark.run("ollama", "phi4:latest")
    # benchmark.run("ollama", "qwen3:4b")
    # benchmark.run("ollama", "qwen3:8b")
    # benchmark.run("ollama", "taozhiyuai/openbiollm-llama-3:8b_q8_0")

    # https://platform.openai.com/docs/models
    benchmark.run("openai", "gpt-3.5-turbo")
    # benchmark.run("openai", "gpt-4.1")

    # https://docs.anthropic.com/en/docs/about-claude/models/overview
    benchmark.run("anthropic", "claude-3-5-haiku-20241022")  # fastest
    # benchmark.run("anthropic", "claude-sonnet-4-20250514")  # high intelligence, balanced performance
    # benchmark.run("anthropic", "claude-opus-4-20250514")  # most capable

    # https://docs.mistral.ai/getting-started/models/models_overview/
    benchmark.run("mistral", "mistral-medium-latest")  # frontier-class multimodal model
    # benchmark.run("mistral", "magistral-medium-latest")  # frontier-class reasoning
    # benchmark.run("mistral", "mistral-large-latest")  # top-tier large model, high complexity tasks
    # benchmark.run("mistral", "mistral-small-latest")


    # expensive
    # benchmark.run("ollama", "deepseek-r1:32b")
    # benchmark.run("ollama", "gemma3:27b")
    # benchmark.run("ollama", "llama3.3:latest")
    # benchmark.run("ollama", "PetrosStav/gemma3-tools:27b")

    # broken?
    # benchmark.run_external_model("ollama", "llama3-gradient:latest")
    # benchmark.run("ollama", "oscardp96/medcpt-query:latest")

    # generate reports and visualizations
    benchmark.export()


if __name__ == "__main__":
    main()
