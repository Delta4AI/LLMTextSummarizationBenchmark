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

import json
import logging
import argparse
import pickle
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
from numpy import floating
from tqdm import tqdm
import re

from rouge_score import rouge_scorer

try:
    from bert_score import score as bert_score
except ImportError:
    print("Warning: bert_score not installed. BERTScore evaluation will be skipped.")
    bert_score = None

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
from text_summarization.llm_apis.local_client import LocalClient
from text_summarization.config import MIN_WORDS, MAX_WORDS, OUTPUT_DIR, PAPERS_DATA_FILE
from text_summarization.utilities import extract_response
from text_summarization.visualization import SummarizationVisualizer


@dataclass
class Paper:
    """Data class for scientific papers with gold-standard summaries."""
    title: str
    abstract: str
    id: str
    summaries: List[str]  # Gold-standard reference summaries


@dataclass
class EvaluationResult:
    """Data class for evaluation results."""
    method_name: str
    rouge1: float
    rouge2: float
    rougeL: float
    bert_score: float
    execution_time: float
    summaries: List[str]
    length_stats: Dict


class EvaluationMetrics:
    """Collection of evaluation metrics for multi-reference evaluation."""

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def calculate_rouge_scores(self, generated: List[str], references: List[List[str]]) -> dict[str, floating[Any]]:
        """Calculate ROUGE scores against multiple references (max score)."""
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

        for gen, ref_list in zip(generated, references):
            max_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

            for ref in ref_list:
                scores = self.rouge_scorer.score(ref, gen)
                max_scores['rouge1'] = max(max_scores['rouge1'], scores['rouge1'].fmeasure)
                max_scores['rouge2'] = max(max_scores['rouge2'], scores['rouge2'].fmeasure)
                max_scores['rougeL'] = max(max_scores['rougeL'], scores['rougeL'].fmeasure)

            rouge_scores['rouge1'].append(max_scores['rouge1'])
            rouge_scores['rouge2'].append(max_scores['rouge2'])
            rouge_scores['rougeL'].append(max_scores['rougeL'])

        return {
            'rouge1': np.mean(rouge_scores['rouge1']),
            'rouge2': np.mean(rouge_scores['rouge2']),
            'rougeL': np.mean(rouge_scores['rougeL'])
        }

    @staticmethod
    def calculate_bert_score(generated: List[str], references: List[List[str]]) -> float:
        """Calculate BERTScore using best reference for each generated summary."""
        if bert_score is None:
            logger.warning("BERTScore not available")
            return 0.0

        try:
            best_scores = []

            for gen, ref_list in zip(generated, references):
                if not ref_list:
                    continue

                # Calculate BERTScore against all references for this summary
                _P, _R, F1 = bert_score([gen] * len(ref_list), ref_list, lang="en", verbose=False)

                # Take the maximum F1 score
                best_scores.append(F1.max().item())

            return np.mean(best_scores) if best_scores else 0.0

        except Exception as e:
            logger.error(f"BERTScore calculation failed: {e}")
            return 0.0


class SummarizationBenchmark:
    """Main benchmarking framework with length constraint tracking."""

    def __init__(self):

        script_dir = Path(__file__).parent
        if not Path(OUTPUT_DIR).is_absolute():
            self.output_dir = script_dir / OUTPUT_DIR
        else:
            self.output_dir = Path(OUTPUT_DIR)

        self.output_dir.mkdir(exist_ok=True)

        self.min_words = MIN_WORDS
        self.max_words = MAX_WORDS

        self.evaluation_metrics = EvaluationMetrics()

        self.results_fn = self.output_dir / "results.pkl"
        self.results = {}
        self.papers = []
        self.visualizer = SummarizationVisualizer(
            output_dir=self.output_dir,
            min_words=self.min_words,
            max_words=self.max_words,
        )

        self._load_api_clients()

    def _load_api_clients(self):
        self.api_clients = {}

        for _key, _class in [
            ("ollama", OllamaClient),
            ("openai", OpenAIClient),
            ("anthropic", AnthropicClient),
            ("mistral", MistralClient),
            ("huggingface", HuggingFaceClient),
            ("local", LocalClient),
        ]:
            try:
                self.api_clients[_key] = _class()
            except Exception as e:
                logger.warning(f"Failed to load {_key} API client: {e}")

    def save_results(self):
        with open(self.results_fn, "wb") as f:
            pickle.dump(self.results, f)
        logger.info(f"Saved benchmark results to {self.results_fn}")

    def load_results_from_cache(self):
        try:
            with open(self.results_fn, "rb") as f:
                self.results = pickle.load(f)
            logger.info(f"Loaded benchmark results from {self.results_fn}")
        except FileNotFoundError:
            logger.info("No benchmark results found. Will run benchmarking.")

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
            'avg_length': np.mean(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'std_length': np.std(lengths) if lengths else 0,
            'target_min': self.min_words,
            'target_max': self.max_words,
            'all_lengths': lengths
        }

    def load_papers(self, json_file_path: str | Path):
        """Load papers from JSON file."""
        try:
            with open(json_file_path, mode='r', encoding='utf-8') as f:
                data = json.load(f)

                self.papers = []
                for i, item in enumerate(data):

                    if not all(key in item for key in ['title', 'abstract', 'id', 'summaries']):
                        raise ValueError(f"Paper {i} missing required fields (title, abstract, id, summaries)")

                    if not isinstance(item['summaries'], list) or len(item['summaries']) == 0:
                        raise ValueError(f"Paper {i} has invalid summaries field (must be non-empty list)")

                    paper = Paper(
                        title=item['title'],
                        abstract=item['abstract'],
                        id=item['id'],
                        summaries=item['summaries']
                    )
                    self.papers.append(paper)

                logger.info(f"Successfully loaded {len(self.papers)} papers from {json_file_path}")

                total_summaries = sum(len(paper.summaries) for paper in self.papers)
                avg_summaries = total_summaries / len(self.papers) if self.papers else 0
                logger.info(f"Average number of reference summaries per paper: {avg_summaries:.2f}")

        except Exception as e:
            raise ValueError(f"Failed to load papers from {json_file_path}: {e}") from e

        logger.info(f"Loaded {len(self.papers)} papers for benchmarking")

    def run(self, platform: str, model_name: str, parameter_overrides: dict[str, Any] | None = None):
        """Run external model evaluation."""
        method_name = f"{platform}_{model_name}"

        if method_name in self.results:
            logger.info(f"Skipping interference for existing method: {method_name}")
            return

        if hasattr(self.api_clients[platform], 'warmup'):
            try:
                self.api_clients[platform].warmup(model_name=model_name)
            except Exception as e:
                logger.error(f"Warmup failed: {e}")

        results = []
        start_time = time.time()
        failed_papers = []

        for paper in tqdm(self.papers, desc=f"Processing {method_name}"):
            try:
                formatted_publication_text = f"Title: {paper.title}\n\nAbstract: \n{paper.abstract}"

                summary = self.api_clients[platform].summarize(
                    text=formatted_publication_text,
                    model_name=model_name,
                    system_prompt_override=None,
                    parameter_overrides=parameter_overrides
                )
                summary = extract_response(summary)

                if summary and summary.strip():
                    results.append((paper, summary))
                else:
                    logger.warning(f"Empty summary for paper {paper.id} with {method_name}")
                    failed_papers.append(paper.id)

            except Exception as e:
                logger.error(f"Error processing paper {paper.id} with {method_name}: {e}")
                failed_papers.append(paper.id)
                continue

        if failed_papers:
            logger.error(
                f"Method {method_name} failed on {len(failed_papers)} papers: {failed_papers} - skipping evaluation")
            return

        if not results:
            logger.error(f"Method {method_name} failed on all papers - skipping evaluation")
            return

        # Convert tuple to list to fix BERTScore error
        successful_papers, generated_summaries = zip(*results)
        successful_papers = list(successful_papers)
        generated_summaries = list(generated_summaries)

        logger.info(f"Method {method_name} succeeded on {len(successful_papers)}/{len(self.papers)} papers")

        execution_time = time.time() - start_time

        length_stats = self.calculate_length_stats(generated_summaries)

        # Calculate multi-reference metrics
        all_references = [paper.summaries for paper in successful_papers]
        rouge_scores = self.evaluation_metrics.calculate_rouge_scores(
            generated_summaries, all_references
        )
        bert_score_avg = self.evaluation_metrics.calculate_bert_score(
            generated_summaries, all_references
        )

        result = EvaluationResult(
            method_name=method_name,
            rouge1=rouge_scores['rouge1'],
            rouge2=rouge_scores['rouge2'],
            rougeL=rouge_scores['rougeL'],
            bert_score=bert_score_avg,
            execution_time=execution_time,
            summaries=generated_summaries,
            length_stats=length_stats
        )

        self.results[method_name] = result
        logger.info(f"Completed evaluation of {method_name}")
        self.save_results()

    def generate_comparison_report(self):
        """Generate comparison report with length compliance statistics."""
        comparison_data = []
        for method_name, result in self.results.items():
            length_stats = result.length_stats

            comparison_data.append({
                'Method': method_name,
                'ROUGE-1': result.rouge1,
                'ROUGE-2': result.rouge2,
                'ROUGE-L': result.rougeL,
                'BERTScore': result.bert_score,
                'Execution Time (s)': result.execution_time,
                'Avg Length': length_stats['avg_length'],
                'Within Bounds (%)': length_stats['within_bounds_pct'],
                'Too Short (%)': length_stats['too_short_pct'],
                'Too Long (%)': length_stats['too_long_pct'],
                'Min Length': length_stats['min_length'],
                'Max Length': length_stats['max_length'],
                'Target Range': f"{self.min_words}-{self.max_words}"
            })

        df = pd.DataFrame(comparison_data)
        df = df.sort_values('ROUGE-1', ascending=False)

        # Save to file
        report_path = self.output_dir / "comparison_report.csv"
        df.to_csv(report_path, index=False)
        logger.info(f"Comparison report saved to {report_path}")

    def generate_length_compliance_report(self):
        """Generate detailed length compliance report."""
        compliance_data = []
        for method_name, result in self.results.items():
            stats = result.length_stats
            compliance_data.append({
                'Method': method_name,
                'Total_Summaries': stats['total_summaries'],
                'Within_Bounds': stats['within_bounds'],
                'Too_Short': stats['too_short'],
                'Too_Long': stats['too_long'],
                'Within_Bounds_Pct': stats['within_bounds_pct'],
                'Too_Short_Pct': stats['too_short_pct'],
                'Too_Long_Pct': stats['too_long_pct'],
                'Avg_Length': stats['avg_length'],
                'Std_Length': stats['std_length'],
                'Min_Length': stats['min_length'],
                'Max_Length': stats['max_length'],
                'Target_Min': stats['target_min'],
                'Target_Max': stats['target_max']
            })

        df = pd.DataFrame(compliance_data)
        df = df.sort_values('Within_Bounds_Pct', ascending=False)

        report_path = self.output_dir / "length_compliance_report.csv"
        df.to_csv(report_path, index=False)
        logger.info(f"Length compliance report saved to {report_path}")

    def save_detailed_results_as_json(self):
        """Save detailed results to JSON including length statistics."""
        detailed_results = {}

        for method_name, result in self.results.items():
            detailed_results[method_name] = {
                'rouge1': result.rouge1,
                'rouge2': result.rouge2,
                'rougeL': result.rougeL,
                'bert_score': result.bert_score,
                'execution_time': result.execution_time,
                'length_statistics': result.length_stats,
                'summaries': result.summaries
            }

        results_path = self.output_dir / "detailed_results.json"
        with open(results_path, mode='w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Detailed results saved to {results_path}")

    def create_visualizations(self):
        """Create all visualization plots."""
        if not self.results:
            logger.warning("No results to visualize")
            return

        self.visualizer.set_results(self.results)
        self.visualizer.create_all_visualizations()


def main():
    """Main execution function with length constraints."""
    benchmark = SummarizationBenchmark()
    benchmark.load_results_from_cache()

    papers_data_file = Path(__file__).parents[3] / PAPERS_DATA_FILE
    if not Path(papers_data_file).exists():
        logger.error(f"Data file not found: {papers_data_file}")
        return

    try:
        benchmark.load_papers(papers_data_file)
    except ValueError as e:
        logger.error(f"Failed to load papers: {e}")
        return

    logger.info(f"Target length constraints: {MIN_WORDS}-{MAX_WORDS} words")

    benchmark.run("local", "textrank")
    benchmark.run("local", "textrank-simple")
    benchmark.run("local", "frequency")

    # https://huggingface.co/models?pipeline_tag=summarization
    benchmark.run("huggingface", "bart-large-cnn")
    benchmark.run("huggingface", "t5-base")

    benchmark.run("ollama", "deepseek-r1:1.5b", {})
    benchmark.run("ollama", "deepseek-r1:7b", {})
    benchmark.run("ollama", "deepseek-r1:8b", {})
    benchmark.run("ollama", "gemma3:1b", {})
    benchmark.run("ollama", "gemma3:4b", {})
    benchmark.run("ollama", "gemma3:12b", {})
    benchmark.run("ollama", "granite3.3:2b", {})
    benchmark.run("ollama", "granite3.3:8b", {})
    benchmark.run("ollama", "llama3.1:8b", {})
    benchmark.run("ollama", "llama3.2:1b", {})
    benchmark.run("ollama", "llama3.2:3b", {})
    benchmark.run("ollama", "meditron:7b", {})
    benchmark.run("ollama", "medllama2:7b", {})
    benchmark.run("ollama", "mistral:7b", {})
    benchmark.run("ollama", "mistral-nemo:latest", {})
    benchmark.run("ollama", "PetrosStav/gemma3-tools:4b", {})
    benchmark.run("ollama", "phi3:3.8b", {})
    benchmark.run("ollama", "phi4:14b", {})
    benchmark.run("ollama", "phi4:latest", {})
    benchmark.run("ollama", "qwen3:4b", {})
    benchmark.run("ollama", "qwen3:8b", {})
    benchmark.run("ollama", "taozhiyuai/openbiollm-llama-3:8b_q8_0", {})

    # https://platform.openai.com/docs/models
    benchmark.run("openai", "gpt-3.5-turbo")
    benchmark.run("openai", "gpt-4.1")

    # https://docs.anthropic.com/en/docs/about-claude/models/overview
    benchmark.run("anthropic", "claude-3-5-haiku-20241022")  # fastest
    benchmark.run("anthropic", "claude-sonnet-4-20250514")  # high intelligence, balanced performance
    benchmark.run("anthropic", "claude-opus-4-20250514")  # most capable

    # https://docs.mistral.ai/getting-started/models/models_overview/
    benchmark.run("mistral", "mistral-medium-latest")  # frontier-class multimodal model
    benchmark.run("mistral", "magistral-medium-latest")  # frontier-class reasoning
    benchmark.run("mistral", "mistral-large-latest")  # top-tier large model, high complexity tasks
    benchmark.run("mistral", "mistral-small-latest")


    # expensive
    # benchmark.run("ollama", "deepseek-r1:14b")
    # benchmark.run("ollama", "deepseek-r1:32b")
    # benchmark.run("ollama", "gemma3:27b")
    # benchmark.run("ollama", "llama3.3:latest")
    # benchmark.run("ollama", "PetrosStav/gemma3-tools:27b")

    # broken?
    # benchmark.run_external_model("ollama", "llama3-gradient:latest")
    # benchmark.run("ollama", "oscardp96/medcpt-query:latest")


    # Generate reports
    benchmark.generate_comparison_report()
    benchmark.generate_length_compliance_report()

    benchmark.create_visualizations()
    benchmark.save_detailed_results_as_json()

    logger.info(f"All results saved to: {benchmark.output_dir}")


if __name__ == "__main__":
    main()