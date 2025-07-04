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
import warnings
from typing import List, Dict, Callable
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import re

# Evaluation metrics
from rouge_score import rouge_scorer

try:
    from bert_score import score as bert_score
except ImportError:
    print("Warning: bert_score not installed. BERTScore evaluation will be skipped.")
    bert_score = None

# Transformers for advanced models
try:
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM,
        pipeline, BartTokenizer, BartForConditionalGeneration,
        T5Tokenizer, T5ForConditionalGeneration
    )
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers not installed. Advanced models will be skipped.")
    TRANSFORMERS_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

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

from llm_apis.ollama import OllamaClient
from llm_apis.perplexity import PerplexityClient
from llm_apis.anthropic import AnthropicClient
from llm_apis.openai import OpenAIClient


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
    multi_ref_rouge1: float
    multi_ref_rouge2: float
    multi_ref_rougeL: float
    length_stats: Dict


class TextPreprocessor:
    """Text preprocessing utilities."""

    def __init__(self):
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Remove very short sentences
        sentences = sent_tokenize(text)
        sentences = [s for s in sentences if len(s.split()) > 5]

        return ' '.join(sentences)

    def extract_sentences(self, text: str, max_sentences: int = None) -> List[str]:
        """Extract sentences from text."""
        sentences = sent_tokenize(text)
        if max_sentences:
            sentences = sentences[:max_sentences]
        return sentences


class SummarizationMethods:
    """Collection of different summarization methods with length constraints awareness."""

    def __init__(self, target_min_words: int = 15, target_max_words: int = 35, llm_prompt: str = None):
        self.target_min_words = target_min_words
        self.target_max_words = target_max_words
        self.preprocessor = TextPreprocessor()
        self.llm_prompt = llm_prompt
        self._load_models()
        self._load_api_clients()

    def _load_models(self):
        """Load pre-trained models."""
        self.models = {}

        if TRANSFORMERS_AVAILABLE:
            try:
                # Load BART model
                self.models['bart_tokenizer'] = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
                self.models['bart_model'] = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
                logger.info("BART model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load BART model: {e}")

            try:
                # Load T5 model
                self.models['t5_tokenizer'] = T5Tokenizer.from_pretrained('t5-base')
                self.models['t5_model'] = T5ForConditionalGeneration.from_pretrained('t5-base')
                logger.info("T5 model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load T5 model: {e}")

    def _load_api_clients(self):
        self.api_clients = {}

        try:
            self.api_clients["ollama"] = OllamaClient()
        except Exception as e:
            logger.warning(f"Failed to load Ollama API client: {e}")

        try:
            self.api_clients["openai"] = OpenAIClient()
        except Exception as e:
            logger.warning(f"Failed to load OpenAI API client: {e}")

        try:
            self.api_clients["anthropic"] = AnthropicClient()
        except Exception as e:
            logger.warning(f"Failed to load Anthropic API client: {e}")

        try:
            self.api_clients["perplexity"] = PerplexityClient()
        except Exception as e:
            logger.warning(f"Failed to load Perplexity API client: {e}")


    def first_sentence_plus_title(self, text: str) -> str:
        """Baseline: Title + first sentence of abstract, length-aware."""
        lines = text.split('. ')
        if len(lines) >= 2:
            result = f"{lines[0]}. {lines[1]}"
        else:
            result = lines[0] if lines else text[:100]

        # Try to adjust to target length if possible
        words = result.split()
        if len(words) < self.target_min_words and len(lines) > 2:
            # Try to add more content
            for i in range(2, min(len(lines), 5)):
                extended = f"{result}. {lines[i]}"
                if len(extended.split()) <= self.target_max_words:
                    result = extended
                else:
                    break

        return result

    def key_sentences_extractive(self, text: str) -> str:
        """Extract key sentences based on position and content, length-aware."""
        sentences = self.preprocessor.extract_sentences(text)
        if len(sentences) <= 1:
            return text

        # Start with first sentence
        result = sentences[0]
        words_count = len(result.split())

        # Add sentences until we reach target range
        for i in range(1, len(sentences)):
            candidate = f"{result} {sentences[i]}"
            candidate_words = len(candidate.split())

            if candidate_words <= self.target_max_words:
                result = candidate
                words_count = candidate_words
                if words_count >= self.target_min_words:
                    break
            else:
                break

        return result

    def textrank_summarize(self, text: str) -> str:
        """TextRank-based extractive summarization, length-aware."""
        sentences = self.preprocessor.extract_sentences(text)
        if len(sentences) <= 1:
            return text

        # Calculate sentence scores
        sentence_scores = self._calculate_sentence_scores(sentences)

        # Sort sentences by score
        scored_sentences = list(sentence_scores.items())
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        # Select sentences to meet target length
        result_sentences = []
        total_words = 0

        for sentence, score in scored_sentences:
            sentence_words = len(sentence.split())
            if total_words + sentence_words <= self.target_max_words:
                result_sentences.append(sentence)
                total_words += sentence_words
                if total_words >= self.target_min_words:
                    break

        # Sort selected sentences by original order
        if result_sentences:
            result_sentences.sort(key=lambda x: sentences.index(x))
            return ' '.join(result_sentences)
        else:
            # Fallback to highest scoring sentence
            return scored_sentences[0][0]

    def _calculate_sentence_scores(self, sentences: List[str]) -> Dict[str, float]:
        """Calculate sentence scores for TextRank."""
        scores = {}

        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [word for word in words if word.isalnum() and word not in self.preprocessor.stop_words]

            # Simple scoring based on word frequency and length
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

            if words:
                scores[sentence] = sum(word_freq.values()) / len(words)
            else:
                scores[sentence] = 0

        return scores

    def bart_summarize(self, text: str) -> str:
        """BART-based abstractive summarization with target length guidance."""
        if 'bart_tokenizer' not in self.models or 'bart_model' not in self.models:
            logger.warning("BART model not available, falling back to extractive method")
            return self.first_sentence_plus_title(text)

        try:
            tokenizer = self.models['bart_tokenizer']
            model = self.models['bart_model']

            inputs = tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)

            # Use target lengths as guidance for generation
            target_min_tokens = max(int(self.target_min_words * 1.3), 10)  # Approximate token count
            target_max_tokens = max(int(self.target_max_words * 1.3), target_min_tokens + 5)

            with torch.no_grad():
                summary_ids = model.generate(
                    inputs,
                    max_length=target_max_tokens,
                    min_length=target_min_tokens,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )

            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary

        except Exception as e:
            logger.error(f"BART summarization failed: {e}")
            return self.first_sentence_plus_title(text)

    def t5_summarize(self, text: str) -> str:
        """T5-based abstractive summarization with target length guidance."""
        if 't5_tokenizer' not in self.models or 't5_model' not in self.models:
            logger.warning("T5 model not available, falling back to extractive method")
            return self.first_sentence_plus_title(text)

        try:
            tokenizer = self.models['t5_tokenizer']
            model = self.models['t5_model']

            # T5 requires task prefix
            input_text = f"summarize: {text}"
            inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)

            # Use target lengths as guidance
            target_min_tokens = max(int(self.target_min_words * 1.3), 10)
            target_max_tokens = max(int(self.target_max_words * 1.3), target_min_tokens + 5)

            with torch.no_grad():
                summary_ids = model.generate(
                    inputs,
                    max_length=target_max_tokens,
                    min_length=target_min_tokens,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )

            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary

        except Exception as e:
            logger.error(f"T5 summarization failed: {e}")
            return self.first_sentence_plus_title(text)

    def external_model_summarize(self, text: str, platform: str, model_name: str) -> str:
        """Generic external model summarization with length constraints passed as context."""
        try:
            return self.api_clients[platform].summarize(
                text=text,
                model_name=model_name,
                prompt=self.llm_prompt,
                fallback_summary=self.first_sentence_plus_title(text)
            )
        except Exception as e:
            logger.error(f"External model summarization failed for {platform}/{model_name}: {e}")
            return self.first_sentence_plus_title(text)

class EvaluationMetrics:
    """Collection of evaluation metrics for multi-reference evaluation."""

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def calculate_rouge_single_ref(self, generated: List[str], reference: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores against single reference."""
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

        for gen, ref in zip(generated, reference):
            scores = self.rouge_scorer.score(ref, gen)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)

        return {
            'rouge1': np.mean(rouge_scores['rouge1']),
            'rouge2': np.mean(rouge_scores['rouge2']),
            'rougeL': np.mean(rouge_scores['rougeL'])
        }

    def calculate_rouge_multi_ref(self, generated: List[str], references: List[List[str]]) -> Dict[str, float]:
        """Calculate ROUGE scores against multiple references (max score)."""
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

        for gen, ref_list in zip(generated, references):
            # Calculate score against each reference and take maximum
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

    def calculate_bert_score(self, generated: List[str], reference: List[str]) -> float:
        """Calculate BERTScore."""
        if bert_score is None:
            logger.warning("BERTScore not available")
            return 0.0

        try:
            P, R, F1 = bert_score(generated, reference, lang="en", verbose=False)
            return F1.mean().item()
        except Exception as e:
            logger.error(f"BERTScore calculation failed: {e}")
            return 0.0


class DataLoader:
    """Data loading utilities for scientific papers with gold-standard summaries."""

    def load_papers_from_json(self, file_path: str) -> List[Paper]:
        """
        Load papers from JSON file.

        Expected JSON format:
        [
            {
                "title": "Paper title",
                "abstract": "Paper abstract",
                "id": "unique_identifier",
                "summaries": [
                    "Reference summary 1",
                    "Reference summary 2",
                    "Reference summary 3"
                ]
            },
            ...
        ]
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            papers = []
            for i, item in enumerate(data):
                # Validate required fields
                if not all(key in item for key in ['title', 'abstract', 'id', 'summaries']):
                    logger.error(f"Paper {i} missing required fields (title, abstract, id, summaries)")
                    continue

                # Validate summaries field
                if not isinstance(item['summaries'], list) or len(item['summaries']) == 0:
                    logger.error(f"Paper {i} has invalid summaries field (must be non-empty list)")
                    continue

                paper = Paper(
                    title=item['title'],
                    abstract=item['abstract'],
                    id=item['id'],
                    summaries=item['summaries']
                )
                papers.append(paper)

            logger.info(f"Successfully loaded {len(papers)} papers from {file_path}")

            # Log summary statistics
            total_summaries = sum(len(paper.summaries) for paper in papers)
            avg_summaries = total_summaries / len(papers) if papers else 0
            logger.info(f"Average number of reference summaries per paper: {avg_summaries:.2f}")

            return papers

        except Exception as e:
            logger.error(f"Failed to load papers from {file_path}: {e}")
            return []


class SummarizationBenchmark:
    """Main benchmarking framework with length constraint tracking."""

    def __init__(self, output_dir: str = "benchmark_results", min_words: int = 15, max_words: int = 35,
                 llm_prompt: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Length constraints
        self.min_words = min_words
        self.max_words = max_words

        self.summarization_methods = SummarizationMethods(
            target_min_words=min_words,
            target_max_words=max_words,
            llm_prompt=llm_prompt
        )
        self.evaluation_metrics = EvaluationMetrics()
        self.data_loader = DataLoader()

        self.results = {}
        self.papers = []

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

    def load_papers(self, json_file_path: str):
        """Load papers from JSON file."""
        self.papers = self.data_loader.load_papers_from_json(json_file_path)
        if not self.papers:
            raise ValueError(f"No papers loaded from {json_file_path}")

        logger.info(f"Loaded {len(self.papers)} papers for benchmarking")

    def evaluate_method(self, method_name: str, method_function: Callable):
        """Evaluate a single summarization method."""
        logger.info(f"Evaluating method: {method_name} (Target: {self.min_words}-{self.max_words} words)")

        generated_summaries = []
        start_time = time.time()

        for paper in tqdm(self.papers, desc=f"Processing {method_name}"):
            try:
                # Use title + abstract as input for condensation
                full_text = f"{paper.title}. {paper.abstract}"
                summary = method_function(full_text)
                generated_summaries.append(summary)
            except Exception as e:
                logger.error(f"Error processing paper {paper.id} with {method_name}: {e}")
                generated_summaries.append(f"{paper.title}. {paper.abstract[:50]}...")  # Fallback

        execution_time = time.time() - start_time

        # Calculate length statistics
        length_stats = self.calculate_length_stats(generated_summaries)

        # Log length statistics
        logger.info(f"{method_name} - Length Statistics:")
        logger.info(f"  Average length: {length_stats['avg_length']:.1f} words")
        logger.info(
            f"  Within bounds ({self.min_words}-{self.max_words}): {length_stats['within_bounds']}/{length_stats['total_summaries']} ({length_stats['within_bounds_pct']:.1f}%)")
        logger.info(
            f"  Too short (<{self.min_words}): {length_stats['too_short']} ({length_stats['too_short_pct']:.1f}%)")
        logger.info(f"  Too long (>{self.max_words}): {length_stats['too_long']} ({length_stats['too_long_pct']:.1f}%)")

        # Calculate metrics using first reference for single-ref evaluation
        first_references = [paper.summaries[0] for paper in self.papers]
        rouge_scores = self.evaluation_metrics.calculate_rouge_single_ref(
            generated_summaries, first_references
        )
        bert_score_avg = self.evaluation_metrics.calculate_bert_score(
            generated_summaries, first_references
        )

        # Calculate multi-reference ROUGE scores
        all_references = [paper.summaries for paper in self.papers]
        multi_ref_rouge = self.evaluation_metrics.calculate_rouge_multi_ref(
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
            multi_ref_rouge1=multi_ref_rouge['rouge1'],
            multi_ref_rouge2=multi_ref_rouge['rouge2'],
            multi_ref_rougeL=multi_ref_rouge['rougeL'],
            length_stats=length_stats
        )

        self.results[method_name] = result
        logger.info(f"Completed evaluation of {method_name}")

    def run_base_methods(self):
        """Run all base summarization methods."""
        base_methods = {
            'First_Sentence_Plus_Title': self.summarization_methods.first_sentence_plus_title,
            'Key_Sentences_Extractive': self.summarization_methods.key_sentences_extractive,
            'TextRank_Length_Aware': self.summarization_methods.textrank_summarize,
            'BART_Length_Guided': self.summarization_methods.bart_summarize,
            'T5_Length_Guided': self.summarization_methods.t5_summarize
        }

        for method_name, method_function in base_methods.items():
            self.evaluate_method(method_name, method_function)

    def run_external_model(self, platform: str, model_name: str):
        """Run external model evaluation."""
        display_name = f"{platform}_{model_name}"
        method_function = lambda text: self.summarization_methods.external_model_summarize(
            text, platform, model_name
        )

        self.evaluate_method(display_name, method_function)

    def generate_comparison_report(self) -> pd.DataFrame:
        """Generate comparison report with length compliance statistics."""
        if not self.results:
            logger.warning("No results to compare")
            return pd.DataFrame()

        comparison_data = []
        for method_name, result in self.results.items():
            length_stats = result.length_stats

            comparison_data.append({
                'Method': method_name,
                'ROUGE-1 (Single)': result.rouge1,
                'ROUGE-2 (Single)': result.rouge2,
                'ROUGE-L (Single)': result.rougeL,
                'ROUGE-1 (Multi)': result.multi_ref_rouge1,
                'ROUGE-2 (Multi)': result.multi_ref_rouge2,
                'ROUGE-L (Multi)': result.multi_ref_rougeL,
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
        df = df.sort_values('ROUGE-1 (Multi)', ascending=False)

        # Save to file
        report_path = self.output_dir / "comparison_report.csv"
        df.to_csv(report_path, index=False)
        logger.info(f"Comparison report saved to {report_path}")

        return df

    def generate_length_compliance_report(self) -> pd.DataFrame:
        """Generate detailed length compliance report."""
        if not self.results:
            return pd.DataFrame()

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

        # Save to file
        report_path = self.output_dir / "length_compliance_report.csv"
        df.to_csv(report_path, index=False)
        logger.info(f"Length compliance report saved to {report_path}")

        return df

    def create_visualizations(self):
        """Create comprehensive visualization plots."""
        if not self.results:
            logger.warning("No results to visualize")
            return

        # Create main quality comparison plot
        self._create_quality_comparison_plot()

        # Create separate length analysis plot
        self._create_length_analysis_plot()

        logger.info(f"Visualizations saved to {self.output_dir}")

    def _create_quality_comparison_plot(self):
        """Create comprehensive quality metrics comparison plot."""
        methods = list(self.results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Summarization Quality Comparison', fontsize=16)

        x = np.arange(len(methods))
        width = 0.35

        # 1. ROUGE-1 comparison (single vs multi-reference)
        single_rouge1 = [self.results[m].rouge1 for m in methods]
        multi_rouge1 = [self.results[m].multi_ref_rouge1 for m in methods]

        axes[0, 0].bar(x - width / 2, single_rouge1, width, label='Single Reference', color='skyblue', alpha=0.8)
        axes[0, 0].bar(x + width / 2, multi_rouge1, width, label='Multi Reference', color='darkblue', alpha=0.8)
        axes[0, 0].set_title('ROUGE-1 Scores')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(methods, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1)

        # 2. ROUGE-2 comparison (single vs multi-reference)
        single_rouge2 = [self.results[m].rouge2 for m in methods]
        multi_rouge2 = [self.results[m].multi_ref_rouge2 for m in methods]

        axes[0, 1].bar(x - width / 2, single_rouge2, width, label='Single Reference', color='lightcoral', alpha=0.8)
        axes[0, 1].bar(x + width / 2, multi_rouge2, width, label='Multi Reference', color='darkred', alpha=0.8)
        axes[0, 1].set_title('ROUGE-2 Scores')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(methods, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1)

        # 3. ROUGE-L comparison (single vs multi-reference)
        single_rougeL = [self.results[m].rougeL for m in methods]
        multi_rougeL = [self.results[m].multi_ref_rougeL for m in methods]

        axes[1, 0].bar(x - width / 2, single_rougeL, width, label='Single Reference', color='lightgreen', alpha=0.8)
        axes[1, 0].bar(x + width / 2, multi_rougeL, width, label='Multi Reference', color='darkgreen', alpha=0.8)
        axes[1, 0].set_title('ROUGE-L Scores')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(methods, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0, 1)

        # 4. BERTScore and Execution Time
        bert_scores = [self.results[m].bert_score for m in methods]
        exec_times = [self.results[m].execution_time for m in methods]

        # Create secondary y-axis for execution time
        ax4 = axes[1, 1]
        ax4_twin = ax4.twinx()

        # BERTScore bars
        bars1 = ax4.bar(x - width / 2, bert_scores, width, label='BERTScore', color='mediumpurple', alpha=0.8)
        ax4.set_ylabel('BERTScore', color='mediumpurple')
        ax4.tick_params(axis='y', labelcolor='mediumpurple')
        ax4.set_ylim(0, 1)

        # Execution time bars
        bars2 = ax4_twin.bar(x + width / 2, exec_times, width, label='Execution Time', color='orange', alpha=0.8)
        ax4_twin.set_ylabel('Execution Time (s)', color='orange')
        ax4_twin.tick_params(axis='y', labelcolor='orange')

        ax4.set_title('BERTScore & Execution Time')
        ax4.set_xticks(x)
        ax4.set_xticklabels(methods, rotation=45, ha='right')

        # Combined legend
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()

        # Save quality comparison plot
        quality_plot_path = self.output_dir / "quality_comparison.png"
        plt.savefig(quality_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_length_analysis_plot(self):
        """Create dedicated length analysis visualization."""
        methods = list(self.results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Length Analysis (Target: {self.min_words}-{self.max_words} words)', fontsize=16)

        # 1. Length distribution histogram
        ax1 = axes[0, 0]
        for i, method in enumerate(methods):
            lengths = self.results[method].length_stats['all_lengths']
            ax1.hist(lengths, alpha=0.7, label=method, color=colors[i], bins=20, edgecolor='black', linewidth=0.5)

        ax1.axvline(self.min_words, color='red', linestyle='--', linewidth=2, label=f'Min ({self.min_words})')
        ax1.axvline(self.max_words, color='red', linestyle='--', linewidth=2, label=f'Max ({self.max_words})')
        ax1.fill_between([self.min_words, self.max_words], 0, ax1.get_ylim()[1], alpha=0.2, color='green',
                         label='Target Range')
        ax1.set_xlabel('Word Count')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Summary Length Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Box plot comparison
        ax2 = axes[0, 1]
        length_data = [self.results[method].length_stats['all_lengths'] for method in methods]
        box_plot = ax2.boxplot(length_data, labels=methods, patch_artist=True,
                               showmeans=True, meanline=True)

        # Color the boxes
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.axhline(self.min_words, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax2.axhline(self.max_words, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax2.fill_between(ax2.get_xlim(), self.min_words, self.max_words, alpha=0.2, color='green')
        ax2.set_ylabel('Word Count')
        ax2.set_title('Length Distribution by Method')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        # 3. Compliance breakdown (stacked bar chart)
        ax3 = axes[1, 0]
        x = np.arange(len(methods))
        within_bounds_pct = [self.results[m].length_stats['within_bounds_pct'] for m in methods]
        too_short_pct = [self.results[m].length_stats['too_short_pct'] for m in methods]
        too_long_pct = [self.results[m].length_stats['too_long_pct'] for m in methods]

        bars1 = ax3.bar(x, within_bounds_pct, label='Within Bounds', color='lightgreen', alpha=0.8)
        bars2 = ax3.bar(x, too_short_pct, bottom=within_bounds_pct, label='Too Short', color='lightcoral', alpha=0.8)
        bottom_combined = [w + s for w, s in zip(within_bounds_pct, too_short_pct)]
        bars3 = ax3.bar(x, too_long_pct, bottom=bottom_combined, label='Too Long', color='lightyellow', alpha=0.8)

        # Add percentage labels on bars
        for i, (within, short, long) in enumerate(zip(within_bounds_pct, too_short_pct, too_long_pct)):
            if within > 5:  # Only show label if segment is large enough
                ax3.text(i, within / 2, f'{within:.1f}%', ha='center', va='center', fontweight='bold')
            if short > 5:
                ax3.text(i, within + short / 2, f'{short:.1f}%', ha='center', va='center', fontweight='bold')
            if long > 5:
                ax3.text(i, within + short + long / 2, f'{long:.1f}%', ha='center', va='center', fontweight='bold')

        ax3.set_xlabel('Method')
        ax3.set_ylabel('Percentage')
        ax3.set_title('Length Compliance (%)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(methods, rotation=45, ha='right')
        ax3.legend()
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)

        # 4. Average length with error bars
        ax4 = axes[1, 1]
        avg_lengths = [self.results[m].length_stats['avg_length'] for m in methods]
        std_lengths = [self.results[m].length_stats['std_length'] for m in methods]

        bars = ax4.bar(methods, avg_lengths, color=colors, alpha=0.8,
                       yerr=std_lengths, capsize=5, error_kw={'linewidth': 2})

        # Add value labels on bars
        for i, (avg, std) in enumerate(zip(avg_lengths, std_lengths)):
            ax4.text(i, avg + std + 0.5, f'{avg:.1f}±{std:.1f}', ha='center', va='bottom', fontweight='bold')

        ax4.axhline(self.min_words, color='red', linestyle='--', alpha=0.8, linewidth=2,
                    label=f'Min ({self.min_words})')
        ax4.axhline(self.max_words, color='red', linestyle='--', alpha=0.8, linewidth=2,
                    label=f'Max ({self.max_words})')
        ax4.fill_between(ax4.get_xlim(), self.min_words, self.max_words, alpha=0.2, color='green', label='Target Range')
        ax4.set_ylabel('Average Word Count')
        ax4.set_title('Average Length ± Standard Deviation')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save length analysis plot
        length_plot_path = self.output_dir / "length_analysis.png"
        plt.savefig(length_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_combined_performance_plot(self):
        """Create a combined plot showing quality vs length compliance."""
        methods = list(self.results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Performance vs Length Compliance', fontsize=16)

        # 1. Quality vs Length Compliance Scatter
        ax1 = axes[0]
        multi_rouge1 = [self.results[m].multi_ref_rouge1 for m in methods]
        within_bounds_pct = [self.results[m].length_stats['within_bounds_pct'] for m in methods]

        for i, method in enumerate(methods):
            ax1.scatter(within_bounds_pct[i], multi_rouge1[i],
                        color=colors[i], s=100, alpha=0.8, label=method)
            ax1.annotate(method, (within_bounds_pct[i], multi_rouge1[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax1.set_xlabel('Length Compliance (%)')
        ax1.set_ylabel('ROUGE-1 Multi-Reference')
        ax1.set_title('Quality vs Length Compliance')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 1)

        # 2. Radar chart for top methods
        ax2 = axes[1]

        # Select top 3 methods by ROUGE-1 multi-ref
        top_methods = sorted(methods, key=lambda m: self.results[m].multi_ref_rouge1, reverse=True)[:3]

        # Metrics to include in radar chart
        metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore', 'Length\nCompliance']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        for i, method in enumerate(top_methods):
            result = self.results[method]
            values = [
                result.multi_ref_rouge1,
                result.multi_ref_rouge2,
                result.multi_ref_rougeL,
                result.bert_score,
                result.length_stats['within_bounds_pct'] / 100  # Normalize to 0-1
            ]
            values += values[:1]  # Complete the circle

            ax2.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
            ax2.fill(angles, values, alpha=0.25, color=colors[i])

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metrics)
        ax2.set_ylim(0, 1)
        ax2.set_title('Top 3 Methods Performance Profile')
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax2.grid(True)

        plt.tight_layout()

        # Save combined performance plot
        combined_plot_path = self.output_dir / "performance_analysis.png"
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def create_visualizations(self):
        """Create all visualization plots."""
        if not self.results:
            logger.warning("No results to visualize")
            return

        self._create_quality_comparison_plot()
        self._create_length_analysis_plot()
        self._create_combined_performance_plot()

        logger.info(f"Visualizations saved to {self.output_dir}")
        logger.info("Generated plots:")
        logger.info("  - quality_comparison.png: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore comparison")
        logger.info("  - length_analysis.png: Detailed length distribution and compliance analysis")
        logger.info("  - performance_analysis.png: Combined quality vs length compliance analysis")

    def save_detailed_results(self):
        """Save detailed results to JSON including length statistics."""
        detailed_results = {}

        for method_name, result in self.results.items():
            detailed_results[method_name] = {
                'rouge1_single': result.rouge1,
                'rouge2_single': result.rouge2,
                'rougeL_single': result.rougeL,
                'rouge1_multi': result.multi_ref_rouge1,
                'rouge2_multi': result.multi_ref_rouge2,
                'rougeL_multi': result.multi_ref_rougeL,
                'bert_score': result.bert_score,
                'execution_time': result.execution_time,
                'length_statistics': result.length_stats,
                'summaries': result.summaries
            }

        results_path = self.output_dir / "detailed_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Detailed results saved to {results_path}")


def main():
    """Main execution function with length constraints."""
    parser = argparse.ArgumentParser(description="Scientific Paper Summarization Benchmark")
    parser.add_argument("--data-file", required=True,
                        help="Path to JSON file containing papers with reference summaries")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    parser.add_argument("--min-words", type=int, default=15, help="Minimum target word count for summaries")
    parser.add_argument("--max-words", type=int, default=35, help="Maximum target word count for summaries")

    args = parser.parse_args()

    llm_prompt = f"""Please provide a concise summary of the following scientific paper. 
The summary MUST be between {args.min_words} and {args.max_words} words long.
Focus on the key findings, methodology, and conclusions.

{{text}}

Summary ({args.min_words} - {args.max_words} words):
"""

    if not Path(args.data_file).exists():
        logger.error(f"Data file not found: {args.data_file}")
        return

    # Initialize benchmark with length constraints
    benchmark = SummarizationBenchmark(
        output_dir=args.output_dir,
        min_words=args.min_words,
        max_words=args.max_words,
        llm_prompt=llm_prompt
    )

    try:
        benchmark.load_papers(args.data_file)
    except ValueError as e:
        logger.error(f"Failed to load papers: {e}")
        return

    logger.info(f"Target length constraints: {args.min_words}-{args.max_words} words")

    # Run base methods
    benchmark.run_base_methods()

    # Run external models
    benchmark.run_external_model("ollama", "granite3.3:2b")
    benchmark.run_external_model("ollama", "granite3.3:8b")
    benchmark.run_external_model("ollama", "gemma3:12b")
    benchmark.run_external_model("ollama", "llama3.1:8b")
    benchmark.run_external_model("ollama", "deepseek-r1:7b")

    # Generate reports
    comparison_df = benchmark.generate_comparison_report()
    print("\n" + "=" * 100)
    print("SUMMARIZATION METHODS COMPARISON")
    print("=" * 100)
    print(comparison_df.to_string(index=False))

    # Generate length compliance report
    compliance_df = benchmark.generate_length_compliance_report()
    print("\n" + "=" * 80)
    print("LENGTH COMPLIANCE REPORT")
    print("=" * 80)
    print(compliance_df.to_string(index=False))

    benchmark.create_visualizations()
    benchmark.save_detailed_results()

    print(f"\nAll results saved to: {benchmark.output_dir}")
    print(f"Target length range: {args.min_words}-{args.max_words} words")


if __name__ == "__main__":
    main()