import logging
import networkx as nx
import numpy as np
import re
from typing import Optional, List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

from text_summarization.llm_apis.base_client import BaseClient

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass


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


class LocalClient(BaseClient):
    def __init__(self, target_min_words: int = 15, target_max_words: int = 35):
        self.target_min_words = target_min_words
        self.target_max_words = target_max_words
        self.preprocessor = TextPreprocessor()

    def summarize(self, text: str, model_name: str, prompt: Optional[str] = None) -> str:
        """
        Local algorithm summarization.

        Args:
            text: Input text to summarize
            model_name: Algorithm name ("textrank", "textrank-simple", "frequency", etc.)
            prompt: Not used for local algorithms

        Returns:
            Generated summary
        """
        if model_name == "textrank":
            return self._textrank_summarize_advanced(text)
        elif model_name == "textrank-simple":
            return self._textrank_summarize_simple(text)
        elif model_name == "frequency":
            return self._frequency_summarize(text)
        else:
            raise Exception(f"Unsupported local algorithm: {model_name}")

    def _textrank_summarize_advanced(self, text: str) -> str:
        """Advanced TextRank using TF-IDF and cosine similarity."""
        sentences = self.preprocessor.extract_sentences(text)

        if len(sentences) <= 2:
            return ' '.join(sentences)

        # Create TF-IDF vectors for sentences
        try:
            vectorizer = TfidfVectorizer(
                stop_words='english',
                lowercase=True,
                max_features=1000,
                ngram_range=(1, 2)  # Include bigrams
            )

            tfidf_matrix = vectorizer.fit_transform(sentences)

            # Calculate cosine similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Set diagonal to 0 (no self-similarity)
            np.fill_diagonal(similarity_matrix, 0)

        except Exception as e:
            logger.warning(f"TF-IDF vectorization failed: {e}, falling back to simple method")
            return self._textrank_summarize_simple(text)

        # Build graph and apply PageRank
        try:
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph, max_iter=100, tol=1e-4)
        except Exception as e:
            logger.warning(f"PageRank failed: {e}, falling back to simple method")
            return self._textrank_summarize_simple(text)

        # Sort sentences by score
        ranked_sentences = sorted(
            [(scores[i], sentences[i]) for i in range(len(sentences))],
            key=lambda x: x[0], reverse=True
        )

        # Select sentences based on target word count
        selected_sentences = []
        total_words = 0

        for score, sentence in ranked_sentences:
            sentence_words = len(sentence.split())
            if total_words + sentence_words <= self.target_max_words:
                selected_sentences.append(sentence)
                total_words += sentence_words
                if total_words >= self.target_min_words:
                    break

        if not selected_sentences:
            selected_sentences = [ranked_sentences[0][1]]

        # Reorder sentences to maintain original order
        original_order = []
        for sentence in sentences:
            if sentence in selected_sentences:
                original_order.append(sentence)

        summary = ' '.join(original_order)
        logger.info(f"Advanced TextRank summary: {len(summary.split())} words")
        return summary

    def _textrank_summarize_simple(self, text: str) -> str:
        """Simple TextRank-based extractive summarization (your original implementation)."""
        sentences = self.preprocessor.extract_sentences(text)
        if len(sentences) <= 1:
            return text

        # Calculate sentence scores using simple word frequency
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
            summary = ' '.join(result_sentences)
        else:
            # Fallback to highest scoring sentence
            summary = scored_sentences[0][0]

        logger.info(f"Simple TextRank summary: {len(summary.split())} words")
        return summary

    def _calculate_sentence_scores(self, sentences: List[str]) -> Dict[str, float]:
        """Calculate sentence scores for simple TextRank."""
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

    def _frequency_summarize(self, text: str) -> str:
        """Frequency-based extractive summarization."""
        sentences = self.preprocessor.extract_sentences(text)
        if len(sentences) <= 1:
            return text

        # Calculate word frequencies
        all_words = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [word for word in words if word.isalnum() and word not in self.preprocessor.stop_words]
            all_words.extend(words)

        word_freq = {}
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Score sentences based on word frequencies
        sentence_scores = {}
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [word for word in words if word.isalnum() and word not in self.preprocessor.stop_words]

            if words:
                sentence_scores[sentence] = sum(word_freq.get(word, 0) for word in words) / len(words)
            else:
                sentence_scores[sentence] = 0

        # Select sentences based on length constraints
        scored_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)

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
            summary = ' '.join(result_sentences)
        else:
            summary = scored_sentences[0][0]

        logger.info(f"Frequency summary: {len(summary.split())} words")
        return summary

    def first_sentence_plus_title(self, text: str) -> str:
        """Fallback method: return first sentence plus title if available."""
        sentences = self.preprocessor.extract_sentences(text)
        if sentences:
            return sentences[0]
        return text[:100] + "..." if len(text) > 100 else text