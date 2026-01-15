import logging
import networkx as nx
import numpy as np
from typing import Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from llm_apis.base_client import SummaryClient

logger = logging.getLogger(__name__)

try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    logger.error(f"Import error in local_client.py: {e}")


class TextPreprocessor:
    """Text preprocessing utilities."""

    def __init__(self, min_words: int, max_words: int):
        self.min_words = min_words
        self.max_words = max_words
        self.stop_words = set(stopwords.words('english'))

    @staticmethod
    def extract_sentences(text: str, max_sentences: int = None) -> list[str]:
        """Extract sentences from text."""
        sentences = sent_tokenize(text)
        if max_sentences:
            sentences = sentences[:max_sentences]
        return sentences

    def select_sentences(self, scored_sentences: list[tuple[str, float]]) -> list[str]:
        result_sentences = []
        total_words = 0

        for sentence, score in scored_sentences:
            sentence_words = len(sentence.split())
            if total_words + sentence_words <= self.max_words:
                result_sentences.append(sentence)
                total_words += sentence_words
                if total_words >= self.min_words:
                    break

        return result_sentences


class TextRankSummarizer(SummaryClient):
    def __init__(self):
        self.preprocessor = TextPreprocessor(
            min_words=self.min_words,
            max_words=self.max_words
        )

        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            max_features=1000,
            ngram_range=(1, 2)  # Include bigrams
        )

    def warmup(self, model_name: str, train_corpus: list[str] | None = None):
        if not train_corpus or len(train_corpus) == 0:
            logger.warning("No training corpus provided for TextRankSummarizer")
            return

        # Extract sentences from all documents in the corpus
        all_sentences = []
        for document in train_corpus:
            sentences = self.preprocessor.extract_sentences(document)
            all_sentences.extend(sentences)

        # Train the vectorizer on all sentences
        logger.info(
            f"Training TextRank vectorizer on {len(all_sentences)} sentences from {len(train_corpus)} documents")
        self.vectorizer.fit(all_sentences)
        logger.info(f"TextRank vectorizer vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def summarize(self, text: str, model_name: str, system_prompt_override: Optional[str] = None,
                  parameter_overrides: dict[str, Any] | None = None,
                  tokenizer_overrides: dict[str, Any] | None = None) -> tuple[str, int, int]:

        sentences = self.preprocessor.extract_sentences(text)

        if len(sentences) <= 2:
            return ' '.join(sentences), 0, 0

        # Apply TF-IDF to the current document's sentences
        tfidf_matrix = self.vectorizer.transform(sentences)

        # Calculate similarity matrix for this document
        similarity_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(similarity_matrix, 0)

        # Build graph and apply PageRank for this document
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph, max_iter=100, tol=1e-4)

        # Sort sentences by score
        ranked_sentences = sorted(
            [(scores[i], sentences[i]) for i in range(len(sentences))],
            key=lambda x: x[0], reverse=True
        )

        # Select sentences based on target word count
        sentence_scores = {sentences[i]: scores[i] for i in range(len(sentences))}
        scored_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)

        # Select sentences
        selected_sentences = self.preprocessor.select_sentences(scored_sentences=scored_sentences)

        if not selected_sentences:
            selected_sentences = [ranked_sentences[0][1]]

        # Reorder sentences to maintain original order
        original_order = []
        for sentence in sentences:
            if sentence in selected_sentences:
                original_order.append(sentence)

        summary = ' '.join(original_order)
        logger.info(f"TextRank summary: {len(summary.split())} words")
        return summary, 0, 0


class FrequencySummarizer(SummaryClient):
    def __init__(self):
        self.preprocessor = TextPreprocessor(
            min_words=self.min_words,
            max_words=self.max_words
        )

    def summarize(self, text: str, model_name: str, system_prompt_override: Optional[str] = None,
                  parameter_overrides: dict[str, Any] | None = None,
                  tokenizer_overrides: dict[str, Any] | None = None) -> tuple[str, int, int]:

        sentences = self.preprocessor.extract_sentences(text)
        if len(sentences) <= 1:
            return text, 0, 0

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
        selected_sentences = self.preprocessor.select_sentences(scored_sentences=scored_sentences)

        # Sort selected sentences by original order
        if selected_sentences:
            selected_sentences.sort(key=lambda x: sentences.index(x))
            summary = ' '.join(selected_sentences)
        else:
            summary = scored_sentences[0][0]

        logger.info(f"Frequency summary: {len(summary.split())} words")
        return summary, 0, 0
