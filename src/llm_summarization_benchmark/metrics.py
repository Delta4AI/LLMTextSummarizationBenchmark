import logging
import gc
import time

from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score.rouge_scorer import RougeScorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from alignscore import AlignScore

from llm_apis.huggingface_client import init_hf_cache_dir
from llm_summarization_benchmark.summarization_utilities import get_min_max_mean_std
from exploration_utilities import get_project_root

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)

ROUGE_TYPES = ['rouge1', 'rouge2', 'rougeL']
ROUGE_SCORER = RougeScorer(ROUGE_TYPES, use_stemmer=True)
OUT_DIR = get_project_root() / "Output" / "llm_summarization_benchmark"

init_hf_cache_dir()


class ModelCache:
    """Cache for reusing models to avoid reloading"""

    def __init__(self):
        self.sentence_transformers = {}
        self.bert_models = {}

    def get_sentence_transformer(self, model_name: str) -> SentenceTransformer:
        """Get or create a SentenceTransformer model"""
        if model_name not in self.sentence_transformers:
            logger.info(f"Loading SentenceTransformer model: {model_name}")

            device = self._get_best_device()

            self.sentence_transformers[model_name] = SentenceTransformer(
                model_name,
                device=device
            )

            self._log_memory_usage(f"After loading {model_name}")

        return self.sentence_transformers[model_name]

    @staticmethod
    def _get_best_device() -> str:
        """Determine the best device to use"""
        if torch and torch.cuda.is_available():
            # Check available GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
            free_memory = gpu_memory - allocated

            # Use GPU only if we have enough free memory (at least 1GB)
            if free_memory > 1.0:
                return 'cuda'
            else:
                logger.warning(f"Low GPU memory ({free_memory:.2f}GB), using CPU for SentenceTransformer")
                return 'cpu'
        return 'cpu'

    @staticmethod
    def _log_memory_usage(context: str = ""):
        """Log current memory usage"""
        if torch and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
            cached = torch.cuda.memory_reserved(0) / 1024 ** 3
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            free = total - allocated
            logger.info(f"{context} - GPU: {allocated:.2f}GB allocated, {cached:.2f}GB cached, {free:.2f}GB free")

    def cleanup_all(self):
        """Clean up all cached models"""
        self.sentence_transformers.clear()
        self.bert_models.clear()

        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(0.5)
            torch.cuda.empty_cache()

        self._log_memory_usage("After Cleanup")


_model_cache = ModelCache()


def get_length_scores(summaries: list[str], min_words: int, max_words: int) -> dict:
    """Calculate length compliance statistics for a set of summaries."""
    lengths = [len(summary.split()) for summary in summaries]

    too_short = sum(1 for length in lengths if length < min_words)
    too_long = sum(1 for length in lengths if length > max_words)
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
        'target_min': min_words,
        'target_max': max_words,
        'all_lengths': lengths
    }


def get_rouge_scores(generated: list[str], references: list[list[str]]) -> dict[str, dict[str, float]]:
    """Calculate ROUGE scores against multiple references (max score)."""
    rouge_scores = {rouge_type: [] for rouge_type in ROUGE_TYPES}

    for gen, ref_list in zip(generated, references):
        max_scores = dict.fromkeys(ROUGE_TYPES, 0.0)

        for ref in ref_list:
            scores = ROUGE_SCORER.score(ref, gen)
            for rouge_type in ROUGE_TYPES:
                max_scores[rouge_type] = max(max_scores[rouge_type], scores[rouge_type].fmeasure)

        for rouge_type in ROUGE_TYPES:
            rouge_scores[rouge_type].append(max_scores[rouge_type])

    return {
        rouge_type: get_min_max_mean_std(rouge_scores[rouge_type])
        for rouge_type in ROUGE_TYPES
    }


def get_meteor_scores(generated: list[str], references: list[list[str]]) -> dict[str, float]:
    """Calculate METEOR score using best reference for each generated summary."""
    meteor_scores = []

    try:
        for gen, ref_list in zip(generated, references):
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

    return get_min_max_mean_std(meteor_scores)


def get_bert_scores(generated: list[str], references: list[list[str]], model: str) -> dict[str, dict[str, float]]:
    """Calculate BERTScore using best reference for each generated summary."""
    best_precision = []
    best_recall = []
    best_f1 = []

    try:
        logger.info(f"Calculating BERTScore with model: {model}")

        # Clear cache before BERTScore to free memory
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

        batch_size = 2  # Process 2 documents at a time

        for i in range(0, len(generated), batch_size):
            batch_gen = generated[i:i + batch_size]
            batch_ref = references[i:i + batch_size]

            for gen, ref_list in zip(batch_gen, batch_ref):
                # Calculate BERTScore against all references for this summary
                with torch.no_grad():  # Disable gradient computation
                    P, R, F1 = bert_score(
                        cands=[gen] * len(ref_list),
                        refs=ref_list,
                        model_type=model,
                        lang="en",
                        verbose=False,
                        device='cuda' if torch and torch.cuda.is_available() else 'cpu'
                    )

                # Take the maximum scores
                best_precision.append(P.max().item())
                best_recall.append(R.max().item())
                best_f1.append(F1.max().item())

                # Clean up tensors
                del P, R, F1

                # Clear cache after each document
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"BERTScore calculation failed: {e}")
        # Force cleanup on error
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        'precision': get_min_max_mean_std(best_precision),
        'recall': get_min_max_mean_std(best_recall),
        'f1': get_min_max_mean_std(best_f1)
    }


def get_bleu_scores(generated: list[str], references: list[list[str]]) -> dict[str, float]:
    """Calculate BLEU score using best reference for each generated summary."""
    bleu_scores = []
    try:
        for gen, ref_list in zip(generated, references):
            scores = sentence_bleu(references=ref_list, hypothesis=gen)
            bleu_scores.append(scores)
    except Exception as e:
        logger.error(f"BLEU calculation failed: {e}")

    return get_min_max_mean_std(bleu_scores)


def get_sentence_transformer_similarity(generated: list[str], source_documents: list[str], model_name: str) -> dict[
    str, float]:
    """Calculate sentence transformer similarity using cached model."""
    similarities = []

    try:
        logger.info(f"Calculating SentenceTransformer similarity with model: {model_name}")
        model = _model_cache.get_sentence_transformer(model_name)

        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

        batch_size = 4

        for i in range(0, len(generated), batch_size):
            batch_gen = generated[i:i + batch_size]
            batch_src = source_documents[i:i + batch_size]

            for gen, src in zip(batch_gen, batch_src):
                with torch.no_grad():  # Disable gradient computation
                    embeddings = model.encode([gen, src], convert_to_tensor=True)
                    similarity = cosine_similarity(
                        embeddings[0].cpu().numpy().reshape(1, -1),
                        embeddings[1].cpu().numpy().reshape(1, -1)
                    )[0][0]
                    similarities.append(float(similarity))

                # Clean up tensors
                del embeddings

                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Sentence Transformer embedding similarity {model_name} failed: {e}")
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return get_min_max_mean_std(similarities)

def get_alignscore_scores(generated: list[str], references: list[str]) -> dict[str, float]:
    """
    Calculate AlignScore between generated summaries and abstracts.
    """
    scores = []

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    try:
        logger.info(f"Calculating AlignScore on device: {device}")

        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

        aligner = AlignScore(
            model="roberta-large",
            ckpt_path=OUT_DIR / "AlignScore-large.ckpt",
            batch_size=16,
            device=device,
        )

        batch_size = 16
        for i in range(0, len(generated), batch_size):
            batch_gen = generated[i:i + batch_size]
            batch_ref = references[i:i + batch_size]

            try:
                batch_scores = aligner.score(batch_ref, batch_gen)
                scores.extend(batch_scores)
            except Exception as e:
                logger.warning(f"AlignScore batch {i // batch_size + 1} failed: {e}")
                continue

            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(f"AlignScore computed for {len(scores)} pairs")

    except Exception as e:
        logger.error(f"AlignScore calculation failed: {e}")
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return get_min_max_mean_std(scores)


def cleanup_metrics_cache():
    """Clean up all cached models in metrics"""
    logger.info("Cleaning up metrics cache")
    _model_cache.cleanup_all()
