import logging
import gc
import time
from typing import TYPE_CHECKING

from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score.rouge_scorer import RougeScorer
from bert_score import score as bert_score, BERTScorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from alignscore import AlignScore

from llm_apis.huggingface_client import init_hf_cache_dir

from utilities import get_project_root, get_min_max_mean_std

try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    torch = None
    DEVICE = "cpu"

if TYPE_CHECKING:
    from llm_summarization_benchmark.benchmark import InterferenceRunContainer

logger = logging.getLogger(__name__)

ROUGE_TYPES = ['rouge1', 'rouge2', 'rougeL']
ROUGE_SCORER = RougeScorer(ROUGE_TYPES, use_stemmer=True)
OUT_DIR = get_project_root() / "Output" / "llm_summarization_benchmark"
USE_MODEL_CACHE = False
METRIC_TYPES = [
    "rouge_scores", "roberta_scores", "deberta_scores", "meteor_scores", "bleu_scores",
    "mpnet_content_coverage_scores", "alignscore_scores", "summac_scores", "factcc_scores",
    "minicheck_ft5_scores", "minicheck_7b_scores"
]


init_hf_cache_dir()


class ModelCache:
    """Cache for reusing models to avoid reloading"""

    def __init__(self) -> None:
        self.sentence_transformers = {}
        self.bert_models = {}

    def get_sentence_transformer(self, model_name: str) -> SentenceTransformer:
        """Get or create a SentenceTransformer model"""
        if not USE_MODEL_CACHE:
            return SentenceTransformer(model_name, device=self._get_best_device())

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
        empty_cuda_cache(sync=True)
        self._log_memory_usage("After Cleanup")
        time.sleep(5)


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


def get_rouge_scores(generated: list[str], references: list[list[str]],
                     irc: 'InterferenceRunContainer') -> dict[str, dict[str, float]]:
    """Calculate ROUGE scores against multiple references (max score)."""
    rouge_scores = {rouge_type: [] for rouge_type in ROUGE_TYPES}

    for gen, ref_list, paper in zip(generated, references, irc.papers):
        max_scores = dict.fromkeys(ROUGE_TYPES, 0.0)

        for ref in ref_list:
            scores = ROUGE_SCORER.score(ref, gen)
            for rouge_type in ROUGE_TYPES:
                max_scores[rouge_type] = max(max_scores[rouge_type], scores[rouge_type].fmeasure)
                paper.scores[rouge_type].append(scores[rouge_type].fmeasure)

        for rouge_type in ROUGE_TYPES:
            rouge_scores[rouge_type].append(max_scores[rouge_type])

    time.sleep(5)

    return {
        rouge_type: get_min_max_mean_std(rouge_scores[rouge_type])
        for rouge_type in ROUGE_TYPES
    }


def get_meteor_scores(generated: list[str], references: list[list[str]],
                      irc: 'InterferenceRunContainer') -> dict[str, float]:
    """Calculate METEOR score using best reference for each generated summary."""
    meteor_scores = []

    try:
        for gen, ref_list, paper in zip(generated, references, irc.papers):
            # Pre-tokenize the generated summary
            gen_tokens = gen.split()

            # For each reference, calculate the METEOR score
            ref_scores = []
            for ref in ref_list:
                ref_tokens = ref.split()
                score = meteor_score([ref_tokens], gen_tokens)
                ref_scores.append(score)
                paper.scores["meteor"].append(score)

            # Take the best score for this document
            best_score = max(ref_scores) if ref_scores else 0.0
            meteor_scores.append(best_score)

    except Exception as e:
        logger.error(f"METEOR calculation failed: {str(e)}")
        raise

    time.sleep(5)

    return get_min_max_mean_std(meteor_scores)


def get_bert_scores(generated: list[str], references: list[list[str]], model: str,
                    irc: 'InterferenceRunContainer') -> dict[str, dict[str, float]]:
    """Calculate BERTScore using best reference for each generated summary."""
    best_precision = []
    best_recall = []
    best_f1 = []

    try:
        logger.info(f"Calculating BERTScore with model: {model}")
        empty_cuda_cache()

        for idx, (gen, ref_list, paper) in enumerate(zip(generated, references, irc.papers)):
            with torch.no_grad():
                P, R, F1 = bert_score(
                    cands=[gen] * len(ref_list),
                    refs=ref_list,
                    model_type=model,
                    lang="en",
                    verbose=False,
                    device='cuda' if torch and torch.cuda.is_available() else 'cpu',
                    batch_size=8,
                )

            precision = P.max().item()
            recall = R.max().item()
            f1 = F1.max().item()

            best_precision.append(precision)
            best_recall.append(recall)
            best_f1.append(f1)

            paper.scores[f"bert_{model}_precision"].append(precision)
            paper.scores[f"bert_{model}_recall"].append(recall)
            paper.scores[f"bert_{model}_f1"].append(f1)

            del P, R, F1

            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(generated)} documents for {model}")


    except Exception as e:
        logger.error(f"BERTScore calculation failed: {e}")
        raise
    finally:
        if hasattr(BERTScorer, '_model'):
            BERTScorer._model = None

        gc.collect()
        empty_cuda_cache(sync=True)

    return {
        'precision': get_min_max_mean_std(best_precision),
        'recall': get_min_max_mean_std(best_recall),
        'f1': get_min_max_mean_std(best_f1)
    }

def get_bleu_scores(generated: list[str], references: list[list[str]],
                    irc: 'InterferenceRunContainer') -> dict[str, float]:
    """Calculate BLEU score using best reference for each generated summary."""
    bleu_scores = []
    try:
        for gen, ref_list, paper in zip(generated, references, irc.papers):
            scores = sentence_bleu(references=ref_list, hypothesis=gen)
            paper.scores["bleu"].append(scores)
            bleu_scores.append(scores)
    except Exception as e:
        logger.error(f"BLEU calculation failed: {e}")
        raise

    return get_min_max_mean_std(bleu_scores)


def get_sentence_transformer_similarity(generated: list[str], source_documents: list[str], model_name: str,
                                        irc: 'InterferenceRunContainer') -> dict[str, float]:
    """Calculate sentence transformer similarity using cached model."""
    similarities = []

    try:
        logger.info(f"Calculating SentenceTransformer similarity with model: {model_name}")
        model = _model_cache.get_sentence_transformer(model_name)
        empty_cuda_cache()

        batch_size = 4

        for i in range(0, len(generated), batch_size):
            if i % 100 == 0:
                logger.info(f"Processed sentence transformer similarities for {i}/{len(generated)} documents "
                            f"for {model_name}")
            batch_gen = generated[i:i + batch_size]
            batch_src = source_documents[i:i + batch_size]
            batch_papers = irc.papers[i:i + batch_size]

            for gen, src, paper in zip(batch_gen, batch_src, batch_papers):
                with torch.no_grad():  # Disable gradient computation
                    embeddings = model.encode([gen, src], convert_to_tensor=True)
                    similarity = cosine_similarity(
                        embeddings[0].cpu().numpy().reshape(1, -1),
                        embeddings[1].cpu().numpy().reshape(1, -1)
                    )[0][0]
                    similarities.append(float(similarity))
                    paper.scores["sentence_transformer"].append(similarity)

                # Clean up tensors
                del embeddings
                empty_cuda_cache(silent=True)

    except Exception as e:
        logger.error(f"Sentence Transformer embedding similarity {model_name} failed: {e}")
        raise
    finally:
        empty_cuda_cache()
        _model_cache.cleanup_all()

    return get_min_max_mean_std(similarities)

def get_alignscore_scores(generated: list[str], references: list[str],
                          irc: 'InterferenceRunContainer') -> dict[str, float]:
    """
    Calculate AlignScore between generated summaries and abstracts.
    """
    scores = []
    ckpt_path = OUT_DIR / "AlignScore-large.ckpt"

    try:
        logger.info(f"Calculating AlignScore on device: {DEVICE}")
        empty_cuda_cache()

        aligner = AlignScore(
            model="roberta-large",
            ckpt_path=ckpt_path,
            batch_size=16,
            device=DEVICE,
        )

        batch_size = 16
        for i in range(0, len(generated), batch_size):
            batch_gen = generated[i:i + batch_size]
            batch_ref = references[i:i + batch_size]
            batch_papers = irc.papers[i:i + batch_size]

            try:
                batch_scores = aligner.score(batch_ref, batch_gen)
                scores.extend(batch_scores)
                for score, paper in zip(batch_scores, batch_papers):
                    paper.scores["alignscore"].append(score)
            except Exception as e:
                logger.warning(f"AlignScore batch {i // batch_size + 1} failed: {e}")
                continue

            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(f"AlignScore computed for {len(scores)} pairs")

    except Exception as e:
        logger.error(f"AlignScore calculation failed: {e}")
        logger.error(f"Make sure this file exists: {ckpt_path}")
        raise
    finally:
        empty_cuda_cache()

    return get_min_max_mean_std(scores)


def get_summac_scores(generated: list[str], references: list[str],
                      irc: 'InterferenceRunContainer') -> dict[str, float]:
    """Calculate SummaC-ZS (zero-shot) factual consistency scores.

    Uses NLI-based sentence-level entailment aggregation to assess whether
    generated summaries are factually consistent with the source documents.
    """
    from summac.model_summac import SummaCZS

    try:
        logger.info(f"Calculating SummaC-ZS on device: {DEVICE}")
        empty_cuda_cache()

        model = SummaCZS(model_name="vitc", granularity="sentence", device=DEVICE)

        scores = []
        batch_size = 16
        for i in range(0, len(generated), batch_size):
            batch_gen = generated[i:i + batch_size]
            batch_ref = references[i:i + batch_size]
            batch_papers = irc.papers[i:i + batch_size]

            result = model.score(batch_ref, batch_gen)
            batch_scores = result["scores"]
            scores.extend(batch_scores)
            for score, paper in zip(batch_scores, batch_papers):
                paper.scores["summac"].append(score)

            empty_cuda_cache(silent=True)

            if (i + batch_size) % 100 < batch_size:
                logger.info(f"SummaC-ZS: processed {min(i + batch_size, len(generated))}/{len(generated)} documents")

        logger.info(f"SummaC-ZS computed for {len(scores)} pairs")

    except Exception as e:
        logger.error(f"SummaC-ZS calculation failed: {e}")
        raise
    finally:
        empty_cuda_cache()

    return get_min_max_mean_std(scores)


def get_factcc_scores(generated: list[str], references: list[str],
                      irc: 'InterferenceRunContainer') -> dict[str, float]:
    """Calculate FactCC factual consistency scores using manueldeprada/FactCC.

    Uses a BERT-based binary consistency classifier that returns P(CORRECT)
    as a continuous score in [0, 1].
    """
    from transformers import BertForSequenceClassification, BertTokenizer

    model_path = "manueldeprada/FactCC"

    model = None
    tokenizer = None
    try:
        logger.info(f"Calculating FactCC on device: {DEVICE}")
        empty_cuda_cache()

        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path).to(DEVICE)
        model.eval()
        correct_idx = model.config.label2id["CORRECT"]

        scores = []
        batch_size = 16
        for i in range(0, len(generated), batch_size):
            batch_gen = generated[i:i + batch_size]
            batch_ref = references[i:i + batch_size]
            batch_papers = irc.papers[i:i + batch_size]

            inputs = tokenizer(batch_ref, batch_gen, max_length=512,
                               padding="max_length", truncation="only_first",
                               return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            batch_scores = probs[:, correct_idx].tolist()

            scores.extend(batch_scores)
            for score, paper in zip(batch_scores, batch_papers):
                paper.scores["factcc"].append(score)

            del inputs, logits, probs
            empty_cuda_cache(silent=True)

            if (i + batch_size) % 100 < batch_size:
                logger.info(f"FactCC: processed {min(i + batch_size, len(generated))}/{len(generated)} documents")

        logger.info(f"FactCC computed for {len(scores)} pairs")

    except Exception as e:
        logger.error(f"FactCC calculation failed: {e}")
        raise
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        empty_cuda_cache()

    return get_min_max_mean_std(scores)


def get_minicheck_scores(generated: list[str], references: list[str],
                         irc: 'InterferenceRunContainer',
                         model_name: str) -> dict[str, float]:
    """Calculate MiniCheck factual consistency scores via sentence-level verification.

    Decomposes each summary into sentences, scores each (source, sentence) pair,
    and aggregates per-paper scores via mean.

    Args:
        generated: Generated summaries.
        references: Source documents (abstracts) to verify against.
        irc: Interference run container with paper objects.
        model_name: MiniCheck model variant — "flan-t5-large" or "Bespoke-MiniCheck-7B".
    """
    from minicheck.minicheck import MiniCheck
    from nltk.tokenize import sent_tokenize

    score_key = "minicheck_ft5" if "flan" in model_name.lower() else "minicheck_7b"

    scorer = None
    try:
        logger.info(f"Calculating MiniCheck ({model_name}) on device: {DEVICE}")
        empty_cuda_cache()

        scorer = MiniCheck(model_name=model_name)

        total = len(generated)
        skipped = 0
        scores = []
        for idx, (gen, ref, paper) in enumerate(zip(generated, references, irc.papers)):
            sentences = sent_tokenize(gen)
            if not sentences:
                logger.warning(f"MiniCheck ({model_name}): paper {idx + 1}/{total} has no sentences, assigning 0.0")
                scores.append(0.0)
                paper.scores[score_key].append(0.0)
                skipped += 1
                continue

            docs = [ref] * len(sentences)
            _, sent_scores, _, _ = scorer.score(docs=docs, claims=sentences)

            paper_score = sum(sent_scores) / len(sent_scores)
            scores.append(paper_score)
            paper.scores[score_key].append(paper_score)

            if (idx + 1) % 100 == 0:
                logger.info(f"MiniCheck ({model_name}): processed {idx + 1}/{total} papers")

        logger.info(f"MiniCheck ({model_name}) computed for {len(scores)} papers"
                     f"{f', skipped {skipped} (no sentences)' if skipped else ''}")

    except Exception as e:
        logger.error(f"MiniCheck ({model_name}) calculation failed: {e}")
        raise
    finally:
        if scorer is not None:
            del scorer
        gc.collect()
        empty_cuda_cache()

    return get_min_max_mean_std(scores)


def empty_cuda_cache(sync: bool = False, silent: bool = False):
    if DEVICE == "cuda":
        if not silent:
            logger.info("Clearing CUDA cache ..")
        gc.collect()
        torch.cuda.empty_cache()
        if sync:
            if not silent:
                logger.info("Syncing CUDA cache ..")
            torch.cuda.synchronize()


def cleanup_metrics_cache():
    """Clean up all cached models in metrics"""
    logger.info("Cleaning up metrics cache")
    _model_cache.cleanup_all()
