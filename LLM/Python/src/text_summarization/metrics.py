import logging

from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score.rouge_scorer import RougeScorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from text_summarization.utilities import get_min_max_mean_std


logger = logging.getLogger(__name__)
ROUGE_TYPES = ['rouge1', 'rouge2', 'rougeL']
ROUGE_SCORER = RougeScorer(ROUGE_TYPES, use_stemmer=True)

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
        for gen, ref_list in zip(generated, references):

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

def get_sentence_transformer_similarity(generated: list[str], source_documents: list[str], model: str) -> dict[str, float]:
    model = SentenceTransformer(model)
    similarities = []

    try:
        for gen, src in zip(generated, source_documents):
            embeddings = model.encode([gen, src])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            similarities.append(similarity)
    except Exception as e:
        logger.error(f"Sentence Transformer embedding similarity {model} failed: {e}")

    return get_min_max_mean_std(similarities)