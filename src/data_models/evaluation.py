from dataclasses import dataclass, field
from typing import Any

from utilities import get_min_max_mean_std
from data_models import Paper


@dataclass
class InterferenceRunContainer:
    """Data class for run parameters."""
    platform: str
    model_name: str | None = None
    method_name: str | None = None
    model_param_overrides: dict[str, Any] | None = None
    tokenizer_param_overrides: dict[str, Any] | None = None
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
    alignscore_scores: dict[str, float]
    summac_scores: dict[str, float]
    factcc_scores: dict[str, float]
    full_paper_details: list[Paper]

    def __getattr__(self, name: str):
        """Fallback for fields missing on old pickled instances.

        Returns a zero-value stats dict so visualization code doesn't crash
        on old pickled results that lack the new metric fields.
        """
        if name in ("summac_scores", "factcc_scores"):
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def as_json(self, detailed: bool = False) -> dict[str, Any]:
        rouge = {f"{k}_{kk}": vv for k, v in self.rouge_scores.items() for kk, vv in v.items()}
        roberta = {f"roberta_{k}_{kk}": vv for k, v in self.roberta_scores.items() for kk, vv in v.items()}
        deberta = {f"deberta_{k}_{kk}": vv for k, v in self.deberta_scores.items() for kk, vv in v.items()}
        meteor = {f"meteor_{k}": v for k, v in self.meteor_scores.items()}
        bleu = {f"bleu_{k}": v for k, v in self.bleu_scores.items()}
        mpnet_content_coverage = {f"content_coverage_{k}": v for k, v in self.mpnet_content_coverage_scores.items()}
        alignscore = {f"alignscore_{k}": v for k, v in self.alignscore_scores.items()}
        summac = {f"summac_{k}": v for k, v in self.summac_scores.items()}
        factcc = {f"factcc_{k}": v for k, v in self.factcc_scores.items()}
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
            **alignscore,
            **summac,
            **factcc,
            **exec_times,
            **_variable_results
        }
