import argparse
import logging
import math
import json
from copy import deepcopy

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from typing import TYPE_CHECKING, NamedTuple, Any, Callable

if TYPE_CHECKING:
    from llm_summarization_benchmark.benchmark import SummarizationBenchmark

logger = logging.getLogger(__name__)

SURFACE_LEVEL = "Metrics: Surface-level"
EMBEDDING_BASED = "Metrics: Embedding-based"
AGGREGATE = "Metrics: Aggregate"
PERFORMANCE = "Performance"
OVERALL = "Overall (70% metrics, 10% speed/accept./cost)"

# model grouping
MODEL_GROUPS = {
    "Traditional Methods": [
        "local:textrank", "local:frequency"
    ],
    "Encoder-Decoder Models": [
        "huggingface_facebook/bart-large-cnn", "huggingface_facebook/bart-base",
        "huggingface_google-t5/t5-base", "huggingface_google-t5/t5-large",
        "huggingface_csebuetnlp/mT5_multilingual_XLSum",
        "huggingface_google/pegasus-xsum", "huggingface_google/pegasus-large",
        "huggingface_google/pegasus-cnn_dailymail"
    ],
    "General-purpose LLMs": [
        "ollama_gemma3:270M","ollama_gemma3:1b", "ollama_gemma3:4b", "ollama_gemma3:12b",
        "ollama_granite3.3:2b", "ollama_granite3.3:8b", "ollama_llama3.1:8b",
        "ollama_llama3.2:1b", "ollama_llama3.2:3b", "ollama_mistral:7b",
        "ollama_mistral-nemo:12b", "ollama_mistral-small3.2:24b",
        "ollama_PetrosStav/gemma3-tools:4b", "ollama_phi3:3.8b", "ollama_phi4:14b",
        "openai_gpt-3.5-turbo", "openai_gpt-4.1", "openai_gpt-4.1-mini",
        "openai_gpt-4o", "openai_gpt-4o-mini", "anthropic_claude-3-5-haiku-20241022",
        "mistral_mistral-medium-2505", "mistral_mistral-small-2506", "mistral_mistral-large-2411",
        "huggingface:chat_swiss-ai/Apertus-8B-Instruct-2509", "ollama_granite4:tiny-h",
        "ollama_granite4:small-h", "ollama_granite4:micro", "ollama_granite4:micro-h",

    ],
    "Reasoning-oriented LLMs": [
        "ollama_deepseek-r1:1.5b", "ollama_deepseek-r1:7b", "ollama_deepseek-r1:8b",
        "ollama_deepseek-r1:14b", "ollama_qwen3:4b", "ollama_qwen3:8b",
        "ollama_gpt-oss:20b", "anthropic_claude-sonnet-4-20250514",
        "anthropic_claude-opus-4-20250514", "openai_gpt-5-nano-2025-08-07", "openai_gpt-5-mini-2025-08-07",
        "openai_gpt-5-2025-08-07", "anthropic_claude-opus-4-1-20250805"
    ],
    "Specialized Models": [
        "huggingface_AlgorithmicResearchGroup/led_large_16384_arxiv_summarization",
        "ollama_medllama2:7b", "huggingface_google/pegasus-pubmed",
        "huggingface_google/bigbird-pegasus-large-pubmed", "huggingface:completion_microsoft/biogpt",
        "huggingface:chat_Uni-SMART/SciLitLLM1.5-7B", "huggingface:chat_Uni-SMART/SciLitLLM1.5-14B",
        "huggingface:chat_aaditya/OpenBioLLM-Llama3-8B", "huggingface:conversational_BioMistral/BioMistral-7B"
    ],
}

class Metric(NamedTuple):
    label: str
    getter: Callable[[str], Any]
    line_override: dict | None = None
    category: str | None = None
    error_bars: bool = True


class SummarizationVisualizer:
    """Interactive visualization generator for summarization benchmark results."""

    def __init__(self, benchmark_ref: 'SummarizationBenchmark'):
        self.benchmark_ref = benchmark_ref

        self.min_words = benchmark_ref.min_words
        self.max_words = benchmark_ref.max_words
        self.results = {}
        self.methods = None
        self.out_dir = None
        self.max_publications = None

        self.metrics = [
            Metric("ROUGE-1", lambda m: self.results[m].rouge_scores["rouge1"], None, SURFACE_LEVEL),
            Metric("ROUGE-2", lambda m: self.results[m].rouge_scores["rouge2"], None, SURFACE_LEVEL),
            Metric("ROUGE-L", lambda m: self.results[m].rouge_scores["rougeL"], None, SURFACE_LEVEL),
            Metric("METEOR", lambda m: self.results[m].meteor_scores, None, SURFACE_LEVEL),
            Metric("BLEU", lambda m: self.results[m].bleu_scores, None, SURFACE_LEVEL),
            Metric("RoBERTa", lambda m: self.results[m].roberta_scores["f1"], None, EMBEDDING_BASED),
            Metric("DeBERTa", lambda m: self.results[m].deberta_scores["f1"], None, EMBEDDING_BASED),
            Metric("all-mpnet-base-v2", lambda m: self.results[m].mpnet_content_coverage_scores, None, EMBEDDING_BASED),
            Metric("AlignScore", lambda m: self.results[m].alignscore_scores, None, EMBEDDING_BASED),
        ]

        self.aggregates = [
            Metric("Metrics Mean Score", lambda m: self.metric_scores[m],
                   {"color": "black", "width": 4}, AGGREGATE, False),

        ]

        self.performances = [
            Metric("Speed", lambda m: self.normalized_exec_times[m],
                    {"color": "rgb(138, 43, 226)", "width": 3}, PERFORMANCE, False),
            Metric("Acceptance", lambda m: self.acceptance_scores[m],
                   {"color": "rgb(255, 165, 0)", "width": 3}, PERFORMANCE, False),
            Metric("Insufficient Findings", lambda m: self.insufficient_findings_rates[m],
                   {"color": "rgb(106, 137, 247)", "width": 3}, PERFORMANCE, False),
            # Metric("Input Token Cost", lambda m: self.input_token_costs[m],
            #        {"color": "rgb(204, 204, 0)", "width": 3}, PERFORMANCE, False),
            Metric("Output Token Cost", lambda m: self.output_token_costs[m],
                   {"color": "rgb(102, 51, 0)", "width": 3}, PERFORMANCE, False),
        ]

        self.overall = [
            Metric("Overall Score", lambda m: self.combined_final_scores[m],
                   {"color": "rgb(220, 20, 60)", "width": 5}, OVERALL, False)
        ]

        self.length_within_bounds = Metric("Length Within Bounds", lambda m: {
            "mean": self.results[m].length_stats["within_bounds_pct"]/100}, None, None)

        self.metric_scores = {}
        self.normalized_exec_times = {}
        self.acceptance_scores = {}
        self.insufficient_findings_rates = {}
        self.input_token_costs = {}
        self.output_token_costs = {}
        self.combined_final_scores = {}

        # https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express
        self.colors = px.colors.qualitative.Vivid

    def create_all_visualizations(self):
        """Create all visualization plots as separate HTML files."""
        self.results = self.benchmark_ref.results.data[self.benchmark_ref.papers_hash]
        self.methods = [f"{_[0]}_{_[1]}" if _[1] else f"{_[0]}" for _ in self.benchmark_ref.models]
        # temporarily pull methods from pkl file again
        self.methods = list(self.results.keys())
        self._sort_methods()
        self.out_dir = self.benchmark_ref.hashed_and_dated_output_dir

        logger.info("Creating interactive visualizations...")

        self._aggregate_metric_scores()
        self._aggregate_execution_times()
        self._calculate_acceptance_rates()
        self._calculate_insufficient_findings_rates()
        self._calculate_token_costs()
        self._calculate_final_combined_scores()

        self._create_metric_comparison_plot()
        self._create_length_analysis_plot()
        self._create_radar_chart()

        self._create_execution_time_boxplot()
        self._create_grouped_execution_time_boxplot()
        self._create_metric_correlation_matrix()
        self._create_rank_heatmap()
        self._create_tradeoff_3d()
        self._create_pareto_bubble()
        self._create_alignscore_only_plot()
        self._create_group_bar_chart()
        try:
            self._create_llm_comparison_plot()
        except Exception as exc:
            logger.error("Failed to create llm comparison plot: %s", exc)

        logger.info(f"Interactive visualizations saved to {self.out_dir}")

    def _sort_methods(self):
        def sort_key(method):
            if method.startswith('local:'):
                return 0, method
            elif method.startswith('huggingface_'):
                return 1, method
            elif method.startswith('ollama_'):
                return 2, method
            else:
                return 3, method

        self.methods.sort(key=sort_key)

    def _aggregate_metric_scores(self):
        for method in self.methods:
            _m = self.results[method]
            valid_metrics = []

            for metric in self.metrics:
                try:
                    metric_data = metric.getter(_m.method_name)
                    if metric_data["mean"] >= 0.0:
                        valid_metrics.append(metric_data)
                except (KeyError, TypeError):
                    continue

            if valid_metrics:
                self.metric_scores[_m.method_name] = {
                    "mean": np.mean([m["mean"] for m in valid_metrics]),
                    "min": min(m["min"] for m in valid_metrics),
                    "max": max(m["max"] for m in valid_metrics),
                    "std": np.mean([m["std"] for m in valid_metrics])
                }

    def _aggregate_execution_times(self):
        """Calculate normalized execution times (inverted: faster = higher score)."""
        mean_times = {m: np.mean(self.results[m].execution_times) for m in self.methods}

        max_exec_time = max(mean_times.values())
        min_exec_time = min(mean_times.values())

        if max_exec_time == min_exec_time:
            self.normalized_exec_times = dict.fromkeys(self.methods, 1.0)
            return

        self.normalized_exec_times = {
            m: {
                "min": np.min(self.results[m].execution_times),
                "max": np.max(self.results[m].execution_times),
                "mean": 1 - (mean_times[m] - min_exec_time) / (max_exec_time - min_exec_time),
                "std": np.std(self.results[m].execution_times),
            } for m in self.methods
        }

    def _calculate_acceptance_rates(self):
        """Calculate acceptance rates based on how many responses are present."""
        self.max_publications = max(len(self.results[m].execution_times) for m in self.methods)

        for method in self.methods:
            publications_processed = len(self.results[method].execution_times)
            coverage_ratio = publications_processed / self.max_publications if self.max_publications > 0 else 0

            self.acceptance_scores[method] = {
                "mean": coverage_ratio,
                "min": coverage_ratio,
                "max": coverage_ratio,
                "std": 0.0
            }

    def _calculate_insufficient_findings_rates(self):
        """Calculate summary rates and normalize to 0-1 where 1 is best (fewer insufficient findings)."""
        for method in self.methods:
            insufficient_findings = [
                _ for _ in self.results[method].summaries
                if _ == "INSUFFICIENT_FINDINGS"
                   or _.startswith("INSUFFICIENT")
                   or _ == "'**'"
                   or len(_) < 10
            ]
            insufficient_ratio = len(insufficient_findings) / self.max_publications

            normalized_score = 1 - insufficient_ratio

            self.insufficient_findings_rates[method] = {
                "mean": normalized_score,
                "min": normalized_score,
                "max": normalized_score,
                "std": 0.0
            }

    @staticmethod
    def _add_mean_min_max_std_and_handle_none(tokens: list, method: str, target_dict: dict):
        valid_tokens = [t for t in tokens if t is not None]

        if not valid_tokens:
            target_dict[method] = {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0
            }
        else:
            target_dict[method] = {
                "mean": np.mean(valid_tokens),
                "min": np.min(valid_tokens),
                "max": np.max(valid_tokens),
                "std": np.std(valid_tokens)
            }


    def _calculate_token_costs(self):
        """Calculate token costs and normalize to 0-1 where 1 is best (lower cost)."""
        raw_input_costs = {}
        raw_output_costs = {}

        for method in self.methods:
            self._add_mean_min_max_std_and_handle_none(self.results[method].input_tokens, method, raw_input_costs)
            self._add_mean_min_max_std_and_handle_none(self.results[method].output_tokens, method, raw_output_costs)

        output_means = [raw_output_costs[method]["mean"] for method in self.methods]
        max_output = max(output_means)
        min_output = min(output_means)

        if max_output != min_output:
            for method in self.methods:
                raw_mean = raw_output_costs[method]["mean"]
                normalized_mean = 1 - (raw_mean - min_output) / (max_output - min_output)

                self.output_token_costs[method] = {
                    "mean": normalized_mean,
                    "min": normalized_mean,
                    "max": normalized_mean,
                    "std": 0.0
                }
        else:
            for method in self.methods:
                self.output_token_costs[method] = {
                    "mean": 1.0,
                    "min": 1.0,
                    "max": 1.0,
                    "std": 0.0
                }

        self.input_token_costs = raw_input_costs

    def _calculate_final_combined_scores(self):
        """Calculate final combined score from all normalized metrics."""
        # check whether to include costs
        output_token_means = {method: self.output_token_costs[method]["mean"] for method in self.methods}
        include_cost = any(tokens > 0 for tokens in output_token_means.values())

        if include_cost:
            quality_weight = 0.70
            speed_weight = 0.1 # 0.2
            acceptance_weight = 0.1 # 0.05
            cost_weight = 0.1 # 0.05
        else:
            quality_weight = 0.70
            speed_weight = 0.15 # 0.225
            acceptance_weight = 0.15 # 0.075
            cost_weight = 0.0

        for method in self.methods:
            metric_sum_score = self.metric_scores.get(method, {}).get("mean", 0)
            speed_score = self.normalized_exec_times.get(method, {}).get("mean", 0)
            acceptance_score = self.acceptance_scores.get(method, {}).get("mean", 0)
            cost_score = self.output_token_costs.get(method, {}).get("mean", 0) if include_cost else 0

            combined_score = (
                    metric_sum_score * quality_weight +
                    speed_score * speed_weight +
                    acceptance_score * acceptance_weight +
                    cost_score * cost_weight
            )

            metric_sum_std = self.metric_scores.get(method, {}).get("std", 0)
            speed_std = self.normalized_exec_times.get(method, {}).get("std", 0)
            acceptance_std = self.acceptance_scores.get(method, {}).get("std", 0)

            combined_std = np.sqrt(
                (metric_sum_std * quality_weight) ** 2 +
                (speed_std * speed_weight) ** 2 +
                (acceptance_std * acceptance_weight) ** 2
            )

            self.combined_final_scores[method] = {
                "mean": combined_score,
                "min": combined_score - combined_std,
                "max": combined_score + combined_std,
                "std": combined_std
            }

    def _create_metric_comparison_plot(self):
        """Create ROUGE comparison plot with BERTScore with filled confidence bands."""

        group_map = MODEL_GROUPS

        model_to_group = {model: group for group, models in group_map.items() for model in models}
        # For drawing brackets
        self.group_labels = [model_to_group.get(model, "UNCATEGORIZED") for model in self.methods]

        best_overall_method = max(self.methods, key=lambda m: self.combined_final_scores[m]["mean"])

        fig = go.Figure()

        metrics = [
            *self.metrics,
            *self.aggregates,
            *self.performances,
            *self.overall,
        ]

        for i, metric in enumerate(metrics):
            color = self.colors[i % len(self.colors)] if metric.line_override is None else metric.line_override["color"]
            low_opacity_color = color.replace("rgb(", "rgba(").replace(")", ", 0.2)")

            # Main line (mean values)
            fig.add_trace(
                go.Scatter(
                    x=self.methods,
                    y=[metric.getter(m)["mean"] for m in self.methods],
                    mode='lines+markers',
                    name=metric.label,
                    legendgroup=metric.category if metric.category else "Other",
                    legendgrouptitle_text=metric.category if metric.category else "Other",
                    line={
                        "color": color,
                        "width": 2
                    } if metric.line_override is None else metric.line_override,
                    marker={
                        "size": 8
                    },
                    opacity=1.0,
                    hovertemplate=str(f'<b>%{{x}}</b>'
                                      f'<br>{metric.label}: %{{y:.3f}}'
                                      f'<br>Min: %{{customdata[0]:.3f}}'
                                      f'<br>Max: %{{customdata[1]:.3f}}'
                                      f'<br>Std: %{{customdata[2]:.3f}}'
                                      f'<extra></extra>'),
                    customdata=[[metric.getter(m)["min"],
                                 metric.getter(m)["max"],
                                 metric.getter(m)["std"]] for m in self.methods]
                )
            )

            if not metric.error_bars:
                continue

            # Upper bound (invisible line)
            fig.add_trace(
                go.Scatter(
                    x=self.methods,
                    y=[metric.getter(m)["max"] for m in self.methods],
                    mode='lines',
                    line={"width": 0},
                    showlegend=False,
                    legendgroup=metric.category if metric.category else "Other",
                    hoverinfo='skip'
                )
            )

            # Lower bound with fill
            fig.add_trace(
                go.Scatter(
                    x=self.methods,
                    y=[metric.getter(m)["min"] for m in self.methods],
                    mode='lines',
                    line={"width": 0},
                    fill='tonexty',
                    fillcolor=low_opacity_color,
                    showlegend=False,
                    legendgroup=metric.category if metric.category else "Other",
                    hoverinfo='skip'
                )
            )

        # Prepare threshold data
        threshold_scores = {
            "Overall": {
                "scores": [self.combined_final_scores[m]["mean"] for m in self.methods],
                "dash_style": "dash",
            },
            "Metrics Mean": {
                "scores": [self.metric_scores[m]["mean"] for m in self.methods],
                "dash_style": "dot",
            },
            "Speed": {
                "scores": [self.normalized_exec_times[m]["mean"] for m in self.methods],
                "dash_style": "dashdot",
            }
        }

        # Generate threshold lines and annotations
        shapes, annotations = self._create_threshold_lines_and_annotations(threshold_scores)

        # Create enhanced labels using overall scores for ranking
        enhanced_labels = []
        overall_scores = [self.combined_final_scores[m]["mean"] for m in self.methods]
        percentile_75 = np.percentile(overall_scores, 75)
        percentile_90 = np.percentile(overall_scores, 90)

        for method in self.methods:
            score = self.combined_final_scores[method]["mean"]

            if method == best_overall_method:
                prefix = '<span style="color:red;">🥇 BEST</span> '
            elif score >= percentile_90:
                prefix = '<span style="color:orange;">🥈 TOP 90%</span> '
            elif score >= percentile_75:
                prefix = '<span style="color:blue;">🥉 TOP 75%</span> '
            else:
                prefix = ''

            enhanced_labels.append(f"{prefix}{method}")

        fig.update_layout(
            title=f"Summary Quality and Performance Analysis based on {self.max_publications} publications",
            xaxis_title="Methods",
            yaxis_title="Score",
            hovermode='closest',
            yaxis={
                "range": [0, 1.01]
            },
            xaxis={
                "tickmode": "array",
                "tickvals": list(range(len(self.methods))),
                "ticktext": enhanced_labels,
                "tickangle": -45,
                "tickfont": {"size": 10}
            },
            margin={"b": 10},
            legend={
                "groupclick": "togglegroup"
            },
            shapes=shapes,
            annotations=annotations
        )

        output_path = self.out_dir / "metric_comparison.html"
        pyo.plot(fig, filename=str(output_path), auto_open=False)

    def _create_threshold_lines_and_annotations(self, scores_dict):
        """Create threshold lines and annotations"""
        shapes = []
        annotations = []

        for score_name, config in scores_dict.items():
            scores = config["scores"]
            dash_style = config.get("dash_style", "dash")

            percentile_75 = np.percentile(scores, 75)
            percentile_90 = np.percentile(scores, 90)

            # Create threshold lines
            for percentile, color in [(percentile_90, "orange"), (percentile_75, "blue")]:
                shapes.append({
                    "type": "line",
                    "x0": -0.5,
                    "x1": len(self.methods) - 0.5,
                    "y0": percentile,
                    "y1": percentile,
                    "line": {
                        "color": color,
                        "width": 2,
                        "dash": dash_style
                    }
                })

            # Create annotations
            x_pos = len(self.methods) - 0.5
            x_anchor = "left"

            annotations.extend([
                {
                    "x": x_pos,
                    "y": percentile_90,
                    "text": f"{score_name} 90th: {percentile_90:.3f}",
                    "showarrow": False,
                    "xanchor": x_anchor,
                    "yanchor": "bottom",
                    "bgcolor": "rgba(255,255,255,0.8)",
                    "bordercolor": "orange",
                    "borderwidth": 1,
                    "font": {"color": "orange", "size": 10}
                },
                {
                    "x": x_pos,
                    "y": percentile_75,
                    "text": f"{score_name} 75th: {percentile_75:.3f}",
                    "showarrow": False,
                    "xanchor": x_anchor,
                    "yanchor": "top",
                    "bgcolor": "rgba(255,255,255,0.8)",
                    "bordercolor": "blue",
                    "borderwidth": 1,
                    "font": {"color": "blue", "size": 10}
                }
            ])

        return shapes, annotations

    def _create_group_bar_chart(self):
        """Bar chart showing Metric Mean Score for each model group."""

        group_map = MODEL_GROUPS

        group_names = list(group_map.keys())

        metric_means = []

        for group in group_names:
            models = group_map[group]
            metric_scores = [self.metric_scores[m]["mean"] for m in models if m in self.metric_scores]
            metric_means.append(np.mean(metric_scores) if metric_scores else 0)

        # create grouped bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=group_names,
            y=metric_means,
            name="Metric Mean Score",
            marker_color="rgb(26, 118, 255)",
            text=[f"{v:.3f}" for v in metric_means],
            textposition="auto"
        ))

        fig.update_layout(
            title="Comparison of Model Groups by Metric Mean Score",
            xaxis_title="Model Groups",
            yaxis_title="Metric Mean Score",
            yaxis=dict(range=[0, 1.0]),
            barmode="group",
            bargap=0.3,
            legend=dict(x=0.5, y=1.1, orientation="h", xanchor="center"),
            margin=dict(l=60, r=40, t=80, b=100),
            height=500
        )

        output_path = self.out_dir / "group_bar_chart.html"
        pyo.plot(fig, filename=str(output_path), auto_open=False)

    def _create_llm_comparison_plot(self):
        """Compare General-purpose vs Reasoning-oriented LLMs across key metrics."""
        group_map = {
            "General-purpose LLMs": {"models": deepcopy(MODEL_GROUPS["General-purpose LLMs"])},
            "Reasoning-oriented LLMs": {"models": deepcopy(MODEL_GROUPS["Reasoning-oriented LLMs"])},
        }

        metrics = {
            "Surface-Level Metrics": lambda m: np.mean([
                self.results[m].rouge_scores["rouge1"]["mean"],
                self.results[m].rouge_scores["rouge2"]["mean"],
                self.results[m].rouge_scores["rougeL"]["mean"],
                self.results[m].bleu_scores["mean"],
                self.results[m].meteor_scores["mean"]
            ]),
            "Embedding-Based Metrics": lambda m: np.mean([
                self.results[m].roberta_scores["f1"]["mean"],
                self.results[m].deberta_scores["f1"]["mean"],
                self.results[m].mpnet_content_coverage_scores["mean"],
                self.results[m].alignscore_scores["mean"]
            ]),
            "Execution Time": lambda m: self.normalized_exec_times.get(m, {}).get("mean"),
            "% Within Bounds": lambda m: self.results[m].length_stats["within_bounds_pct"] / 100,
            "Metric Mean Score": lambda m: self.metric_scores.get(m, {}).get("mean")  # moved to end
        }

        categories = list(metrics.keys())
        group_values = {group: [] for group in group_map}

        for group, info in group_map.items():
            for metric_label, fn in metrics.items():
                vals = [fn(m) for m in info["models"] if fn(m) is not None]
                group_values[group].append(np.mean(vals) if vals else 0)

        fig = go.Figure()

        colors = {
            "General-purpose LLMs": "rgb(99, 110, 250)",
            "Reasoning-oriented LLMs": "rgb(239, 85, 59)"
        }

        for group, values in group_values.items():
            outlines = [
                3 if category == "Metric Mean Score" else 1 for category in categories
            ]

            fig.add_trace(go.Bar(
                x=categories,
                y=values,
                name=group,
                marker=dict(
                    color=colors.get(group),
                    line=dict(
                        color="black",
                        width=outlines
                    )
                ),
                text=[f"{v:.3f}" for v in values],
                textposition="auto"
            ))

        fig.update_layout(
            title="Detailed Comparison: General-purpose vs Reasoning-oriented LLMs",
            yaxis_title="Score (normalized)",
            barmode="group",
            xaxis_tickangle=-45,
            legend_title="Model Group",
            margin=dict(l=40, r=40, t=60, b=120)
        )

        output_path = self.out_dir / "llm_comparison.html"
        pyo.plot(fig, filename=str(output_path), auto_open=False)

    def _create_alignscore_only_plot(self, out_name="alignscore_only.html"):
        """Bar chart of AlignScore per model with asymmetric error bars (min/max)."""
        rows = []
        for m in self.methods:
            s = getattr(self.results[m], "alignscore_scores", None)
            if not s or s.get("mean") is None:
                continue
            rows.append({
                "method": m,
                "mean": float(s["mean"]),
                "min": float(s["min"]),
                "max": float(s["max"]),
                "std": float(s["std"])
            })

        if not rows:
            logger.warning("No overlapping methods between results and AlignScore; skipping plot.")
            return

        rows.sort(key=lambda r: r["mean"], reverse=True)

        x = [r["method"] for r in rows]
        y = [r["mean"] for r in rows]
        err_plus  = [max(0.0, r["max"] - r["mean"]) for r in rows]
        err_minus = [max(0.0, r["mean"] - r["min"]) for r in rows]

        def family(name: str) -> str:
            if name.startswith("local:"): return "Local"
            if name.startswith("huggingface_"): return "HuggingFace"
            if name.startswith("ollama_"): return "Ollama"
            if name.startswith("openai_"): return "OpenAI"
            if name.startswith("anthropic_"): return "Anthropic"
            if name.startswith("mistral_"): return "Mistral"
            return "Other"

        families = [family(m) for m in x]
        fam_set = list(dict.fromkeys(families))
        color_map = {fam: self.colors[i % len(self.colors)] for i, fam in enumerate(fam_set)}
        bar_colors = [color_map[f] for f in families]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=x,
                    y=y,
                    marker={"color": bar_colors},
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=err_plus,
                        arrayminus=err_minus,
                        thickness=1.5,
                        width=2,
                        visible=True
                    ),
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "AlignScore (mean): %{y:.4f}<br>"
                        "min–max: %{customdata[0]:.4f} – %{customdata[1]:.4f}<br>"
                        "std: %{customdata[2]:.4f}<extra></extra>"
                    ),
                    customdata=[[r["min"], r["max"], r["std"]] for r in rows],
                )
            ]
        )

        for fam, color in color_map.items():
            fig.add_trace(
                go.Bar(
                    x=[None], y=[None],
                    marker={"color": color},
                    name=fam,
                    showlegend=True
                )
            )

        fig.update_layout(
            title=f"AlignScore by Model (n={len(rows)})",
            xaxis_title="Model",
            yaxis_title="AlignScore",
            yaxis=dict(range=[0, 1.01]),
            bargap=0.2,
            hovermode="closest",
            legend_title_text="Family",
            xaxis=dict(tickangle=-45, tickfont={"size": 10}),
            margin={"b": 40}
        )

        output_path = self.out_dir / out_name
        pyo.plot(fig, filename=str(output_path), auto_open=False)

    def _create_length_analysis_plot(self):
        """Create length analysis plot with compliance breakdown."""

        # shorten some method names
        def shorten_name(name):
            short = name.replace("huggingface_", "hf_")
            if "hf_AlgorithmicResearchGroup/" in short:
                short = short.replace("hf_AlgorithmicResearchGroup/", "hf_AlgorithmicResearchGroup/<br>")
            return short

        shortened_methods = [shorten_name(m) for m in self.methods]

        mid_idx = len(self.methods) // 2
        methods_top = shortened_methods[:mid_idx]
        methods_bottom = shortened_methods[mid_idx:]

        def split_percentages(percent_list):
            return percent_list[:mid_idx], percent_list[mid_idx:]

        within_bounds_top, within_bounds_bottom = split_percentages(
            [self.results[m].length_stats['within_bounds_pct'] for m in self.methods])
        too_short_top, too_short_bottom = split_percentages(
            [self.results[m].length_stats['too_short_pct'] for m in self.methods])
        too_long_top, too_long_bottom = split_percentages(
            [self.results[m].length_stats['too_long_pct'] for m in self.methods])

        fig = make_subplots(
            rows=2, cols=1,
            shared_yaxes=True,
            vertical_spacing=0.2
        )

        # helper to add traces
        def add_bar_traces(methods, wb, ts, tl, row, showlegend):
            fig.add_trace(go.Bar(
                x=methods,
                y=wb,
                name='Within Bounds',
                marker_color='lightgreen',
                text=[f'{v:.1f}%' if v >= 5 else '' for v in wb],
                textposition='inside',
                hovertemplate='<b>%{x}</b><br>Within Bounds: %{y:.1f}%<extra></extra>',
                showlegend=showlegend
            ), row=row, col=1)

            fig.add_trace(go.Bar(
                x=methods,
                y=ts,
                name='Too Short',
                marker_color='lightcoral',
                text=[f'{v:.1f}%' if v >= 5 else '' for v in ts],
                textposition='inside',
                hovertemplate='<b>%{x}</b><br>Too Short: %{y:.1f}%<extra></extra>',
                showlegend=showlegend
            ), row=row, col=1)

            fig.add_trace(go.Bar(
                x=methods,
                y=tl,
                name='Too Long',
                marker_color='lightyellow',
                text=[f'{v:.1f}%' if v >= 5 else '' for v in tl],
                textposition='inside',
                hovertemplate='<b>%{x}</b><br>Too Long: %{y:.1f}%<extra></extra>',
                showlegend=showlegend
            ), row=row, col=1)

        # add top and bottom rows
        add_bar_traces(methods_top, within_bounds_top, too_short_top, too_long_top, row=1, showlegend=True)
        add_bar_traces(methods_bottom, within_bounds_bottom, too_short_bottom, too_long_bottom, row=2, showlegend=False)


        # final layout
        fig.update_layout(
            title_text=f"Summary Length Compliance (Target: {self.min_words}-{self.max_words} words)",
            title_x=0.5,
            hovermode='closest',
            barmode='stack',
            height=950,
            margin=dict(t=120, b=100)
        )

        # axis labels
        fig.update_yaxes(title_text="Percentage (%)", row=1, col=1)
        fig.update_xaxes(title_text="Methods", row=2, col=1)

        # fontsize and angle for model names
        fig.update_xaxes(tickfont=dict(size=10), tickangle=30)
        fig.update_yaxes(range=[0, 105])

        output_path = self.out_dir / "length_analysis.html"
        pyo.plot(fig, filename=str(output_path), auto_open=False)

    def _create_radar_chart(self, top_n: int = 10):
        """Create radar chart for top n performing methods ranked by aggregate score."""
        metrics = [
            *self.metrics,
            *self.aggregates,
            *self.performances,
            self.length_within_bounds,
        ]

        top_methods = dict(sorted(
            {
                _method: {
                    metric.label: metric.getter(_method)["mean"]
                    for metric in metrics
                } for _method in self.methods
            }.items(),
            key=lambda x: sum(x[1].values()),
            reverse=True
        )[:top_n])

        fig = go.Figure()

        for i, (method, vals) in enumerate(top_methods.items()):
            aggregate_score = sum(vals.values())
            rank = i + 1

            hover_text = f'<b>{method}</b><br>'
            hover_text += f'Aggregate Score: {sum(vals.values()):.3f}<br>'
            hover_text += '<br>'.join([f'{metric}: {score:.3f}' for metric, score in vals.items()])

            fig.add_trace(
                go.Scatterpolar(
                    r=list(vals.values()),
                    theta=list(vals.keys()),
                    fill='toself',
                    name=f"#{rank} {method} ({aggregate_score:.3f})",
                    line_color=self.colors[i % len(self.colors)],
                    hovertemplate=f'<b>{method}</b><br>' +
                                  'Metric: %{theta}<br>' +
                                  'Score: %{r:.3f}<extra></extra>',
                    text=hover_text,
                    hoverinfo='text+name'
                )
            )

        fig.update_layout(
            polar={
                "radialaxis": {
                    "visible": True,
                    "range": [0, 1]
                },
            },
            title=f"Top {top_n} Methods by Aggregate Performance Score",
            hovermode='closest'
        )

        output_path = self.out_dir / "radar_chart.html"
        pyo.plot(fig, filename=str(output_path), auto_open=False)

    def _create_execution_time_boxplot(self):
        """Create boxplot for execution time distribution per method."""
        fig = go.Figure()

        # exclude Traditional Methods and Encoder-Decoder Models
        excluded_models = set(
            MODEL_GROUPS["Traditional Methods"] + MODEL_GROUPS["Encoder-Decoder Models"]
        )

        plot_methods = [
            m for m in self.methods
            if m in self.results and m not in excluded_models
        ]

        for i, method in enumerate(plot_methods):
            times = self.results[method].execution_times
            fig.add_trace(go.Box(
                y=times,
                name=method,
                marker_color=self.colors[i % len(self.colors)],
                boxpoints="outliers",
                hovertemplate=f'<b>{method}</b><br>Time: %{{y:.2f}} sec<extra></extra>'
            ))

        # vertical guidelines for each boxplot
        shapes = []
        for i in range(len(plot_methods)):
            shapes.append(dict(
                type="line",
                xref="x",
                yref="paper",
                x0=plot_methods[i],
                x1=plot_methods[i],
                y0=0,
                y1=1,
                line=dict(
                    color="rgba(0, 0, 0, 0.1)",
                    width=1,
                    dash="dot"
                ),
                layer="below"
            ))

        fig.update_layout(
            title="Execution Time Distribution per Method (excluding 'Traditional Methods' & 'Encoder-Decoder Models')",
            yaxis_title="Time (seconds, log scale)",
            xaxis_title="Methods",
            boxmode="group",
            boxgap=0.3,
            boxgroupgap=0.2,
            showlegend=False,
            yaxis=dict(
                type="log",
                tickvals=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
                ticktext=["1s", "2s", "5s", "10s", "20s", "50s", "100s", "200s", "500s", "1000s"]
            ),
            shapes=shapes
        )

        fig.update_traces(
            width=0.6,
            line=dict(width=2)
        )

        output_path = self.out_dir / "execution_time_distribution.html"
        pyo.plot(fig, filename=str(output_path), auto_open=False)

    def _create_grouped_execution_time_boxplot(self):
        """Create grouped boxplot comparing execution times across general-purpose vs reasoning-oriented LLMs."""

        group_map = {
            "General-purpose LLMs": {"models": deepcopy(MODEL_GROUPS["General-purpose LLMs"])},
            "Reasoning-oriented LLMs": {"models": deepcopy(MODEL_GROUPS["Reasoning-oriented LLMs"])},
        }

        fig = go.Figure()

        group_colors = {
            "General-purpose LLMs": "rgba(66, 135, 245, 0.6)",
            "Reasoning-oriented LLMs": "rgba(245, 135, 66, 0.6)"
        }

        for group_name, group_info in group_map.items():
            all_times = []

            for model in group_info["models"]:
                if model in self.results and hasattr(self.results[model], "execution_times"):
                    all_times.extend(self.results[model].execution_times)

            fig.add_trace(go.Box(
                y=all_times,
                name=group_name,
                marker_color=group_colors.get(group_name, "rgba(100,100,100,0.6)"),
                boxpoints="outliers",
                hovertemplate=f"<b>{group_name}</b><br>Time: %{{y:.2f}} sec<extra></extra>"
            ))

        fig.update_layout(
            title="Execution Time: General-purpose vs Reasoning-oriented LLMs",
            yaxis_title="Time (seconds, log scale)",
            xaxis_title="Model Group",
            boxmode="group",
            boxgap=0.4,
            boxgroupgap=0.2,
            showlegend=False,
            yaxis=dict(
                type="log",
                tickvals=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
                ticktext=["1s", "2s", "5s", "10s", "20s", "50s", "100s", "200s", "500s", "1000s"]
            )
        )

        fig.update_traces(
            width=0.5,
            line=dict(width=2)
        )

        output_path = self.out_dir / "grouped_execution_time_distribution.html"
        pyo.plot(fig, filename=str(output_path), auto_open=False)

    
    def _create_metric_correlation_matrix(self):
        """
        Correlate evaluation metrics across models (each metric is a variable; observations = models).
        Only includes true quality metrics (e.g., ROUGE, BERTScore, METEOR, BLEU, mpnet).
        Uses Pearson correlation coefficient.
        """
        chosen_metrics = [*self.metrics]
        labels = [m.label for m in chosen_metrics]
        categories = [m.category for m in chosen_metrics]

        data = []
        for metric in chosen_metrics:
            row = []
            for method in self.methods:
                try:
                    val = metric.getter(method)["mean"]
                except Exception:
                    val = np.nan
                row.append(val)
            data.append(row)

        A = np.array(data, dtype=float)

        with np.errstate(invalid="ignore"):
            corr = np.ma.corrcoef(np.ma.masked_invalid(A))

        corr_filled = np.array(corr.filled(np.nan))
        np.fill_diagonal(corr_filled, 1.0)
        corr_filled = np.nan_to_num(corr_filled, nan=0.0)

        def get_category_range(cat_name):
            idxs = [i for i, c in enumerate(categories) if c == cat_name]
            return (min(idxs), max(idxs)) if idxs else (None, None)

        surf_start, surf_end = get_category_range(SURFACE_LEVEL)
        emb_start, emb_end = get_category_range(EMBEDDING_BASED)

        category_colors = {
            SURFACE_LEVEL: "rgba(65, 105, 225, 0.4)",
            EMBEDDING_BASED: "rgba(34, 139, 34, 0.4)",
        }

        shapes = []
        annotations = []

        def add_category_band(start, end, color, label_text, text_color):
            if start is None:
                return

            shapes.append(dict(
                type="rect",
                xref="x", yref="paper",
                x0=start - 0.5, x1=end + 0.5,
                y0=1.02, y1=1.07,
                fillcolor=color,
                line=dict(width=0)
            ))

            shapes.append(dict(
                type="rect",
                xref="paper", yref="y",
                x0=1.01, x1=1.06,
                y0=start - 0.5, y1=end + 0.5,
                fillcolor=color,
                line=dict(width=0)
            ))

            annotations.append(dict(
                x=(start + end) / 2,
                y=1.045,
                xref="x",
                yref="paper",
                text=label_text,
                showarrow=False,
                font=dict(size=13, color=text_color, family="Arial", weight="bold"),
                yanchor="middle"
            ))

            annotations.append(dict(
                x=1.050,
                y=(start + end) / 2,
                xref="paper",
                yref="y",
                text=label_text,
                showarrow=False,
                font=dict(size=13, color=text_color, family="Arial", weight="bold"),
                textangle=90,
                yanchor="middle"
            ))

        add_category_band(surf_start, surf_end, category_colors[SURFACE_LEVEL], "Surface-level", "royalblue")
        add_category_band(emb_start, emb_end, category_colors[EMBEDDING_BASED], "Embedding-based", "forestgreen")

        # heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_filled,
                x=labels,
                y=labels,
                zmin=-1, zmax=1,
                colorscale="RdBu",
                reversescale=True,
                colorbar=dict(title="ρ", x=1.12)
            )
        )

        for i in range(len(labels)):
            for j in range(len(labels)):
                if j >= i:
                    annotations.append(dict(
                        x=labels[j], y=labels[i],
                        text=f"{corr_filled[i, j]:.2f}",
                        showarrow=False,
                        font=dict(size=11, color="black")
                    ))

        fig.update_layout(
            title="Correlation Between Metrics (across models)",
            xaxis=dict(tickangle=60),
            yaxis=dict(autorange="reversed"),
            annotations=annotations,
            shapes=shapes,
            margin=dict(l=140, r=200, t=100, b=140),
            height=750,
            width=880
        )

        output_path = self.out_dir / "metric_correlation_matrix.html"
        pyo.plot(fig, filename=str(output_path), auto_open=False)
    
    def _create_rank_heatmap(self):
        """
        Shows ranks of models per metric in a heatmap.
        Lower rank = better (1 is best).
        Models sorted by average rank across metrics (best at top).
        """
        # metrics to rank (quality only)
        metrics = [*self.metrics]
        metric_labels = [m.label for m in metrics]

        value_rows = []
        for method in self.methods:
            row = []
            for metric in metrics:
                try:
                    val = metric.getter(method)["mean"]
                except Exception:
                    val = math.nan
                row.append(val)
            value_rows.append(row)
        
        A = np.array(value_rows, dtype=float)

        # convert to rank matrix (1 = best)
        ranks = np.zeros_like(A, dtype=float)
        for j in range(A.shape[1]):
            col = A[:, j]
            if np.all(np.isnan(col)):
                ranks[:, j] = np.nan
                continue
            col_nonan = np.nan_to_num(col, nan=-1e12)
            sorted_idx = np.argsort(-col_nonan)
            rank_col = np.empty_like(sorted_idx, dtype=float)
            rank_col[sorted_idx] = np.arange(1, len(sorted_idx) + 1)
            rank_col[np.isnan(col)] = np.nan
            ranks[:, j] = rank_col

        # compute aggregate rank per model to sort (mean of available ranks)
        avg_ranks = np.nanmean(ranks, axis=1)
        sort_idx = np.argsort(avg_ranks)

        # reorder methods and matrices
        methods_sorted = [self.methods[i] for i in sort_idx]
        ranks_sorted = ranks[sort_idx, :]

        # shorten some method names
        display_methods = [
            m.replace("huggingface_", "hf_") if m.startswith("huggingface_") else m
            for m in methods_sorted
        ]

        if np.all(np.isnan(ranks_sorted)):
            max_rank = 1.0
        else:
            max_rank = int(np.nanmax(ranks_sorted))

        Z = -ranks_sorted
        zmin, zmax = -max_rank, -1

        # green (best) → yellow → red (worst)
        colorscale = [
            [0.0, "rgb(200, 0, 0)"],
            [0.5, "rgb(255, 215, 0)"],
            [1.0, "rgb(0, 150, 0)"],
        ]

        n_ticks = min(6, max_rank)
        if n_ticks <= 1:
            tickvals = [-1]
            ticktext = ["1"]
        else:
            tickvals = np.linspace(-1, -max_rank, n_ticks)
            ticktext = [str(int(-v)) for v in tickvals]

        fig = go.Figure(
            data=go.Heatmap(
                z=Z,
                x=metric_labels,
                y=display_methods,
                zmin=zmin,
                zmax=zmax,
                colorscale=colorscale,
                zauto=False,
                colorbar=dict(
                    title="Rank (1 = best)",
                    tickmode="array",
                    tickvals=tickvals.tolist(),
                    ticktext=ticktext,
                ),
            )
        )

        # annotations: show rank number + 🥇🥈🥉 for top-3
        annotations = []
        for i in range(ranks_sorted.shape[0]):
            for j in range(ranks_sorted.shape[1]):
                r = ranks_sorted[i, j]
                if np.isnan(r):
                    continue
                r_int = int(r)
                medal = " 🥇" if r_int == 1 else (" 🥈" if r_int == 2 else (" 🥉" if r_int == 3 else ""))
                annotations.append(
                    dict(
                        x=metric_labels[j],
                        y=display_methods[i],
                        text=f"{r_int}{medal}",
                        showarrow=False,
                        font=dict(size=11, color="white"),
                    )
                )

        fig.update_layout(
            title="Model Ranks per Metric (higher is better; models sorted by average rank)",
            annotations=annotations,
            xaxis=dict(tickangle=45),
            margin=dict(l=200, r=40, t=60, b=120),
        )

        fig.update_yaxes(autorange="reversed")
        
        output_path = self.out_dir / "rank_heatmap.html"
        pyo.plot(fig, filename=str(output_path), auto_open=False)

    def _create_tradeoff_3d(self):
        """
        3D trade-off: Quality (↑) vs Speed (↑) vs Tokens (↓).
        All models as points; Pareto frontier shown as a translucent surface.
        """
        points = []
        names = []
        hover = []
        for m in self.methods:
            q = self.metric_scores.get(m, {}).get("mean", None)
            s = self.normalized_exec_times.get(m, {}).get("mean", None)

            t = self.input_token_costs.get(m, {}).get("mean", None)
            if q is None or s is None or t is None or np.isnan(q) or np.isnan(s) or np.isnan(t):
                continue
            points.append((q, s, t))
            names.append(m)
            hover.append(f"<b>{m}</b><br>Quality: {q:.3f}<br>Speed: {s:.3f}<br>Tokens: {t:.1f}")

        if not points:
            logger.warning("No points available for tradeoff 3D plot.")
            return
        
        P = np.array(points)
        Q = P[:, 0]
        S = P[:, 1]
        T = P[:, 2]
        
        def dominates(a, b):
            # a dominates b if: Q_a>=Q_b, S_a>=S_b, T_a<=T_b and at least one strict
            return (a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and
                    ((a[0] > b[0]) or (a[1] > b[1]) or (a[2] < b[2])))
        
        pareto_idx = []
        for i in range(len(P)):
            dominated = False
            for j in range(len(P)):
                if i != j and dominates(P[j], P[i]):
                    dominated = True
                    break
            if not dominated:
                pareto_idx.append(i)
        pareto_idx = np.array(pareto_idx, dtype=int)
        P_pareto = P[pareto_idx]
        names_pareto = [names[i] for i in pareto_idx]

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=Q, y=S, z=T,
            mode="markers+text",
            text=names,
            textposition="top center",
            marker=dict(size=6, opacity=0.9),
            hovertext=hover,
            hoverinfo="text",
            name="Models"
        ))

        # highlight pareto points
        fig.add_trace(go.Scatter3d(
            x=P_pareto[:, 0], y=P_pareto[:, 1], z=P_pareto[:, 2],
            mode="markers",
            marker=dict(size=8, symbol="diamond", color="crimson"),
            hovertext=[hover[i] for i in pareto_idx],
            hoverinfo="text",
            name="Pareto frontier (points)"
        ))

        mesh_added = False
        if len(P_pareto) >= 4:
            try:
                P_trans = P_pareto.copy()
                P_trans[:, 2] = -P_trans[:, 2]
                hull = ConvexHull(P_trans)
                simplices = hull.simplices

                fig.add_trace(go.Mesh3d(
                    x=P_pareto[:, 0],
                    y=P_pareto[:, 1],
                    z=P_pareto[:, 2],
                    i=simplices[:, 0],
                    j=simplices[:, 1],
                    k=simplices[:, 2],
                    opacity=0.25,
                    color="crimson",
                    name="Pareto frontier (surface)",
                    hoverinfo="skip"
                ))
                mesh_added = True
            except QhullError:
                logger.warning("Convex hull failed; plotting Pareto points only.")
            except Exception as e:
                logger.warning(f"Pareto surface construction error: {e}")
        
        title_suffix = " (with surface)" if mesh_added else " (points only)"
        fig.update_layout(
            title="Quality–Speed–Tokens Trade-off" + title_suffix,
            scene=dict(
                xaxis_title="Quality (Metric Mean, ↑ better)",
                yaxis_title="Speed (Normalized, ↑ faster)",
                zaxis_title="Tokens (Mean input, ↓ better)",
                xaxis=dict(range=[max(0, float(np.nanmin(Q)) - 0.05),
                                min(1.0, float(np.nanmax(Q)) + 0.05)]),
                yaxis=dict(range=[max(0, float(np.nanmin(S)) - 0.05),
                                min(1.0, float(np.nanmax(S)) + 0.05)]),
            ),
            legend=dict(itemsizing="constant")
        )

        output_path = self.out_dir / "tradeoff_3d.html"
        pyo.plot(fig, filename=str(output_path), auto_open=False)
    
    def _create_pareto_bubble(self):
        """
        Pareto View considering Quality (Metric Mean Score) vs Speed (Mean Execution Time).
        Each model is a bubble; Pareto frontier shown as a thick dashed line.
        """
        x_quality = []
        y_time = []
        colors = []
        texts = []

        for m in self.methods:
            q = self.metric_scores.get(m, {}).get("mean", 0.0)
            t = float(np.mean(self.results[m].execution_times)) if len(self.results[m].execution_times) else np.nan
            x_quality.append(q)
            y_time.append(t)
            colors.append(m)
            texts.append(m)
        
        fig = px.scatter(
            x=x_quality,
            y=y_time,
            size=[10 for _ in texts],
            color=colors,
            hover_name=texts,
            labels=dict(x="Quality (Metric Mean Score)", y="Mean Execution Time (s)", color="Model"),
            title="Pareto View: Quality vs Speed"
        )

        # invert Y axis visually (top = faster)
        fig.update_yaxes(autorange="reversed")

        # compute Pareto frontier (upper-left: high quality, low time)
        pts = sorted([(x_quality[i], y_time[i], texts[i]) for i in range(len(texts)) if not np.isnan(y_time[i])],
                    key=lambda p: (-p[0], p[1]))
        frontier = []
        best_time = float("inf")
        for q, t, name in pts:
            if t < best_time:
                frontier.append((q, t, name))
                best_time = t

        if len(frontier) >= 2:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in frontier],
                y=[p[1] for p in frontier],
                mode="lines+markers",
                name="Pareto frontier",
                line=dict(width=3, dash="dash", color="red"),
                marker=dict(size=6, color="red")
            ))
        
        # annotate only Pareto frontier models
        for q, t, name in frontier:
            fig.add_annotation(
                x=q,
                y=t,
                text=name,
                showarrow=False,
                font=dict(size=10, color="black"),
                yshift=10
            )
        
        output_path = self.out_dir / "pareto_quality_speed_bubble.html"
        pyo.plot(fig, filename=str(output_path), auto_open=False)

if __name__ == "__main__":
    """This script runs the visualization component of the text summarization benchmark pipeline without benchmarking.
    The following files have to be in place:

    The following files must exist:    
        - /path/to/Repositories/exploration/LLM/Output/text_summarization_benchmark/benchmark.pkl
        - /path/to/Repositories/exploration/LLM/Resources/text_summarization_goldstandard_data.json
    
    Usage:
        cd /path/to/Repositories/exploration/LLM
        uv run Python/src/text_summarization/visualization.py
    """
    from llm_summarization_benchmark.benchmark import SummarizationBenchmark, GOLD_STANDARD_DATA, EvaluationResult
    benchmark = SummarizationBenchmark()
    benchmark.load_papers(GOLD_STANDARD_DATA)
    benchmark.load_results()
    benchmark.visualizer.create_all_visualizations()
