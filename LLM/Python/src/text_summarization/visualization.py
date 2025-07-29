import logging
from collections import namedtuple

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import numpy as np

from typing import TYPE_CHECKING, NamedTuple, Any, Callable

if TYPE_CHECKING:
    from text_summarization.benchmark import SummarizationBenchmark

logger = logging.getLogger(__name__)

SURFACE_LEVEL = "Metrics: Surface-level"
REFERENCE_SIMILARITY = "Metrics: Reference Similarity"
CONTENT_COVERAGE = "Metrics: Content Coverage"
AGGREGATE = "Metrics: Aggregate"
PERFORMANCE = "Performance"
OVERALL = "Overall (70% met, 20% spd, 10% cov)"


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
            Metric("RoBERTa", lambda m: self.results[m].roberta_scores["f1"], None, REFERENCE_SIMILARITY),
            Metric("DeBERTa", lambda m: self.results[m].deberta_scores["f1"], None, REFERENCE_SIMILARITY),
            Metric("METEOR", lambda m: self.results[m].meteor_scores, None, SURFACE_LEVEL),
            Metric("BLEU", lambda m: self.results[m].bleu_scores, None, SURFACE_LEVEL),
            Metric("all-mpnet-base-v2", lambda m: self.results[m].mpnet_content_coverage_scores, None, CONTENT_COVERAGE)
        ]

        self.aggregates = [
            Metric("Metrics Mean Score", lambda m: self.metric_scores[m],
                   {"color": "black", "width": 4}, AGGREGATE, False),

        ]

        self.performances = [
            Metric("Speed Performance", lambda m: self.normalized_exec_times[m],
                    {"color": "rgb(138, 43, 226)", "width": 3}, PERFORMANCE, False),
            Metric("Success Rate", lambda m: self.coverage_scores[m],
                    {"color": "rgb(255, 165, 0)", "width": 3}, PERFORMANCE, False)
        ]

        self.overall = [
            Metric("Overall Score", lambda m: self.combined_final_scores[m],
                   {"color": "rgb(220, 20, 60)", "width": 5}, OVERALL, False)
        ]

        self.length_within_bounds = Metric("Length Within Bounds", lambda m: {
            "mean": self.results[m].length_stats["within_bounds_pct"]/100}, None, None)

        self.metric_scores = {}
        self.normalized_exec_times = {}
        self.coverage_scores = {}
        self.combined_final_scores = {}

        # https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express
        self.colors = px.colors.qualitative.Vivid

    def create_all_visualizations(self):
        """Create all visualization plots as separate HTML files."""
        self.results = self.benchmark_ref.results.data[self.benchmark_ref.papers_hash]
        self.methods = list(self.results.keys())
        self._sort_methods()
        self.out_dir = self.benchmark_ref.hashed_and_dated_output_dir

        logger.info("Creating interactive visualizations...")

        self._aggregate_metric_scores()
        self._aggregate_execution_times()
        self._calculate_coverage_scores()
        self._calculate_final_combined_scores()

        self._create_metric_comparison_plot()
        self._create_length_analysis_plot()
        self._create_radar_chart()

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
        for _m in self.results.values():
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
        methods = self.results.keys()

        mean_times = {m: np.mean(self.results[m].execution_times) for m in methods}

        max_exec_time = max(mean_times.values())
        min_exec_time = min(mean_times.values())

        if max_exec_time == min_exec_time:
            self.normalized_exec_times = dict.fromkeys(methods, 1.0)
            return

        self.normalized_exec_times = {
            m: {
                "min": np.min(self.results[m].execution_times),
                "max": np.max(self.results[m].execution_times),
                "mean": 1 - (mean_times[m] - min_exec_time) / (max_exec_time - min_exec_time),
                "std": np.std(self.results[m].execution_times),
            } for m in methods
        }

    def _calculate_coverage_scores(self):
        """Calculate coverage scores based on how many publications are summarized."""
        methods = self.results.keys()

        self.max_publications = max(len(self.results[m].execution_times) for m in methods)

        for method in methods:
            publications_processed = len(self.results[method].execution_times)
            coverage_ratio = publications_processed / self.max_publications if self.max_publications > 0 else 0

            self.coverage_scores[method] = {
                "mean": coverage_ratio,
                "min": coverage_ratio,
                "max": coverage_ratio,
                "std": 0.0
            }

    def _calculate_final_combined_scores(self):
        """Calculate final combined score from metric sum performance, speed performance, and coverage."""
        methods = self.results.keys()

        quality_weight = 0.70
        speed_weight = 0.20
        coverage_weight = 0.10

        for method in methods:
            metric_sum_score = self.metric_scores.get(method, {}).get("mean", 0)
            speed_score = self.normalized_exec_times.get(method, {}).get("mean", 0)
            coverage_score = self.coverage_scores.get(method, {}).get("mean", 0)

            combined_score = (
                    metric_sum_score * quality_weight +
                    speed_score * speed_weight +
                    coverage_score * coverage_weight
            )

            metric_sum_std = self.metric_scores.get(method, {}).get("std", 0)
            speed_std = self.normalized_exec_times.get(method, {}).get("std", 0)
            coverage_std = self.coverage_scores.get(method, {}).get("std", 0)

            combined_std = np.sqrt(
                (metric_sum_std * quality_weight) ** 2 +
                (speed_std * speed_weight) ** 2 +
                (coverage_std * coverage_weight) ** 2
            )

            self.combined_final_scores[method] = {
                "mean": combined_score,
                "min": combined_score - combined_std,
                "max": combined_score + combined_std,
                "std": combined_std
            }

    def _create_metric_comparison_plot(self):
        """Create ROUGE comparison plot with BERTScore with filled confidence bands."""
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

    def _create_length_analysis_plot(self):
        """Create length analysis plot with compliance breakdown and box plot only."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Compliance', 'Distribution'),
            vertical_spacing=0.15
        )

        within_bounds = [self.results[m].length_stats['within_bounds_pct'] for m in self.methods]
        too_short = [self.results[m].length_stats['too_short_pct'] for m in self.methods]
        too_long = [self.results[m].length_stats['too_long_pct'] for m in self.methods]

        fig.add_trace(
            go.Bar(
                x=self.methods,
                y=within_bounds,
                name='Within Bounds',
                marker_color='lightgreen',
                text=[f'{w:.1f}%' for w in within_bounds],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Within Bounds: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=self.methods,
                y=too_short,
                name='Too Short',
                marker_color='lightcoral',
                text=[f'{s:.1f}%' for s in too_short],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Too Short: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=self.methods,
                y=too_long,
                name='Too Long',
                marker_color='lightyellow',
                text=[f'{l:.1f}%' for l in too_long],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Too Long: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )

        # 2. Box plot for length distribution
        for i, method in enumerate(self.methods):
            lengths = self.results[method].length_stats['all_lengths']
            fig.add_trace(
                go.Box(
                    y=lengths,
                    name=method,
                    marker_color=self.colors[i % len(self.colors)],
                    showlegend=False,
                    hovertemplate=f'<b>{method}</b><br>Length: %{{y}}<extra></extra>',
                ),
                row=2, col=1
            )

        # Add target range lines to box plot
        for _y in [self.min_words, self.max_words]:
            fig.add_hline(
                y=_y,
                line_dash="dash",
                line_color="red",
                row=2, col=1
            )

        fig.update_layout(
            title_text=f"Summary Length Analysis (Target: {self.min_words}-{self.max_words} words)",
            title_x=0.5,
            hovermode='closest',
            barmode='stack',
            xaxis_title="Methods",
            xaxis2_title="Methods",
        )

        fig.update_yaxes(title_text="Percentage (%)", row=1, col=1)
        fig.update_yaxes(title_text="Length (Words)", row=2, col=1)

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

