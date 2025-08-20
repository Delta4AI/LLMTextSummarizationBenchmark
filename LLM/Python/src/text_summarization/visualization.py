import argparse
import logging
import math

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from typing import TYPE_CHECKING, NamedTuple, Any, Callable

if TYPE_CHECKING:
    from text_summarization.benchmark import SummarizationBenchmark

logger = logging.getLogger(__name__)

SURFACE_LEVEL = "Metrics: Surface-level"
REFERENCE_SIMILARITY = "Metrics: Reference Similarity"
CONTENT_COVERAGE = "Metrics: Content Coverage"
AGGREGATE = "Metrics: Aggregate"
PERFORMANCE = "Performance"
OVERALL = "Overall (70% metrics, 10% speed/accept./cost)"


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
        self._create_insufficient_findings_bar()
        self._create_metric_correlation_matrix()
        self._create_rank_heatmap()
        self._create_tradeoff_3d()
        self._create_pareto_bubble()

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

    def _calculate_acceptance_rates(self):
        """Calculate acceptance rates based on how many responses are present."""
        methods = self.results.keys()

        self.max_publications = max(len(self.results[m].execution_times) for m in methods)

        for method in methods:
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
        for method in self.results.keys():
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

    def _calculate_token_costs(self):
        """Calculate token costs and normalize to 0-1 where 1 is best (lower cost)."""
        methods = list(self.results.keys())

        raw_input_costs = {}
        raw_output_costs = {}

        for method in methods:
            input_tokens = self.results[method].input_tokens
            output_tokens = self.results[method].output_tokens

            raw_input_costs[method] = {
                "mean": np.mean(input_tokens),
                "min": np.min(input_tokens),
                "max": np.max(input_tokens),
                "std": np.std(input_tokens)
            }

            raw_output_costs[method] = {
                "mean": np.mean(output_tokens),
                "min": np.min(output_tokens),
                "max": np.max(output_tokens),
                "std": np.std(output_tokens)
            }

        output_means = [raw_output_costs[method]["mean"] for method in methods]
        max_output = max(output_means)
        min_output = min(output_means)

        if max_output != min_output:
            for method in methods:
                raw_mean = raw_output_costs[method]["mean"]
                normalized_mean = 1 - (raw_mean - min_output) / (max_output - min_output)

                self.output_token_costs[method] = {
                    "mean": normalized_mean,
                    "min": normalized_mean,
                    "max": normalized_mean,
                    "std": 0.0
                }
        else:
            for method in methods:
                self.output_token_costs[method] = {
                    "mean": 1.0,
                    "min": 1.0,
                    "max": 1.0,
                    "std": 0.0
                }

        self.input_token_costs = raw_input_costs

    def _calculate_final_combined_scores(self):
        """Calculate final combined score from all normalized metrics."""
        methods = self.results.keys()

        # check whether to include costs
        output_token_means = {method: self.output_token_costs[method]["mean"] for method in methods}
        include_cost = any(tokens > 0 for tokens in output_token_means.values())

        if include_cost:
            quality_weight = 0.70
            speed_weight = 0.1
            acceptance_weight = 0.1
            cost_weight = 0.1
        else:
            quality_weight = 0.70
            speed_weight = 0.15
            acceptance_weight = 0.15
            cost_weight = 0.0

        for method in methods:
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

    def _create_execution_time_boxplot(self):
        """Create boxplot for execution time distribution per method."""
        fig = go.Figure()
        for i, method in enumerate(self.methods):
            times = self.results[method].execution_times
            fig.add_trace(go.Box(
                y=times,
                name=method,
                marker_color=self.colors[i % len(self.colors)],
                boxpoints="outliers",
                hovertemplate=f'<b>{method}</b><br>Time: %{{y:.2f}} sec<extra></extra>'
            ))

        fig.update_layout(
            title="Execution Time Distribution per Method",
            yaxis_title="Time (seconds, log scale)",
            xaxis_title="Methods",
            boxmode="group",
            boxgap=0.3,
            boxgroupgap=0.2,
            yaxis=dict(
                type="log",
                tickvals=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
                ticktext=["1s", "2s", "5s", "10s", "20s", "50s", "100s", "200s", "500s", "1000s"]
            )
        )

        fig.update_traces(
            width=0.6,
            line=dict(width=2)
        )

        output_path = self.out_dir / "execution_time_distribution.html"
        pyo.plot(fig, filename=str(output_path), auto_open=False)

    def _create_insufficient_findings_bar(self):
        """Create bar chart for insufficient findings per method."""
        fig = go.Figure()

        for i, method in enumerate(self.methods):
            insufficient = [
                _ for _ in self.results[method].summaries
                if _ == "INSUFFICIENT_FINDINGS"
                or _.startswith("INSUFFICIENT")
                or _ == "'**'"
                or len(_) < 10
            ]
            ratio = len(insufficient) / self.max_publications * 100

            fig.add_trace(go.Bar(
                x=[method],
                y=[ratio],
                marker_color=self.colors[i % len(self.colors)],
                name=method,
                hovertemplate=f"<b>{method}</b><br>Insufficient Rate: {ratio:.1f}%<extra></extra>"
            ))
        
        fig.update_layout(
            title="Rate of INSUFFICIENT Findings per Method",
            xaxis_title="Method",
            yaxis_title="Percentage (%)",
            yaxis=dict(range=[0, 12.5], tick0=0, dtick=2.5),
        )

        output_path = self.out_dir / "insufficient_findings_bar.html"
        pyo.plot(fig, filename=str(output_path), auto_open=False)
    
    def _create_metric_correlation_matrix(self):
        """
        Correlate evaluation metrics across models (each metric is a variable; observations = models).
        Only includes true quality metrics (e.g., ROUGE, BERTScore, METEOR, BLEU, mpnet).
        """
        chosen_metrics = [*self.metrics]  
        labels = [m.label for m in chosen_metrics]

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

        # heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_filled,
                x=labels,
                y=labels,
                zmin=-1, zmax=1,
                colorscale="RdBu",
                reversescale=True,
                colorbar=dict(title="ρ")
            )
        )

        # annotate only upper triangle and correlations above 0.5
        annotations = []
        for i in range(len(labels)):
            for j in range(len(labels)):
                if j >= i and abs(corr_filled[i, j]) >= 0.5:
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
            margin=dict(l=120, r=40, t=60, b=120),
            height=700,
            width=700
        )

        output_path = self.out_dir / "metric_correlation_matrix.html"
        pyo.plot(fig, filename=str(output_path), auto_open=False)
    
    def _create_rank_heatmap(self):
        """
        Shows ranks of models per metric in a heatmap.
        """
        # metrics to rank (quality only)
        metrics = [*self.metrics]
        metric_labels = [m.label for m in metrics]

        rank_matrix = []
        for method in self.methods:
            row = []
            for metric in metrics:
                try:
                    val = metric.getter(method)["mean"]
                except Exception:
                    val = math.nan
                row.append(val)
            rank_matrix.append(row)
        
        A = np.array(rank_matrix, dtype=float)

        ranks = np.zeros_like(A)
        for j in range(A.shape[1]):
            col = A[:, j]
            # higher is better; handle NaNs by pushing them to bottom
            order = np.argsort(np.argsort(np.nan_to_num(-col, nan=-1e9))) + 1
            if np.all(np.isnan(col)):
                ranks[:, j] = np.nan
            else:
                ranks[:, j] = np.where(np.isnan(col), np.nan, order)
        
        fig = go.Figure(
            data=go.Heatmap(
                z=ranks,
                x=metric_labels,
                y=self.methods,
                colorbar=dict(title="Rank (1=best)"),
                colorscale="Viridis",
                reversescale=True,
                zauto=False
            )
        )

        annotations = []
        for i in range(ranks.shape[0]):
            for j in range(ranks.shape[1]):
                if not np.isnan(ranks[i, j]):
                    annotations.append(dict(
                        x=metric_labels[j], y=self.methods[i],
                        text=str(int(ranks[i, j])),
                        showarrow=False, font=dict(size=10, color="white")
                    ))
        fig.update_layout(
            title="Model Ranks per Metric (lower is better)",
            annotations=annotations,
            xaxis=dict(tickangle=45),
            margin=dict(l=180, r=40, t=60, b=120)
        )
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
    from text_summarization.benchmark import SummarizationBenchmark, GOLD_STANDARD_DATA, EvaluationResult
    benchmark = SummarizationBenchmark()
    benchmark.load_papers(GOLD_STANDARD_DATA)
    benchmark.load_results()
    benchmark.visualizer.create_all_visualizations()
