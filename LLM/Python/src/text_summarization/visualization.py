import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark import SummarizationBenchmark

logger = logging.getLogger(__name__)


class SummarizationVisualizer:
    """Interactive visualization generator for summarization benchmark results."""

    def __init__(self, benchmark_ref: 'SummarizationBenchmark'):
        self.benchmark_ref = benchmark_ref

        self.min_words = benchmark_ref.min_words
        self.max_words = benchmark_ref.max_words
        self.results = {}
        self.methods = None
        self.out_dir = None

        self.metric_scores = {}
        self.exec_time_stats = {}
        self.normalized_exec_time_stats = {}

        # https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express
        self.colors = px.colors.qualitative.Vivid

    def create_all_visualizations(self):
        """Create all visualization plots as separate HTML files."""
        self.results = self.benchmark_ref.results.data[self.benchmark_ref.papers_hash]
        self.methods = list(self.results.keys())
        self.out_dir = self.benchmark_ref.hashed_and_dated_output_dir

        logger.info("Creating interactive visualizations...")

        self._aggregate_metric_scores()
        self._aggregate_execution_times()

        self._create_metric_comparison_plot()
        self._create_length_analysis_plot()
        self._create_radar_chart()

        logger.info(f"Interactive visualizations saved to {self.out_dir}")

    def _aggregate_metric_scores(self):
        for _m in self.results.values():
            valid_metrics = []

            if _m.meteor_scores["mean"] >= 0.0:
                valid_metrics.append(_m.meteor_scores)

            for rouge_type in self.benchmark_ref.rouge_types:
                if _m.rouge_scores[rouge_type]["mean"] >= 0.0:
                    valid_metrics.append(_m.rouge_scores[rouge_type])

            if _m.bert_scores["f1"]["mean"] >= 0.0:
                valid_metrics.append(_m.bert_scores["f1"])

            if valid_metrics:
                self.metric_scores[_m.method_name] = {
                    "mean": np.mean([m["mean"] for m in valid_metrics]),
                    "min": min(m["min"] for m in valid_metrics),
                    "max": max(m["max"] for m in valid_metrics),
                    "std": np.mean([m["std"] for m in valid_metrics])
                }

    def _aggregate_execution_times(self):
        self.exec_time_stats = {}
        for method in self.methods:
            times = self.results[method].execution_times
            self.exec_time_stats[method] = {
                "min": np.min(times),
                "max": np.max(times),
                "mean": np.mean(times),
                "std": np.std(times)
            }

        all_exec_times = [self.exec_time_stats[m]["mean"] for m in self.methods]
        min_time = min(all_exec_times)
        max_time = max(all_exec_times)

        self.normalized_exec_time_stats = {}
        for method in self.methods:
            stats = self.exec_time_stats[method]
            self.normalized_exec_time_stats[method] = {
                "min": 1 - (stats["max"] - min_time) / (max_time - min_time),
                "max": 1 - (stats["min"] - min_time) / (max_time - min_time),
                "mean": 1 - (stats["mean"] - min_time) / (max_time - min_time),
                "std": stats["std"] / (max_time - min_time)
            }

        print("K")
        # TODO: why do i get:
        #  local:frequency
        #  Min: -0.239
    def _normalize_execution_times(self):
        """Calculate normalized execution times (inverted: faster = higher score)."""
        methods = self.results.keys()

        mean_times = {m: np.mean(self.results[m].execution_times) for m in methods}

        max_exec_time = max(mean_times.values())
        min_exec_time = min(mean_times.values())

        if max_exec_time == min_exec_time:
            self.normalized_exec_times = dict.fromkeys(methods, 1.0)
            return

        self.normalized_exec_times = {
            m: 1 - (mean_times[m] - min_exec_time) / (max_exec_time - min_exec_time)
            for m in methods
        }

    def _create_metric_comparison_plot(self):
        """Create ROUGE comparison plot with BERTScore with filled confidence bands."""
        methods = list(self.results.keys())
        best_method = max(methods, key=lambda m: self.metric_scores[m]["mean"])

        # Calculate percentiles using means
        scores = [self.metric_scores[m]["mean"] for m in methods]
        percentile_75 = np.percentile(scores, 75)
        percentile_90 = np.percentile(scores, 90)

        fig = go.Figure()

        metrics = [
            ("ROUGE-1", lambda m: self.results[m].rouge_scores["rouge1"], None),
            ("ROUGE-2", lambda m: self.results[m].rouge_scores["rouge2"], None),
            ("ROUGE-L", lambda m: self.results[m].rouge_scores["rougeL"], None),
            ("BERTScore", lambda m: self.results[m].bert_scores["f1"], None),
            ("METEOR", lambda m: self.results[m].meteor_scores, None),
            ("Average", lambda m: self.metric_scores[m], {"color": "black", "width": 4}),
            ("Execution Speed", lambda m: self.normalized_exec_time_stats[m], {"color": "rgb(138, 43, 226)", "width": 3}),
        ]

        for i, (metric_name, metric_getter, line_override) in enumerate(metrics):
            color = self.colors[i % len(self.colors)] if line_override is None else line_override["color"]
            low_opacity_color = color.replace("rgb(", "rgba(").replace(")", ", 0.2)")

            # Main line (mean values)
            fig.add_trace(
                go.Scatter(
                    x=methods,
                    y=[metric_getter(m)["mean"] for m in methods],
                    mode='lines+markers',
                    name=metric_name,
                    legendgroup=metric_name,
                    line={
                        "color": color,
                        "width": 2
                    } if line_override is None else line_override,
                    marker={
                        "size": 8
                    },
                    opacity=1.0,
                    hovertemplate=str(f'<b>%{{x}}</b>'
                                      f'<br>{metric_name}: %{{y:.3f}}'
                                      f'<br>Min: %{{customdata[0]:.3f}}'
                                      f'<br>Max: %{{customdata[1]:.3f}}'
                                      f'<br>Std: %{{customdata[2]:.3f}}'
                                      f'<extra></extra>'),
                    customdata=[[metric_getter(m)["min"],
                                 metric_getter(m)["max"],
                                 metric_getter(m)["std"]] for m in methods]
                )
            )

            if metric_name == "Average":
                continue

            # Upper bound (invisible line)
            fig.add_trace(
                go.Scatter(
                    x=methods,
                    y=[metric_getter(m)["max"] for m in methods],
                    mode='lines',
                    line={"width": 0},
                    showlegend=False,
                    legendgroup=metric_name,
                    hoverinfo='skip'
                )
            )

            # Lower bound with fill
            fig.add_trace(
                go.Scatter(
                    x=methods,
                    y=[metric_getter(m)["min"] for m in methods],
                    mode='lines',
                    line={"width": 0},
                    fill='tonexty',
                    fillcolor=low_opacity_color,
                    showlegend=False,
                    legendgroup=metric_name,
                    hoverinfo='skip'
                )
            )

        annotation_styles = {
            'best method': {'arrowcolor': 'darkred', 'bgcolor': 'lightcoral', 'bordercolor': 'red'},
            'top 90%': {'arrowcolor': 'gold', 'bgcolor': 'yellow', 'bordercolor': 'orange'},
            'top 75%': {'arrowcolor': 'steelblue', 'bgcolor': 'lightblue', 'bordercolor': 'blue'}
        }

        for method in methods:
            score = self.metric_scores[method]["mean"]

            if method == best_method:
                style_key = "best method"
            elif score >= percentile_90:
                style_key = "top 90%"
            elif score >= percentile_75:
                style_key = "top 75%"
            else:
                continue

            fig.add_annotation(
                x=method,
                y=score,
                text=f"{style_key}<br>{method}: {score:.3f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                **annotation_styles[style_key],
                borderwidth=2
            )

        fig.update_layout(
            title="Metric Comparison",
            xaxis_title="Methods",
            yaxis_title="Score",
            hovermode='closest',
            yaxis={
                "range": [0, 1]
            }
        )

        output_path = self.out_dir / "metric_comparison.html"
        pyo.plot(fig, filename=str(output_path), auto_open=False)

    def _create_length_analysis_plot(self):
        """Create length analysis plot with compliance breakdown and box plot only."""
        methods = list(self.results.keys())

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Length Compliance Breakdown', 'Box Plot Distribution'),
            vertical_spacing=0.15
        )

        # 1. Stacked bar chart for compliance
        within_bounds = [self.results[m].length_stats['within_bounds_pct'] for m in methods]
        too_short = [self.results[m].length_stats['too_short_pct'] for m in methods]
        too_long = [self.results[m].length_stats['too_long_pct'] for m in methods]

        fig.add_trace(
            go.Bar(
                x=methods,
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
                x=methods,
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
                x=methods,
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
        for i, method in enumerate(methods):
            lengths = self.results[method].length_stats['all_lengths']
            fig.add_trace(
                go.Box(
                    y=lengths,
                    name=method,
                    marker_color=self.colors[i % len(self.colors)],
                    showlegend=False,
                    hovertemplate=f'<b>{method}</b><br>Length: %{{y}}<extra></extra>'
                ),
                row=2, col=1
            )

        # Add target range lines to box plot
        fig.add_hline(
            y=self.min_words,
            line_dash="dash",
            line_color="red",
            row=2, col=1
        )
        fig.add_hline(
            y=self.max_words,
            line_dash="dash",
            line_color="red",
            row=2, col=1
        )

        fig.update_layout(
            title_text=f"Length Analysis (Target: {self.min_words}-{self.max_words} words)",
            title_x=0.5,
            hovermode='closest',
            barmode='stack'
        )

        # Save plot
        output_path = self.out_dir / "length_analysis.html"
        pyo.plot(fig, filename=str(output_path), auto_open=False)

    def _create_radar_chart(self):
        """Create radar chart for top 8 performing methods with processing time."""
        methods = list(self.results.keys())

        # Select top 8 methods by ROUGE-1
        top_methods = sorted(methods, key=lambda m: self.results[m].rouge1, reverse=True)[:8]

        fig = go.Figure()

        metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore', 'Length Compliance', 'Processing Speed']

        max_exec_time = max(self.results[m].execution_time for m in methods)

        for i, method in enumerate(top_methods):
            result = self.results[method]
            values = [
                result.rouge1,
                result.rouge2,
                result.rougeL,
                result.bert_score,
                result.length_stats['within_bounds_pct'] / 100,  # Normalize to 0-1
                1 - (result.execution_time / max_exec_time)  # Invert and normalize (higher is better)
            ]

            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=metrics,
                    fill='toself',
                    name=method,
                    line_color=self.colors[i % len(self.colors)],
                    hovertemplate=f'<b>{method}</b><br>' +
                                  'Metric: %{theta}<br>' +
                                  'Score: %{r:.3f}<extra></extra>'
                )
            )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Top 8 Methods Performance Profile",
            hovermode='closest'
        )

        output_path = self.out_dir / "radar_chart.html"
        pyo.plot(fig, filename=str(output_path), auto_open=False)

