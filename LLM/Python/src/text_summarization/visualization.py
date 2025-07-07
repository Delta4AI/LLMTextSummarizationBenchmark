import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
from typing import Dict, Any
import pickle
import numpy as np

logger = logging.getLogger(__name__)


class SummarizationVisualizer:
    """Interactive visualization generator for summarization benchmark results."""

    def __init__(self, output_dir: Path, min_words: int, max_words: int):
        self.output_dir = Path(output_dir)
        self.min_words = min_words
        self.max_words = max_words
        self.results = {}

        self.average_scores = {}
        self.normalized_exec_times = {}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_fn = self.output_dir / "results.pkl"

        self.colors = px.colors.qualitative.Set3

    def set_results(self, results: Dict[str, Any]):
        """Set the benchmark results to visualize."""
        with open(self.results_fn, "wb") as f:
            pickle.dump(results, f)
        self.results = results

        logger.info(f"Saved benchmark results to {self.results_fn}")

    def create_all_visualizations(self):
        """Create all visualization plots as separate HTML files."""
        if not self.results:
            logger.warning("No results to visualize")
            return

        logger.info("Creating interactive visualizations...")

        self._get_average_metric_scores()
        self._normalize_execution_times()

        self._create_metric_comparison_plot()
        self._create_length_analysis_plot()
        self._create_radar_chart()

        logger.info(f"Interactive visualizations saved to {self.output_dir}")

    def _get_average_metric_scores(self):
        metrics = ["rouge1", "rouge2", "rougeL", "bert_score"]
        for method in self.results:
            avg = 0
            for metric in metrics:
                avg += self.results[method].__dict__[metric]
            avg /= len(metrics)
            self.average_scores[method] = avg

    def _normalize_execution_times(self):
        """Calculate normalized execution times (inverted: faster = higher score)."""
        methods = self.results.keys()
        exec_times = [self.results[m].execution_time for m in methods]
        max_exec_time = max(exec_times)
        min_exec_time = min(exec_times)

        if max_exec_time == min_exec_time:
            self.normalized_exec_times = {m: 1.0 for m in methods}
            return

        self.normalized_exec_times = {
            m: 1 - (self.results[m].execution_time - min_exec_time) / (max_exec_time - min_exec_time)
            for m in methods
        }

    def _create_metric_comparison_plot(self):
        """Create ROUGE comparison plot with BERTScore (multi-ref scores only)."""
        methods = list(self.results.keys())
        best_method = max(methods, key=lambda m: self.average_scores[m])
        best_score = self.average_scores[best_method]

        fig = go.Figure()

        # ROUGE-1 Multi-ref
        fig.add_trace(
            go.Scatter(
                x=methods,
                y=[self.results[m].rouge1 for m in methods],
                mode='lines+markers',
                name='ROUGE-1',
                line=dict(color='blue', width=2),
                marker=dict(size=8),
                opacity=0.8,
                hovertemplate='<b>%{x}</b><br>ROUGE-1: %{y:.3f}<extra></extra>'
            )
        )

        # ROUGE-2 Multi-ref
        fig.add_trace(
            go.Scatter(
                x=methods,
                y=[self.results[m].rouge2 for m in methods],
                mode='lines+markers',
                name='ROUGE-2',
                line=dict(color='red', width=2),
                marker=dict(size=8),
                opacity=0.8,
                hovertemplate='<b>%{x}</b><br>ROUGE-2: %{y:.3f}<extra></extra>'
            )
        )

        # ROUGE-L Multi-ref
        fig.add_trace(
            go.Scatter(
                x=methods,
                y=[self.results[m].rougeL for m in methods],
                mode='lines+markers',
                name='ROUGE-L',
                line=dict(color='green', width=2),
                marker=dict(size=8),
                opacity=0.8,
                hovertemplate='<b>%{x}</b><br>ROUGE-L: %{y:.3f}<extra></extra>'
            )
        )

        # BERTScore
        fig.add_trace(
            go.Scatter(
                x=methods,
                y=[self.results[m].bert_score for m in methods],
                mode='lines+markers',
                name='BERTScore',
                line=dict(color='purple', width=2),
                marker=dict(size=8),
                opacity=0.8,
                hovertemplate='<b>%{x}</b><br>BERTScore: %{y:.3f}<extra></extra>'
            )
        )

        # Average Scores
        fig.add_trace(
            go.Scatter(
                x=methods,
                y=[self.average_scores[m] for m in methods],
                mode='lines+markers',
                name='Rogue-N+BERTScore Averaged',
                line=dict(color='black', width=4),
                marker=dict(size=8),
                hovertemplate='<b>%{x}</b><br>Average: %{y:.3f}<extra></extra>'
            )
        )

        fig.add_annotation(
            x=best_method,
            y=best_score,
            text=f"{best_method}: {best_score:.3f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="gold",
            bgcolor="yellow",
            bordercolor="orange",
            borderwidth=2
        )

        # Normalized execution times
        fig.add_trace(
            go.Scatter(
                x=methods,
                y=[self.normalized_exec_times[m] for m in methods],
                mode='lines+markers',
                name='Processing Speed Score',
                line=dict(color='orange', width=2),
                marker=dict(size=8),
                hovertemplate='<b>%{x}</b><br>Processing Speed Score: %{y:.2f}<extra></extra>'
            )
        )

        fig.update_layout(
            title="ROUGE and BERTScore Comparison (Multi-Reference)",
            xaxis_title="Methods",
            yaxis_title="Score",
            hovermode='closest',
            yaxis=dict(range=[0, 1])
        )

        output_path = self.output_dir / "metric_comparison.html"
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
        output_path = self.output_dir / "length_analysis.html"
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

        output_path = self.output_dir / "radar_chart.html"
        pyo.plot(fig, filename=str(output_path), auto_open=False)

