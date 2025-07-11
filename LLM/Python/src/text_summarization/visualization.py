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

        self.metrics = [
            ("ROUGE-1", lambda m: self.results[m].rouge_scores["rouge1"], None),
            ("ROUGE-2", lambda m: self.results[m].rouge_scores["rouge2"], None),
            ("ROUGE-L", lambda m: self.results[m].rouge_scores["rougeL"], None),
            ("BERTScore", lambda m: self.results[m].bert_scores["f1"], None),
            ("METEOR", lambda m: self.results[m].meteor_scores, None),
        ]

        self.overall_qualities = (
            "Overall Quality", lambda m: self.metric_scores[m], {"color": "black", "width": 4})
        self.performances = (
            "Speed Performance", lambda m: self.normalized_exec_times[m], {"color": "rgb(138, 43, 226)", "width": 3})
        self.length_within_bounds = (
            "Length Within Bounds", lambda m: {"mean": self.results[m].length_stats["within_bounds_pct"]/100}, None)

        self.metric_scores = {}
        self.normalized_exec_times = {}

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
            *self.metrics,
            self.overall_qualities,
            self.performances,
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

            if metric_name in ["Overall Quality", "Speed Performance"]:
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
            title="Summary Quality and Performance Analysis",
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
            self.overall_qualities,
            self.performances,
            self.length_within_bounds,
        ]

        top_methods = dict(sorted(
            {
                _method: {
                    _metric[0]: _metric[1](_method)["mean"]
                    for _metric in metrics
                } for _method in self.methods
            }.items(),
            key=lambda x: sum(x[1].values()),
            reverse=True
        )[:top_n])

        fig = go.Figure()

        for i, (method, vals) in enumerate(top_methods.items()):
            hover_text = f'<b>{method}</b><br>'
            hover_text += f'Aggregate Score: {sum(vals.values()):.3f}<br>'
            hover_text += '<br>'.join([f'{metric}: {score:.3f}' for metric, score in vals.items()])

            fig.add_trace(
                go.Scatterpolar(
                    r=list(vals.values()),
                    theta=list(vals.keys()),
                    fill='toself',
                    name=method,
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

