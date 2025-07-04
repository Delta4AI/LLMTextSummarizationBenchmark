# Scientific Paper Summarization Benchmark

Benchmarking tool for evaluating text summarization methods on scientific papers.

## Quick Start

```bash
# Install dependencies
uv add transformers rouge-score bert-score nltk scikit-learn matplotlib seaborn pandas numpy tqdm torch ollama

# Run with default settings (15-35 words)
python benchmark.py --data-file papers.json

# Run with custom length constraints
python benchmark.py --data-file papers.json --min-words 20 --max-words 30

```

## Input Format
1-N reference summaries can be provided per paper. Multiple reference summaries improve evaluation robustness 
and reduce single-annotator bias.

```json
[
  {
    "title": "Paper Title",
    "abstract": "Paper abstract text...",
    "id": "paper_001",
    "summaries": [
      "This paper analyzes ..",
      "The paper investigates .. "
    ]
  }
]
```

### `testdata.json`
- summary source #1: https://www.semanticscholar.org/ TLDR
- summary source #2: claude 4 sonnet via jetbrains AI assistant
- summary source #3: chatgpt o4-mini with reasoning


#### Used prompt
```
Please provide a concise summary of the following scientific paper. 
The summary MUST be between 15 and 35 words long.
Focus on the key findings, methodology, and conclusions.


Title: <TITLE>

Abstract: <ABSTRACT>

Summary (15 - 35 words):
```

## Summarization Methods

### Built-in Methods
- **First Sentence Plus Title**: Baseline using title + first abstract sentence (length-aware)
- **Key Sentences Extractive**: Position and content-based sentence extraction
- **TextRank Length Aware**: Graph-based extractive with target length optimization
- **BART Length Guided**: `facebook/bart-large-cnn` with length constraints
- **T5 Length Guided**: `t5-base` with target length parameters

### External Platforms
- **Ollama**, **OpenAI**, **Perplexity**, **Anthropic** and a number of models

## Evaluation Metrics

### Quality Metrics
- **ROUGE-1/2/L**: N-gram overlap with reference summaries
  - Single-reference evaluation (first reference)
  - Multi-reference evaluation (best score across all references)
- **BERTScore**: Semantic similarity using BERT embeddings
- **Execution Time**: Processing time for performance comparison

### Length Compliance Metrics
- **Within Bounds**: Percentage meeting length constraints
- **Too Short/Long**: Violation statistics with percentages
- **Average Length**: Mean word count with standard deviation
- **Length Distribution**: Detailed statistical analysis

## Output Files

### Reports
- `comparison_report.csv` - Main performance comparison with length statistics
- `length_compliance_report.csv` - Detailed length constraint analysis
- `detailed_results.json` - Complete results with individual summaries

### Visualizations
- `quality_comparison.png` - ROUGE-1/2/L and BERTScore comparison charts
- `length_analysis.png` - Length distribution and compliance analysis
- `performance_analysis.png` - Quality vs length compliance correlation
