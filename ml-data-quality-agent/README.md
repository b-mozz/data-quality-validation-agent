# ML Data Quality Agent

An intelligent data validation agent that detects quality issues in datasets before they enter ML pipelines, with specialized support for healthcare data.

## Features

- **Schema Validation**: Detects empty datasets, duplicate columns, duplicate rows, and duplicate IDs
- **Completeness Checks**: Identifies missing values with severity levels (critical >50%, warning >10%)
- **Type Consistency**: Flags mixed types within columns
- **Domain Validation**: Validates clinical ranges using ML-based column name matching (via sentence embeddings)
- **Statistical Outliers**: Detects outliers using z-score analysis
- **ML Anomaly Detection**: Ensemble approach with Isolation Forest + Local Outlier Factor (LOF)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.agents.data_quality_agent import validate_dataset
import pandas as pd

# Load your data
df = pd.read_csv("data/sample/healthcare_with_issues.csv")

# Validate
report = validate_dataset(df)

# View results
print(f"Status: {report.status.value}")
print(f"Critical Issues: {len(report.critical_issues)}")
print(f"Warnings: {len(report.warnings)}")
```

## Usage

**Validate a CSV file:**

```python
from src.agents.data_quality_agent import validate_file, format_report_text

report = validate_file("path/to/data.csv")
print(format_report_text(report))
```

**Configure validation thresholds:**

```python
config = {
    "null_threshold_warning": 0.05,      # 5% nulls = warning
    "null_threshold_critical": 0.30,     # 30% nulls = critical
    "outlier_zscore_threshold": 2.5,     # Custom z-score threshold
    "anomaly_contamination": 0.10,       # Expected % of anomalies
}

report = validate_dataset(df, config=config)
```

## Testing

```bash
pytest tests/test_data_quality_agent.py -v
```

## Project Structure

```
ml-data-quality-agent/
├── src/agents/data_quality_agent.py   # Main agent implementation
├── tests/test_data_quality_agent.py   # Unit tests
├── data/sample/                       # Sample datasets
├── notebooks/demo.ipynb               # Demo notebook
└── requirements.txt                   # Dependencies
```

## ML Methods

- **Sentence Transformers** (`all-MiniLM-L6-v2`): Automatic column name matching to clinical terms
- **Isolation Forest**: Unsupervised anomaly detection
- **Local Outlier Factor (LOF)**: Density-based outlier detection

## License

Part of the Nimblemind Agentic AI Framework extension.
