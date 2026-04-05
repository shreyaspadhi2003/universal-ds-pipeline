# Universal Data Science Pipeline

An automated a reusable Data Science framework that identifies the best performing model for any tabular dataset by benchmarking 100+ algorithms across 7 categories with zero manual configuration.

## Features
- Auto-detects column types, target variable, and problem type
- Supports single or multiple relational datasets (CSV/Excel)
- Runs 100+ models across regression, classification, clustering, time series, anomaly detection
- Association rules / market basket analysis
- Hybrid ensemble models (voting, stacking, blending)
- SHAP explainability
- Full EDA with visualizations
- Feature importance and selection

## How to Run
```bash
cd universal_ds_pipeline
python main.py
```

## Requirements
```bash
pip install -r requirements.txt
```