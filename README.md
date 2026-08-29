# Calgary Water Main Failure Prediction

This repository contains an end-to-end machine learning pipeline for predicting water main failures in Calgary, AB, Canada. The model uses XGBoost and CatBoost with a 70-year historical failure record to predict the probability of failure for each pipe in the network, enabling proactive maintenance and renewal planning.

## Pipeline Overview

1. Data loading and preprocessing
2. Spatial joins to assign pressure zones and match failures to pipes
3. Panel construction with a fixed 2018 sampling frame for never-broken pipes
4. Feature engineering with a custom annual-renewal Lorenz curve evaluation
5. Model training (XGBoost, CatBoost) with temporal split and spatial block CV
6. Full-network scoring and evaluation
7. Ablation study on break-history features
8. SHAP interpretability analysis

## Results

**Full-Network Evaluation (206,619 pipes, 2019-2025)**
- Annual-renewal Lorenz curve:
  - 55.9% of failures captured at 5% renewal (mean over 2019-2025)
  - 78.3% of failures captured at 10% renewal
- Precision-Recall AUC (PR-AUC): 0.0062
- ROC AUC: 0.8991

**Model Selection and Validation**
- Temporal split (train ≤2018, test 2019-2025):
  - Sampled test PR-AUC: 0.0146 (XGBoost), 0.0156 (CatBoost)
  - Sampled test ROC-AUC: 0.9148 (XGBoost), 0.9174 (CatBoost)
- Spatial block cross-validation (pressure zones): 0.9027 ± 0.0396

**Interpretability and Ablation**
- SHAP feature importance: break history, material, and pressure zone are top drivers
- Ablation study: break-history features contribute 5 percentage points to full-network Lorenz@5%

**Operational Impact**
- Top-100 pipes: 24% confirmed breaks within the 2019-2025 horizon (network base rate 0.6%)
- Per-year ROC-AUC: 0.88-0.92 (stable across a 7-year horizon)

## Getting Started

1. Clone the repository
2. Download the data files (see links in `data/README.md`)
3. Install dependencies: `pip install -r requirements.txt`
4. Run the pipeline: `python pipeline.py`

## Repository Structure

- `pipeline.py`: Main pipeline script
- `data/`: Input data files (not included in repo)
- `results/`: Output files (CSV tables, plots, SHAP summaries)

## Data Sources

- City of Calgary Open Data Portal: water mains, pressure zones, break records
- Alberta Geological Survey: surficial geology (soil types)
- City of Calgary Transportation: traffic volumes

## Methodology

Evaluation design follows Forero-Ortiz et al. (2026), "Water Mains Failure Prediction: A Case Study for Barcelona", Applied Water Science:

- Sampling frame fixed at the last train year (2018) to prevent selection on test-period outcomes
- Temporal split: train ≤2018, test 2019-2025 (near-future validation)
- Annual-renewal Lorenz curve as the headline metric: each year, rank every in-service pipe, compute the fraction of failures captured at each renewal threshold, average across years