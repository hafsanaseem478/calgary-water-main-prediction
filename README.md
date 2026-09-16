# Calgary Water Main Failure Prediction for Renewal Prioritisation

## 1. Overview

Water utilities can renew only a small share of an ageing network each year, so the practical question is not simply whether a pipe may fail, but **which pipes should be prioritised first**. This project develops a machine-learning workflow for Calgary's water-distribution network that ranks in-service pipe segments by next-year failure risk. Using a strictly future-year test period (2019–2025), the final CatBoost model placed **49.5% ± 2.9% of observed failures within the top 5% of network length** on average. The project focuses on defensible temporal validation, realistic full-network evaluation, and understanding where the model's ranking power comes from.

## 2. Research Question

> **Can a temporally validated machine-learning model identify a small fraction of Calgary's water-main network that contains a disproportionately large share of future failures?**

## 3. Data

The study uses publicly available infrastructure data for Calgary, Alberta. The network master contains **206,619 pipe segments**, with approximately **37,000 recorded breaks from 1956–2025**. Pipe geometry, material, diameter, installation year, break records, and pressure zones come from the City of Calgary Open Data portal. Surficial geology is from the Alberta Geological Survey and is simplified into broad soil classes.

The 2019–2025 evaluation contains **1,434,067 pipe-years** and **1,222 positive pipe-years** (**0.0852%** prevalence).

Because the mains layer is a later snapshot, a pipe enters the historical risk set **the year after installation**, and breaks dated in or before installation are excluded to reduce replacement-related temporal leakage.

## 4. What I Did

- Matched recorded breaks to the nearest pipe segment within 20 m, joined pressure-zone and surficial-geology information, and built annual pipe-level histories.
- Constructed lagged predictors using only information available before each prediction year, including pipe age, material, diameter, length, spatial context, cumulative prior breaks, and time since last failure.
- Compared XGBoost and CatBoost using a **pre-test validation period** and a pre-specified PR-AUC selection rule; CatBoost was selected before the 2019–2025 test years were evaluated.
- Scored every eligible in-service pipe each test year and evaluated ranking performance using an annual-renewal Lorenz metric: the percentage of failures captured within the top 1%, 5%, and 10% of network length.
- Tested robustness with pressure-zone block cross-validation, bootstrap uncertainty, break-history ablation, and material-stratified analysis.
- Used SHAP for model interpretation, not causal inference.

## 5. Key Results

The final CatBoost model achieved the following full-network results across 2019–2025:

| Metric | Result |
|---|---:|
| Failures captured in top 1% of network length | **16.3% ± 1.7%** |
| Failures captured in top 5% | **49.5% ± 2.9%** |
| Failures captured in top 10% | **70.3% ± 4.0%** |
| Pooled AUC-ROC | **0.9104** |
| Pooled PR-AUC | **0.0113** |
| PR-AUC lift over the 0.0852% base rate | **13.2×** |

Annual AUC-ROC remained between **0.8975 and 0.9198** across the seven future test years.

For 2019, the top 5% contained **54.7% of failures**; 500 bootstrap resamples gave a 95% interval of **48.1%–61.5%**.

![Annual-renewal Lorenz curve for the 2019 full network](results/lorenz_curve.png)

The interpretation analysis shows that **material is the dominant model feature**, followed by geographic variables, pressure zone, age, and length.

![Mean absolute SHAP value by feature](results/shap_bar.png)

Material-specific analysis showed stronger separation between broad risk groups than among individual cast-iron pipes.

![Material-stratified SHAP comparison](results/shap_ci_vs_pvc.png)

## 6. Reliability and Limitations

- **Future-year testing:** model selection was completed before the 2019–2025 test period. The final test years were not used to choose the model.
- **Rare-event evaluation:** headline metrics are calculated on the natural-prevalence full network rather than only on a sampled case-control dataset.
- **Spatial generalisation is weaker:** pressure-zone block CV gave **AUC = 0.771 ± 0.057**, showing that location-specific structure contributes to performance.
- **Break history was not the main ranking signal:** in the XGBoost ablation, Lorenz@5% was **53.0% with all features and 57.6% without break-history variables**. This is a Calgary-specific result.
- **Historical networks are reconstructed:** the installation-year rule reduces leakage, but pipes removed before the later snapshot may be absent from earlier networks.
- **Outputs are rankings, not calibrated failure probabilities.**
- **Single-city study:** Calgary performance does not establish transferability elsewhere.

## 7. Repository Structure and How to Run

```text
config.py                 study settings, paths and features
common.py                 shared utilities and evaluation functions
geo.py                    spatial processing
panels.py                 pipe-year panel construction
01_load_data.py           load and spatially match source data
02_build_panel.py         construct historical risk sets
03_features.py            engineer and encode predictors
04_train_models.py        temporal model development and selection
05_full_network_eval.py   full-network future-year evaluation
06_ablation.py            break-history ablation
07_evaluation.py          bootstrap, spatial CV and diagnostics
08_interpretation.py      SHAP and material-specific analyses
results/                  generated tables and figures
```

Run the scripts in numerical order after placing the public input datasets in `data/`:

```bash
python 01_load_data.py
python 02_build_panel.py
python 03_features.py
python 04_train_models.py
python 05_full_network_eval.py
python 06_ablation.py
python 07_evaluation.py
python 08_interpretation.py
```

## 8. References

1. **City of Calgary Open Data.** *Public Water Main* and *Water Main Breaks* datasets.  
   https://data.calgary.ca/

2. Fenton, M. M., Waters, E. J., Pawley, S. M., Atkinson, N., Utting, D. J., & McKay, K. (2013). *Surficial Geology of Alberta, 1:1,000,000 scale (GIS data, polygon features)*. Alberta Geological Survey, DIG 2013-0002.  
   https://ags.aer.ca/publications/all-publications/dig-2013-0002

3. Forero-Ortiz, E., Sanchez-Juny, M., Martinez-Gomariz, E., Cardus Gonzalez, J., Cucchietti, F., & Baque Viader, F. (2026). Near-future prediction of pipe failures in water supply networks: a key determinant for water pipe renewal policies through a machine learning approach. *Applied Water Science, 16*, 70.  
   https://doi.org/10.1007/s13201-025-02738-1

4. Boloukasli ahmadgourabi, F., & Dziedzic, R. (2024). Developing an open repository of water main break prediction models in Kitchener. *Engineering Proceedings, 69*(1), 13.  
   https://doi.org/10.3390/engproc2024069013

5. Gharaati, S., & Dziedzic, R. (2024). Analysis of factors driving water main breaks across 13 Canadian utilities. *Environmental Systems Research, 13*, 9.  
   https://doi.org/10.1186/s40068-024-00334-x

6. Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432.  
   https://doi.org/10.1371/journal.pone.0118432

