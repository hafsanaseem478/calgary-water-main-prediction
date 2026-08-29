# Interpretable ML for Water Main Failure Prediction — Calgary

Network-wide risk prioritization using XGBoost + CatBoost with SHAP, Lorenz curve evaluation, and material-stratified analysis. Extends Behrooz & Ilbeigi (2025) pipeline with soil geology, traffic data, and Barcelona 2026 evaluation methodology.

## Key Results

| Metric | Value | Barcelona 2026 Benchmark |
|---|---|---|
| **CatBoost AUC-ROC** (temporal) | **0.887** | — |
| **Lorenz @ 1% renewal** | **37.0%** | 10.32% |
| **Lorenz @ 5% renewal** | **57.9%** | 30.2% |
| **Top-100 confirmed** | **57/100 (57%)** | — |
| Spatial CV AUC | 0.930 ± 0.027 | — |
| Cost savings (optimal threshold) | $18.4M | — |

By renewing just 5% of Calgary's network guided by the model, **57.9% of failures are prevented** — nearly twice the Barcelona 2026 benchmark (30.2%).

## Research Question

*Can interpretable machine learning reliably identify the highest-risk pipes across Calgary's distribution network under realistic temporal evaluation, and what infrastructure, environmental, and operational factors drive failure?*

## Methodology

**Data sources:** Calgary Public Water Main (60,741 records), Water Main Breaks (37,498 records, 1956–2026), Water Pressure Zones (80 polygons), daily weather, Alberta Geological Survey soil polygons (3,280), Calgary Traffic Volumes 2023 (326 segments).

**Pipeline:**
1. Spatial join breaks → pipes (20m threshold, EPSG:3762)
2. Real pressure zones assigned to ALL pipes via spatial join
3. Per-pipe × per-year panel with lagged break history (prevents target leakage)
4. Negative sampling (3:1 ratio) with real attributes for network-wide coverage
5. Soil genesis classification (8 types from Alberta Geological Survey)
6. Traffic volume integration (nearest road segment ADWT within 200m)
7. XGBoost + CatBoost comparison (23 features)
8. Lorenz curve, MCC, bootstrap CI, spatial block CV, SHAP, cost-benefit analysis

**Methodological corrections to prior work:**
- Age formula corrected (`year - INSTALLED_YEAR`, not reversed)
- Break-history features lagged by 1 year (prevents target leakage)
- Break-type dummies removed from features (they encode the target)
- Real pressure zones for all pipes (no artificial UNKNOWN labels)

## SHAP Feature Importance

Top failure drivers (CatBoost, temporal split):
1. **Age** — older pipes deteriorate through corrosion and fatigue
2. **TSLF (prev yr)** — time since last failure signals ongoing vulnerability
3. **X, Y coordinates** — proxy for unmeasured spatial variables (soil chemistry, groundwater)
4. **Material** — metallic pipes (CI, DI) break more than PVC
5. **Cumulative Breaks** — prior failures predict future failures
6. **CDD, Precipitation** — climate contributes at annual scale

**Material-stratified analysis:** Cast Iron failures driven by Age + Cumulative Breaks (corrosion-driven). PVC failures driven by Age + Location (installation-quality-driven). Different materials require different prioritization strategies.

## Material-Stratified Lorenz (5% renewal)

| Material | Pipes | Failures | Lorenz@5% |
|---|---|---|---|
| PVC | 53,168 | 85 | 54.1% |
| AC | 900 | 34 | 29.4% |
| YDI | 9,512 | 433 | 27.2% |
| CI | 13,907 | 1,462 | 14.2% |

Cast Iron is hardest to predict — diverse corrosion mechanisms require richer features (operating pressure, wall thickness) not available in open data.

## Limitations

1. **Spatial autocorrelation**: X, Y coordinates rank high in SHAP. Random temporal split may leak spatial information. Spatial CV (AUC 0.93) confirms strong spatial generalization.
2. **Annual climate resolution**: Masks sub-annual freeze-thaw dynamics. Related Kitchener work shows daily climate features improve temporal prediction.
3. **Missing operational features**: Operating pressure, burial depth, water chemistry unavailable in open data.
4. **Traffic coverage**: Only 2.9% of pipes have traffic volume data (326 monitored road segments).
5. **Negative sampling**: 3:1 ratio with real attributes; verified stable across random seeds.

## How to Reproduce

### 1. Download data

Place these files in `data/` folder:
- **Public Water Main**: [data.calgary.ca](https://data.calgary.ca) → rename to `Public_Water_Main.csv`
- **Water Main Breaks**: [data.calgary.ca](https://data.calgary.ca) → rename to `Water_Main_Breaks.csv`
- **Water Pressure Zones**: [data.calgary.ca](https://data.calgary.ca) → rename to `Water_Pressure_Zones.csv`
- **Weather**: [weatherstats.ca/calgary](https://calgary.weatherstats.ca) → rename to `weatherstats_calgary_daily.csv`
- **Soil**: [open.canada.ca](https://open.canada.ca/data/en/dataset/627213e5-6fc8-4551-a464-5dbb4b61bc31) → download SHP ZIP, unzip into `data/soil_data/`
- **Traffic**: [data.calgary.ca — Traffic Volumes 2023](https://data.calgary.ca/Transportation-Transit/Traffic-Volumes-for-2023/bjag-w7zi) → rename to `Traffic_Volumes_for_2023.csv`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
python pipeline.py
```

Runtime: ~30 minutes (20 min panel building, 10 min modeling + evaluation). Results saved to `results/`.

## Project Structure

```
calgary-water-main-prediction/
├── pipeline.py              # complete pipeline (707 lines)
├── requirements.txt
├── .gitignore
├── README.md
├── data/                    # (gitignored — download separately)
│   ├── Public_Water_Main.csv
│   ├── Water_Main_Breaks.csv
│   ├── Water_Pressure_Zones.csv
│   ├── weatherstats_calgary_daily.csv
│   ├── Traffic_Volumes_for_2023.csv
│   └── soil_data/
│       └── surf_py_ll.shp (+ .dbf, .shx, .prj)
└── results/                 # generated outputs
    ├── lorenz_curve.png
    ├── roc_pr_curves.png
    ├── shap_summary.png
    ├── shap_bar.png
    ├── shap_ci_vs_pvc.png
    ├── cost_threshold.png
    ├── bootstrap_ci.csv
    ├── mcc_thresholds.csv
    ├── material_lorenz.csv
    └── top100_risky_pipes.csv
```

## References

- Forero-Ortiz, E., et al. (2026). Machine learning for pipe failure prediction in Barcelona's WDS. *Applied Water Science*, 16, 63.
- Behrooz, H., & Ilbeigi, M. (2025). Probabilistic forecasting of water main breaks using LSTM. *J. Comput. Civ. Eng.*
- Kimutai, E., et al. (2015). Comparison of statistical models for predicting pipe failures: Calgary. *J. Pipeline Sys. Eng. Practice*, 6(4).
- Kleiner, Y., & Rajani, B. (2001). Comprehensive review of structural deterioration of water mains. *Urban Water*, 3(3), 131–150.
- Boloukasli, S., & Dziedzic, R. (2024). Climate-informed water main failure prediction. *WDSA/CCWI Conference.*

## License

MIT
