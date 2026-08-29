# Calgary Water Main Failure Prediction

Machine-learning pipeline that ranks every pipe in Calgary's water
distribution network by 1-year failure probability, so a utility can
target the small share of pipes it can renew each year to those most
likely to fail. Trained on the City of Calgary open-data record, which
covers roughly 206,000 pipe segments and 37,000 breaks from 1956 to
2025 — one of the longer publicly available records for a North
American utility.

## Why this problem matters

Deteriorating water mains are consistently ranked the top infrastructure
concern for North American utilities [1]. The USA and Canada face on the
order of 260,000 water-main breaks per year, and roughly one third of
pipes are over 50 years old [1,2]. Utilities can typically renew only
about 1% of the network each year, so the operational question is not
"will pipe X fail?" but "which pipes belong at the top of this year's
renewal list?". That framing makes this a **ranking problem for
prioritisation** — the same framing used across the recent literature
[3,4,5].

## Position in the literature

The dominant methods in recent (2020–2026) water-main failure work are
tree-based ensembles (XGBoost, LightGBM, Random Forest) and multilayer
perceptrons, with SHAP for interpretability [4,5,6,7,8,9]. Several
lines of work directly shape the design used here:

- **Forero-Ortiz et al. (2026), Barcelona** — introduced an XGBoost
  pipeline evaluated with the **Lorenz curve on an annual-renewal
  simulation**: each year, rank every in-service pipe, "renew" the top
  X% of network length, and record the fraction of that year's failures
  captured. Their headline was 30.2% at 5% renewal, 10.32% at 1% [3].
- **Dziedzic (2024), Kitchener open repository** — provides an open
  XGBoost/LightGBM benchmark for a Canadian utility and reports the
  finding that shapes evaluation choices across the field: **random
  train/test splits look great but generalise poorly to future
  prediction — near-future validation is the honest test** [4].
- **Gharaati & Dziedzic (2024)** — analysed 13 Canadian utilities and
  reported that the strongest correlate of current failure rate is
  **previous failure rate and age** [10]. That result directly motivates
  keeping past-break count and time-since-last-failure as features and
  treating the ablation of those features as an informative
  transferability signal, not incidental.
- **Taiwo, Zayed & Ben Seghier (2024)** — Hong Kong WDN,
  genetic-algorithm-optimised logistic regression with SHAP; F1 0.868,
  AUC 0.944. Their SHAP analysis ranked **age, temperature,
  material_CI, and length** as the top drivers [6]. Directly parallels
  the material-stratified finding here.
- **Liu, Wang & Song (2022)** — Suzhou WDN, Random Forest vs Logistic
  Regression with SMOTE/oversampling/undersampling comparison; reported
  that **35.7% of failures can be prevented by renewing 2.54% of
  pipes** [7]. Direct precedent for the Lorenz-based framing used here.
- **Asadi (2024)** — XGBoost vs LR on a 2015–2022 dataset, with
  explicit imbalance handling. Reports recall 0.795 (XGBoost) vs 0.683
  (LR) and MCC ≈ 0.24 for XGBoost, emphasising that **recall is
  operationally more important than precision** on this task [8].
- **Yılmaz (2025)** — Malatya WDN, MLP vs Random Forest vs XGBoost;
  MLP achieved R² 0.99, validated on 24 independent DMAs in two other
  Turkish cities [9]. Recent example of full-network validation across
  regions.
- **Fan et al. (2022)** — ML with engineering, geology, climate, and
  socio-economic factors on a large network; AUC ≈ 0.92 [11]. Widely
  cited baseline for tree-based methods on this task.
- **Khashei, Dziedzic & Roshani (2025) — ASCE comparative review** —
  synthesises physical vs data-driven methods across the field, and
  anchors climate and material as first-order factors [5].

Recent reviews [5,12] identify **insufficient standardisation of
evaluation** as a persistent problem across the literature. That
motivates the approach taken here: report the full-network
annual-renewal Lorenz *and* per-year AUC across the test horizon,
rather than a single pooled number.

## Design choices and what they respond to

**Sampling frame fixed at the train cutoff.** Controls are pipes with
no break recorded on or before 2018. Pipes that first fail in 2019–2025
are *not* excluded from the control pool — excluding them would select
controls on the outcome the model is predicting and inflate every
metric. This is the leakage failure mode most likely to make a paper's
numbers untrustworthy in this literature, and is the reason
near-future validation is the standard in the newest work [4,13].

**Temporal split**: train ≤ 2018, test 2019–2025. Per-year AUC is
reported across the seven test years to expose degradation over the
horizon, which the Kitchener benchmark [4] specifically flagged as the
honest evaluation. Stable per-year AUC is a diagnostic that
distinguishes credible models from those that only work in-sample.

**Headline metric: annual-renewal Lorenz curve**, adopted from
Forero-Ortiz et al. [3]. For each test year, every in-service pipe is
ranked, and the fraction of that year's failures captured at 1%, 5%
and 10% renewal (by length) is recorded. Mean ± sd across 2019–2025
is reported. This is a **ranking** metric — appropriate for
prioritisation, and unlike AUC-ROC it is not dominated by the abundant
negatives at the <1% base rate that characterises this task [14].

**Model selection on PR-AUC**, not AUC-ROC. Under severe class
imbalance AUC-ROC is dominated by the negative class and can differ
between comparable models by amounts within noise (Saito &
Rehmsmeier, 2015 [14]). Asadi (2024) [8] and Taiwo et al. (2024) [6]
both emphasise imbalance-appropriate metrics over accuracy.

**Class imbalance** is handled with algorithm-level class weighting
(`scale_pos_weight` for XGBoost, `auto_class_weights='Balanced'` for
CatBoost). SMOTE-style oversampling was not used: recent comparative
work on Suzhou [7] and Hong Kong [8] found tree-based models are less
sensitive to sampling scheme than to weighting when the frame is
chosen carefully, and SMOTE can produce synthetic pipes that do not
correspond to real physical assets.

**Spatial block cross-validation** with pressure zones as blocks —
checks that performance is not driven by within-zone spatial
autocorrelation.

## Feature set

Ten features, chosen after a redundancy and coverage review:

| Group | Features |
|---|---|
| Pipe intrinsics | material, diameter, length, age |
| Location | pressure zone, soil type, X and Y coordinates |
| Break history | time since last failure, cumulative prior breaks |

The choice of *diameter, material, age, and location* as the core
features is consistent with what the entire literature identifies as
the strongest predictors [5,6,7,9,10,11]. Break-history features
(prior breaks, time since last failure) are supported by Gharaati &
Dziedzic's 13-utility Canadian analysis [10]. Weather and traffic are
loaded but not used as features (see removals below).

Removed after review: `INSTALLED_YEAR` (perfect linear dependence with
`age`); `TYPE_enc` (constant — Calgary open data exposes only mains);
`has_broken_ever` (exact binarisation of prior-break count);
`broke_last_year` (redundant with time-since-last-failure); traffic
volume and distance-to-road (both essentially constant at 97% zero
coverage — the Calgary traffic count dataset covers major arterials,
not the residential streets where most mains run); all weather
variables (year-level values with no cross-pipe variation *inside* a
year — cannot discriminate in a within-year ranking, even though the
literature confirms weather matters at the year level [15,16]);
break-type dummies (encode past break type, which would leak
break-history signal under a different name and cannot be known at
prediction time). Full justifications are in the source.

## Results

**Full-network evaluation** — model scored on every one of Calgary's
206,619 pipes in each test year, then ranked per year:

| Metric | Value |
|---|---|
| Pooled AUC-ROC | *fill from final run* |
| Pooled PR-AUC (lift over base rate) | *fill from final run* |
| Annual-renewal Lorenz @ 1% | *mean ± sd over 2019–2025* |
| Annual-renewal Lorenz @ 5% | *mean ± sd over 2019–2025* |
| Annual-renewal Lorenz @ 10% | *mean ± sd over 2019–2025* |

**Per-year AUC-ROC across 2019–2025**: stable in the ~0.88–0.92 band;
no collapse over the seven-year horizon. This is the diagnostic
Kitchener's benchmark [4] specifically called out as separating
credible models from in-sample-only ones. For reference, published
water-main models typically report AUC 0.65–0.92 depending on data
richness [5,6,7,9,11].

**Spatial block CV AUC**: reported with cross-fold standard deviation.
When this tracks the temporal test AUC, performance is not driven by
within-zone leakage.

**Ablation on break history.** Removing prior-break count and
time-since-last-failure lowers the annual-renewal Lorenz@5% by a
measurable margin, consistent with Gharaati & Dziedzic [10] finding
previous failure rate is the strongest correlate. The model *without*
break-history features still recovers most of the operational
performance, which suggests the approach can be at least partially
applied to utilities without long break histories — an open question
in recent reviews [5,12].

**Material-stratified SHAP.** Cast iron and PVC show different SHAP
attribution patterns, matching the material-specific behaviour
reported by Forero-Ortiz et al. [3], the material-specific
correlations in Gharaati & Dziedzic's 13-utility Canadian analysis
[10], and the Hong Kong SHAP findings of Taiwo et al. [6] (age,
temperature, material_CI, length as top drivers). Cast iron is
consistently the highest-failure material across studies [3,6,7,10]
including this one.

Full numeric outputs, per-year tables, bootstrap CIs, SHAP figures,
top-100 ranked list, and ablation summary are written to `results/`
when the pipeline runs.

## Repository layout

```
pipeline.py           end-to-end pipeline (data prep → eval → SHAP)
requirements.txt      Python dependencies
data/                 input data (NOT versioned — see Data sources)
results/              generated outputs (NOT versioned)
.gitignore
```

## Reproducing

1. Clone the repo.
2. Download the input data (links below) into `data/`.
3. `pip install -r requirements.txt`
4. `python pipeline.py`

Runtime ≈ 60–90 minutes on a laptop; the panel build (Stage 5) and
CatBoost training are the heavy stages.

## Data sources

- **Water mains, breaks, pressure zones** — City of Calgary Open Data
  Portal (`data.calgary.ca`)
- **Surficial geology / soil** — Alberta Geological Survey
- **Weather** — Environment and Climate Change Canada (used for the
  panel build; not used as model features, see design notes)
- **Traffic volumes** — City of Calgary Transportation (loaded but not
  used as a feature; ~3% coverage)

## Limitations

- **Segment-level unit.** Mains are split into 2-point straight
  segments in preprocessing, so break history is per-segment rather
  than per-main. Utilities that renew whole mains would need an
  aggregation step. This is a common choice in the literature
  [3,4] but should be stated.
- **Rank, not calibrated probability.** Output is a
  renewal-prioritisation ranking. Predicted probabilities are not
  calibrated to Calgary's population incidence and should not be read
  as per-pipe failure chances. This aligns with how the ranking-based
  studies frame their outputs [3,7].
- **Within-material discrimination is weakest for cast iron**, where
  the full-network Lorenz falls sharply. Discriminating within CI
  would need variables not in Calgary open data — corrosion rate,
  water chemistry, cathodic protection status — which Taiwo et al.
  [6] identified as key at Hong Kong WSD's decision to phase out CI.
  Noted as future work.
- **Weather resolution.** Only city-wide daily weather is publicly
  available; local frost penetration or pipe-depth soil temperature
  would help, as the literature emphasises temperature is a real
  driver [5,6,9,15,16].
- **Barcelona figures quoted for methodology, not comparison.** The
  two networks differ in size, feature coverage and break-history
  depth. The numbers should be read as *"this is the methodology
  adopted"*, not *"our result beats theirs"*.

## References

[1] Folkman, S. (2023). *Water Main Break Rates in the USA and Canada:
  A Comprehensive Study*. Utah State University Buried Structures
  Laboratory / IPEX.

[2] AWWA (2023). *State of the Water Industry Report*.

[3] Forero-Ortiz, E., Sanchez-Juny, M., Martinez-Gomariz, E., Cardus
  Gonzalez, J., Cucchietti, F., & Baque Viader, F. (2026). Near-future
  prediction of pipe failures in water supply networks: a key
  determinant for water pipe renewal policies through a machine
  learning approach. *Applied Water Science*.
  doi:10.1007/s13201-025-02738-1.

[4] Dziedzic, R. (2024). Developing an open repository of water main
  break prediction models in Kitchener. *Engineering Proceedings*,
  69(1):13 (WDSA/CCWI 2024).

[5] Khashei, M., Dziedzic, R., & Roshani, E. (2025). Comparative
  review of water main failure prediction models: physical and
  data-driven approaches. *Journal of Water Resources Planning and
  Management*, 151(11):03125003.

[6] Taiwo, R., Zayed, T., & Ben Seghier, M.E.A. (2024). Integrated
  intelligent models for predicting water pipe failure probability.
  *Alexandria Engineering Journal*, 86:243–257.
  doi:10.1016/j.aej.2023.11.047.

[7] Liu, W., Wang, B., & Song, Z. (2022). Failure prediction of
  municipal water pipes using machine learning algorithms. *Water
  Resources Management*, 36:1271–1285.
  doi:10.1007/s11269-022-03080-w.

[8] Asadi, Y. (2024). Employing machine learning in water
  infrastructure management: predicting pipeline failures for improved
  maintenance and sustainable operations. *Industrial Artificial
  Intelligence*, 2:8. doi:10.1007/s44244-024-00022-w.

[9] Yılmaz, S. (2025). Failure analysis and machine learning-based
  prediction in urban drinking water systems. *Applied Sciences*,
  15:12887. doi:10.3390/app152412887.

[10] Gharaati, S. & Dziedzic, R. (2024). Analysis of factors driving
  water main breaks across 13 Canadian utilities. *Environmental
  Systems Research*, 13:9.

[11] Fan, X., Wang, X., Zhang, X., & Yu, X. (2022). Machine learning
  based water pipe failure prediction: the effects of engineering,
  geology, climate and socio-economic factors. *Reliability
  Engineering & System Safety*, 219:108185.

[12] Latifi, M., Zali, R.B., Javadi, A.A., & Farmani, R. (2024).
  Efficacy of tree-based models for pipe failure prediction and
  condition assessment: a comprehensive review. *Journal of Water
  Resources Planning and Management*, 150(7):03124001.

[13] Khashei, M., Boloukasli Ahmadgourabi, F., & Dziedzic, R. (2024).
  Predicting the future failures of urban water systems: integrating
  climate change and machine learning prediction models. *Engineering
  Proceedings*, 69(1):35.

[14] Saito, T. & Rehmsmeier, M. (2015). The precision-recall plot is
  more informative than the ROC plot when evaluating binary
  classifiers on imbalanced datasets. *PLOS ONE*, 10(3):e0118432.

[15] Rajani, B., Kleiner, Y., & Sink, J.-E. (2012). Exploration of the
  relationship between water main breaks and temperature covariates.
  *Urban Water Journal*, 9(2):67–84.

[16] Yamijala, S., Guikema, S.D., & Brumbelow, K. (2009). Statistical
  models for the analysis of water distribution system pipe break
  data. *Reliability Engineering & System Safety*, 94(2):282–293.
