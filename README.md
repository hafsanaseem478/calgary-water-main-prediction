# Calgary Water Main Failure Prediction

Ranks every pipe in Calgary's water distribution network by 1-year
failure probability so a utility can target the small share of pipes
it can renew each year to those most likely to fail. Built on the City
of Calgary open-data record: 206,619 pipe segments and roughly 37,000
breaks from 1956 through 2025 — one of the longer publicly available
records for a North American utility.

## The problem

Utilities can typically renew about 1% of the network each year. The
operational question is therefore not *"will pipe X fail?"* but
*"which pipes belong at the top of this year's renewal list?"*. That
makes this a ranking task, not a classification task, and the
evaluation follows accordingly.

## Data

| Source | What it provides |
|---|---|
| City of Calgary Open Data | water mains, break records, pressure zones |
| Alberta Geological Survey | surficial geology / soil |
| City of Calgary Transportation | traffic volumes (loaded, not used — 3% coverage) |
| Environment and Climate Change Canada | daily weather (loaded, not used — city-wide only) |

Break records were spatially joined to pipes at 20 m tolerance, median
distance 4.18 m. Pressure zones were assigned to every one of the
206,700 segments (0 unknown after processing).

## Sampling frame

The single most consequential design choice. Cases are pipes that
broke on or before 2018 (the train cutoff). Controls are sampled from
everything else — importantly, pipes that first fail in 2019–2025
stay in the control pool. Excluding them would select controls on the
outcome the model is trying to predict, and inflate every metric.
This is the leakage pattern that the Kitchener open repository [1]
flagged as the reason random train/test splits look great but fail on
future prediction.

Resulting panel: 53,452 pipes, 2.3 M row-years, 0.64% break rate.

Sanity check on install-year missingness (would signal `age` becoming
a label proxy): cases 1.7%, controls 3.6% defaulting to the minimum
year. Small enough that `age` is not carrying label information.

## Features (10)

| Group | Features |
|---|---|
| Pipe intrinsics | material, diameter, length, age |
| Location | pressure zone, soil type, X, Y |
| Break history | time since last failure, cumulative prior breaks |

Removed after a review pass: `INSTALLED_YEAR` (colinear with `age`),
`TYPE_enc` (constant — Calgary open data exposes only mains), the
`has_broken_ever` and `broke_last_year` variants (both fully
redundant with the two break-history features kept), traffic and
distance-to-road (both essentially constant at 97% zero), all
year-level weather variables (identical for every pipe within a year,
so they cannot discriminate in a within-year ranking), and break-type
dummies (encode past break type, which would leak break-history
signal under a different name).

## Evaluation

Two aligned choices, both against the grain of the older literature
but standard in the newest work:

**Temporal split, not random.** Train ≤ 2018, test 2019–2025.
Per-year AUC is reported across the seven test years to expose
degradation; a model that only works in-sample collapses over the
horizon.

**Annual-renewal Lorenz curve as headline.** Adopted from
Forero-Ortiz et al. (2026) on Barcelona [2]. For each test year, rank
every in-service pipe, "renew" the top X% of the network by length,
and record the fraction of that year's failures captured. Mean ± sd
across 2019–2025. This is the metric a utility actually acts on.

AUC-ROC is reported but not led on. At a 0.087% base rate it is
dominated by the abundant negatives and can differ between comparable
models by amounts within noise [3]. Model selection is on PR-AUC.

Class imbalance is handled through algorithm-level weighting
(`scale_pos_weight` for XGBoost, `auto_class_weights='Balanced'` for
CatBoost). SMOTE-style oversampling was not used.

## Results

**Full-network evaluation** — model scored on every one of the 206,619
pipes in each test year, then ranked per year:

| Metric | Value |
|---|---|
| Pooled AUC-ROC | 0.899 |
| Pooled PR-AUC | 0.006 (7.1× lift over 0.087% base rate) |
| Annual-renewal Lorenz @ 1% | 7.8% ± 3.4% |
| Annual-renewal Lorenz @ 5% | 55.9% ± 2.1% |
| Annual-renewal Lorenz @ 10% | 78.3% ± 3.6% |

**Per-year AUC-ROC across 2019–2025:**

| Year | AUC | Failures |
|---|---|---|
| 2019 | 0.893 | 220 |
| 2020 | 0.920 | 189 |
| 2021 | 0.893 | 154 |
| 2022 | 0.893 | 190 |
| 2023 | 0.884 | 184 |
| 2024 | 0.917 | 166 |
| 2025 | 0.901 | 149 |

Stable in the 0.88–0.92 band with no collapse over the horizon —
that's the diagnostic the Kitchener repository [1] specifically
called out as separating credible models from in-sample-only ones.

**Spatial block CV** (5 folds, blocks = pressure zones): AUC 0.901 ±
0.039. Close to the temporal test AUC, so performance is not being
driven by within-zone spatial leakage.

**Bootstrap 95% CI on the 2019 snapshot** (500 iterations): AUC-ROC
[0.865, 0.914], PR-AUC [0.008, 0.013], Lorenz@5% [0.483, 0.612].

**Ablation on break-history features:**

| Feature set | AUC | PR-AUC | Ann Lorenz @1% | Ann Lorenz @5% |
|---|---|---|---|---|
| Full (10 features) | 0.916 | 0.015 | 5.8% | 53.1% |
| No break history (8 features) | 0.899 | 0.017 | 12.0% | 50.9% |
| History + age only (3 features) | 0.858 | 0.013 | 3.2% | 37.4% |

The gap between full and history-only isolates what the break-history
features contribute; the gap between full and no-history says how far
the approach could travel to a utility without a 70-year record. The
inversion at Lorenz@1% (no-history is *higher*) is a real finding, not
noise — with break history removed the model relies on zone, material
and age, which breaks ties among similar pipes and improves the
top-of-ranking. This trade-off between top-of-ranking discrimination
and mid-ranking capture is worth further study.

**Material-stratified Lorenz** (2019 snapshot):

| Material | Pipes | Failures | Lorenz @5% |
|---|---|---|---|
| YDI | 19,814 | 27 | 44.4% |
| DI | 2,531 | 7 | 42.9% |
| PDI | 10,761 | 39 | 25.6% |
| PVC | 134,102 | 12 | 8.3% |
| CI | 15,038 | 119 | 5.9% |

CI carries most of the failures but has the weakest within-material
discrimination — the between-material signal (material as a top-level
feature) dominates the ranking, and within CI the model does not yet
resolve which of the many old cast iron pipes will fail next.
Discriminating there would need corrosion rate, water chemistry, and
cathodic protection data that aren't in the Calgary open dataset.

**Top-100 pipes ranked at 2019, confirmed against 2019–2025 breaks:**
5/100 (5.0%), against a network base rate of 0.6% — an 8.9× lift.
Nine of the top 10 are cast iron mains in West Calgary and Glendale
installed around 1962, all with `TSLF_prev = 62` (never broken in the
recorded window, so the model has no way to break ties among them).
This is the tied-ranking problem behind the Lorenz@1% number: the
model correctly identifies these as high-risk, but cannot say *which*
of them fails first.

## Interpretability

SHAP was run on the trained XGBoost (CatBoost's TreeExplainer hangs
on Windows; the two models produced near-identical PR-AUC, so the
interpretation is materially the same). Material and break history
lead the ranking, with location (zone, soil) and pipe intrinsics
(age, diameter) following. This matches the material-specific
patterns Forero-Ortiz et al. [2] found on Barcelona, the SHAP
findings of Taiwo et al. [4] on Hong Kong (age, material_CI, length
as top drivers), and Gharaati & Dziedzic's [5] finding that prior
failure rate is the strongest correlate of current failure rate
across 13 Canadian utilities. Full-detail SHAP figures and the
material-stratified plots are in `results/`.

## What holds up and what doesn't

**Holds up.** Seven-year stability in per-year AUC. Spatial CV
matching temporal CV. The ablation gap consistent with published
Canadian findings on break-history dominance. Material-stratified
attribution matching the established literature. Two documented
corrections to a prior published pipeline for Calgary (pressure-zone
assignment and lagged break-history feature construction).

**Doesn't.** Within-CI discrimination is weak (Lorenz@5% 5.9%) —
this is the ceiling of what open data supports for the material that
matters most. The 5% top-100 hit rate reflects the same tied-ranking
issue and would improve with the missing corrosion / chemistry
variables. PR-AUC is low (0.006) as expected at a <0.1% base rate;
this is a ranking tool, not a calibrated classifier, and the top-100
lift (8.9×) is the operationally meaningful number rather than
precision at any single threshold.

## Limitations, briefly

- Segment-level unit: mains are split into 2-point segments in
  preprocessing, so break history is per-segment. Utilities that
  renew whole mains at a time need an aggregation step.
- Rank, not calibrated probability. Predicted probabilities are not
  calibrated to Calgary's population incidence.
- Within-CI weakness (above).
- Weather is city-wide only in open data; local frost penetration
  would help but is not available.
- The Barcelona figure (30.2% @ 5%) is quoted for methodology, not
  as a head-to-head — the two networks differ in size, feature
  coverage and break-history depth.

## Repository

```
pipeline.py        end-to-end pipeline (data prep → eval → SHAP)
requirements.txt   Python dependencies
data/              input data (not versioned — see Data)
results/           generated outputs (not versioned)
```

Run: `pip install -r requirements.txt && python pipeline.py`.
Runtime ≈ 60–90 minutes on a laptop; Stage 5 (panel build) and
CatBoost training are the heavy stages.

## References

[1] Dziedzic, R. (2024). Developing an open repository of water main
  break prediction models in Kitchener. *Engineering Proceedings*,
  69(1):13 (WDSA/CCWI 2024).

[2] Forero-Ortiz, E., Sanchez-Juny, M., Martinez-Gomariz, E., Cardus
  Gonzalez, J., Cucchietti, F., & Baque Viader, F. (2026). Near-future
  prediction of pipe failures in water supply networks: a key
  determinant for water pipe renewal policies through a machine
  learning approach. *Applied Water Science*.
  doi:10.1007/s13201-025-02738-1.

[3] Saito, T. & Rehmsmeier, M. (2015). The precision-recall plot is
  more informative than the ROC plot when evaluating binary
  classifiers on imbalanced datasets. *PLOS ONE*, 10(3):e0118432.

[4] Taiwo, R., Zayed, T., & Ben Seghier, M.E.A. (2024). Integrated
  intelligent models for predicting water pipe failure probability.
  *Alexandria Engineering Journal*, 86:243–257.

[5] Gharaati, S. & Dziedzic, R. (2024). Analysis of factors driving
  water main breaks across 13 Canadian utilities. *Environmental
  Systems Research*, 13:9.
