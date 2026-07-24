# Statistical Analysis Plan: 2x2 Separability Matrix
**Date Documented:** July 23, 2026 (Note: Documented following the initial 5-model exploratory run to constrain further analysis).

## Primary Objective
Evaluate if hardware constraints (8-channel clinical vs. 2-channel commodity) compound temporal degradation.
- **Primary Endpoint:** Absolute accuracy drop (Day 1 Within - Day 29 Cross_Max).
- **Test:** Paired Wilcoxon signed-rank test (two-sided).
- **Confidence Intervals:** 95% Bootstrap CI (10,000 resamples) on the paired difference of drops, calculated for all models. (Note: CI evaluates the mean, while Wilcoxon evaluates ranks; both are reported to diagnose skew/subgroup heterogeneity).

## Scope & Multiplicity
- **Models Evaluated:** LDA, KNN, Linear SVM, Random Forest, XGBoost (5 models).
- **Multiple Comparisons:** Bonferroni correction applied across the 5 primary interaction tests. Alpha threshold = 0.05 / 5 = 0.01. (RF survives at p=0.0091).

## Secondary & Sensitivity Analyses
- **Descriptive Endpoint:** Day 8 (Session 2) evaluated alongside Day 29 to track the degradation curve.
- **Headroom-Normalized Interaction:** `(Within - Cross_Max) / (Within - 0.333)`. **Note:** This secondary endpoint is uncorrected for multiple comparisons and is interpreted strictly descriptively.
- **Exclusion Criteria (Secondary Only):** Subjects scoring <= 40% Day 1 accuracy under *any* model in *either* condition are excluded from the normalized sensitivity analysis (n=41 remaining) to prevent ratio artifacts. 

## Post-Hoc Exploratory Observations (Added July 23, 2026)
- **Model-Ceiling Spread:** Model selection yields a ~9.3% robustness spread on clinical hardware but only ~2.3% on commodity hardware at Day 29.
- **Deployability Corollary:** Deploying computationally cheap models (LDA) on constrained edge hardware costs negligible accuracy (~1.6%) compared to high-capacity ensembles (RF) at Day 29.