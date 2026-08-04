# Statistical Analysis Plan: 2x2 Separability Matrix
**Date Documented:** July 23, 2026 (Note: Documented following the initial 5-model exploratory run to constrain further analysis).

July 24, 2026: Confidence intervals for paired differences across hardware conditions are calculated using SciPy's bootstrap method, which defaults to the Bias-Corrected and accelerated (BCa) algorithm to adjust for skewness in the bootstrap distribution.

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

## 2.5. Hardware Ablation Analysis
To isolate the mechanical cause of accuracy loss, we evaluate the Random Forest model across five degradation conditions compared to the `clinical` baseline:
* `rate_only`: 1000 Hz, unquantized, 8 channels
* `depth_only`: 2048 Hz, 10-bit quantized, 8 channels
* `rate_depth`: 1000 Hz, 10-bit quantized, 8 channels
* `spatial_only`: 2048 Hz, unquantized, 2 channels (F1F9, F5F13)
* `commodity`: 1000 Hz, 10-bit quantized, 2 channels

**Endpoints and Statistics:**
* **Primary Endpoint:** The difference in cross-session degradation (Day 1 accuracy minus Day 29 accuracy) between each condition and the clinical baseline.
* **Secondary Endpoint:** The difference in Day 1 baseline accuracy between each condition and the clinical baseline.
* **Testing:** 95% confidence intervals on the paired per-subject differences are calculated using SciPy's bootstrap method with the Bias-Corrected and accelerated (BCa) algorithm. No multiplicity corrections are applied as the goal is to estimate bounded effect sizes, not assert threshold significance.

## 2.6. SHAP Mechanism Evaluation
To test the spatial redundancy hypothesis, we measure feature attribution shifts between Day 1 and Day 29 using two metrics on the shared physical locations (F1F9 and F5F13).

1. **Attribution Migration:** The fraction of total SHAP attribution on the 8 shared features. 
   - *Prediction:* In the 32-channel `clinical` condition, this fraction will significantly shift (up or down) from Day 1 to Day 29. A decrease implies the shared channels degraded more than auxiliary channels; an increase implies auxiliary channels degraded more. In either case, the shift proves the static model's reliance on training-time split distribution. The `spatial_only` condition is pinned at 1.0 by construction.
2. **Profile Shift (Excess Cosine Distance):** The cosine distance of the normalized 8-feature profile across sessions, minus the empirical baseline noise.
   - *Baseline Noise:* Average of within-day cosine distances (Day 1 Trial 6 vs 7, and Day 29 Trial 6 vs 7).
   - *Longitudinal Shift:* Average of cross-day cosine distances (T6 vs T6, T7 vs T7).
   - *Excess:* Longitudinal Shift - Baseline Noise.
   - *Prediction:* The `spatial_only` excess shift will be significantly larger than the `clinical` excess shift, reflecting forced reliance on drifting features without the buffer of auxiliary channel splits.

   ## 2.6. SHAP Mechanism Evaluation (Results Logged July 25)
*   **Attribution Migration:** Null result (p > 0.76). The fraction of weight on shared channels remained perfectly flat across sessions.
*   **Profile Shift:** Directionally consistent with the spatial redundancy hypothesis (8-channel models shifted less than 2-channel models), but did not survive Bonferroni correction (minimum p = 0.0178 vs threshold 0.0063).
*   **Sub-optimal Placement Finding:** The 8 shared features (25% of the subset) consistently received only ~15% of the model's total attribution, indicating standard anatomical placements are sub-optimal for classification compared to auxiliary channels.
*   **Conclusion:** Spatial redundancy on edge devices operates entirely at training-time (split dilution) rather than inference-time (dynamic redistribution).