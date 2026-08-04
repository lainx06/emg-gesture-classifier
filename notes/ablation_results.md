# Hardware Ablation & Mechanism Notes

## 1. Primary Finding (Descriptive)
Initial means suggest degradation on edge EMG hardware is driven by spatial constraints (channel reduction), while signal constraints (sample rate and bit depth) exert minimal penalty on Random Forest accuracy.

| Condition    | Day 1 (%) | Day 29 (%) | Drop (%) |
|--------------|-----------|------------|----------|
| Clinical     | 97.2      | 91.8       | 5.4      |
| Rate_only    | 97.3      | 92.1       | 5.2      |
| Depth_only   | 97.1      | 91.8       | 5.3      |
| Rate_depth   | 97.3      | 92.2       | 5.1      |
| Spatial_only | 86.0      | 71.6       | 14.5     |
| Commodity    | 86.9      | 73.0       | 14.0     |

*(Note: Pending bootstrap confidence intervals to bound the exact effect size of the rate and depth penalties).*

## 2. Methodological Note: V_MAX Pooling
The `depth_only` condition is handicapped. It applies a 0.6407 mV threshold pooled across 8 channels, which quantizes the primary channels (F1/F9) roughly 1.9× more coarsely than the 0.3329 mV threshold used in the `commodity` condition. If `depth_only` shows negligible accuracy loss, this asymmetry acts as a conservative robustness check.

## 3. Deployability Hypotheses
* **Hypothesis:** RF Flash memory footprint scales monotonically (inversely) with subject accuracy.
* **Status:** KNN fits in Flash (18,379 bytes) but will fail inference latency constraints. RF maximum footprint (29,334 bytes) will likely fail Flash constraints assuming ~2 KB overhead. 

## 4. Phase 2 Mechanism Prediction (SHAP)
* **Hypothesis:** Random Forest survives on clinical hardware by relying on spatial redundancy across all 8 channels. When electrodes shift, it routes decision-making away from degraded channels to stable ones.
* **Prediction 1 (Shared Channels):** Isolating the physical electrodes shared by both devices (F1F9 and F5F13), the clinical model will actively shift attribution weight *away* from them across sessions as they degrade. The commodity model, lacking other channels, will be forced to maintain weight on them.
* **Prediction 2 (Saturation Shape):** SHAP attribution shifts will mirror the accuracy degradation curves. The commodity model's attribution shift will saturate by Day 8, while the clinical model will continue dynamically shifting through Day 29.
* **Falsification:** If `depth_only` shows massive attribution shifts, or if the clinical model does not redistribute weight away from the shared channels, the spatial-redundancy mechanism is wrong.