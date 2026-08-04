"""SHAP mechanism analysis: attribution migration and profile shift across sessions.

Tests the spatial redundancy hypothesis by measuring how Random Forest feature
attribution on the two shared physical channels (F1F9, F5F13) changes between
session 1 and session 3.

Six conditions. The four 8-channel conditions retain spatial redundancy; the two
2-channel conditions do not. Under the redundancy hypothesis the four should
pattern together and the two should pattern together, regardless of sample rate
or bit depth.
"""

import numpy as np
import os
import shap
from scipy.spatial.distance import cosine
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# condition -> channel count
CONDITIONS = {
    "clinical":     8,
    "rate_only":    8,
    "depth_only":   8,
    "rate_depth":   8,
    "spatial_only": 2,
    "commodity":    2,
}


def get_shap_means(explainer, X):
    """Mean absolute SHAP value per feature, summed across classes."""
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        return np.sum([np.abs(sv) for sv in shap_vals], axis=0).mean(axis=0)
    elif len(shap_vals.shape) == 3:
        return np.abs(shap_vals).mean(axis=0).sum(axis=1)
    elif len(shap_vals.shape) == 2:
        return np.abs(shap_vals).mean(axis=0)
    raise ValueError(f"Unexpected SHAP shape: {np.array(shap_vals).shape}")


def get_shared_profile(mean_abs, target_indices):
    """Normalized 8-feature attribution profile over the shared channels."""
    shared_shap = mean_abs[target_indices]
    total = np.sum(shared_shap)
    return np.ones(8) / 8.0 if total == 0 else shared_shap / total


def get_migration_fraction(mean_abs, target_indices):
    """Fraction of total attribution sitting on the shared channels."""
    return np.sum(mean_abs[target_indices]) / np.sum(mean_abs)


def eval_subject_shap(X_s, Y_s, G_s, feat_names):
    train_m = (G_s[:, 1] == 1) & (G_s[:, 2] <= 5)

    t6_d1  = (G_s[:, 1] == 1) & (G_s[:, 2] == 6)
    t7_d1  = (G_s[:, 1] == 1) & (G_s[:, 2] == 7)
    t6_d29 = (G_s[:, 1] == 3) & (G_s[:, 2] == 6)
    t7_d29 = (G_s[:, 1] == 3) & (G_s[:, 2] == 7)

    if (sum(train_m) == 0 or sum(t6_d1) == 0 or sum(t7_d1) == 0
            or sum(t6_d29) == 0 or sum(t7_d29) == 0):
        return None

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_s[train_m])

    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X_tr, Y_s[train_m])

    explainer = shap.TreeExplainer(clf)

    target_indices = [i for i, f in enumerate(feat_names)
                      if 'F1F9' in f or 'F5F13' in f]
    if len(target_indices) != 8:
        raise ValueError(f"Expected exactly 8 shared features, found {len(target_indices)}")

    # All four profiles come from a single trial: 37 windows each, matched.
    m_t6_d1  = get_shap_means(explainer, scaler.transform(X_s[t6_d1]))
    m_t7_d1  = get_shap_means(explainer, scaler.transform(X_s[t7_d1]))
    m_t6_d29 = get_shap_means(explainer, scaler.transform(X_s[t6_d29]))
    m_t7_d29 = get_shap_means(explainer, scaler.transform(X_s[t7_d29]))

    frac_d1 = (get_migration_fraction(m_t6_d1, target_indices)
               + get_migration_fraction(m_t7_d1, target_indices)) / 2.0
    frac_d29 = (get_migration_fraction(m_t6_d29, target_indices)
                + get_migration_fraction(m_t7_d29, target_indices)) / 2.0

    p_t6_d1  = get_shared_profile(m_t6_d1,  target_indices)
    p_t7_d1  = get_shared_profile(m_t7_d1,  target_indices)
    p_t6_d29 = get_shared_profile(m_t6_d29, target_indices)
    p_t7_d29 = get_shared_profile(m_t7_d29, target_indices)

    # Empirical null: same day, same electrodes, different repetition.
    base_d1  = cosine(p_t6_d1,  p_t7_d1)
    base_d29 = cosine(p_t6_d29, p_t7_d29)
    avg_baseline = (base_d1 + base_d29) / 2.0

    # Longitudinal: same repetition, 28 days apart.
    long_6 = cosine(p_t6_d1, p_t6_d29)
    long_7 = cosine(p_t7_d1, p_t7_d29)
    avg_longitudinal = (long_6 + long_7) / 2.0

    return {
        "frac_d1":      frac_d1,
        "frac_d29":     frac_d29,
        "baseline":     avg_baseline,
        "longitudinal": avg_longitudinal,
        "excess":       avg_longitudinal - avg_baseline,
    }


def process_condition(cache_path):
    data = np.load(cache_path)
    X, Y, GROUPS = data['X'], data['Y'], data['GROUPS']
    if 'feature_names' not in data:
        raise KeyError(f"feature_names missing from {cache_path}")
    feat_names = data['feature_names']
    subjects = np.unique(GROUPS[:, 0])

    results = {}
    for subj in subjects:
        subj_mask = GROUPS[:, 0] == subj
        res = eval_subject_shap(X[subj_mask], Y[subj_mask],
                                GROUPS[subj_mask], feat_names)
        if res is not None:
            results[subj] = res
    return results


def safe_wilcoxon(a, b):
    """Wilcoxon that returns None rather than raising when all diffs are zero."""
    if np.allclose(np.asarray(a), np.asarray(b)):
        return None
    return wilcoxon(a, b)[1]


def main():
    base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")

    all_results = {}
    for cond in CONDITIONS:
        print(f"Calculating SHAP shifts for {cond}...")
        all_results[cond] = process_condition(
            os.path.join(base, f"{cond}_cache.npz"))

    # Subjects present in every condition
    common = set(all_results["clinical"].keys())
    for cond in CONDITIONS:
        common &= set(all_results[cond].keys())
    common = sorted(list(common))

    def arr(cond, key):
        return np.array([all_results[cond][s][key] for s in common])

    print(f"\n{'='*74}")
    print(f"SHAP MECHANISM ANALYSIS  (Random Forest, n={len(common)} subjects)")
    print(f"{'='*74}")

    # --- Metric 1: attribution migration -------------------------------------
    print("\n--- Attribution Migration ---")
    print("Fraction of total decision weight on F1F9 + F5F13 (8 shared features)")
    print(f"\n{'Condition':<15} {'Ch':>3} {'Day 1':>9} {'Day 29':>9} "
          f"{'Shift':>9} {'p':>10}")
    print("-" * 60)
    for cond, n_ch in CONDITIONS.items():
        f1, f29 = arr(cond, "frac_d1"), arr(cond, "frac_d29")
        p = safe_wilcoxon(f1, f29)
        p_str = "pinned" if p is None else f"{p:.4f}"
        print(f"{cond:<15} {n_ch:>3} {np.mean(f1)*100:>8.1f}% "
              f"{np.mean(f29)*100:>8.1f}% "
              f"{(np.mean(f29) - np.mean(f1))*100:>+8.1f}% {p_str:>10}")
    print("\n2-channel conditions are pinned at 100% by construction.")

    # --- Metric 2: profile shift ---------------------------------------------
    print("\n--- Profile Shift on Shared Channels (cosine distance) ---")
    print(f"\n{'Condition':<15} {'Ch':>3} {'Baseline':>11} {'Longitudinal':>14} "
          f"{'Excess':>10}")
    print("-" * 58)
    for cond, n_ch in CONDITIONS.items():
        print(f"{cond:<15} {n_ch:>3} {np.mean(arr(cond,'baseline')):>11.4f} "
              f"{np.mean(arr(cond,'longitudinal')):>14.4f} "
              f"{np.mean(arr(cond,'excess')):>10.4f}")

    # --- Paired tests: each 8-channel condition vs each 2-channel condition ---
    print("\n--- Excess Shift: 8-channel vs 2-channel (paired Wilcoxon) ---")
    eight = [c for c, n in CONDITIONS.items() if n == 8]
    two = [c for c, n in CONDITIONS.items() if n == 2]
    print(f"\n{'Comparison':<32} {'Diff':>10} {'p':>10}")
    print("-" * 54)
    for c8 in eight:
        for c2 in two:
            a, b = arr(c8, "excess"), arr(c2, "excess")
            p = safe_wilcoxon(a, b)
            p_str = "n/a" if p is None else f"{p:.4f}"
            print(f"{c8 + ' vs ' + c2:<32} "
                  f"{np.mean(a) - np.mean(b):>+10.4f} {p_str:>10}")
    print(f"\n{len(eight)*len(two)} comparisons; Bonferroni alpha = "
          f"{0.05/(len(eight)*len(two)):.4f}")

    # --- Falsification control ------------------------------------------------
    print("\n--- Falsification Control ---")
    print("rate_only, depth_only, and rate_depth retain all 8 channels.")
    print("Under the redundancy hypothesis their excess shift should resemble")
    print("clinical, not spatial_only. If they resemble spatial_only instead,")
    print("the effect is not driven by channel count.")
    clin_exc = np.mean(arr("clinical", "excess"))
    spat_exc = np.mean(arr("spatial_only", "excess"))
    print(f"\n  clinical     {clin_exc:>8.4f}   (8ch reference)")
    for cond in ["rate_only", "depth_only", "rate_depth"]:
        v = np.mean(arr(cond, "excess"))
        closer = "clinical" if abs(v - clin_exc) < abs(v - spat_exc) else "spatial_only"
        print(f"  {cond:<12} {v:>8.4f}   closer to {closer}")
    print(f"  spatial_only {spat_exc:>8.4f}   (2ch reference)")


if __name__ == "__main__":
    main()