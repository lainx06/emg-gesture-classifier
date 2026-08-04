import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy import stats
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def run_evaluation(cache_path, clf, mode="within"):
    """
    Subject-dependent evaluation: one model per subject, session-wise splits.
    Training parity is enforced -- every mode trains on session 1, trials 1-5.
    Only the test session changes.
    """
    data = np.load(cache_path)
    X, Y, GROUPS = data['X'], data['Y'], data['GROUPS']
    subjects = np.unique(GROUPS[:, 0])
    subject_results = {}

    for subj in subjects:
        subj_mask = (GROUPS[:, 0] == subj)
        X_subj, Y_subj, G_subj = X[subj_mask], Y[subj_mask], GROUPS[subj_mask]

        train_mask = (G_subj[:, 1] == 1) & (G_subj[:, 2] <= 5)

        if mode == "within":        # Day 1
            test_mask = (G_subj[:, 1] == 1) & (G_subj[:, 2] > 5)
        elif mode == "cross_mid":   # Day 8
            test_mask = (G_subj[:, 1] == 2) & (G_subj[:, 2] > 5)
        elif mode == "cross_max":   # Day 29
            test_mask = (G_subj[:, 1] == 3) & (G_subj[:, 2] > 5)
        else:
            raise ValueError("Invalid mode")

        X_train, Y_train = X_subj[train_mask], Y_subj[train_mask]
        X_test, Y_test = X_subj[test_mask], Y_subj[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        # Scaler is part of the model: fit on train only, transform test.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf.fit(X_train_scaled, Y_train)
        preds = clf.predict(X_test_scaled)
        subject_results[subj] = accuracy_score(Y_test, preds)

    return subject_results


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    clin_path = os.path.join(base_dir, "cache", "clinical_cache.npz")
    comm_path = os.path.join(base_dir, "cache", "commodity_cache.npz")

    models = {
        "LDA": LinearDiscriminantAnalysis(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": LinearSVC(dual="auto", max_iter=10000, random_state=42),
        "RF":  RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        "XGB": XGBClassifier(n_estimators=50, random_state=42, n_jobs=-1,
                             eval_metric="logloss"),
    }

    chance = 1.0 / 3.0
    res = {}

    # Pass 1: run every model x condition x timepoint, hold results in memory
    for m, clf in models.items():
        res[m] = {
            "cw":     run_evaluation(clin_path, clf, "within"),
            "cc_mid": run_evaluation(clin_path, clf, "cross_mid"),
            "cc_max": run_evaluation(clin_path, clf, "cross_max"),
            "mw":     run_evaluation(comm_path, clf, "within"),
            "mc_mid": run_evaluation(comm_path, clf, "cross_mid"),
            "mc_max": run_evaluation(comm_path, clf, "cross_max"),
        }

    # Safe intersection across every model and condition
    all_subjs = set(res["LDA"]["cw"].keys())
    for m in models:
        for k in res[m]:
            all_subjs = all_subjs.intersection(set(res[m][k].keys()))
    all_subjs = sorted(list(all_subjs))

    # Union exclusion mask for the normalized secondary endpoint only.
    # Denominator (within - chance) must be bounded away from zero.
    excluded_norm = set()
    for s in all_subjs:
        for m in models:
            if res[m]["cw"][s] <= 0.40 or res[m]["mw"][s] <= 0.40:
                excluded_norm.add(s)
    norm_subjs = sorted(list(set(all_subjs) - excluded_norm))

    print(f"Primary Cohort (Absolute Drop): n={len(all_subjs)}")
    print(f"Secondary Cohort (Normalized):  n={len(norm_subjs)} "
          f"(Excluded {len(excluded_norm)})\n")

    # Pass 2: statistics
    for m in models:
        print(f"{'='*40}\n MODEL: {m}\n{'='*40}")

        cw_arr     = np.array([res[m]["cw"][s]     for s in all_subjs])
        cc_mid_arr = np.array([res[m]["cc_mid"][s] for s in all_subjs])
        cc_max_arr = np.array([res[m]["cc_max"][s] for s in all_subjs])
        mw_arr     = np.array([res[m]["mw"][s]     for s in all_subjs])
        mc_mid_arr = np.array([res[m]["mc_mid"][s] for s in all_subjs])
        mc_max_arr = np.array([res[m]["mc_max"][s] for s in all_subjs])

        print("[Degradation Curve (Descriptive)]")
        print(f"Clinical:  Day 1: {np.mean(cw_arr)*100:.1f}% -> "
              f"Day 8: {np.mean(cc_mid_arr)*100:.1f}% -> "
              f"Day 29: {np.mean(cc_max_arr)*100:.1f}%")
        print(f"Commodity: Day 1: {np.mean(mw_arr)*100:.1f}% -> "
              f"Day 8: {np.mean(mc_mid_arr)*100:.1f}% -> "
              f"Day 29: {np.mean(mc_max_arr)*100:.1f}%")

        # Primary endpoint: absolute interaction, full cohort
        clin_abs_drops = cw_arr - cc_max_arr
        comm_abs_drops = mw_arr - mc_max_arr
        stat_abs, p_abs = stats.wilcoxon(clin_abs_drops, comm_abs_drops)

        print("\n[Primary Endpoint: Absolute Interaction (Day 1 vs Day 29)]")
        print(f"Clinical Drop:  {np.mean(clin_abs_drops)*100:.1f}%")
        print(f"Commodity Drop: {np.mean(comm_abs_drops)*100:.1f}%")
        print(f"Wilcoxon (Two-Sided) p-value: "
              f"{p_abs:.4e}" if p_abs < 0.001 else
              f"Wilcoxon (Two-Sided) p-value: {p_abs:.4f}")

        # Bootstrap CI on the paired difference of drops
        np.random.seed(42)
        diffs = (comm_abs_drops - clin_abs_drops) * 100
        bootstrapped_means = np.empty(10000)
        for i in range(10000):
            sample = np.random.choice(diffs, size=len(diffs), replace=True)
            bootstrapped_means[i] = np.mean(sample)
        ci_lower = np.percentile(bootstrapped_means, 2.5)
        ci_upper = np.percentile(bootstrapped_means, 97.5)
        print(f"Bootstrap 95% CI (Difference of Drops): "
              f"[{ci_lower:.2f}%, {ci_upper:.2f}%]")

        # Secondary endpoint: headroom-normalized, masked cohort
        cw_n     = np.array([res[m]["cw"][s]     for s in norm_subjs])
        cc_max_n = np.array([res[m]["cc_max"][s] for s in norm_subjs])
        mw_n     = np.array([res[m]["mw"][s]     for s in norm_subjs])
        mc_max_n = np.array([res[m]["mc_max"][s] for s in norm_subjs])

        clin_norm_drops = (cw_n - cc_max_n) / (cw_n - chance)
        comm_norm_drops = (mw_n - mc_max_n) / (mw_n - chance)
        stat_norm, p_norm = stats.wilcoxon(clin_norm_drops, comm_norm_drops)

        print("\n[Secondary Endpoint: Headroom-Normalized Interaction]")
        print(f"Clinical Normalised Loss:  {np.mean(clin_norm_drops)*100:.1f}% of capacity")
        print(f"Commodity Normalised Loss: {np.mean(comm_norm_drops)*100:.1f}% of capacity")
        print(f"Wilcoxon (Two-Sided) p-value: "
              f"{p_norm:.4e}\n" if p_norm < 0.001 else
              f"Wilcoxon (Two-Sided) p-value: {p_norm:.4f}\n")