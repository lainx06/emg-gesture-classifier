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

# Suppress XGBoost deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def run_evaluation(cache_path, clf, mode="within"):
    data = np.load(cache_path)
    X, Y, GROUPS = data['X'], data['Y'], data['GROUPS']
    subjects = np.unique(GROUPS[:, 0])
    subject_results = {}

    for subj in subjects:
        subj_mask = (GROUPS[:, 0] == subj)
        X_subj, Y_subj, G_subj = X[subj_mask], Y[subj_mask], GROUPS[subj_mask]

        train_mask = (G_subj[:, 1] == 1) & (G_subj[:, 2] <= 5)

        if mode == "within":       
            test_mask = (G_subj[:, 1] == 1) & (G_subj[:, 2] > 5)
        elif mode == "cross_mid":  
            test_mask = (G_subj[:, 1] == 2) & (G_subj[:, 2] > 5)
        elif mode == "cross_max":  
            test_mask = (G_subj[:, 1] == 3) & (G_subj[:, 2] > 5)
        else:
            raise ValueError("Invalid mode")

        X_train, Y_train = X_subj[train_mask], Y_subj[train_mask]
        X_test, Y_test = X_subj[test_mask], Y_subj[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

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
        "RF": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        "XGB": XGBClassifier(n_estimators=50, random_state=42, n_jobs=-1, eval_metric="logloss")
    }

    chance = 1.0 / 3.0
    res = {}

    # Pass 1: Run all evaluations and store in memory
    for m, clf in models.items():
        res[m] = {
            "cw": run_evaluation(clin_path, clf, "within"),
            "cc_mid": run_evaluation(clin_path, clf, "cross_mid"),
            "cc_max": run_evaluation(clin_path, clf, "cross_max"),
            "mw": run_evaluation(comm_path, clf, "within"),
            "mc_mid": run_evaluation(comm_path, clf, "cross_mid"),
            "mc_max": run_evaluation(comm_path, clf, "cross_max")
        }

    # Extract safe 4-way intersection across all conditions and models
    all_subjs = set(res["LDA"]["cw"].keys())
    for m in models:
        for k in res[m]:
            all_subjs = all_subjs.intersection(set(res[m][k].keys()))
    all_subjs = sorted(list(all_subjs))

    # Build union mask for normalized test: exclude if <=40% Day 1 on ANY model
    excluded_norm = set()
    for s in all_subjs:
        for m in models:
            if res[m]["cw"][s] <= 0.40 or res[m]["mw"][s] <= 0.40:
                excluded_norm.add(s)
    norm_subjs = sorted(list(set(all_subjs) - excluded_norm))
    
    print(f"Primary Cohort (Absolute Drop): n={len(all_subjs)}")
    print(f"Secondary Cohort (Normalized):  n={len(norm_subjs)} (Excluded {len(excluded_norm)})\n")

    # Pass 2: Calculate and print statistics
    for m in models:
        print(f"{'='*40}\n MODEL: {m}\n{'='*40}")
        
        # 1. Primary Arrays (Full Cohort)
        cw_arr = np.array([res[m]["cw"][s] for s in all_subjs])
        cc_mid_arr = np.array([res[m]["cc_mid"][s] for s in all_subjs])
        cc_max_arr = np.array([res[m]["cc_max"][s] for s in all_subjs])
        
        mw_arr = np.array([res[m]["mw"][s] for s in all_subjs])
        mc_mid_arr = np.array([res[m]["mc_mid"][s] for s in all_subjs])
        mc_max_arr = np.array([res[m]["mc_max"][s] for s in all_subjs])

        print("[Degradation Curve (Descriptive)]")
        print(f"Clinical:  Day 1: {np.mean(cw_arr)*100:.1f}% -> Day 8: {np.mean(cc_mid_arr)*100:.1f}% -> Day 29: {np.mean(cc_max_arr)*100:.1f}%")
        print(f"Commodity: Day 1: {np.mean(mw_arr)*100:.1f}% -> Day 8: {np.mean(mc_mid_arr)*100:.1f}% -> Day 29: {np.mean(mc_max_arr)*100:.1f}%")

        # 2. Primary Endpoint: Absolute Interaction
        clin_abs_drops = cw_arr - cc_max_arr
        comm_abs_drops = mw_arr - mc_max_arr
        stat_abs, p_abs = stats.wilcoxon(clin_abs_drops, comm_abs_drops) # two-sided by default
        
        print("\n[Primary Endpoint: Absolute Interaction (Day 1 vs Day 29)]")
        print(f"Clinical Drop:  {np.mean(clin_abs_drops)*100:.1f}%")
        print(f"Commodity Drop: {np.mean(comm_abs_drops)*100:.1f}%")
        print(f"Wilcoxon (Two-Sided) p-value: {p_abs:.4e}" if p_abs < 0.001 else f"Wilcoxon (Two-Sided) p-value: {p_abs:.4f}")

        # 3. Bootstrap CI (10,000 resamples)
        np.random.seed(42)
        diffs = (comm_abs_drops - clin_abs_drops) * 100 
        bootstrapped_means = np.empty(10000)
        for i in range(10000):
            sample = np.random.choice(diffs, size=len(diffs), replace=True)
            bootstrapped_means[i] = np.mean(sample)
        ci_lower = np.percentile(bootstrapped_means, 2.5)
        ci_upper = np.percentile(bootstrapped_means, 97.5)
        print(f"Bootstrap 95% CI (Difference of Drops): [{ci_lower:.2f}%, {ci_upper:.2f}%]")

        # 4. Secondary Arrays (Masked Cohort)
        cw_norm_arr = np.array([res[m]["cw"][s] for s in norm_subjs])
        cc_max_norm_arr = np.array([res[m]["cc_max"][s] for s in norm_subjs])
        mw_norm_arr = np.array([res[m]["mw"][s] for s in norm_subjs])
        mc_max_norm_arr = np.array([res[m]["mc_max"][s] for s in norm_subjs])

        clin_norm_drops = (cw_norm_arr - cc_max_norm_arr) / (cw_norm_arr - chance)
        comm_norm_drops = (mw_norm_arr - mc_max_norm_arr) / (mw_norm_arr - chance)
        
        stat_norm, p_norm = stats.wilcoxon(clin_norm_drops, comm_norm_drops)
        
        print("\n[Secondary Endpoint: Headroom-Normalized Interaction]")
        print(f"Clinical Normalised Loss:  {np.mean(clin_norm_drops)*100:.1f}% of capacity")
        print(f"Commodity Normalised Loss: {np.mean(comm_norm_drops)*100:.1f}% of capacity")
        print(f"Wilcoxon (Two-Sided) p-value: {p_norm:.4e}" if p_norm < 0.001 else f"Wilcoxon (Two-Sided) p-value: {p_norm:.4f}\n")