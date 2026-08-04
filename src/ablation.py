"""Evaluate RF degradation across all 6 hardware ablation axes."""
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def eval_rf(cache_path):
    data = np.load(cache_path)
    X, Y, GROUPS = data['X'], data['Y'], data['GROUPS']
    subjects = np.unique(GROUPS[:, 0])
    
    d1_accs, d29_accs = [], []
    for subj in subjects:
        subj_mask = (GROUPS[:, 0] == subj)
        X_s, Y_s, G_s = X[subj_mask], Y[subj_mask], GROUPS[subj_mask]

        train_m = (G_s[:, 1] == 1) & (G_s[:, 2] <= 5)
        test_d1 = (G_s[:, 1] == 1) & (G_s[:, 2] > 5)
        test_d29 = (G_s[:, 1] == 3) & (G_s[:, 2] > 5)

        # Added guard for test_d1
        if sum(train_m) == 0 or sum(test_d1) == 0 or sum(test_d29) == 0: 
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_s[train_m])
        X_test_d1 = scaler.transform(X_s[test_d1])
        X_test_d29 = scaler.transform(X_s[test_d29])

        clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        clf.fit(X_train, Y_s[train_m])
        
        d1_accs.append(accuracy_score(Y_s[test_d1], clf.predict(X_test_d1)))
        d29_accs.append(accuracy_score(Y_s[test_d29], clf.predict(X_test_d29)))

    # Return full, unmasked per-subject arrays for Wilcoxon paired tests
    return np.array(d1_accs), np.array(d29_accs)

if __name__ == "__main__":
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
    caches = ["clinical", "spatial_only", "rate_only", "depth_only", "rate_depth", "commodity"]
    
    print("\n--- Hardware Ablation: Random Forest Degradation ---")
    print(f"{'Condition':<15} | {'Day 1':<6} | {'Day 29':<6} | {'Drop':<5}")
    print("-" * 45)
    
    results = {}
    for c in caches:
        d1_arr, d29_arr = eval_rf(os.path.join(base, f"{c}_cache.npz"))
        results[c] = {"d1": d1_arr, "d29": d29_arr}
        
        d1_mean = np.mean(d1_arr) * 100
        d29_mean = np.mean(d29_arr) * 100
        print(f"{c.capitalize():<15} | {d1_mean:5.1f}% | {d29_mean:5.1f}% | {(d1_mean - d29_mean):4.1f}%")