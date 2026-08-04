"""Run bootstrap CIs for ablation conditions and Spearman correlation for RF size."""
import numpy as np
import os
from scipy.stats import bootstrap, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def get_rf_size(clf):
    total = 0
    for est in clf.estimators_:
        tree = est.tree_
        leaves = np.sum(tree.children_left == -1)
        internals = tree.node_count - leaves
        total += (internals * 9) + (leaves * 1)
    return total

def eval_subject(X, Y, train_m, test_d1, test_d29):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[train_m])
    X_t1 = scaler.transform(X[test_d1])
    X_t29 = scaler.transform(X[test_d29])
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X_tr, Y[train_m])
    
    acc1 = accuracy_score(Y[test_d1], clf.predict(X_t1))
    acc29 = accuracy_score(Y[test_d29], clf.predict(X_t29))
    size = get_rf_size(clf)
    
    return acc1, acc29, size

def main():
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
    conditions = ["clinical", "rate_only", "depth_only", "rate_depth", "spatial_only", "commodity"]
    
    data = {}
    for c in conditions:
        d = np.load(os.path.join(base, f"{c}_cache.npz"))
        data[c] = {"X": d['X'], "Y": d['Y'], "GROUPS": d['GROUPS']}
        
    ref_groups = data['clinical']['GROUPS']
    for c in conditions[1:]:
        np.testing.assert_array_equal(ref_groups, data[c]['GROUPS'], err_msg=f"GROUPS mismatch in {c}")

    subjects = np.unique(ref_groups[:, 0])
    results = {c: {"d1": [], "d29": [], "drop": [], "size": []} for c in conditions}

    print("Evaluating models across all conditions...")
    for subj in subjects:
        mask = ref_groups[:, 0] == subj
        G_s = ref_groups[mask]
        
        train_m = (G_s[:, 1] == 1) & (G_s[:, 2] <= 5)
        test_d1 = (G_s[:, 1] == 1) & (G_s[:, 2] > 5)
        test_d29 = (G_s[:, 1] == 3) & (G_s[:, 2] > 5)

        if sum(train_m) == 0 or sum(test_d1) == 0 or sum(test_d29) == 0: 
            continue

        for c in conditions:
            a1, a29, size = eval_subject(data[c]['X'][mask], data[c]['Y'][mask], train_m, test_d1, test_d29)
            results[c]["d1"].append(a1)
            results[c]["d29"].append(a29)
            results[c]["drop"].append(a1 - a29)
            results[c]["size"].append(size)

    clin_d1 = np.array(results["clinical"]["d1"])
    clin_drop = np.array(results["clinical"]["drop"])
    
    print("\n--- 95% Bootstrap CIs: Difference from Clinical ---")
    print(f"{'Condition':<15} | {'Day 1 Diff (CI)':<30} | {'Drop Diff (CI)':<30}")
    print("-" * 80)
    
    for c in ["rate_only", "depth_only", "rate_depth", "spatial_only", "commodity"]:
        # Day 1 CI
        diff_d1 = np.array(results[c]["d1"]) - clin_d1
        res_d1 = bootstrap((diff_d1,), np.mean, confidence_level=0.95, random_state=42)
        m_d1 = np.mean(diff_d1) * 100
        l_d1, h_d1 = res_d1.confidence_interval.low * 100, res_d1.confidence_interval.high * 100
        d1_str = f"{m_d1:+5.2f}% [{l_d1:+5.2f}, {h_d1:+5.2f}]"
        
        # Drop CI
        diff_drop = np.array(results[c]["drop"]) - clin_drop
        res_drop = bootstrap((diff_drop,), np.mean, confidence_level=0.95, random_state=42)
        m_dr = np.mean(diff_drop) * 100
        l_dr, h_dr = res_drop.confidence_interval.low * 100, res_drop.confidence_interval.high * 100
        dr_str = f"{m_dr:+5.2f}% [{l_dr:+5.2f}, {h_dr:+5.2f}]"
        
        print(f"{c.capitalize():<15} | {d1_str:<30} | {dr_str:<30}")

    print("\n--- RF Flash Size vs Day 1 Accuracy (Commodity) ---")
    comm_acc = results["commodity"]["d1"]
    comm_sizes = results["commodity"]["size"]
    corr, p_corr = spearmanr(comm_sizes, comm_acc)
    print(f"Spearman rho: {corr:.3f} (p = {p_corr:.3f})")

    print("\n--- Direct Paired CI: Commodity vs Spatial-Only ---")
    for endpoint in ["d1", "drop"]:
        a = np.array(results["spatial_only"][endpoint])
        b = np.array(results["commodity"][endpoint])
        diff = (b - a) * 100
        res = bootstrap((diff,), np.mean, confidence_level=0.95, random_state=42)
        print(f"{endpoint}: {np.mean(diff):+.2f} "
              f"[{res.confidence_interval.low:+.2f}, {res.confidence_interval.high:+.2f}]")

if __name__ == "__main__":
    main()