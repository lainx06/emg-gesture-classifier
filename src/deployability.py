"""Estimate Arduino Uno Flash and SRAM use for EMG classifiers.

Calculates the raw deployed C++ byte footprint of fitted models.
"""

import json
import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

BYTES_PER_VALUE = 4
COMMODITY_SAMPLE_RATE_HZ = 1000
WINDOW_MS = 200
COMMODITY_CHANNELS = 2
FLASH_BUDGET_BYTES = 30720 # Provisional: 32KB - ~2KB for bootloader/serial/DSP

def get_models():
    return {
        "LDA": LinearDiscriminantAnalysis(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": LinearSVC(dual="auto", max_iter=10000, random_state=42),
        "RF": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        "XGB": XGBClassifier(n_estimators=50, random_state=42, n_jobs=-1, eval_metric="logloss"),
    }

def load_training_data(X, Y, GROUPS, subject):
    train_mask = (GROUPS[:, 0] == subject) & (GROUPS[:, 1] == 1) & (GROUPS[:, 2] <= 5)
    if not np.any(train_mask):
        raise ValueError(f"No session-1 training rows found for subject {subject}.")
    return X[train_mask], Y[train_mask]

# --- Flash (PROGMEM) Calculations ---

def scaler_bytes(scaler):
    return (scaler.mean_.size + scaler.scale_.size) * BYTES_PER_VALUE

def linear_bytes(clf):
    return (clf.coef_.size + clf.intercept_.size) * BYTES_PER_VALUE

def knn_bytes(features, labels):
    return (features.size * BYTES_PER_VALUE) + (labels.size * 1) # Labels as uint8

def rf_tree_bytes(clf):
    total = 0
    for est in clf.estimators_:
        tree = est.tree_
        leaves = np.sum(tree.children_left == -1)
        internals = tree.node_count - leaves
        total += (internals * 9) + (leaves * 1)
    return total

def xgb_node_bytes(node):
    if "leaf" in node:
        return 4 
    return 9 + sum(xgb_node_bytes(c) for c in node.get("children", []))

def xgb_tree_bytes(clf):
    tree_json = clf.get_booster().get_dump(dump_format="json")
    return sum(xgb_node_bytes(json.loads(t)) for t in tree_json)

# --- SRAM Calculations ---

def sram_bytes():
    samples = int(COMMODITY_SAMPLE_RATE_HZ * WINDOW_MS / 1000)
    naive_bytes = samples * COMMODITY_CHANNELS * BYTES_PER_VALUE
    int16_bytes = samples * COMMODITY_CHANNELS * 2
    
    # 50% Overlap Accumulators (Corrected for carry state overlap)
    streaming_bytes = (43 * COMMODITY_CHANNELS) + 2 # ~88 bytes total
    
    return naive_bytes, int16_bytes, streaming_bytes

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(os.path.dirname(current_dir), "cache", "commodity_cache.npz")
    
    if not os.path.isfile(cache_path):
        print(f"Commodity cache not found: {cache_path}")
        return

    with np.load(cache_path) as data:
        X, Y, GROUPS = data["X"], data["Y"], data["GROUPS"]

    print("\n--- Memory Profile (Day 1 Train) ---")
    
    naive, int16_b, streaming = sram_bytes()
    print("\n[SRAM Estimates - Active Window]")
    print(f"Naive (float32):        {naive} bytes ({(naive/2048)*100:.1f}% of 2KB)")
    print(f"Optimized (int16_t):    {int16_b} bytes ({(int16_b/2048)*100:.1f}% of 2KB)")
    print(f"Overlapping Accumulator: {streaming} bytes ({(streaming/2048)*100:.1f}% of 2KB)\n")

    print(f"{'Model':<8} {'Flash (Bytes)':>22} {'Fits 30KB?':>15}")
    print("-" * 48)

    for name, clf_template in get_models().items():
        if name in ["LDA", "SVM", "KNN"]:
            features, labels = load_training_data(X, Y, GROUPS, 1)
            clf = get_models()[name]
            model = make_pipeline(StandardScaler(), clf)
            model.fit(features, labels)
            
            flash = scaler_bytes(model.named_steps['standardscaler'])
            if name in ["LDA", "SVM"]:
                flash += linear_bytes(clf)
            else:
                flash += knn_bytes(features, labels)
                
            fits = "Yes" if flash <= FLASH_BUDGET_BYTES else "No"
            print(f"{name:<8} {flash:>22,} {fits:>15}")
        else:
            sizes = []
            for subj in range(1, 44):
                features, labels = load_training_data(X, Y, GROUPS, subj)
                clf = get_models()[name]
                model = make_pipeline(StandardScaler(), clf)
                model.fit(features, labels)
                
                flash = scaler_bytes(model.named_steps['standardscaler'])
                flash += rf_tree_bytes(clf) if name == "RF" else xgb_tree_bytes(clf)
                sizes.append(flash)
                
            min_f, max_f = min(sizes), max(sizes)
            fits = "Yes" if max_f <= FLASH_BUDGET_BYTES else "No"
            print(f"{name:<8} {min_f:,} - {max_f:,} {fits:>15}")
            
    print("\n*Note: 30KB Flash budget is a provisional placeholder until final C++ compilation.")

if __name__ == "__main__":
    main()