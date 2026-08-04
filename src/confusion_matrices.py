"""Generate Day 29 confusion matrices for Clinical vs Commodity conditions."""
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

def get_cm(cache_path, test_session):
    data = np.load(cache_path)
    X, Y, G = data['X'], data['Y'], data['GROUPS']
    cm_total = np.zeros((3, 3), dtype=int)

    for subj in np.unique(G[:, 0]):
        s = G[:, 0] == subj
        Xs, Ys, Gs = X[s], Y[s], G[s]

        train_m = (Gs[:, 1] == 1) & (Gs[:, 2] <= 5)
        test_m  = (Gs[:, 1] == test_session) & (Gs[:, 2] > 5)
        if train_m.sum() == 0 or test_m.sum() == 0:
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(Xs[train_m])
        X_te = scaler.transform(Xs[test_m])

        clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        clf.fit(X_tr, Ys[train_m])
        cm_total += confusion_matrix(Ys[test_m], clf.predict(X_te), labels=[0, 1, 2])

    return cm_total

def show(name, cm):
    labels = ["Extension", "Close", "Rest"]
    acc = np.trace(cm) / cm.sum() * 100
    print(f"\n{name}  (accuracy {acc:.1f}%)")
    print(f"{'true \\ pred':<12}" + "".join(f"{l:>11}" for l in labels) + f"{'recall':>10}")
    for i, l in enumerate(labels):
        row = cm[i]
        pct = [f"{v/row.sum()*100:>10.1f}%" for v in row]
        print(f"{l:<12}" + "".join(pct) + f"{row[i]/row.sum()*100:>9.1f}%")

if __name__ == "__main__":
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
    for cond in ["clinical", "commodity"]:
        for sess, day in [(1, "Day 1"), (3, "Day 29")]:
            show(f"{cond} --- {day}", get_cm(os.path.join(base, f"{cond}_cache.npz"), sess))