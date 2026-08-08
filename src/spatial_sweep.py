#!/usr/bin/env python3
"""
Exploratory spatial budget + placement sweep.

Pre-specified design:
  2 channels: [0,4], [1,5], [2,6], [3,7]
  4 channels: [0,2,4,6], [1,3,5,7]
  8 channels: [0,1,2,3,4,5,6,7]

Models:
  RandomForestClassifier(n_estimators=50, random_state=42)
  LinearDiscriminantAnalysis()

Endpoints:
  Day-1 accuracy
  Day-1 - Day-29 cross-session drop

The script enforces a hard validation gate:
  clinical cache columns for [0,4] must exactly equal spatial_only cache,
  including Y/GROUPS and resulting RF/LDA predictions.

Usage (from your analysis/scripts folder):
  python spatial_sweep.py
or:
  python spatial_sweep.py --cache-dir ../cache --out-dir ../results/spatial_sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


LAYOUTS = {
    8: [(0,1,2,3,4,5,6,7)],
    4: [(0,2,4,6), (1,3,5,7)],
    2: [(0,4), (1,5), (2,6), (3,7)],
}

MODEL_NAMES = ("RF", "LDA")
SEED = 42


def feature_columns(layout):
    cols = []
    for d in layout:
        cols.extend(range(4*d, 4*d + 4))
    return np.array(cols, dtype=int)


def new_model(name):
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=50, random_state=SEED, n_jobs=-1
        )
    if name == "LDA":
        return LinearDiscriminantAnalysis()
    raise ValueError(name)


def load_cache(path: Path):
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k].copy() for k in z.files}
    for k in ("X", "Y", "GROUPS"):
        if k not in out:
            raise KeyError(f"{path} missing required key {k}")
    return out


def masks_for_subject(groups_s):
    train = (groups_s[:, 1] == 1) & (groups_s[:, 2] <= 5)
    d1 = (groups_s[:, 1] == 1) & (groups_s[:, 2] > 5)
    d29 = (groups_s[:, 1] == 3) & (groups_s[:, 2] > 5)
    if not (np.any(train) and np.any(d1) and np.any(d29)):
        raise RuntimeError("Missing train/D1/D29 rows for a subject.")
    return train, d1, d29


def eval_one_subject(X_s, y_s, g_s, model_name):
    train, d1, d29 = masks_for_subject(g_s)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_s[train])
    Xd1 = scaler.transform(X_s[d1])
    Xd29 = scaler.transform(X_s[d29])

    clf = new_model(model_name)
    clf.fit(Xtr, y_s[train])

    p1 = clf.predict(Xd1)
    p29 = clf.predict(Xd29)

    return {
        "d1": accuracy_score(y_s[d1], p1),
        "d29": accuracy_score(y_s[d29], p29),
        "drop": accuracy_score(y_s[d1], p1) - accuracy_score(y_s[d29], p29),
        "pred_d1": p1,
        "pred_d29": p29,
    }


def validate_gate(clinical, spatial):
    print("\n=== VALIDATION GATE ===")

    if not np.array_equal(clinical["Y"], spatial["Y"]):
        raise SystemExit("FAIL: clinical and spatial_only Y arrays differ.")
    if not np.array_equal(clinical["GROUPS"], spatial["GROUPS"]):
        raise SystemExit("FAIL: clinical and spatial_only GROUPS arrays differ.")

    cols_04 = feature_columns((0,4))
    X_sel = clinical["X"][:, cols_04]

    if X_sel.shape != spatial["X"].shape:
        raise SystemExit(
            f"FAIL: selected clinical shape {X_sel.shape} != "
            f"spatial_only shape {spatial['X'].shape}"
        )

    if not np.array_equal(X_sel, spatial["X"]):
        diff = np.abs(X_sel - spatial["X"])
        raise SystemExit(
            "FAIL: selected clinical [0,4] features do not exactly equal "
            f"spatial_only. max_abs_diff={np.nanmax(diff):.17g}, "
            f"n_unequal={np.count_nonzero(X_sel != spatial['X'])}"
        )

    print("PASS 1/2: feature matrices, labels, and GROUPS match exactly.")

    subjects = np.unique(clinical["GROUPS"][:, 0])
    for model_name in MODEL_NAMES:
        for subj in subjects:
            mc = clinical["GROUPS"][:, 0] == subj
            ms = spatial["GROUPS"][:, 0] == subj

            rc = eval_one_subject(
                X_sel[mc], clinical["Y"][mc], clinical["GROUPS"][mc], model_name
            )
            rs = eval_one_subject(
                spatial["X"][ms], spatial["Y"][ms], spatial["GROUPS"][ms], model_name
            )

            if not np.array_equal(rc["pred_d1"], rs["pred_d1"]):
                raise SystemExit(f"FAIL: {model_name} subject {subj} D1 predictions differ.")
            if not np.array_equal(rc["pred_d29"], rs["pred_d29"]):
                raise SystemExit(f"FAIL: {model_name} subject {subj} D29 predictions differ.")
            if rc["d1"] != rs["d1"] or rc["d29"] != rs["d29"]:
                raise SystemExit(f"FAIL: {model_name} subject {subj} accuracies differ.")

    print("PASS 2/2: RF and LDA predictions/accuracies match for every subject.")
    print("VALIDATION GATE PASSED. Sweep may proceed.\n")


def run_layouts(clinical):
    X = clinical["X"]
    y = clinical["Y"]
    groups = clinical["GROUPS"]
    subjects = np.unique(groups[:, 0])

    rows = []
    for count in (2, 4, 8):
        for layout in LAYOUTS[count]:
            cols = feature_columns(layout)
            Xlayout = X[:, cols]
            layout_name = "[" + ",".join(map(str, layout)) + "]"
            print(f"Running {count}ch {layout_name} ...")

            for model_name in MODEL_NAMES:
                for subj in subjects:
                    m = groups[:, 0] == subj
                    r = eval_one_subject(Xlayout[m], y[m], groups[m], model_name)
                    rows.append({
                        "model": model_name,
                        "count": count,
                        "layout": layout_name,
                        "subject": int(subj),
                        "day1_accuracy": r["d1"],
                        "day29_accuracy": r["d29"],
                        "drop": r["drop"],
                    })
    return pd.DataFrame(rows)


def bca_mean_ci(x, n_resamples=10000):
    x = np.asarray(x, dtype=float)
    res = bootstrap(
        (x,),
        np.mean,
        confidence_level=0.95,
        n_resamples=n_resamples,
        method="BCa",
        vectorized=False,
        rng=np.random.default_rng(SEED),
    )
    return float(np.mean(x)), float(res.confidence_interval.low), float(res.confidence_interval.high)


def aggregate_within_subject(layout_df):
    agg = (
        layout_df.groupby(["model", "count", "subject"], as_index=False)
        .agg(day1_accuracy=("day1_accuracy", "mean"),
             day29_accuracy=("day29_accuracy", "mean"))
    )
    agg["drop"] = agg["day1_accuracy"] - agg["day29_accuracy"]
    return agg


def primary_contrasts(agg, n_resamples):
    rows = []
    contrasts = [(4,8), (2,8), (2,4)]  # low-count minus high-count
    for model in MODEL_NAMES:
        dm = agg[agg["model"] == model].set_index(["subject", "count"])
        subjects = sorted(agg.loc[agg["model"] == model, "subject"].unique())

        for low, high in contrasts:
            d1 = np.array([
                dm.loc[(s, low), "day1_accuracy"] - dm.loc[(s, high), "day1_accuracy"]
                for s in subjects
            ])
            drop = np.array([
                dm.loc[(s, low), "drop"] - dm.loc[(s, high), "drop"]
                for s in subjects
            ])

            d1_mean, d1_lo, d1_hi = bca_mean_ci(d1, n_resamples)
            dr_mean, dr_lo, dr_hi = bca_mean_ci(drop, n_resamples)

            rows.append({
                "model": model,
                "contrast": f"{low}vs{high}",
                "endpoint": "day1_accuracy_difference",
                "mean_pp": 100*d1_mean,
                "ci_low_pp": 100*d1_lo,
                "ci_high_pp": 100*d1_hi,
            })
            rows.append({
                "model": model,
                "contrast": f"{low}vs{high}",
                "endpoint": "drop_difference",
                "mean_pp": 100*dr_mean,
                "ci_low_pp": 100*dr_lo,
                "ci_high_pp": 100*dr_hi,
            })
    return pd.DataFrame(rows)


def placement_summary(layout_df):
    means = (
        layout_df.groupby(["model", "count", "layout"], as_index=False)
        .agg(day1_mean=("day1_accuracy", "mean"),
             day29_mean=("day29_accuracy", "mean"),
             drop_mean=("drop", "mean"))
    )

    varying = layout_df[layout_df["count"].isin([2,4])]
    spreads = []
    for (model, count, subject), g in varying.groupby(["model","count","subject"]):
        spreads.append({
            "model": model,
            "count": int(count),
            "subject": int(subject),
            "day1_spread": g["day1_accuracy"].max() - g["day1_accuracy"].min(),
            "drop_spread": g["drop"].max() - g["drop"].min(),
        })
    return means, pd.DataFrame(spreads)


def model_contrast(agg, n_resamples):
    rf = agg[agg["model"] == "RF"].set_index(["subject","count"])
    lda = agg[agg["model"] == "LDA"].set_index(["subject","count"])
    subjects = sorted(agg["subject"].unique())

    rows = []
    advantage_by_count = {}
    for count in (2,4,8):
        advantage = np.array([
            lda.loc[(s,count), "drop"] - rf.loc[(s,count), "drop"]
            for s in subjects
        ])
        advantage_by_count[count] = advantage
        mean, lo, hi = bca_mean_ci(advantage, n_resamples)
        rows.append({
            "contrast": f"RF_robustness_advantage_at_{count}ch",
            "mean_pp": 100*mean,
            "ci_low_pp": 100*lo,
            "ci_high_pp": 100*hi,
        })

    change = advantage_by_count[8] - advantage_by_count[2]
    mean, lo, hi = bca_mean_ci(change, n_resamples)
    rows.append({
        "contrast": "change_in_RF_advantage_8ch_minus_2ch",
        "mean_pp": 100*mean,
        "ci_low_pp": 100*lo,
        "ci_high_pp": 100*hi,
    })
    return pd.DataFrame(rows)


def count_summary(agg):
    return (
        agg.groupby(["model","count"], as_index=False)
        .agg(day1_mean=("day1_accuracy","mean"),
             day29_mean=("day29_accuracy","mean"),
             drop_mean=("drop","mean"))
    )


def find_cache_dir(arg):
    if arg:
        return Path(arg).resolve()

    here = Path.cwd()
    candidates = [
        here / "cache",
        here.parent / "cache",
        Path(__file__).resolve().parent / "cache",
        Path(__file__).resolve().parent.parent / "cache",
    ]
    for c in candidates:
        if (c / "clinical_cache.npz").exists() and (c / "spatial_only_cache.npz").exists():
            return c
    raise SystemExit(
        "Could not locate cache directory. Pass --cache-dir PATH containing "
        "clinical_cache.npz and spatial_only_cache.npz."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir")
    ap.add_argument("--out-dir", default="spatial_sweep_results")
    ap.add_argument("--n-resamples", type=int, default=10000)
    args = ap.parse_args()

    cache_dir = find_cache_dir(args.cache_dir)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    clinical_path = cache_dir / "clinical_cache.npz"
    spatial_path = cache_dir / "spatial_only_cache.npz"

    print(f"clinical:     {clinical_path}")
    print(f"spatial_only: {spatial_path}")
    print(f"outputs:      {out_dir}")

    clinical = load_cache(clinical_path)
    spatial = load_cache(spatial_path)

    validate_gate(clinical, spatial)

    layout_df = run_layouts(clinical)
    agg = aggregate_within_subject(layout_df)
    summary = count_summary(agg)
    contrasts = primary_contrasts(agg, args.n_resamples)
    layout_means, spreads = placement_summary(layout_df)
    model_cmp = model_contrast(agg, args.n_resamples)

    layout_df.to_csv(out_dir / "layout_subject_results.csv", index=False)
    agg.to_csv(out_dir / "count_subject_aggregated.csv", index=False)
    summary.to_csv(out_dir / "count_summary.csv", index=False)
    contrasts.to_csv(out_dir / "primary_contrasts_bca.csv", index=False)
    layout_means.to_csv(out_dir / "placement_layout_means.csv", index=False)
    spreads.to_csv(out_dir / "placement_subject_spreads.csv", index=False)
    model_cmp.to_csv(out_dir / "rf_lda_model_contrast.csv", index=False)

    meta = {
        "layouts": {str(k): [list(x) for x in v] for k, v in LAYOUTS.items()},
        "models": list(MODEL_NAMES),
        "seed": SEED,
        "n_resamples": args.n_resamples,
        "independent_unit": "subject",
        "n_subjects": int(agg["subject"].nunique()),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    print("\n=== COUNT SUMMARY (means across subjects; %) ===")
    pretty = summary.copy()
    for c in ("day1_mean","day29_mean","drop_mean"):
        pretty[c] *= 100
    print(pretty.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== PRIMARY CONTRASTS (percentage points, BCa 95% CI) ===")
    print(contrasts.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== PER-LAYOUT MEANS (%) ===")
    p = layout_means.copy()
    for c in ("day1_mean","day29_mean","drop_mean"):
        p[c] *= 100
    print(p.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== RF vs LDA ROBUSTNESS CONTRAST (percentage points) ===")
    print(model_cmp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()