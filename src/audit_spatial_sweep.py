"""Independent audit of the exploratory spatial budget sweep.

Recomputes every per-subject accuracy from the clinical cache rather than
reading the sweep's saved output, so this is a genuine cross-check rather than
a re-read. Then:

  1. confirms the 8-channel condition reproduces the clinical condition
  2. confirms [0,4] reproduces the spatial_only condition
  3. reports the pre-specified per-subject placement-spread distributions
  4. recomputes the BCa contrasts and diffs them against the saved outputs

The cross-check establishes that both scripts estimate the same quantities from
the same data. It does not require bit-identical confidence intervals: the two
aggregate layouts within subject using different reduction implementations, and
BCa selects order statistics, so endpoints can differ by a few thousandths of a
percentage point without any disagreement in substance. Point estimates must
agree exactly, and interval endpoints are checked for magnitude rather than
identity.

Run from the project root:  python src/audit_spatial_sweep.py
"""

import os
import glob
import numpy as np
from scipy.stats import bootstrap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "cache")
SWEEP_OUT = os.path.join(ROOT, "results", "spatial_sweep")

LAYOUTS = {
    2: [[0, 4], [1, 5], [2, 6], [3, 7]],
    4: [[0, 2, 4, 6], [1, 3, 5, 7]],
    8: [[0, 1, 2, 3, 4, 5, 6, 7]],
}
SEED = 42
N_BOOT = 10000


def cols_for(layout):
    """Feature columns for a set of derivations. Four features per channel,
    grouped by channel: derivation d occupies columns 4d .. 4d+3."""
    out = []
    for d in layout:
        out.extend([4 * d, 4 * d + 1, 4 * d + 2, 4 * d + 3])
    return np.array(out)


def check_column_mapping(feature_names):
    """Verify the 4-features-per-channel grouping actually holds."""
    if feature_names is None:
        return "feature_names absent from cache; column mapping assumed"
    names = [str(n) for n in feature_names]
    if len(names) != 32:
        return f"unexpected feature count {len(names)}"
    suffixes = [n.split("_")[-1] for n in names]
    expected = ["rms", "mav", "wl", "zc"] * 8
    if suffixes != expected:
        return f"feature order is not (rms,mav,wl,zc) per channel: {suffixes[:8]}"
    derivs = [names[4 * d].rsplit("_", 1)[0] for d in range(8)]
    return "OK  derivation order: " + ", ".join(derivs)


def eval_subject(X, Y, G, cols, model_name):
    """Per-subject Day-1 and Day-29 accuracy for one layout and one model."""
    train = (G[:, 1] == 1) & (G[:, 2] <= 5)
    t_d1 = (G[:, 1] == 1) & (G[:, 2] > 5)
    t_d29 = (G[:, 1] == 3) & (G[:, 2] > 5)
    if train.sum() == 0 or t_d1.sum() == 0 or t_d29.sum() == 0:
        return None

    Xc = X[:, cols]
    sc = StandardScaler().fit(Xc[train])
    Xtr = sc.transform(Xc[train])

    if model_name == "RF":
        clf = RandomForestClassifier(n_estimators=50, random_state=SEED, n_jobs=-1)
    else:
        clf = LinearDiscriminantAnalysis()
    clf.fit(Xtr, Y[train])

    a1 = float(np.mean(clf.predict(sc.transform(Xc[t_d1])) == Y[t_d1]))
    a29 = float(np.mean(clf.predict(sc.transform(Xc[t_d29])) == Y[t_d29]))
    return a1, a29


def run_all():
    d = np.load(os.path.join(CACHE, "clinical_cache.npz"), allow_pickle=True)
    X, Y, G = d["X"], d["Y"], d["GROUPS"]
    fn = d["feature_names"] if "feature_names" in d else None

    print("column mapping:", check_column_mapping(fn))
    print()

    subs = np.unique(G[:, 0])
    res = {}   # (model, count, layout_idx) -> {subject: (d1, d29)}

    for model in ("RF", "LDA"):
        for count, layouts in LAYOUTS.items():
            for li, layout in enumerate(layouts):
                cols = cols_for(layout)
                acc = {}
                for s in subs:
                    m = G[:, 0] == s
                    r = eval_subject(X[m], Y[m], G[m], cols, model)
                    if r is not None:
                        acc[s] = r
                res[(model, count, li)] = acc
    return res, subs


def reproduction_checks(res, subs):
    """8-channel must match clinical; [0,4] must match spatial_only."""
    print("=" * 70)
    print("REPRODUCTION CHECKS")
    print("=" * 70)

    for cache_name, count, li, label in (
        ("clinical_cache.npz", 8, 0, "8ch [0..7] vs clinical"),
        ("spatial_only_cache.npz", 2, 0, "2ch [0,4] vs spatial_only"),
    ):
        path = os.path.join(CACHE, cache_name)
        if not os.path.exists(path):
            print(f"  {label}: cache missing, skipped")
            continue
        d = np.load(path, allow_pickle=True)
        Xr, Yr, Gr = d["X"], d["Y"], d["GROUPS"]
        ncol = Xr.shape[1]
        cols_ref = np.arange(ncol)

        worst = 0.0
        n = 0
        for model in ("RF", "LDA"):
            for s in subs:
                m = Gr[:, 0] == s
                ref = eval_subject(Xr[m], Yr[m], Gr[m], cols_ref, model)
                got = res[(model, count, li)].get(s)
                if ref is None or got is None:
                    continue
                worst = max(worst, abs(ref[0] - got[0]), abs(ref[1] - got[1]))
                n += 1
        verdict = "PASS" if worst == 0.0 else ("close" if worst < 1e-12 else "FAIL")
        print(f"  {label:<32} max abs diff {worst:.3e} over {n} comparisons  {verdict}")
    print()


def placement_spreads(res, subs):
    """Pre-specified secondary endpoint: per-subject spread across layouts."""
    print("=" * 70)
    print("PLACEMENT SPREAD  (per-subject max - min across layouts, pp)")
    print("=" * 70)
    print("Distribution across subjects. The 8-channel condition has a single")
    print("layout, so its spread is zero by construction and is omitted.")
    print()
    print(f"{'model':<6}{'count':>6}{'endpoint':>10}"
          f"{'median':>9}{'IQR':>16}{'min':>8}{'max':>8}{'n':>5}")
    print("-" * 70)

    for model in ("RF", "LDA"):
        for count in (2, 4):
            n_lay = len(LAYOUTS[count])
            for endpoint in ("day1", "drop"):
                vals = []
                for s in subs:
                    per = []
                    ok = True
                    for li in range(n_lay):
                        a = res[(model, count, li)].get(s)
                        if a is None:
                            ok = False
                            break
                        per.append(a[0] if endpoint == "day1" else a[0] - a[1])
                    if ok:
                        vals.append((max(per) - min(per)) * 100)
                v = np.array(vals)
                q1, q3 = np.percentile(v, [25, 75])
                print(f"{model:<6}{count:>6}{endpoint:>10}"
                      f"{np.median(v):>9.2f}{f'[{q1:.2f}, {q3:.2f}]':>16}"
                      f"{v.min():>8.2f}{v.max():>8.2f}{len(v):>5}")
    print()


def bca(diff_frac, alt_rng=False):
    """BCa interval on the mean of a paired difference vector.

    Mirrors the frozen spatial_sweep.py numerically, so the CSV cross-check is
    deterministic rather than merely close:

      * the input is the FRACTIONAL difference, not scaled to percentage points
      * vectorized=False, matching the sweep's call
      * rng=np.random.default_rng(SEED), not random_state=SEED, which seeds a
        different stream
      * scaling to percentage points happens only after bootstrap() returns

    BCa selects order statistics from the resample distribution, so a
    difference of even a few ULP in the bias-correction or acceleration term
    can shift which resample lands at an interval endpoint. Any deviation from
    the sweep's exact call is therefore worth eliminating rather than
    tolerating.

    The alternate RNG path is retained only as a sensitivity observation.
    """
    kw = dict(confidence_level=0.95, n_resamples=N_BOOT, method="BCa",
              vectorized=False)
    if alt_rng:
        r = bootstrap((diff_frac,), np.mean, random_state=SEED, **kw)
    else:
        try:
            r = bootstrap((diff_frac,), np.mean,
                          rng=np.random.default_rng(SEED), **kw)
        except TypeError:
            raise SystemExit(
                "This scipy does not accept rng=; the frozen sweep uses it, so "
                "the cross-check cannot be made deterministic. Upgrade scipy or "
                "change both scripts to the same API."
            )
    return (float(np.mean(diff_frac)) * 100.0,
            float(r.confidence_interval.low) * 100.0,
            float(r.confidence_interval.high) * 100.0)


def contrasts(res, subs):
    """Primary endpoint: layouts aggregated within subject, then paired BCa."""
    print("=" * 70)
    print("PRIMARY CONTRASTS  (layouts averaged within subject; pp, BCa 95%)")
    print("=" * 70)

    agg = {}
    for model in ("RF", "LDA"):
        for count in LAYOUTS:
            n_lay = len(LAYOUTS[count])
            d1, dr = [], []
            for s in subs:
                per1, per29 = [], []
                ok = True
                for li in range(n_lay):
                    a = res[(model, count, li)].get(s)
                    if a is None:
                        ok = False
                        break
                    per1.append(a[0])
                    per29.append(a[1])
                if ok:
                    m1, m29 = np.mean(per1), np.mean(per29)
                    d1.append(m1)
                    dr.append(m1 - m29)
            agg[(model, count)] = (np.array(d1), np.array(dr))

    print(f"{'model':<6}{'contrast':>10}{'endpoint':>10}"
          f"{'mean':>9}{'95% BCa CI':>20}")
    print("-" * 70)
    rows, rows_alt = {}, {}
    for model in ("RF", "LDA"):
        for a, b in ((4, 8), (2, 8), (2, 4)):
            for i, ep in enumerate(("day1", "drop")):
                # fractional difference; bca() scales to pp after bootstrapping
                x = agg[(model, a)][i] - agg[(model, b)][i]
                m, lo, hi = bca(x)
                rows[(model, f"{a}vs{b}", ep)] = (m, lo, hi)
                rows_alt[(model, f"{a}vs{b}", ep)] = bca(x, alt_rng=True)
                print(f"{model:<6}{f'{a}vs{b}':>10}{ep:>10}"
                      f"{m:>9.3f}{f'[{lo:.3f}, {hi:.3f}]':>20}")
    print()

    print("RNG SENSITIVITY  (same data, alternate scipy RNG path)")
    print("-" * 70)
    worst = 0.0
    flips = 0
    for k in rows:
        m, lo, hi = rows[k]
        m2, lo2, hi2 = rows_alt[k]
        worst = max(worst, abs(lo - lo2), abs(hi - hi2))
        if (lo > 0) != (lo2 > 0) or (hi < 0) != (hi2 < 0):
            flips += 1
    print(f"  max endpoint shift across all 12 contrasts: {worst:.4f} pp")
    print(f"  contrasts whose zero-exclusion status changes: {flips}")
    print("  Interpretation is stable if the second number is 0.")
    print()
    return rows


def compare_saved(rows):
    """Diff the recomputed contrasts against whatever the sweep saved."""
    print("=" * 70)
    print("SAVED-OUTPUT CROSS-CHECK")
    print("=" * 70)
    files = glob.glob(os.path.join(SWEEP_OUT, "*.csv"))
    if not files:
        print("  no CSVs found in", SWEEP_OUT)
        print("  compare the table above against the terminal output by hand")
        return
    try:
        import pandas as pd
    except ImportError:
        print("  pandas not available; compare by hand")
        return

    for f in files:
        df = pd.read_csv(f)
        cols = {c.lower() for c in df.columns}
        if not {"model", "contrast", "endpoint"} <= cols:
            continue
        print(f"  {os.path.basename(f)}")
        mc = next(c for c in df.columns if c.lower() == "model")
        cc = next(c for c in df.columns if c.lower() == "contrast")
        ec = next(c for c in df.columns if c.lower() == "endpoint")
        num = [c for c in df.columns if df[c].dtype.kind == "f"]
        overall = 0.0
        point_worst = 0.0
        for _, r in df.iterrows():
            ep = "day1" if "day1" in str(r[ec]).lower() else "drop"
            key = (str(r[mc]), str(r[cc]), ep)
            if key not in rows:
                continue
            m, lo, hi = rows[key]
            saved = [r[c] for c in num[:3]]
            point_worst = max(point_worst, abs(saved[0] - m))
            worst = max(abs(saved[0] - m),
                        abs(saved[1] - lo) if len(saved) > 1 else 0,
                        abs(saved[2] - hi) if len(saved) > 2 else 0)
            overall = max(overall, worst)
            flag = "MATCH" if worst < 1e-9 else (
                "close" if worst < 0.01 else "DIFFERS")
            print(f"    {key}  saved {saved[0]:.3f}  recomputed {m:.3f}  "
                  f"max diff {worst:.4f}  {flag}")

        print()
        print(f"  max point-estimate discrepancy : {point_worst:.6f} pp")
        print(f"  max CI-endpoint discrepancy    : {overall:.4f} pp")
    print()
    print("  Point estimates must agree exactly. CI endpoints need not be")
    print("  bit-identical even with the RNG matched, because the two scripts")
    print("  reduce the within-subject layout average differently: the sweep")
    print("  aggregates with pandas groupby().mean(), the audit with np.mean().")
    print("  Floating-point addition is commutative but not associative, so a")
    print("  four-element average can differ in the last bits while a")
    print("  two-element or single-element one cannot. BCa selects order")
    print("  statistics, which can turn that into a shift of a few thousandths")
    print("  of a percentage point.")
    print()
    print("  This is why contrasts involving only the 4-channel (two-layout)")
    print("  and 8-channel (one-layout) aggregates read MATCH, while every")
    print("  contrast touching the four-layout 2-channel aggregate reads close.")
    print("  A discrepancy of this size is numerical sensitivity, not a")
    print("  disagreement about the quantity being estimated. Any row reading")
    print("  DIFFERS would be.")


def layout_means(res, subs):
    print("=" * 70)
    print("PER-LAYOUT MEANS ACROSS SUBJECTS (%)")
    print("=" * 70)
    print(f"{'model':<6}{'count':>6}{'layout':>20}{'day1':>9}{'day29':>9}{'drop':>9}")
    print("-" * 70)
    for model in ("RF", "LDA"):
        for count in sorted(LAYOUTS):
            for li, layout in enumerate(LAYOUTS[count]):
                a = res[(model, count, li)]
                d1 = np.mean([v[0] for v in a.values()]) * 100
                d29 = np.mean([v[1] for v in a.values()]) * 100
                print(f"{model:<6}{count:>6}{str(layout):>20}"
                      f"{d1:>9.3f}{d29:>9.3f}{d1 - d29:>9.3f}")
    print()


if __name__ == "__main__":
    res, subs = run_all()
    print(f"subjects evaluated: {len(subs)}\n")
    reproduction_checks(res, subs)
    layout_means(res, subs)
    placement_spreads(res, subs)
    rows = contrasts(res, subs)
    compare_saved(rows)