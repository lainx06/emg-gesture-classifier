# EMG Gesture Classifier — Project Context

**Read this before editing anything.** This file records design decisions that are
deliberate and were reached through analysis. Several of them look wrong or
"unoptimized" from the code alone. They are not.

---

## What this project is

Surface EMG gesture classification. Two tracks:

**Computational (the paper).** The GRABMyo dataset (PhysioNet, 43 subjects, 3
sessions on days 1 / 8 / 29, 2048 Hz, 16 forearm electrodes) is computationally
degraded to simulate commodity hardware — 2 channels instead of 8, 1000 Hz
instead of 2048, 10-bit instead of float. Classical ML only (LDA, KNN, linear
SVM, Random Forest, XGBoost) plus a SHAP explainability layer.

**Embedded.** MyoWare 2.0 sensors + Arduino Uno (ATmega328P: 32 KB flash,
2 KB SRAM, 16 MHz, no FPU). Per-stage latency instrumentation and a
flash/SRAM deployability table.

**The research question** is the *interaction*: does cheap hardware make
cross-session degradation worse, or do the two penalties simply add?

---

## Hard rules

- **No deep learning.** Not enough data, won't fit on the MCU, not the point.
- **One dataset: GRABMyo.** Settled. Do not propose alternatives.
- **Three classes only:** gesture 11 (wrist extension), 16 (hand close),
  17 (rest), remapped to 0 / 1 / 2.
- **Nothing self-collected enters the paper.** Demo calibration data is
  presentation-only.
- **Do not restructure `preprocess.py`, `pipeline.py`, or `build_cache.py`**
  without explicit review. Every constant in them is defended below.

---

## File map

```
EMG_CLASS/
├─ src/
│  ├─ preprocess.py       # WFDB loading, bipolar derivation, trimming
│  ├─ pipeline.py         # degradation chain + feature extraction
│  ├─ build_cache.py      # generates feature caches for all conditions
│  ├─ calibrate_vmax.py   # measures ADC full-scale range
│  └─ separability.py     # evaluation harness + statistics
├─ data/grabmyo/          # gitignored, ~1.7 GB
├─ cache/                 # gitignored, .npz feature caches
└─ notes/                 # decision records, analysis plan, results
```

---

## Locked design decisions

### Channel selection — F1F9 and F5F13

GRABMyo's forearm setup is two rings of 8 electrodes, 2 cm apart along the arm.
Electrode 1 sits on the centerline of the elbow crease (anterior). Eight
electrodes evenly spaced = 45° per step, so 4 steps = 180° = electrode 5,
on the posterior midline. F1 is over the flexor compartment, F5 over the
extensors — the antagonistic pair driving hand close and wrist extension.

The 180° relationship is direction-invariant, so clockwise vs counterclockwise
numbering does not matter.

**Channels are bipolar derivations, not raw columns.** Raw monopolar F1 and F9
correlate at 0.987; differencing removes that common-mode content (measured
post-difference correlation: 0.139). Ring 1 and ring 2 at the same angle,
2 cm apart, is exactly the geometry of a bipolar electrode pair, and matches
the MyoWare's own 2 cm contact spacing (measured).

- Commodity: 2 channels — `sig[:,0] - sig[:,8]`, `sig[:,4] - sig[:,12]`
- Clinical: 8 channels — `sig[:,i] - sig[:,i+8]` for i in 0..7

Commodity is a strict subset of clinical (positions 0 and 4), so the two arms
differ only in channel count, not derivation. Removing that confound was
deliberate.

Channels are anchored to an anatomical landmark, **not** tuned per subject —
a commodity armband gets positioned by landmark, not by muscle mapping. Per-subject
flexor/extensor dominance is inconsistent as a result (some subjects invert).
This is expected and is a Discussion point, not a bug.

### Degradation chain order

`difference → trim → resample → quantize → causal filter → window → features`

Quantization comes **before** filtering because on real hardware nothing can
filter the signal before the ADC sees it. Filtering first would smooth
quantization noise, which is the thing being measured.

### Resampling

`resample_poly(up=1000, down=2048)` — 2048/1000 is not an integer, so
`decimate` cannot be used. `resample_poly` applies an anti-aliasing FIR as an
intrinsic part of polyphase resampling; this is not optional and must not be
"optimized away."

### Quantization

10-bit, symmetric bipolar map over 1023 intervals, clipping at the rails.
V_MAX is **per condition**, measured at the exact point in the chain where the
quantizer runs (post-resample for resampled conditions, pre-resample for
`depth_only`). Measuring it anywhere else gives the wrong range.

Rationale for per-condition ranges: a hardware designer sets front-end gain so
the input fills the ADC span. This isolates quantization *resolution*, not the
behavior of a fixed-range ADC fed an unscaled signal.

### Filtering

4th-order Butterworth, 20 Hz **high-pass only**, applied to both arms.

- **`sosfilt`, not `sosfiltfilt`.** `filtfilt` is non-causal — it uses future
  samples. No real-time system can do that. Using it would make offline features
  cleaner than the hardware can ever produce. This is intentional; do not
  "improve" it.
- **No lowpass.** The dataset arrives with a 10–500 Hz Butterworth and 60 Hz
  notch already applied (verified by PSD). Leaving the high end unfiltered lets
  the sample-rate axis carry its real bandwidth penalty.
- 20 Hz clears motion artifact and electrode-impedance fluctuation at the cost
  of a sliver of low-frequency sEMG.

### Transient trim

200 ms discarded after filtering, using `int()` floor rather than `round()`.
The floor is load-bearing: `round(200 * 2048/1000)` = 410 samples = 200.195 ms,
which trims slightly too much and costs the clinical arm its final window,
breaking window-count symmetry between arms. Verified empirically that the
filter has settled by 200 ms.

### Windowing

200 ms window, 100 ms step, **computed in the time domain** and converted to
sample indices per arm. This guarantees identical window counts (37) across
all conditions despite differing sample rates. Do not rewrite in samples.

### Features — RMS, MAV, WL, ZC per channel

- WL and ZC are **rate-normalized** (per sample and per second respectively)
  because window length in samples differs between arms.
- ZC uses a 20 µV deadzone (`EPSILON = 0.02` mV). Value validated empirically:
  at 20 µV rest goes to ~0 Hz and the two gestures stay distinct; at 3.2 µV
  rest is submerged in baseline noise (~170–210 Hz).
- `np.sign(0)` is forced to 1. After quantization, samples land exactly on the
  zero level frequently, and `sign` returning 0 would double-count crossings.
- Feature names are keyed by **derivation** (`F1F9_rms`), not position index,
  so the same physical channel has the same label across conditions. SHAP
  comparisons depend on this.

### Evaluation protocol

- **Subject-dependent**: one model per subject, accuracies averaged across 43.
- **Session-wise splits**, never shuffled. Training is always session 1,
  trials 1–5 (parity across all conditions). Testing is trials 6–7 of the
  target session.
- Trial is the smallest independent unit. Windows overlap by 50%, so splitting
  below trial level leaks.
- `StandardScaler` fits on **train only**, transforms test. The scaler is part
  of the model, not preprocessing.
- Statistics: paired Wilcoxon signed-rank across subjects, two-sided, Bonferroni
  across 5 models (α = 0.01), bootstrap 95% CIs on all interaction tests.
  See `notes/analysis_plan.md`.

---

## Known-fragile things

These have caused silent bugs before. Preserve them.

- `load_trial` asserts the WFDB column map is F1–F16 at indices 0–15 before
  indexing. Do not remove.
- `load_trial` raises on records shorter than 2048 samples rather than
  returning a short array.
- Exception handling in `build_cache` counts skips and errors separately and
  prints a tally. Never replace with a bare `except: continue`.
- `build_cache` asserts `X`, `Y`, and `GROUPS` have matching row counts.
- Paths use `os.path.dirname(os.path.abspath(__file__))`, not relative strings.
  Relative paths resolve against the working directory and silently write the
  cache outside the project.
- Any parameter measured from data must be measured on **the exact signal at
  the point in the chain where it is used**. This has been the source of
  several bugs (aliasing fraction, V_MAX).

---

## Current state (as of 24 July 2026)

**Done:** full pipeline, six-condition feature cache (43 subjects × 3 sessions
× 3 gestures × 7 trials × 37 windows), five-model evaluation, degradation curve
(days 1 / 8 / 29), interaction tests with bootstrap CIs.

**Headline results (LDA):** clinical within-session 96.2%, commodity
within-session 86.1%, clinical cross-session 82.5%, commodity cross-session
71.4%. Both main effects p < 0.001. Interaction is model-dependent —
significant under Random Forest (p = 0.009), null under LDA and KNN.

**Key observation:** commodity Day-29 accuracy is flat across all five models
(70.7–73.0%) while clinical spans 82.5–91.8%. Model choice buys 9.3 points of
cross-session robustness on 8 channels, 2.3 points on 2.

**Outstanding:** V_MAX pooled-vs-per-channel decision, cache rebuild for the
six-condition ablation, SHAP (Phase 2, mentor requirement), embedded latency
and deployability measurements, Methods + Results draft (due 3 August 2026).

---

## Deadlines

- **3 Aug 2026** — Methods and Results sections to mentor
- **7 Aug 2026** — full finalized draft
- **13 Aug 2026** — internship ends; presentation (5 minutes)
- Conference submission happens after the internship, following public
  release review. Venue TBD.
