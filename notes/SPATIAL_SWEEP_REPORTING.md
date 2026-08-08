# Spatial Sweep — Reporting Structure and Scope of Change

Written after the sweep but before any manuscript edit. Nothing here is
inserted until the audit passes and the placement-spread distributions exist.

---

## 1. Proposed reporting structure

One table and one figure, in a subsection placed **after** Degradation Axis
Attribution and labelled explicitly as post-hoc and exploratory.

### Table — spatial budget and placement, two panels

One table, two panels. Panel A is the raw material; Panel B is the
pre-specified secondary endpoint. They belong together because Panel B
characterizes variation that Panel A cannot show.

**Panel A — per-layout means across subjects (%)**

| Channels | Layout | RF Day 1 | RF drop | LDA Day 1 | LDA drop |
|---|---|---|---|---|---|
| 8 | 0–7 | 97.15 | 5.40 | 96.18 | 13.68 |
| 4 | 0,2,4,6 | 95.22 | 10.75 | 93.97 | 13.84 |
| 4 | 1,3,5,7 | 97.43 | 6.13 | 96.38 | 7.71 |
| 2 | 0,4 | 86.05 | 14.49 | 84.62 | 16.58 |
| 2 | 1,5 | 95.32 | 7.82 | 93.66 | 5.07 |
| 2 | 2,6 | 94.76 | 10.64 | 94.19 | 7.89 |
| 2 | 3,7 | 92.06 | 13.44 | 92.78 | 12.64 |

**Panel B — per-subject placement spread, max minus min across layouts (pp)**

Distribution over the 43 subjects. Reported alongside the layout means because
group-level layout means can obscure substantial within-subject placement
variability.

| Model | Channels | Endpoint | Median | IQR | Range |
|---|---|---|---|---|---|
| RF | 2 | Day 1 | 14.86 | 5.86 – 20.72 | 0.45 – 42.34 |
| RF | 2 | drop | 30.18 | 20.95 – 36.04 | 4.50 – 89.19 |
| RF | 4 | Day 1 | 0.90 | 0.00 – 4.50 | 0.00 – 16.67 |
| RF | 4 | drop | 12.61 | 3.60 – 21.40 | 0.45 – 48.65 |
| LDA | 2 | Day 1 | 14.86 | 5.63 – 21.17 | 0.00 – 39.64 |
| LDA | 2 | drop | 30.63 | 21.85 – 44.59 | 9.46 – 77.48 |
| LDA | 4 | Day 1 | 1.35 | 0.45 – 5.86 | 0.00 – 26.13 |
| LDA | 4 | drop | 13.96 | 4.50 – 25.90 | 0.00 – 63.96 |

The caption must state that `[0,4]` is the configuration used throughout the
primary analysis, so a reader can locate it immediately. The 8-channel
condition has a single layout, so its spread is zero by construction and is
omitted from Panel B.

Two guards on how Panel B is read, both pre-specified:

**A maximum-minus-minimum over four estimates is inflated by noise alone.**
With roughly 222 test windows per subject per session, the binomial standard
error on an accuracy is about 2 pp and on a drop about 2.8 pp; the expected
range over four independent draws is therefore around 6 pp before any placement
effect. A naive independent-window calculation gives a nonzero expected range
even without a placement effect, and overlapping windows increase that floor.
The observed approximately 30-pp median at two channels is substantially larger
than the naive independent-window reference, but no formal noise-only null
distribution was estimated.

**The per-subject maximum is selection-biased.** The best layout is identified
on the same data used to evaluate it. These figures therefore quantify how much
outcomes vary with placement; they do not estimate what per-subject placement
selection would achieve. That would require the layout to be chosen on training
data and evaluated on held-out data, which is a different experiment.

### Figure — one panel per endpoint

- x-axis: channel count (2, 4, 8), categorical
- small markers: individual layouts
- large marker with CI: subject-averaged aggregate
- RF and LDA distinguished by colour

This shows the count trend and the placement spread in the same view, which is
the whole point. Two panels, Day-1 accuracy and cross-session drop.

Do not fit a line through three points.

### Text

Three short paragraphs at most:

1. Aggregated contrasts, RF and LDA, both endpoints
2. Placement spread — the pre-specified per-subject distribution, not the
   layout means, since group-level means can obscure within-subject variability
3. The RF-minus-LDA robustness differential across counts, stated as
   model-dependent evidence consistent with the redundancy account rather than
   as support for it over alternatives

---

## 2. What the sweep changes

### Magnitude of the reported spatial penalty is placement-sensitive

The primary analysis uses `[0,4]`, which is the weakest of the four regular
2-channel rotations tested. Averaging across rotations within subject:

| Endpoint | `[0,4]` vs 8ch | Placement-averaged 2ch vs 8ch | Reduction |
|---|---|---|---|
| Day-1 accuracy | −11.10 pp | −5.10 pp `[−6.49, −3.96]` | 54% |
| Cross-session drop | +9.09 pp | +6.20 pp `[+3.19, +8.63]` | 32% |

The Day-1 penalty roughly halves. The degradation penalty does not — it falls
by about a third and remains substantial under every rotation tested.

### Placement variation within subjects is large

The spread across layout means understates the effect substantially. Under RF
at 2 channels, layout means span 9.3 pp on Day-1 accuracy and 6.7 pp on the
drop, while the median individual subject sees 14.9 pp and 30.2 pp
respectively.

The larger within-subject spread than the range of group-level layout means is
consistent with substantial subject-specific placement sensitivity. Determining
whether preferred rotations systematically differ across subjects would require
explicitly examining per-subject layout rankings, which is not done here. This
is consistent with the manuscript's existing framing that a rigid,
landmark-anchored placement reflects how a consumer armband is actually worn,
and it quantifies what that generalized fit costs relative to alternative fixed
rotations.

### The claim wording needs to widen from count to configuration

"Channel count is the dominant driver" becomes something closer to "spatial
configuration — channel count together with placement — is the dominant
constraint." The ordering against rate and depth is unaffected; the attribution
to count alone is.

### An intermediate point now exists

4 channels recovers nearly all Day-1 accuracy (−0.82 pp `[−1.52, −0.24]` under
RF) but only part of the robustness (+3.04 pp drop `[+0.26, +5.58]`). That is
new information rather than a correction, and it is the most directly
actionable result for a hardware designer.

### The fixed-layout result sits at the adverse end of the tested placement range

`[0,4]` was specified in advance on anatomical grounds and the result reported
for it is valid for that layout. But among the four regular 2-channel rotations
tested, it produced the largest penalty on both endpoints. The reported figures
should therefore be described as pertaining to a predetermined fixed placement
that falls toward the pessimistic end of the tested placement range, not as a
conservative estimate of 2-channel performance in general.

The distinction matters. "Conservative estimate" invites the reading that
2-channel systems are at least this good. The finding is narrower: this
particular layout performs worse than the other rotations examined, so the
headline penalty is not representative of 2-channel configurations as a class.

### Limitations text is narrowed, not removed

The sweep adds one intermediate channel budget (4) and regular rotated
placements at 2 and 4 channels. It does not densely sample channel count, and
it does not examine arbitrary electrode subsets — only evenly spaced rotations
of the eight-position ring. Three counts is not a continuous sweep.

The existing limitation should therefore be rewritten to reflect what is now
covered and what still is not, rather than deleted.

---

## 3. What the sweep does not change

**The six-condition ablation remains primary.** It answers a different
question — which of three degradation axes matters — and the sweep does not
revisit rate or depth at all.

**Rate and depth remain negligible.** Untouched by this analysis. The tested
reductions changed both endpoints by well under a point, and nothing here bears
on that.

**The ordering across axes holds.** Even the best 2-channel rotation, `[1,5]`
under RF, costs 1.83 pp at Day 1 and 2.42 pp of additional drop relative to 8
channels. Rate and depth cost roughly 0.2. The spatial axis remains an order of
magnitude larger under every placement tested.

**Model dependence of the interaction is unchanged**, and the sweep is
consistent with it: LDA shows no benefit from additional channels on the drop
endpoint at any count, with all intervals including zero.

**All embedded and deployability results are unaffected.** Memory, flash,
latency, and the implementation verification concern the 2-channel commodity
configuration as built and are independent of which rotation is used.

**The SHAP results are unaffected.** Attribution was measured within the
8-channel model, and the sweep evaluates standalone layouts. The two do not
speak to each other and neither should be used to interpret the other.

**The 2×2 main effects are unaffected.**

---

## 4. Interpretations explicitly not being made

**The physiological rationale for `[0,4]` is not contradicted.** The manuscript
claims that opposing flexor and extensor placement provides a rationale for
separability and that the choice was landmark-anchored rather than optimized.
It never predicts that `[0,4]` is optimal. The finding is that the fixed-layout
penalty is placement-sensitive and that this particular layout is the weakest of
those tested.

**No mechanism is attributed to the placement effect.** The sweep does not
isolate any single derivation, and 8-channel attribution shares do not rank
standalone 2-channel configurations. Any amplitude- or anatomy-based explanation
would be a new post-result hypothesis requiring its own analysis.

**Placement is not claimed to dominate channel count.** The best 2-channel
rotation matching or exceeding the worst 4-channel layout shows that placement
can partially offset channel budget for particular configurations. It does not
establish that placement is the larger determinant, and the frozen plan does not
license that reading.

**The RF/LDA differential is not proof of redundancy.** It is model-dependent
evidence consistent with the account. Differing bias-variance behaviour with
feature count, or differing capacity to represent feature interactions, predict
similar patterns. The manuscript's existing hedging stands.

---

## 5. Numerical provenance

Recorded from the independent audit so the manuscript can state what was
verified and to what precision.

**Reproduction gates — both exact.**

| Check | Max absolute difference | Comparisons |
|---|---|---|
| 8ch `[0..7]` vs the clinical condition | 0.000e+00 | 86 |
| 2ch `[0,4]` vs the `spatial_only` condition | 0.000e+00 | 86 |

Column mapping verified against `feature_names`: derivations 0–7 are F1F9
through F8F16, so derivation 0 is F1F9 and derivation 4 is F5F13 — the pair
used throughout the primary analysis.

**Independent recomputation.** Every per-layout mean and every contrast point
estimate reproduced exactly. Confidence-interval endpoints agreed to within
**0.004 pp**.

That residual is numerical, not substantive. The frozen sweep aggregates
layouts within subject using `pandas.groupby().mean()`; the audit uses
`np.mean()`. Floating-point addition is commutative but not associative, so a
four-element average can differ in the last bits while a two- or one-element
average cannot. BCa selects order statistics from the resample distribution,
which can convert a last-bit difference into a shift of a few thousandths of a
percentage point.

The pattern confirms this: every contrast involving only the 4-channel
(two-layout) and 8-channel (one-layout) aggregates agreed exactly, while every
contrast touching the four-layout 2-channel aggregate showed the small
difference.

**RNG sensitivity.** Re-running the intervals through scipy's alternate
`random_state=` path rather than `rng=np.random.default_rng(SEED)` shifted
endpoints by up to **0.2655 pp**. **Zero** contrasts changed their
zero-exclusion status.

**Summary for the manuscript.** Across all contrasts, alternate bootstrap RNG
handling shifted interval endpoints by at most 0.2655 pp and did not change
zero-exclusion status. RF 4-vs-8 drop has the interval closest to zero and
should therefore be reported with its interval and interpreted cautiously
rather than described as strong evidence.

---

## 6. Order of operations

1. Run the audit. Confirm the reproduction gates, obtain the per-subject
   placement-spread distributions, cross-check the BCa outputs.
2. Freeze the sweep results and commit them under a tag.
3. Draft the subsection.
4. Send Dr. Lin the results and the proposed text together, so she reviews a
   concrete change rather than a proposal.

The manuscript is not edited before step 3.
