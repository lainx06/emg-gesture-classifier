# Analysis Plan — Exploratory Spatial Budget Sweep

**Plan frozen:** 8 August 2026, before generating any spatial-sweep results.

**Status:** post-hoc and exploratory. Not part of the locked analysis plan
dated 23 July 2026. Does not replace or modify the six-condition ablation,
which remains the primary spatial result.

**Written before generating any new numbers.** Nothing below was chosen after
seeing an outcome.

**Timing:** run beginning 8 August 2026, after the presentation deck was finalized
and rehearsed and before journal submission. The presentation deck remains frozen.
The manuscript is not modified until results exist and are understood.

---

## 1. Questions

**Q1 — Channel budget.** Does cross-session robustness change systematically
as spatial coverage goes 2 → 4 → 8 channels?

**Q2 — Placement sensitivity.** At a fixed channel count, how much does
performance vary with rotated placement on the ring?

These are separate. Q1 is the primary endpoint. Q2 is descriptive and exists
to bound how much of the 2-channel penalty is attributable to a particular
placement rather than to count.

---

## 2. Conditions

All layouts are fixed by array geometry alone and specified here before any
result is observed. Derivation indices refer to the eight clinical bipolar
channels; derivation *d* occupies feature columns 4d through 4d+3.

| Count | Layouts | Angular spacing |
|---|---|---|
| 8 | `[0,1,2,3,4,5,6,7]` | 45° |
| 4 | `[0,2,4,6]`, `[1,3,5,7]` | 90° |
| 2 | `[0,4]`, `[1,5]`, `[2,6]`, `[3,7]` | 180° |

Seven conditions total.

**6 channels is excluded.** Six positions cannot be evenly spaced on an
eight-position ring; any 6-channel layout has mixed 45°/90° spacings and is
therefore not a rotation of a regular pattern. Including it would break the
geometric comparability that makes within-count variation interpretable. Two,
four, and eight also provide natural regular design points on the eight-position array.

**Note on structure:** the two 4-channel layouts partition the array, as do the
four 2-channel layouts. They are complementary rather than independently drawn,
which is a property of the design and should be stated rather than treated as
sampling.

---

## 3. Data and pipeline

**Source:** the existing clinical cache — 8 channels, 2048 Hz, floating point.

**Method:** select feature columns for the derivations in each layout. No
raw-signal reprocessing. Because features are computed per channel
independently, selecting columns after extraction is equivalent to selecting
channels before it.

**Rationale for using clinical rather than a degraded cache:** the question is
about spatial coverage alone. Holding rate and depth at clinical values keeps
the spatial axis unconfounded, and matches the existing `spatial_only`
condition.

---

## 4. Validation gate — mandatory, before any new numbers

Selecting columns `[0,1,2,3,16,17,18,19]` from the clinical cache must
reproduce the existing `spatial_only` condition **exactly**: identical
per-subject Day-1 and Day-29 accuracies, for both models, to floating-point
equality.

Both are the same two channels at the same rate and precision through the same
feature extraction, so any difference indicates a column-indexing error.

**If the gate fails, stop.** Do not generate sweep results until it passes.
Every number downstream depends on the column mapping being correct.

---

## 5. Models

**Random Forest** (50 estimators, `random_state=42`) and **LDA**.

Two models, not five. RF is where the compounding interaction survived
Bonferroni correction; LDA is where it was null. The contrast is the point.

**On interpretation:** if RF's robustness advantage over LDA widens with
channel count, that is *model-dependent evidence consistent with* training-time
spatial redundancy. It is not proof. Other explanations — differing bias-variance
behaviour with feature count, differing capacity to represent feature
interactions — predict similar patterns. The manuscript's existing hedging on
this mechanism stands regardless of what the sweep shows.

---

## 6. Evaluation protocol

Unchanged from the manuscript. Train on Session 1 trials 1–5 per subject;
scaler fit on the training partition only; test on trials 6–7 of Sessions 1
and 3. Trial-level splits. Subject-dependent.

Day 8 is not used. The two endpoints below are the only ones reported.

---

## 7. Endpoints

Exactly two, matching `tab:ablation`:

- **Day-1 accuracy**
- **Cross-session drop** (Day-1 accuracy − Day-29 accuracy)

No additional endpoints. No headroom normalization. No per-class breakdown.

### Primary: channel-count trend

Layouts are aggregated **within subject** before any comparison, so the
independent unit remains the 43 subjects and no subject contributes multiple
correlated observations.

For subject *s*, model *m*, channel count *c*:

```
acc_d1(s,m,c)  = mean over layouts L of count c of acc_d1(s,m,L)
acc_d29(s,m,c) = mean over layouts L of count c of acc_d29(s,m,L)
drop(s,m,c)    = acc_d1(s,m,c) − acc_d29(s,m,c)
```

Averaging accuracies and then differencing is identical to differencing and
then averaging, since the drop is linear in the accuracies.

Report paired per-subject differences from the 8-channel condition, with 95%
BCa bootstrap confidence intervals — the same estimator and presentation as the
existing ablation table.

Contrasts: 4 vs 8, 2 vs 8, and 2 vs 4.

### Exploratory model contrast

Because the RF--LDA comparison motivates including two models, it is specified here
before results are generated. For each subject and channel count, after averaging
layouts within subject, define the RF robustness advantage as:

```
rf_advantage_drop(s,c) = drop_LDA(s,c) - drop_RF(s,c)
```

Positive values mean RF loses less accuracy across sessions than LDA. Report the
subject-level mean and 95% BCa confidence interval at 2, 4, and 8 channels, plus
the paired change in this advantage from 2 to 8 channels. This remains an
estimation-focused exploratory contrast on the existing cross-session-drop endpoint;
it is evidence consistent with model-dependent spatial redundancy if the advantage
widens with channel count, not proof of that mechanism.

**No multiplicity correction.** This analysis is estimation-focused and
exploratory; intervals are interpreted individually and descriptively, not as
simultaneous family-wise bounds. Stated the same way as the existing ablation.

### Secondary, descriptive: placement variability

At each channel count with more than one layout:

- Per-subject spread across layouts (max − min), reported as a distribution
  across the 43 subjects
- Per-layout means across subjects, so the relative position of each rotation
  is visible

Descriptive only. No hypothesis test on placement.

---

## 8. Pre-specified interpretation

Committing now to what each outcome means, so the reading is not chosen after
the fact.

**Channel-count trend.** Three points do not define a curve; no functional form
will be fitted. Three readings are possible and all are reportable:

- 4 channels close to 8 → spatial coverage saturates below the full array;
  the practical recommendation becomes "four, not eight"
- 4 channels roughly midway → penalty scales with coverage
- 4 channels close to 2 → the penalty is incurred on departing from the full
  array rather than accumulating gradually

**Placement variability.** Within-count layout variation is reported descriptively.
If the best-performing 2-channel layout exceeds the worst-performing 4-channel
layout, that is evidence that placement can partially offset the effect of channel
budget for particular configurations. It does not, by itself, establish that
placement is a larger determinant than channel count.

**On `[0,4]` specifically.** The manuscript argues this pair has a physiological
rationale — opposing flexor and extensor compartments — and states explicitly
that it was landmark-anchored rather than optimized. It does *not* claim the
pair is optimal. Whatever position `[0,4]` occupies among the four rotations is
reported as found.

The 15% attribution share reported in the SHAP analysis is a property of those
features *within the 8-channel model* and does not predict standalone 2-channel
performance. The two quantities are not comparable and neither is used to
anticipate the other.

**No post-hoc layout selection.** Layouts are fixed in Section 2. If one
performs poorly it is reported. The manuscript's channel choice is not revised
to a better-performing rotation, because doing so would substitute an
outcome-selected placement for a physiologically motivated one.

---

## 9. Failure modes to guard against

**This is not an electrode-shift experiment.** Every condition trains and tests
on the same layout. Electrode shift means training on one placement and testing
on another, which is a different question and is not addressed here. The
distinction must be stated explicitly in any text that results, or a reader
will conflate them.

**The 8-channel condition has no placement variability.** With a single layout,
its within-count spread is zero by construction. This asymmetry should be noted
rather than presented as a finding.

**Aggregation hides variability by design.** The primary endpoint averages over
layouts, which is what keeps the pairing valid, but it also means the primary
result cannot speak to placement. That is what the secondary analysis is for.
Neither should be reported without the other.

**Subject-dependent framing is unchanged.** Every model is fit per subject.
Nothing here generalizes across subjects, and the sweep does not change that.

---

## 10. Deliverable if the result is worth reporting

One table and at most one figure, in a subsection labelled explicitly as a
post-hoc spatial budget sensitivity analysis. Placed after the existing
Degradation Axis Attribution results, not merged into them.

If the result is uninformative — for example, wide overlapping intervals with
no discernible trend — it does not need a full main-text subsection, but the
complete result is retained in the analysis record and may be summarized briefly
in the manuscript or supplement if relevant. Results are not omitted solely because
they contradict or complicate the current interpretation.

**The six-condition ablation remains the primary spatial result regardless of
outcome.**

---

## 11. Reasons this may not be worth doing

Stated here so the decision is made on merit rather than momentum:

- The paper is complete and approved by the corresponding author
- The current Limitations already name the absent sweep, which is an acceptable
  disclosure rather than a defect
- A result showing large placement variability would require rewriting the
  Discussion's framing of the spatial penalty, which is real work
- Reviewers frequently request exactly this analysis, in which case it is
  better performed during revision, when its scope can be negotiated

The strongest argument for running it: it is cheap, it is pre-specified, and
having the answer before a reviewer asks is better than after.
