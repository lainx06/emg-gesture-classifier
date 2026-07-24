# Data Exclusion Criteria

## High Baseline Noise / Non-Compliance
A subject or specific session will be excluded entirely prior to feature extraction if their baseline "Rest" state demonstrates excessive amplitude relative to their active states.

**Threshold Rule:**
Exclude if the mean RMS of the `rest` gesture (across all 7 trials) on either channel exceeds **30%** of the mean RMS of the active gestures (`wrist_extension` and `hand_close`) on that same channel.

*Rationale:* Prevents model contamination from artifacts, improper hardware contact, or subjects failing to maintain a true resting state. This rule is applied strictly to baseline noise metrics to avoid the bias of dropping subjects based on poor classifier outcomes.