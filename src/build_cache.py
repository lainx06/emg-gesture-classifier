import numpy as np
import os
from preprocess import load_trial
from pipeline import process_trial

DRY_RUN = False

SUBJECTS = range(1, 3) if DRY_RUN else range(1, 44)
SESSIONS = range(1, 4)
GESTURES = [11, 16, 17]          # wrist extension, hand close, rest
GEST_MAP = {11: 0, 16: 1, 17: 2}
TRIALS = range(1, 8)

current_dir = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(os.path.dirname(current_dir), "cache")

CONDITIONS = [
    "clinical",      # 8ch, 2048 Hz, float   -- baseline
    "spatial_only",  # 2ch, 2048 Hz, float
    "rate_only",     # 8ch, 1000 Hz, float
    "depth_only",    # 8ch, 2048 Hz, 10-bit
    "rate_depth",    # 8ch, 1000 Hz, 10-bit
    "commodity",     # 2ch, 1000 Hz, 10-bit  -- full degradation
]


def get_feature_names(condition):
    """
    Feature names keyed by derivation, not position index, so the same physical
    channel carries the same label in every condition. SHAP comparisons depend
    on this.
    """
    features = ['rms', 'mav', 'wl', 'zc']
    if condition in ["clinical", "rate_only", "depth_only", "rate_depth"]:
        channels = ["F1F9", "F2F10", "F3F11", "F4F12",
                    "F5F13", "F6F14", "F7F15", "F8F16"]
    elif condition in ["commodity", "spatial_only"]:
        channels = ["F1F9", "F5F13"]
    else:
        raise ValueError(f"Invalid condition: {condition}")
    return np.array([f"{ch}_{feat}" for ch in channels for feat in features])


def build_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"\n--- Generating Cache ({'DRY RUN' if DRY_RUN else 'FULL BUILD'}) ---")

    data = {cond: {"X": [], "Y": [], "GROUPS": []} for cond in CONDITIONS}
    skipped = {"missing": 0, "error": 0}

    for subj in SUBJECTS:
        for sess in SESSIONS:
            for gest in GESTURES:
                for trial in TRIALS:
                    try:
                        # Single disk read per trial; all conditions derive from
                        # the pristine 8-channel bipolar baseline.
                        sigs = load_trial(session=sess, participant=subj,
                                          gesture=gest, trial=trial)
                        raw_baseline = sigs["clinical"]

                        for condition in CONDITIONS:
                            features = process_trial(raw_baseline, condition=condition)
                            n_windows = features.shape[0]
                            if n_windows == 0:
                                continue

                            data[condition]["X"].append(features)
                            data[condition]["Y"].extend([GEST_MAP[gest]] * n_windows)
                            for w in range(n_windows):
                                data[condition]["GROUPS"].append([subj, sess, trial, w])

                    except FileNotFoundError:
                        skipped["missing"] += 1
                        continue
                    except Exception as e:
                        skipped["error"] += 1
                        print(f"Error Subj {subj}, Sess {sess}, Gest {gest}, "
                              f"Trial {trial}: {e}")
                        continue

        if subj % 5 == 0 or subj == SUBJECTS[-1]:
            print(f"Processed Subject {subj}/{SUBJECTS[-1]}")

    for condition in CONDITIONS:
        feat_names = get_feature_names(condition)
        X_list = data[condition]["X"]
        X_out = np.vstack(X_list) if X_list else np.empty((0, len(feat_names)))
        Y_out = np.array(data[condition]["Y"], dtype=np.int8)
        GROUPS_out = np.array(data[condition]["GROUPS"], dtype=np.int16)

        assert X_out.shape[0] == Y_out.shape[0] == GROUPS_out.shape[0], \
            f"Length mismatch in {condition} arrays."

        prefix = f"{condition}_dry_run" if DRY_RUN else condition
        out_path = os.path.join(CACHE_DIR, f"{prefix}_cache.npz")
        np.savez_compressed(out_path, X=X_out, Y=Y_out,
                            GROUPS=GROUPS_out, feature_names=feat_names)
        print(f"[{condition.upper()}] {out_path}")
        print(f"Shape X: {X_out.shape} | Y: {Y_out.shape} | GROUPS: {GROUPS_out.shape}")

    print(f"\nMissing: {skipped['missing']} | Errors: {skipped['error']}")


if __name__ == "__main__":
    build_cache()