import numpy as np
from scipy import signal
from preprocess import load_trial

def calibrate_vmax():
    print("Scanning Session 1 across all 43 subjects for V_MAX calibration...")
    SUBJECTS = range(1, 44)
    GESTURES = [11, 16, 17]
    TRIALS = range(1, 8)
    
    amp_raw = []
    amp_resampled = []
    first_error = None
    success_count = 0
    
    for subj in SUBJECTS:
        for gest in GESTURES:
            for trial in TRIALS:
                try:
                    sigs = load_trial(session=1, participant=subj, gesture=gest, trial=trial)
                    sig_clin = sigs["clinical"]
                    
                    # 1. Raw (2048 Hz) for depth_only
                    amp_raw.append(np.abs(sig_clin))
                    
                    # 2. Resampled (1000 Hz) for rate_depth
                    sig_resampled = signal.resample_poly(sig_clin, up=1000, down=2048, axis=0)
                    amp_resampled.append(np.abs(sig_resampled))
                    
                    success_count += 1
                except Exception as e:
                    if first_error is None:
                        first_error = (subj, gest, trial, type(e).__name__, str(e))
                    continue
        if subj % 5 == 0 or subj == 43:
            print(f"Processed Subject {subj}/43")
            
    print(f"\nSuccessfully processed {success_count} trials.")
    if success_count == 0:
        print("--- ERROR DEBUG INFO ---")
        if first_error:
            print(f"Failed on Subj {first_error[0]}, Gest {first_error[1]}, Trial {first_error[2]}")
            print(f"Error Type: {first_error[3]}")
            print(f"Error Message: {first_error[4]}")
        return
        
    # Concatenate along time axis (keeps 8 channels intact)
    global_raw = np.concatenate(amp_raw, axis=0)
    global_resampled = np.concatenate(amp_resampled, axis=0)
    
    # Free memory
    del amp_raw
    del amp_resampled
    
    # Calculate 99.9th percentile PER CHANNEL on raw data for sanity check
    v_max_per_channel = np.percentile(global_raw, 99.9, axis=0)
    
    # Calculate global 99.9th percentiles
    v_max_raw = np.percentile(global_raw, 99.9)
    v_max_resampled = np.percentile(global_resampled, 99.9)
    
    channels = ["F1F9", "F2F10", "F3F11", "F4F12", "F5F13", "F6F14", "F7F15", "F8F16"]

    print("\n--- Calibration Results ---")
    print("99.9th Percentile Per Channel (Raw 2048 Hz):")
    for ch_name, val in zip(channels, v_max_per_channel):
        print(f"  {ch_name}: {val:.4f} mV")
        
    print("\n--- Global V_MAX Targets ---")
    print(f"V_MAX_8CH_2048 (Use for depth_only): {v_max_raw:.4f} mV")
    print(f"V_MAX_8CH_1000 (Use for rate_depth): {v_max_resampled:.4f} mV")

if __name__ == "__main__":
    calibrate_vmax()