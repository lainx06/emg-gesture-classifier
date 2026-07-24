import os
import numpy as np
import wfdb

def load_trial(session, participant, gesture, trial):
    """
    Loads a GRABMyo WFDB trial, derives bipolar channels, and trims edge artifacts.
    Returns both clinical (8ch) and commodity (2ch) arrays to prevent double I/O.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, "..", "data", "grabmyo")
    
    sess_dir = f"Session{session}"
    part_dir = f"session{session}_participant{participant}"
    file_prefix = f"session{session}_participant{participant}_gesture{gesture}_trial{trial}"
    
    file_path = os.path.join(base_dir, sess_dir, part_dir, file_prefix)
    
    record = wfdb.rdrecord(file_path)
    sig = record.p_signal
    
    # Enforce expected channel ordering (F1-F16 in indices 0-15)
    expected_channels = [f"F{i}" for i in range(1, 17)]
    assert record.sig_name[:16] == expected_channels, f"Channel map mismatch in {file_prefix}: {record.sig_name[:16]}"
    
    if sig.shape[0] < 2048:
        raise ValueError(f"Trial {file_prefix} too short: {sig.shape[0]} samples")
    
    # Clinical: 8 pairs (F1-F9 through F8-F16)
    clinical_sig = np.zeros((sig.shape[0], 8))
    for i in range(8):
        clinical_sig[:, i] = sig[:, i] - sig[:, i + 8]
            
    # Commodity: 2 pairs (F1-F9 and F5-F13)
    commodity_sig = np.zeros((sig.shape[0], 2))
    commodity_sig[:, 0] = sig[:, 0] - sig[:, 8]
    commodity_sig[:, 1] = sig[:, 4] - sig[:, 12]
        
    # Trim edge artifacts
    clinical_sig = clinical_sig[1024:-1024, :]
    commodity_sig = commodity_sig[1024:-1024, :]
    
    return {"clinical": clinical_sig, "commodity": commodity_sig}