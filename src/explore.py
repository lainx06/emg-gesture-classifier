import numpy as np
from preprocess import load_trial

FS = 2048
GESTURES = {11: "Wrist Extension", 15: "Hand Open", 16: "Hand Close", 17: "Rest"}

def test_zc_comparison():
    epsilons_uv = [3.2, 20.0]
    
    for eps_uv in epsilons_uv:
        eps = eps_uv / 1000.0 # Convert to mV
        print(f"\n--- Testing ε = {eps_uv} µV ---")
        
        for g, name in GESTURES.items():
            zc_rates_ch0 = []
            zc_rates_ch1 = []
            for t in range(1, 8):
                try:
                    sig = load_trial(1, 1, g, t)
                    if sig is None or len(sig) == 0: continue
                    
                    s0 = sig[:, 0]
                    sc0 = np.diff(np.sign(s0)) != 0
                    ac0 = np.abs(np.diff(s0)) >= eps
                    zc_rates_ch0.append(np.sum(sc0 & ac0) / (len(s0) / FS))
                    
                    s1 = sig[:, 1]
                    sc1 = np.diff(np.sign(s1)) != 0
                    ac1 = np.abs(np.diff(s1)) >= eps
                    zc_rates_ch1.append(np.sum(sc1 & ac1) / (len(s1) / FS))
                except FileNotFoundError:
                    continue
            
            if zc_rates_ch0:
                print(f"[{name:<15}] G{g}: Ch0 = {np.mean(zc_rates_ch0):>5.1f} Hz | Ch1 = {np.mean(zc_rates_ch1):>5.1f} Hz")

if __name__ == "__main__":
    test_zc_comparison()