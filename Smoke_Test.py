import numpy as np
import matplotlib.pyplot as plt
from scipy import signal   # imported to confirm it installed

fs = 1000                          # sampling rate, Hz
t = np.arange(0, 2, 1/fs)          # 2 seconds of timestamps

burst = np.zeros_like(t)
burst[(t > 0.5) & (t < 1.5)] = 1.0
fake_emg = np.random.randn(len(t)) * burst
fake_emg += 0.5 * np.sin(2 * np.pi * 60 * t)   # mains hum

region = (t > 0.5) & (t < 1.5)
rest   = (t < 0.5) | (t > 1.5)
print("burst mean:    ", np.mean(fake_emg[region]))
print("burst mean-abs:", np.mean(np.abs(fake_emg[region])))
print("rest mean:     ", np.mean(fake_emg[rest]))
print("rest mean-abs: ", np.mean(np.abs(fake_emg[rest])))

plt.plot(t, fake_emg)
plt.xlabel("time (s)")
plt.ylabel("amplitude")
plt.show()