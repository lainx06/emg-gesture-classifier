# Channel Selection and Derivation

## Anatomical Targets
* **Flexor Group:** Channels F1 and F9 (Array indices 0 and 8)
* **Extensor Group:** Channels F5 and F13 (Array indices 4 and 12)

## Monopolar vs. Bipolar Evidence
* An initial test on `session1_participant1_gesture16_trial1` yielded a Pearson correlation of **0.987** between F1 and F9.
* These electrodes are located 2 cm apart along the same muscle axis. A near-perfect correlation proves the raw GRABMyo dataset channels are **monopolar** (recorded against a shared, distant reference).

## Hardware Simulation (MyoWare 2.0)
* The MyoWare 2.0 sensor is a differential amplifier that outputs the voltage difference between two active contacts.
* To accurately simulate this commodity hardware, we must subtract the paired monopolar channels to create **bipolar derivations**:
  * **Simulated Channel 1 (Flexor):** `F1 - F9` 
  * **Simulated Channel 2 (Extensor):** `F5 - F13` 

## Defense Note: Spatial Filtering
* Differencing two nearby electrodes acts as a spatial high-pass filter.
* Far-field sources (the heart, distant muscles) reach both contacts equally and cancel out (common-mode rejection).
* Near-field sources (local motor unit action potentials) differ between the 2 cm spacing and survive the subtraction.

## Inter-Subject Variability & Methodological Defense
While channels target the flexor and extensor groups, empirical testing across 5 subjects reveals that the direction of class separation is not uniform. 

| Subject | Wrist Ext. Ratio (Ext/Flex) | Hand Close Ratio (Ext/Flex) | Dominant Gesture |
| :--- | :--- | :--- | :--- |
| P1 | 3.33 | 4.69 | Inverted (Hand Close more extensor-dominant) |
| P2 | 9.89 | 3.35 | Textbook (Wrist Ext. more extensor-dominant) |
| P3 | 4.04 | 1.64 | Textbook |
| P4 | 1.37 | 0.59 | Textbook (Hand close is flexor-dominant) |
| P5 | 0.73 | 2.47 | Inverted (Wrist Ext. is flexor-dominant) |

**Methodological Defense:** 
Channels are selected by fixed anatomical landmark (elbow crease) rather than per-subject optimization. This perfectly mirrors the practical use-case of a commodity wearable (e.g., MyoWare/Myo Armband), which does not map internal muscle bellies. 

Because the architecture utilizes subject-dependent training (train and test within the same subject), this inconsistency does not break the classifier. Furthermore, this high inter-subject spatial variance strongly supports the study's core hypothesis: if the spatial mapping varies this heavily between anatomies, it will likely vary between sessions on the same subject due to slight shifts in electrode placement, causing model degradation.