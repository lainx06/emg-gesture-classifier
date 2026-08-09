# Analysis Environment Provenance

Environment provenance recovered on 2026-08-08.

## Virtual environment

Path at time of recovery:

C:\Users\adnan\AFRL_EMG\.venv

Created:
2026-07-22 13:24:14 local time

Python:
3.14.2

## Scientific software

- NumPy 2.4.6
- SciPy 1.18.0
- scikit-learn 1.9.0
- XGBoost 3.3.0
- SHAP 0.52.0
- WFDB 4.3.1
- Pandas 3.0.3
- Matplotlib 3.11.1

## Provenance evidence

The core scientific packages show installation timestamps of approximately
2026-07-22 13:25–13:27.

Repository chronology:

- 2026-07-22 13:59 — initial commit
- 2026-07-24 — initial analysis pipeline/evaluation
- 2026-08-04 — six-condition ablation, SHAP analysis, embedded profiling
- 2026-08-07 — physical embedded implementation verification
- 2026-08-08 — exploratory spatial sweep and audit/freeze

The package installation timestamps therefore predate the committed analysis
work. Software versions were not prospectively pinned and were recovered
retrospectively from the original project virtual environment.

A pip freeze snapshot is stored in requirements_analysis_frozen.txt.