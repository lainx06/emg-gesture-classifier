# AGENTS.md

EMG gesture classification research project. Python 3, Windows, VS Code.
Full design rationale and locked parameters: see `PROJECT_CONTEXT.md`.

## Do not modify without asking first

`src/preprocess.py`, `src/pipeline.py`, `src/build_cache.py`

Constants in these files were derived through analysis and several look wrong
from the code alone: causal `sosfilt` (not `sosfiltfilt`), `int()` floor on the
transient trim, no lowpass filter, time-domain windowing. Read
`PROJECT_CONTEXT.md` before proposing any change here.

## Free to edit

Plotting, SHAP scripts, statistics, Arduino sketches, `notes/`.

## Conventions

- Paths via `os.path.dirname(os.path.abspath(__file__))`, never relative strings
- Never `except Exception: continue` — catch specific exceptions and count them
- Any parameter measured from data must be measured at the exact point in the
  pipeline where it is used
- Do not regenerate `cache/` without asking — a rebuild takes minutes and
  invalidates recorded results