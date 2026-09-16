# Phase 0: Project scaffold and environment

## What was built

- `pyproject.toml`: package `pgvb`, `src` layout, console script `pgvb`, pytest config.
- `LICENSE` (MIT), `README.md` sections for setup, install, model download, and tests.
- `src/pgvb/__init__.py` (version 0.1.0), `src/pgvb/__main__.py` (argparse subcommands
  `version`, `app`, `demo`; only `version` is implemented), `src/pgvb/config.py` (dataclass
  tree, `load(path=None)` defaulting to `configs/default.yaml`).
- `configs/default.yaml` with every tunable named in PLAN.md Section 2.
- `scripts/download_models.py`: checks `models/hand_landmarker.task` for the expected size
  (7,819,105 bytes) before ever touching the network.
- `scripts/check_env.py`: imports numpy, cv2, mediapipe, pyttsx3, tkinter, yaml, prints
  versions, fails on any import error or on numpy >= 2.
- `tests/test_config.py`, `tests/test_smoke.py`.
- `.gitignore` and `requirements.txt` already matched CLAUDE.md; verified, not rewritten.

## Verification

- `pip install --no-build-isolation -e .` succeeded in the `pgvb` conda env; `pgvb version`
  printed `0.1.0`.
- `python scripts/check_env.py` exited 0 and printed: numpy 1.26.4, cv2 4.11.0,
  mediapipe 0.10.14, pyttsx3 (module has no `__version__`, prints "unknown", import
  succeeds), yaml (PyYAML) 6.0.3, tkinter 8.6.
- `pytest -q`: 5 passed (config defaults for every section, version string, `pgvb.__main__`
  version command, console script `python -m pgvb version`).
- `pip check` printed only the known cosmetic line: "mediapipe 0.10.14 is not supported on
  this platform".
- `python scripts/download_models.py` with the file present: reported it present and exited
  0 without downloading.
- `python scripts/download_models.py` with the file moved aside: this development machine
  has live network access, so the script actually re-downloaded the file from the pinned
  URL rather than failing; the downloaded file matched the original by size (7,819,105
  bytes) and sha256. The original file was restored from a backup taken before the test.
  The failure path (network unreachable) was verified by code inspection instead: the
  `urlretrieve` call is wrapped in a `try/except OSError` that prints a message naming
  `MODEL_URL` and returns exit code 1. This is a deviation from the literal DoD wording
  ("network unplugged") because there is no way to disconnect this machine's network from
  inside the session; the code path was reviewed instead of exercised live.

## Deviations

None to the plan itself. The one deviation is the download-failure verification method,
noted above.
