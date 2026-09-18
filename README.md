# Personalized Gesture Vocabulary Builder

A few-shot, training-free AAC (Augmentative and Alternative Communication) tool. A nonverbal
or minimally verbal user, or a caregiver acting for them, teaches a webcam-based system a
personal hand gesture from 5 to 10 demonstrations. The system then recognizes that gesture
live and responds with a spoken message and an on-screen display.

Course project for VIT BCSE415L Human Computer Interaction.
Team: Aadit Singh (23BCE1413), Kanad Bhattacharya (23BCE1234).

## How it works

MediaPipe extracts 21 hand landmarks per frame. A small embedding network, trained once with a
prototypical-network objective and then frozen, maps the normalized landmarks to a compact
vector. Enrolling a gesture averages a few such vectors into a prototype; no retraining ever
happens, so earlier gestures are never disturbed. Live frames are matched to the nearest
prototype with an open-set threshold (unknown poses are rejected) and majority-vote smoothing
before the message is spoken.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the pipeline diagram and the enrollment
versus recognition data flow. The build plan and per-phase definitions of done are in
[docs/PLAN.md](docs/PLAN.md). The project proposal is [docs/DA-1.pdf](docs/DA-1.pdf).

## Setup

Activate the conda environment (already provisioned, see `CLAUDE.md` for pins):

```
conda activate pgvb
```

After cloning, point git at the repo's hooks once so commit messages are checked:

```
git config core.hooksPath .githooks
```

## Install

```
pip install --no-build-isolation -e .
```

`--no-build-isolation` matters: without it pip creates a temporary build environment and
tries to download `setuptools`, even though it is already installed. See `docs/OFFLINE.md`
for offline install details.

## Download models

```
python scripts/download_models.py
```

Fetches `models/hand_landmarker.task` if it is not already present; on this machine it
already is, so the script exits immediately without touching the network.

## Run tests

```
python scripts/check_env.py
pytest -q
```

`pgvb version` prints the installed version once the package is installed.

## Quick start

```
pgvb app --profile <your name>
```

Opens the desktop app: a live camera view with the 21-point hand overlay, a status strip, a
message board, and the gesture list. A fresh profile has no gestures yet; see below to enroll
one.

## How to enroll a gesture

1. Click **Enroll** in the app.
2. Name the gesture and give it the message it should speak.
3. Hold the pose steady; the wizard auto-captures once it settles, prompting you to vary the
   angle slightly every third sample, until it has enough examples (5 to 10).
4. Review the consistency score and any collision warning against an existing gesture, then
   Save.

The gesture is recognized immediately, no retraining or restart needed. If a pose gets missed
later, use **Flag miss** to add one more example to that gesture only, without touching any
other gesture's prototype.

## How to evaluate

Unit tests cover the pure numpy math (features, prototypes, smoothing, embedding parity,
stability):

```
pytest -q
```

Everything else (accuracy, false triggers, latency, stability across enrollments) is measured
by replaying recorded sessions through the real pipeline. See
[docs/evaluation/protocol.md](docs/evaluation/protocol.md) for how the sessions are recorded,
and run:

```
python scripts/eval_recognition.py --profile profiles/<participant>.json \
    --dir data/sessions/eval/<participant> --nongesture "nongesture*.npz" \
    --out reports/eval_recognition_<participant>
python scripts/eval_stability.py --profile profiles/<participant>.json \
    --dir data/sessions/eval/<participant> --out reports/eval_stability_<participant>
```

## Results

See [reports/results.md](reports/results.md) for the full tables (backbone accuracy, threshold
sweep, runtime, per-participant recognition accuracy and false triggers, stability matrix,
enrollment usability), each produced by a script from a committed file.

## Limitations

- Course prototype, not a clinical or production tool: single user session, laptop webcam,
  static poses only (no dynamic gestures), English text and speech only.
- The embedding backbone is trained on the Kaggle ASL Alphabet dataset (two signers); personal
  gestures are a different pose distribution, so `recognize.max_distance` was calibrated
  directly against real recordings rather than the ASL cross-signer sweep alone (see
  `reports/threshold_sweep.md`).
- One hand at a time (`landmarks.num_hands: 1`); two-handed gestures are out of scope (see the
  Backlog in `docs/PLAN.md`).

## Team

Aadit Singh (23BCE1413), Kanad Bhattacharya (23BCE1234). VIT BCSE415L Human Computer
Interaction course project.
