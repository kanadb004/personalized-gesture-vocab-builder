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

## Status

Under construction. The build plan and per-phase definitions of done are in
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
