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

Setup and usage instructions are added in Phase 0 of the plan.
