# Threshold sweep (Phase 3)

Held-out letters: B, C, L, V, Y. 200 trials, each enrolling
3 of the 5 letters from 8 signer A samples, querying with
signer B samples of the enrolled letters (correct-accept target) and of the remaining
2 unenrolled letters (false-accept probes).

ASL-letter sweep result: **0.39**
(enrolled accuracy 0.983, false accept rate 0.044,
target false accept rate <= 0.05).

Full sweep in `reports/threshold_sweep.csv` and `reports/threshold_sweep.png`.

## Override after live verification

0.39 does not transfer to real personal gestures: replaying the team's own idle-hands
recording (`data/sessions/phase3/idle.npz`) at 0.39 produced 22 false triggers, because an
open, flat hand briefly held during ordinary movement sits almost as close to the enrolled
`open_palm` prototype as `open_palm` itself (min cosine distance 0.017 on that recording). The
ASL-letter sweep only measures cross-signer separation between fingerspelling poses; it never
saw an "idle hand" class, so it cannot calibrate open-set rejection against everyday movement.

Re-swept `max_distance` directly against the team's own enrolled-gesture and idle recordings
(`data/sessions/phase3/*.npz`, three personal gestures each with 3 enrollment + 1 held-out
recording, one unrelated-pose recording, one 60 second idle recording): at **0.10**, held-out
per-frame accuracy is 98.2 percent (thumbs_up), 100 percent (open_palm, peace), the unrelated
pose is rejected 100 percent of the time, and the idle recording produces 0 confirmed triggers
after smoothing (97.8 percent of individual idle frames still correctly decide "no gesture").
`configs/default.yaml` uses this value, not the ASL-letter one.
