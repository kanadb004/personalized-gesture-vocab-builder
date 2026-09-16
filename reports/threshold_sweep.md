# Threshold sweep (Phase 3)

Held-out letters: B, C, L, V, Y. 200 trials, each enrolling
3 of the 5 letters from 8 signer A samples, querying with
signer B samples of the enrolled letters (correct-accept target) and of the remaining
2 unenrolled letters (false-accept probes).

Chosen `recognize.max_distance`: **0.39**
(enrolled accuracy 0.983, false accept rate 0.044,
target false accept rate <= 0.05).

Full sweep in `reports/threshold_sweep.csv` and `reports/threshold_sweep.png`.
