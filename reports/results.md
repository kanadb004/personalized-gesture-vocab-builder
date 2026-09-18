# Results

Every number below is produced by a script from a committed file, cited under each table.
Targets from `docs/PLAN.md` are marked met or explained.

## 1. Backbone few-shot accuracy and distance gap (Phase 2)

Source: `scripts/eval_backbone.py` -> `reports/backbone_v1_eval.md`, `reports/backbone_v1_distances.png`.
Held-out letters B, C, L, V, Y, cross-signer (support signer A, query signer B), 500 episodes.

| Metric | Value | Target | Met |
|---|---|---|---|
| 5-way 5-shot accuracy | 0.988 | >= 0.85 (target 0.90) | yes |
| 5-way 1-shot accuracy | 0.957 | >= 0.70 | yes |
| Mean intra-pose cosine distance | 0.0960 | lower than inter-pose | yes |
| Mean inter-pose cosine distance | 1.0405 | higher than intra-pose | yes |
| 95th percentile intra-pose distance | 0.9570 | (informational) | |

## 2. Threshold sweep summary (Phase 3)

Source: `scripts/calibrate_threshold.py` -> `reports/threshold_sweep.md/.csv/.png`.

The ASL-letter sweep alone picked `max_distance = 0.39`, but that did not transfer to real
personal gestures (22 false triggers replaying an idle-hands recording). `max_distance` was
re-swept directly against the team's own recordings and set to **0.10** in
`configs/default.yaml`; full reasoning in `reports/threshold_sweep.md`.

| Sweep | max_distance | Enrolled accuracy | False accept rate |
|---|---|---|---|
| ASL held-out letters (cross-signer) | 0.39 | 0.983 | 0.044 |
| Team recordings (used in `configs/default.yaml`) | 0.10 | see table 4 (clip accuracy at this threshold) | see table 4 (false triggers/min) |

## 3. Runtime (Phases 1 to 3)

Sources: `docs/phases/phase-1.md`, `docs/phases/phase-2.md`, `docs/phases/phase-3.md`
(per-stage timing printed by `scripts/demo_recognize.py` at exit, embed time from
`models/backbone_v1.json`).

| Stage | Value | Target | Met |
|---|---|---|---|
| Camera fps, 640x480, hand in frame | 14.6 to 30.0 (tracker-bound; see Phase 1 notes) | >= 20 fps | partial, see phase-1 notes |
| Landmark tracking (`track_ms`) | 27 to 31 ms live | (dominates per-frame time) | |
| Embed time (`models/backbone_v1.npz`) | 0.033 ms mean, 1000 calls | < 1 ms | yes |
| Recognize + smooth + trigger, excluding tracking | well under 1 ms mean | < 25 ms | yes |

## 4. Recognition accuracy, false triggers, and latency

Source: `scripts/eval_recognition.py` per participant -> `reports/eval_recognition_<participant>.md/.csv`.

<!-- PENDING: fill after Phase 5 data collection (docs/evaluation/protocol.md). One row per
     participant, plus a combined row, from eval_recognition_<participant>.md. -->

| Participant | Gestures | Clip accuracy | False triggers/min | Algorithmic latency (median) |
|---|---|---|---|---|
| _pending_ | | | | |

Targets: clip accuracy >= 0.85, false triggers/min < 1, algorithmic latency median < 600 ms.

Trigger-to-speech latency (live, `demo_recognize.py --speak`, `docs/evaluation/protocol.md`):

<!-- PENDING: one line per participant, from the "trigger to speech start: median ... ms over N
     triggers" line demo_recognize.py prints at exit. -->

## 5. Stability matrix

Source: `scripts/eval_stability.py` per participant -> `reports/eval_stability_<participant>.md`.

<!-- PENDING: fill after Phase 5 data collection. -->

## 6. Usability: enrollment time and observations

Source: `data/sessions/eval/<participant>/enrollment_times.csv`.

<!-- PENDING: table of enrollment seconds per gesture per participant, plus a short
     observations list (what confused the caregiver, what was fixed), from
     docs/evaluation/protocol.md's HUMAN steps. -->

| Participant | Gesture | Enrollment time (s) |
|---|---|---|
| _pending_ | | |

Observations: _pending_.
