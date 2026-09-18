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

Source: `scripts/eval_recognition.py` -> `reports/eval_recognition_kanadb.md/.csv`. One
participant (kanadb), scope reduced from the plan's original 2 (see `docs/phases/phase-5.md`).
Four gestures: `open_palm` = "Hello", `peace_sign` = "Thank you", `fist` = "I need help",
`thumbs_up` = "Okay". 3 five-second clips per gesture, one 60 second non-gesture clip.

| Participant | Gestures | Clip accuracy | False triggers/min | Algorithmic latency (median) |
|---|---|---|---|---|
| kanadb | 4 | 0.750 (9/12) | 0.000 | 729.6 ms |

Per-gesture: `open_palm` 1.000 (3/3), `peace_sign` 1.000 (3/3), `fist` 1.000 (3/3), `thumbs_up`
**0.000 (0/3)**. Confusion matrix and per-clip numbers in `reports/eval_recognition_kanadb.csv`;
figures in `reports/img/kanadb_confusion.png` and `reports/img/kanadb_latency.png`.

Targets: clip accuracy >= 0.85 (**not met**, overall 0.750, entirely due to `thumbs_up`),
false triggers/min < 1 (met, 0.000), algorithmic latency median < 600 ms (**not met**, 729.6 ms).

- **`thumbs_up` false-reject, not misrecognition.** Every `thumbs_up` frame's nearest prototype
  was correctly `thumbs_up` (checked directly against the profile's prototypes; never confused
  with `fist` or another gesture), but the mean cosine distance to its own prototype on the eval
  clips was 0.134, above `recognize.max_distance = 0.10`. This is a threshold, not a
  misclassification, problem: `thumbs_up` and `fist` were both noticeably slower and less stable
  to enroll (see table 6) than the two open-hand gestures, so the enrolled prototype is a less
  tight fit to later demonstrations of the same pose. The Section 6 fallback ("Threshold that
  rejects unknowns also rejects real gestures": more samples with more angle variety, or a
  per-gesture threshold, already in the Backlog) applies here; not implemented in this session.
- **Algorithmic latency above target.** These eval clips ran at roughly 15 to 16 fps (`fist_2`:
  82 frames in 5 s), consistent with the tracker-bound fps Phase 1 and Phase 3 measured live.
  `smoothing.min_votes = 8` alone needs about 8 frames of continuously correct decisions to
  confirm, which is already 8/15.5 ~= 515 ms at this frame rate, before any ramp-up frames at the
  start of a hold are counted; the measured 729.6 ms median is consistent with that plus a few
  additional frames while the pose settles. The 600 ms target assumed a higher frame rate than
  this webcam delivers with the tracker on the main thread; lowering `smoothing.window`/
  `min_votes` or decoupling tracking into a worker thread (both already noted as fallbacks
  elsewhere in the plan) would improve this, not attempted here to stay in budget.

Trigger-to-speech latency (live, `demo_recognize.py --speak`, `docs/evaluation/protocol.md`):
median **0.1 ms over 50 triggers**. This measures the `Speaker` worker thread's queue-to-dequeue
latency (the `on_started` callback fires as the thread picks the message off its queue, right
before it calls into the TTS backend), not the audible onset of speech, since the TTS engine's
own startup latency happens after that callback. Given the Speaker thread was idle and waiting
on the queue the whole time, sub-millisecond dequeue latency is expected and not itself evidence
that speech is heard instantly; Phase 3's live listening test is still the source for perceived
responsiveness ("`--speak` on the live camera produces audible speech ... fps ... comparable").

## 5. Stability matrix

Source: `scripts/eval_stability.py` -> `reports/eval_stability_kanadb.md`. Figure:
`reports/img/kanadb_stability.png`.

Enrollment order: `open_palm`, `peace_sign`, `fist`, `thumbs_up`.

| Gestures enrolled | open_palm | peace_sign | fist | thumbs_up |
|---|---|---|---|---|
| 1 | 1.000 | | | |
| 2 | 1.000 | 1.000 | | |
| 3 | 1.000 | 1.000 | 1.000 | |
| 4 | 1.000 | 1.000 | 1.000 | 0.000 |

O4 assertion: **passed**. Every gesture's prototype was bit-identical to the fully enrolled
profile's at every earlier enrollment step; `thumbs_up`'s own 0.000 accuracy is the same
threshold issue as table 4, not a stability failure, and it does not change while later
gestures are unaffected by it either.

## 6. Usability: enrollment time and observations

Source: `data/sessions/eval/kanadb/enrollment_times.csv`.

| Participant | Gesture | Enrollment time (s) |
|---|---|---|
| kanadb | open_palm | 5 |
| kanadb | peace_sign | 10 |
| kanadb | fist | ~105 (1.5 to 2 min, timed as a range) |
| kanadb | thumbs_up | ~105 (1.5 to 2 min, timed as a range) |

Observations:

- The two open, flat-hand gestures (`open_palm`, `peace_sign`) enrolled in 5 to 10 seconds:
  the pose settles (passes the stability-gate motion check) almost immediately once held.
- The two closed-hand gestures (`fist`, `thumbs_up`) took 10 to 20 times longer to enroll: the
  wizard's auto-capture kept resetting because small, involuntary finger movement in a closed
  fist or a thumb held against a fist keeps the landmark motion above
  `enroll.stability_max_motion` longer than an open palm does. This is a caregiver-usability
  issue (a closed-hand gesture is a natural choice for an AAC vocabulary and should not be this
  much slower to teach), not a training-free-recognition issue.
- The same `thumbs_up` instability at enrollment time is very likely why its later eval clips sit
  further from its own prototype than the target threshold tolerates (see table 4): a slower,
  less consistent enrollment produced a less representative prototype. Fixing enrollment
  stability detection for closed-hand poses (or simply requiring more samples for a gesture whose
  intra-sample distance during enrollment was high) would likely fix both findings together;
  noted for the Backlog rather than fixed in this session.
