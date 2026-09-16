# Build Plan: Personalized Gesture Vocabulary Builder

Source of truth for what gets built, in what order, and when a phase counts as done. Derived
from `docs/DA-1.pdf` (topic statement, objectives, methodology, evaluation plan, milestones).
Read `CLAUDE.md` first for the session protocol and conventions; this file assumes them.

Each phase is one GitHub issue, one branch, one PR, one session. Phases are ordered by
dependency. A phase is `Done` only when every box in its Definition of Done (DoD) has been
verified by running the listed verification, not by inspection.

## 1. Target system in one page

Objectives from the project document, numbered so DoD items can cite them:

- O1: Vision pipeline on a laptop webcam extracting 21 hand landmarks per frame in real time.
- O2: Few-shot enrollment: add a new personal gesture from 5 to 10 examples, no retraining.
- O3: Open-set rejection: inputs that match no enrolled gesture produce "no gesture".
- O4: Stability: enrolling gesture N never changes recognition of gestures 1..N-1.
- O5: Communication output: each gesture pairs with a spoken message and an on-screen display.
- O6: Evaluated for real-time responsiveness and enrollment usability (caregiver-operable).

Method (document section 5): a small embedding network over normalized landmarks is trained
once on many hand poses with a prototypical-network objective, then frozen. Enrollment averages
the embeddings of the demonstrations into one prototype stored in the user profile. Recognition
embeds each frame, finds the nearest prototype, applies a similarity threshold (open set), and
majority-vote smoothing over a sliding window before speaking. An adaptation loop lets the
caregiver flag a miss and add one more example to that gesture's prototype only.

Scope limits (document section 1.5): course prototype, single user session, laptop webcam,
static and simple poses only, English text and speech. Dynamic gestures and the MobileNetV2
image stream are stretch goals (Phases 11 and 12), not part of the core.

## 2. Fixed technical decisions

These are decided. Do not reopen them inside a phase; if one proves wrong, record it in the
phase notes and change this section in the same PR.

| Area | Decision |
|---|---|
| Language, env | Python 3.11, conda env `pgvb` (see CLAUDE.md). |
| Hand tracking | MediaPipe Tasks API `HandLandmarker`, `num_hands=1`, VIDEO running mode with frame timestamps, model `models/hand_landmarker.task`. |
| Landmark features | 21 x (x, y, z) normalized image landmarks. Normalize: translate so wrist (index 0) is the origin, scale so the wrist to middle-MCP (index 9) distance is 1, mirror x for left hands so both hands share one canonical space. Rotation normalization is a config flag, default off. Result: 63-d float32 vector. |
| Backbone | Keras MLP 63 -> 128 -> 128 -> 64 (ReLU, BatchNorm, Dropout 0.2), output L2-normalized. Trained with episodic prototypical loss (N-way K-shot). Frozen after training. |
| Inference | Numpy forward pass over exported weights `models/backbone_v1.npz`. TensorFlow is a training-time dependency only. |
| Distance | Cosine distance `d = 1 - dot(e, p)` on unit vectors. Accept when `d <= max_distance` (default 0.15, calibrated in Phase 5) and, optionally, `d2 - d1 >= min_margin` (default 0.0). |
| Prototype | Unit-normalized mean of the stored example embeddings. Examples are kept in the profile so refinement is an append plus recompute. |
| Smoothing | Sliding window of `window` frames (default 12). A label is confirmed when it has at least `min_votes` (default 8) votes. State machine with rising-edge firing and a refractory period so a held gesture fires once. |
| Storage | One JSON file per user profile in `profiles/<user>.json`, schema version field, backbone id field. |
| Output | pyttsx3 in a worker thread fed by a queue; message board in the GUI showing the last 5 messages. Mute toggle. |
| GUI | Tkinter (ships with the env), camera frame drawn into a Label via Pillow `ImageTk`. Enrollment wizard is a Toplevel. |
| Config | `configs/default.yaml` loaded by `pgvb.config.load()`. Every tunable lives there. |
| Testing | pytest, no hardware in unit tests. Recorded landmark streams (`data/sessions/*.npz`) make every stage after MediaPipe testable offline. |
| Datasets | Two sources. (a) Kaggle ASL Alphabet, already on disk in `data/raw/asl_alphabet/` (87k images, 29 classes, one signer) and `data/raw/asl_alphabet_test/` (1,740 images, a different signer), run through `scripts/extract_landmarks.py`; this is the bulk of the train split. (b) Self-recorded poses via `scripts/collect_poses.py` (the catalogue, 2 or more recorders); this supplies the held-out poses and cross-person training data. Raw images and video are never committed. |

## 3. Target repository layout

```
personalized-gesture-vocab-builder/
  CLAUDE.md, README.md, LICENSE, pyproject.toml, requirements.txt, .gitignore
  .claude/settings.json          attribution off
  .githooks/commit-msg           rejects attribution trailers and dashes
  configs/default.yaml
  docs/PLAN.md, docs/DA-1.pdf, docs/phases/phase-N.md, docs/evaluation/*.md, docs/architecture.md
  models/hand_landmarker.task    (gitignored, downloaded)
  models/backbone_v1.keras       committed, training artifact
  models/backbone_v1.npz         committed, runtime weights
  models/backbone_v1.json        metadata: id, feature config, train date, held-out accuracy
  data/pose_catalogue.yaml       the 30 poses used for backbone training
  data/landmarks/*.npz           committed if under 20 MB
  data/sessions/*.npz            recorded landmark streams for offline eval, committed
  data/raw/                      gitignored
  profiles/                      gitignored except profiles/example.json
  reports/                       calibration and evaluation outputs (csv, png, md)
  scripts/                       download_models.py, collect_poses.py, extract_landmarks.py,
                                 train_backbone.py, export_backbone.py, calibrate_threshold.py,
                                 record_session.py, eval_recognition.py, eval_stability.py,
                                 demo_landmarks.py, demo_recognize.py
  src/pgvb/
    __init__.py, __main__.py, config.py
    camera.py        webcam capture loop, frame timestamps, fps counter
    landmarks.py     HandLandmarker wrapper, returns HandFrame or None
    features.py      normalization and feature vector
    embedding.py     numpy backbone forward pass, load weights
    prototypes.py    prototype math: mean, normalize, distances
    profile.py       Profile, Gesture dataclasses, JSON load/save, validation
    enroll.py        Enroller: collects samples, quality checks, writes gesture
    recognize.py     Recognizer: per-frame decision (label or None, distance, margin)
    smoothing.py     VoteWindow and TriggerStateMachine
    output.py        Speaker (TTS thread), MessageBoard model
    pipeline.py      glue: frame -> landmarks -> features -> embedding -> decision -> trigger
    replay.py        play recorded sessions through the pipeline
    gui/             app.py, enroll_wizard.py, widgets.py
    train/           dataset.py, episodes.py, model.py, losses.py (Keras, training only)
  tests/
```

## 4. Phase status

Update this table in the PR that completes each phase. Status values: `Todo`, `In progress`,
`Done`. The `Notes` cell links to `docs/phases/phase-N.md`.

| Phase | Title | Milestone | Status | Notes |
|---|---|---|---|---|
| 0 | Project scaffold and environment | DA2 | Done | [phase-0](phases/phase-0.md) |
| 1 | Camera and landmark pipeline | DA2 | Todo | |
| 2 | Pose dataset and collection tooling | DA2 | Todo | |
| 3 | Embedding backbone: train, evaluate, freeze, export | DA2 | Todo | |
| 4 | Profiles, enrollment, and recognition core | DA2 | Todo | |
| 5 | Open-set threshold calibration and temporal smoothing | DA3 | Todo | |
| 6 | Spoken output and message board | DA3 | Todo | |
| 7 | Desktop app: live view and enrollment wizard | DA3 | Todo | |
| 8 | Adaptation loop and heuristic usability pass | DA3 | Todo | |
| 9 | Evaluation harness and results | Final | Todo | |
| 10 | Packaging, documentation, demo, final report material | Final | Todo | |
| 11 | Stretch: MobileNetV2 image stream | Optional | Todo | |
| 12 | Stretch: dynamic gestures | Optional | Todo | |

Milestones from the document: DA2 = backbone trained and frozen, enrollment and recognition
end to end (Phases 0 to 4). DA3 = smoothing and rejection tuned, TTS and message board,
internal usability pass (Phases 5 to 8). Final = full demonstration, evaluation results, report
(Phases 9 and 10).

## 5. Phases

Conventions used below. `HUMAN` marks a task that needs a person at the webcam or filling in a
document; Sonnet builds the tooling and templates, then stops and states exactly what the
team must do. A phase with HUMAN tasks has a code DoD (must be met to merge) and a data DoD
(met when the team has done the human step; the phase notes record which are pending).

### Phase 0: Project scaffold and environment

Goal: a runnable, testable, installable skeleton so every later phase only adds modules.

Depends on: nothing.

Deliverables:
- `pyproject.toml` (package `pgvb`, `src` layout, console script `pgvb = pgvb.__main__:main`,
  pytest config), `LICENSE` (MIT), `README.md` with setup and quick start. `requirements.txt`
  and `.gitignore` already exist; verify them against CLAUDE.md rather than rewriting.
- `src/pgvb/__init__.py` (version), `__main__.py` (argparse with subcommands `app`, `demo`,
  `version`; only `version` works in this phase), `config.py` (load YAML into a dataclass
  tree, `load(path=None)` defaults to `configs/default.yaml`).
- `configs/default.yaml` with every tunable named in Section 2, commented.
- `scripts/download_models.py`: downloads `hand_landmarker.task` to `models/`, verifies size,
  idempotent. The file is already present on this machine (see `docs/OFFLINE.md`), so the
  script must detect it and exit 0 without any network access.
- `scripts/check_env.py`: imports numpy, cv2, mediapipe, pyttsx3, tkinter, yaml, prints
  versions, exits non-zero on any failure or on a numpy 2 install.
- `tests/test_config.py`, `tests/test_smoke.py`.
- `docs/phases/phase-0.md`, `docs/DA-1.pdf` copied into the repo.

Definition of Done:
- [ ] `pip install --no-build-isolation -e .` in `pgvb` succeeds offline and `pgvb version`
      prints the version.
- [ ] `python scripts/check_env.py` exits 0 and prints the pinned versions.
- [ ] `python scripts/download_models.py` finds the existing `models/hand_landmarker.task`
      (7,819,105 bytes) and reports it present without downloading; with the file renamed
      away and the network unplugged it fails with a clear message naming the URL.
- [ ] `pytest -q` passes (config loads, every key in Section 2 present with the default value).
- [ ] `pip check` prints nothing except the known mediapipe platform line.
- [ ] `.gitignore` covers `data/raw/`, `profiles/*.json` (except example), `models/*.task`,
      `__pycache__`, `.pytest_cache`, `*.egg-info`, `.DS_Store`.
- [ ] README has: what it is, env activation, install, download models, run tests.
- [ ] `docs/phases/phase-0.md` written; status table updated.

Verification: `pgvb version`, `python scripts/check_env.py`, `pytest -q`, `pip check`.

### Phase 1: Camera and landmark pipeline

Goal: O1. Live webcam frames become normalized landmark feature vectors at interactive frame
rates, and any landmark stream can be recorded and replayed offline.

Depends on: Phase 0.

Deliverables:
- `pgvb/camera.py`: `Camera` context manager over `cv2.VideoCapture`, yields
  `(frame_bgr, timestamp_ms)`, configurable index, width, height, mirror flag; `FpsMeter`.
- `pgvb/landmarks.py`: `HandTracker` wrapping `HandLandmarker` in VIDEO mode. `track(frame,
  ts_ms) -> HandFrame | None` where `HandFrame` has `landmarks` (21x3 float32), `world`
  (21x3), `handedness` ("Left"/"Right"), `score`, `ts_ms`. Returns None when no hand or
  score below `min_hand_score`.
- `pgvb/features.py`: `normalize_landmarks(lm, handedness, rotate=False) -> (63,) float32`
  exactly as in Section 2; `feature_dim()`; pure numpy.
- `pgvb/replay.py`: `SessionRecorder` (append HandFrame or None per frame with timestamps,
  save to `.npz` with a label string and metadata) and `SessionPlayer` (iterate frames back
  at recorded or max speed). Format documented in the module docstring.
- `scripts/demo_landmarks.py`: live overlay of the 21 points and connections, handedness,
  fps, and the normalized vector's first values; key `r` toggles recording to
  `data/sessions/<name>.npz`, `q` quits.
- `scripts/record_session.py --label <text> --seconds N --out <file>`: headless recorder.
- Tests: `tests/test_features.py` (translation invariance, scale invariance, left-right
  mirroring maps to the same vector, output shape and dtype, rotation flag), `tests/test_replay.py`
  (round trip through npz preserves frames, None frames, timestamps, label).
- One committed sample session: `data/sessions/sample_open_palm.npz` (HUMAN, about 5 seconds).

Definition of Done:
- [ ] `pytest -q` passes including the new tests.
- [ ] `python scripts/demo_landmarks.py` runs on the built-in camera at 20 fps or more
      (fps shown on screen, value recorded in phase notes) with a hand tracked.
- [ ] Removing the hand from view yields None frames (overlay shows "no hand").
- [ ] `python scripts/record_session.py --label open_palm --seconds 5` produces a file that
      `SessionPlayer` reads back with the same frame count.
- [ ] `data/sessions/sample_open_palm.npz` committed and under 1 MB.
- [ ] `docs/phases/phase-1.md` records fps, camera resolution, and the normalization choice.

Verification: `pytest -q`; manual camera run recorded in the phase notes.

### Phase 2: Pose dataset and collection tooling

Goal: a landmark dataset of many distinct hand poses from more than one person, split so that
some poses are never seen in training and can measure few-shot generalization.

Depends on: Phase 1.

Data already on disk (see `docs/OFFLINE.md`): `data/raw/asl_alphabet/asl_alphabet_train/asl_alphabet_train/<class>/*.jpg`
with 29 classes (A to Z, del, nothing, space), 3000 images each, 200x200 pixels, one signer;
`data/raw/asl_alphabet/asl_alphabet_test/` (one image per class) and
`data/raw/asl_alphabet_test/<class>/*.jpg` (1,740 images, 60 per class, a different signer).
Measured on 2026-09-13 with `HandLandmarker` at `min_hand_detection_confidence=0.3`: 75
percent of train images and 94 percent of test images yield a hand at native size; upscaling
does not help. `nothing` has no hand (drop it), and `N`, `X`, `del` detect under 60 percent
(keep whatever detects, report the rate).

Deliverables:
- `data/pose_catalogue.yaml`: every pose with id, name, short description, source
  (`asl_alphabet`, `recorded`, or both), and split (`train` or `heldout`). Train: the ASL
  letters and `space` from the Kaggle data, plus recorded versions of at least 12 of them
  (A, B, C, D, F, I, L, O, U, V, W, Y) so the train split has two or more people. Held-out
  (recorded only, never in Kaggle): 8 poses chosen to be visually distinct from each other
  and from any ASL letter, for example thumbs up, thumbs down, OK sign, pinch, call me, horns,
  claw, flat hand sideways. Pointing and fist are excluded from held-out because they
  resemble D and A or S. Extend the recorded set with numbers 1 to 5 and rock if time allows.
- `scripts/collect_poses.py --recorder <name> --poses all|<ids>`: guided webcam session; for
  each pose shows the name and description, 3 second countdown, records `frames_per_pose`
  (default 150) tracked frames while prompting the recorder to slowly vary angle and
  distance; skips frames with no hand; writes `data/raw/<recorder>/<pose>.npz`.
- `scripts/extract_landmarks.py --images <dir> --out <file> [--limit-per-class N]`: runs an
  image folder (class per subfolder) through `HandLandmarker` IMAGE mode, writes raw landmark
  arrays plus handedness per image to `data/raw/landmarks_<name>.npz`, skips undetected images,
  reports the detection rate per class, and marks the source name. Runs on both Kaggle folders;
  processing 87k images takes a while (about 30 ms per image), so support `--limit-per-class`
  (default 1000) and resume from a partial output.
- `pgvb/train/dataset.py`: `build_dataset(raw_dir) -> data/landmarks/poses_v1.npz` with arrays
  `x` (N, 63), `y` (N,), `recorder` (N,) (the Kaggle signers count as recorders `asl_train`
  and `asl_test`), `pose_id` list, `split` per pose. Also
  `augment(x, rng)`: small rotation about the wrist (plus or minus 15 degrees), scale jitter
  (0.9 to 1.1), Gaussian jitter (sigma 0.01), applied on the fly in training only.
- `scripts/dataset_report.py`: prints samples per pose per recorder, and a 2-D PCA scatter of
  raw features saved to `reports/dataset_pca.png`.
- Tests: `tests/test_dataset.py` (build from a tiny synthetic raw dir; split integrity; augment
  keeps shape and stays close to input).

Definition of Done (code):
- [ ] `pytest -q` passes.
- [ ] `collect_poses.py` runs end to end on a 2 pose subset without errors (HUMAN, 1 minute).
- [ ] `build_dataset` refuses to put a held-out pose into the train split (tested).
- [ ] `dataset_report.py` runs on whatever raw data exists.

Definition of Done (data):
- [ ] `extract_landmarks.py` run on both Kaggle folders; detection rate per class in
      `reports/asl_detection_rates.md`; at least 20 classes with 500 or more detected samples.
- [ ] HUMAN: at least 2 recorders, all catalogue poses marked `recorded`, at least 100 tracked
      frames per pose per recorder.
- [ ] `data/landmarks/poses_v1.npz` committed (under 20 MB; subsample the Kaggle classes to
      at most 800 per class if needed) with `reports/dataset_pca.png`.
- [ ] `docs/phases/phase-2.md` lists sources, recorders, counts, and poses that tracked poorly.

Verification: `pytest -q`; `python scripts/dataset_report.py`.

### Phase 3: Embedding backbone: train, evaluate, freeze, export

Goal: the frozen embedding space (document 5.1). A held-out pose must be separable from other
held-out poses using only 5 examples, without any weight update.

Depends on: Phase 2 (data DoD met).

Deliverables:
- `pgvb/train/model.py`: `build_backbone(input_dim, embed_dim)` Keras model per Section 2.
- `pgvb/train/episodes.py`: episodic sampler (N-way, K-shot, Q-query) over train poses with
  augmentation; recorder-aware sampling so support and query can come from different people.
- `pgvb/train/losses.py`: prototypical loss (softmax over negative squared distances to class
  prototypes) and episode accuracy metric.
- `scripts/train_backbone.py --config configs/train_v1.yaml`: seeds, trains for a fixed number
  of episodes with early stopping on held-out episode accuracy, saves
  `models/backbone_v1.keras`, `reports/train_v1_history.csv`, `reports/train_v1_curves.png`.
- `scripts/eval_backbone.py`: 5-way 5-shot and 5-way 1-shot accuracy on held-out poses over
  1000 episodes, cross-recorder; intra-pose vs inter-pose cosine distance histograms saved to
  `reports/backbone_v1_distances.png`; results table in `reports/backbone_v1_eval.md`.
- `scripts/export_backbone.py`: writes `models/backbone_v1.npz` (weights, BatchNorm folded)
  and `models/backbone_v1.json` (id, input dim, embed dim, normalization flags, training date,
  held-out accuracies, git commit).
- `pgvb/embedding.py`: `Backbone.load(path)`, `embed(x: (63,)) -> (64,) unit vector`,
  `embed_batch`. Pure numpy.
- Tests: `tests/test_embedding.py` (numpy forward equals Keras forward within 1e-5 on 100
  random inputs, marked to skip if TensorFlow is absent; output is unit norm; determinism),
  `tests/test_episodes.py`.

Definition of Done:
- [ ] `pytest -q` passes.
- [ ] Held-out 5-way 5-shot accuracy at least 0.90 and 5-way 1-shot at least 0.75, cross
      recorder, 1000 episodes, numbers in `reports/backbone_v1_eval.md`.
- [ ] Mean intra-pose cosine distance is smaller than mean inter-pose distance on held-out
      poses by a clear gap (report the two means and the 95th percentile of intra-pose).
- [ ] `models/backbone_v1.npz` embeds one vector in under 1 ms (mean over 1000 calls, in notes).
- [ ] Training is reproducible: two runs with the same seed give the same held-out accuracy
      to two decimals (state this in the notes).
- [ ] `docs/phases/phase-3.md` records architecture, episodes, time, metrics, and the model id.

Verification: `pytest -q`; `python scripts/eval_backbone.py`.

### Phase 4: Profiles, enrollment, and recognition core

Goal: O2, O3, O4 as a library with no GUI: enroll from a list of frames, recognize a frame,
store and reload the profile, and prove enrollment never touches existing prototypes.

Depends on: Phase 3.

Deliverables:
- `pgvb/profile.py`: `Gesture` (id, name, message, symbol path or None, examples list of
  64-d vectors, prototype, created_at, updated_at, n_refinements) and `Profile` (schema_version
  1, user_id, backbone_id, gestures). `load`, `save` (atomic write via temp file and rename),
  `validate` (rejects mismatched backbone id or dimension), `add_gesture`, `remove_gesture`,
  `rename`, `set_message`. `profiles/example.json` committed.
- `pgvb/prototypes.py`: `make_prototype(examples)`, `cosine_distances(e, protos)`,
  `nearest(e, protos)` returning (index, d1, d2).
- `pgvb/enroll.py`: `Enroller(backbone, profile, cfg)` with `start(name, message)`,
  `add_frame(hand_frame) -> SampleStatus` (rejects no-hand frames and frames whose landmarks
  are still moving: mean displacement over the last `stability_frames` above
  `stability_max_motion`), `accept_sample()` when stable, `progress`, `quality()` returning
  intra-sample mean distance and the closest existing gesture and its distance (collision
  warning if below `collision_distance`), `commit() -> Gesture`, `cancel()`.
- `pgvb/recognize.py`: `Recognizer(backbone, profile, cfg)` with `decide(hand_frame_or_none)
  -> Decision(label or None, distance, margin, embedding)` applying `max_distance` and
  `min_margin`; `reload_profile()`.
- `scripts/demo_recognize.py --profile <file> [--session <npz>]`: prints per-frame decisions
  from the camera or a recorded session; `--enroll <name> --message <text>` enrolls from a
  recorded session file.
- Tests: `tests/test_profile.py` (round trip, atomic save, validation failures),
  `tests/test_prototypes.py`, `tests/test_enroll.py` (needs 5 to 10 samples, stability gate,
  collision warning), `tests/test_recognize.py` (accept inside threshold, reject outside,
  margin rule, None input gives None), `tests/test_stability.py`: enroll A, B, C from synthetic
  clusters, snapshot prototypes, enroll D, assert A, B, C prototypes are bit-identical and their
  decisions on their own held-out samples are unchanged (O4).

Definition of Done:
- [ ] `pytest -q` passes, including the stability test.
- [ ] Enrolling from `data/sessions/*.npz` recordings of 3 poses (HUMAN, reuse Phase 2 raw
      data if suitable) and recognizing a fourth recording of each gives the correct label on at
      least 90 percent of frames with a hand, and a recording of an unrelated pose yields None on
      at least 90 percent of frames, using the default threshold. Numbers in the notes.
- [ ] Profile file is human readable, under 100 KB for 10 gestures, and reloads identically.
- [ ] `docs/phases/phase-4.md` written; this closes milestone DA2 (state it in the notes).

Verification: `pytest -q`; `python scripts/demo_recognize.py --session ...`.

### Phase 5: Open-set threshold calibration and temporal smoothing

Goal: O3 tuned with data, and per-frame flicker removed so a gesture fires once, reliably.

Depends on: Phase 4.

Deliverables:
- `scripts/calibrate_threshold.py`: simulates enrollment and recognition over held-out poses
  and recorders: for many random (enroll 5 gestures, test on their other samples plus samples
  of 3 unenrolled poses) trials, sweeps `max_distance` and reports accuracy on enrolled,
  false accept rate on unenrolled, and the F1 of "correct decision". Writes
  `reports/threshold_sweep.csv`, `reports/threshold_sweep.png`, and a recommended value
  chosen at false accept rate at most 5 percent. The value is written into
  `configs/default.yaml` in the same PR.
- `pgvb/smoothing.py`: `VoteWindow(window, min_votes)` returning the confirmed label or None,
  and `TriggerStateMachine(release_frames, cooldown_ms)` with states IDLE, ARMED, FIRED,
  HOLD; emits `Trigger(label, ts_ms)` on the rising edge only; a new label can fire only after
  the previous label has been absent for `release_frames` frames or `cooldown_ms` has passed.
- `pgvb/pipeline.py`: `Pipeline(cfg, profile)` composing tracker, features, backbone,
  recognizer, smoother, trigger; `process(frame, ts) -> FrameResult` (hand frame, decision,
  confirmed label, trigger or None, timings per stage). Also `process_hand_frame` for replay.
- `scripts/demo_recognize.py` gains `--smooth` to print triggers instead of raw decisions,
  and prints per-stage timing.
- `data/sessions/`: HUMAN recordings for false trigger testing: `idle_hands_60s.npz`
  (hands resting, typing, scratching head, drinking) and `transitions_30s.npz` (moving
  between enrolled gestures).
- Tests: `tests/test_smoothing.py` (a label needs min_votes; brief flicker does not fire;
  held gesture fires exactly once; release then repeat fires again; cooldown), `tests/test_pipeline.py`
  on a synthetic session.

Definition of Done:
- [ ] `pytest -q` passes.
- [ ] `reports/threshold_sweep.md` states the chosen `max_distance`, the enrolled accuracy and
      the false accept rate at that value, and `configs/default.yaml` matches.
- [ ] Replaying `idle_hands_60s.npz` with 5 enrolled gestures yields 0 triggers.
- [ ] Replaying `transitions_30s.npz` yields exactly one trigger per intended gesture.
- [ ] Per-frame pipeline time (excluding capture) under 25 ms mean on the laptop CPU.
- [ ] `docs/phases/phase-5.md` records the sweep, the chosen values, and trigger counts.

Verification: `pytest -q`; `python scripts/calibrate_threshold.py`;
`python scripts/demo_recognize.py --smooth --session data/sessions/idle_hands_60s.npz`.

### Phase 6: Spoken output and message board

Goal: O5 as a library: a trigger becomes speech and a message board entry without blocking the
frame loop.

Depends on: Phase 5.

Deliverables:
- `pgvb/output.py`: `Speaker` (pyttsx3 engine in a daemon thread consuming a queue; `say(text)`
  returns immediately; `mute` flag; `rate` and `voice` from config; `on_started` and
  `on_finished` callbacks with timestamps for latency measurement; graceful shutdown),
  `MessageBoard` (bounded list of `(ts, gesture_name, message)`, `latest(n)`, `clear`),
  `OutputRouter(speaker, board)` subscribed to pipeline triggers.
- `scripts/demo_recognize.py --speak`: speaks on triggers from camera or session.
- `scripts/test_tts.py`: says one sentence and prints available voices.
- Tests: `tests/test_output.py` with a fake engine (queue order, mute, callbacks, board
  bounds). pyttsx3 itself is not exercised in tests.

Definition of Done:
- [ ] `pytest -q` passes.
- [ ] `python scripts/test_tts.py` produces audible speech on the laptop speakers (HUMAN, note
      the voice used).
- [ ] Speaking does not drop the frame rate: fps during speech within 10 percent of fps when
      silent, measured with `demo_recognize.py --speak` and recorded in the notes.
- [ ] Trigger to speech start latency (from `Trigger.ts_ms` to `on_started`) under 300 ms
      median over 10 triggers.
- [ ] Muting stops audio immediately and the board still updates.
- [ ] `docs/phases/phase-6.md` written.

Verification: `pytest -q`; manual audio run.

### Phase 7: Desktop app: live view and enrollment wizard

Goal: O2 and O5 as a product a caregiver can operate without the terminal: enroll a gesture,
see recognition, read the board, hear the message.

Depends on: Phase 6.

Deliverables:
- `pgvb/gui/app.py`: main window. Left: camera view with landmark overlay and a status strip
  (hand found or not, current confirmed label, distance). Right: message board in large text,
  and the gesture list (name, message, example count). Buttons: Enroll new gesture, Flag last
  recognition (wired in Phase 8, disabled here), Mute, Settings (user profile picker, camera
  index). Refresh via `after()` at the camera rate; pipeline runs in the main loop with
  timing kept under one frame period, or in a worker thread if not.
- `pgvb/gui/enroll_wizard.py`: three steps. (1) Name, message, optional symbol image. (2)
  Capture: camera view, 3-2-1 countdown, automatic capture when the hand is stable, progress
  "n of 8", a prompt every third sample to change the angle slightly, retake last sample, or
  press space to capture manually. (3) Review: consistency score, collision warning naming the
  closest existing gesture, Save or Redo. Cancel at any step leaves the profile untouched.
- `pgvb/gui/widgets.py`: reusable pieces (camera label, big message label, countdown).
- `pgvb/__main__.py`: `pgvb app [--profile name] [--camera idx]` launches the GUI.
- Tests: `tests/test_gui_logic.py` for the wizard's state machine and view models, with Tk
  not instantiated (separate the state from the widgets so this is possible).

Definition of Done:
- [ ] `pytest -q` passes.
- [ ] `pgvb app` starts in under 10 seconds from a warm cache and shows live video.
- [ ] A person who has not read the code can enroll a gesture end to end in under 2 minutes
      using only the mouse (HUMAN, one team member times the other).
- [ ] Enrolled gesture is recognized live, spoken, and shown on the board; a non-gesture pose
      shows "no gesture".
- [ ] Closing the window releases the camera and stops the TTS thread (no zombie process).
- [ ] Window is usable at 1280x720 and does not break at 1024x600.
- [ ] `docs/phases/phase-7.md` includes two screenshots (main window, wizard step 2) in
      `docs/img/`.

Verification: `pytest -q`; manual walkthrough recorded in notes.

### Phase 8: Adaptation loop and heuristic usability pass

Goal: document 5.5 (flag a miss, add one example to that prototype only) and the internal
Nielsen heuristic evaluation of the enrollment flow (document 5.6 item 2, first half).

Depends on: Phase 7.

Deliverables:
- `pgvb/enroll.py`: `Refiner(backbone, profile)`: `refine(gesture_id, hand_frame)` appends one
  example, recomputes only that prototype, bumps `n_refinements`; `undo_last(gesture_id)`.
- GUI: Flag button opens a small dialog: pick which gesture was intended (default: the last
  confirmed or the closest), capture one stable sample with countdown, confirm. Also a
  "Manage gestures" dialog: rename, edit message, delete, view example count, undo last
  refinement.
- `docs/evaluation/heuristic_evaluation.md`: Nielsen's 10 heuristics, one row each, severity
  scale 0 to 4, columns for two evaluators, findings list with location and proposed fix.
  Filled in by the team (HUMAN). Findings of severity 3 or 4 fixed in this phase or filed as
  `fix-` issues.
- Tests: `tests/test_refine.py` (only the target prototype changes, bit-identical others; undo
  restores; refinement moves the prototype toward the new example).

Definition of Done:
- [ ] `pytest -q` passes.
- [ ] Flag flow works live: a gesture that was missed is recognized after one refinement in a
      demonstration recorded in the notes (HUMAN).
- [ ] `heuristic_evaluation.md` completed by two evaluators with severity ratings (HUMAN).
- [ ] All severity 3 and 4 findings fixed and listed in the notes, or filed as issues.
- [ ] `docs/phases/phase-8.md` written; this closes milestone DA3 (state it).

Verification: `pytest -q`; manual.

### Phase 9: Evaluation harness and results

Goal: document 5.6, all three lines, with reproducible scripts and a results document that
can be pasted into the final report.

Depends on: Phase 8.

Deliverables:
- `docs/evaluation/protocol.md`: the exact recording protocol. Per participant (proxy users:
  team members and 3 to 5 volunteers, HUMAN): invent 3 to 5 personal gestures, enroll each
  with 8 samples through the app, then record for each gesture 3 test clips of 5 seconds
  (`data/sessions/eval/<participant>/<gesture>_<k>.npz`), plus one 60 second non-gesture clip.
  `scripts/record_session.py` gains `--participant` and `--gesture` naming.
- `scripts/eval_recognition.py`: replays every test clip through the full pipeline with the
  participant's profile; outputs per-gesture accuracy (fraction of clips whose first trigger is
  the correct label and no wrong trigger occurs), confusion matrix, false trigger rate per
  minute on non-gesture clips, latency from first frame with the correct raw decision to
  trigger (algorithmic) and trigger to speech start (from Phase 6 callbacks, live run),
  written to `reports/eval_recognition.md` and `.csv`.
- `scripts/eval_stability.py`: for each participant, enroll gestures one at a time in order;
  after each addition, re-evaluate all previously enrolled gestures on their clips; report the
  accuracy matrix (rows: number enrolled, columns: gesture) and assert prototypes unchanged.
  Output `reports/eval_stability.md`.
- `docs/evaluation/user_test.md`: task list (enroll 3 gestures, use them, fix one), measures
  (time per enrollment, errors, SUS questionnaire, free comments), and a results table filled
  by the team (HUMAN).
- `scripts/make_figures.py`: confusion matrix, latency histogram, stability heatmap to
  `reports/img/`.

Definition of Done (code):
- [ ] `pytest -q` passes; evaluation scripts run on the Phase 5 sample sessions.
- [ ] Every number in the reports is produced by a script from committed session files.

Definition of Done (data, HUMAN):
- [ ] At least 4 participants, at least 3 gestures each, all clips committed.
- [ ] `reports/eval_recognition.md`: per-gesture accuracy, overall accuracy, false triggers per
      minute, latency medians. Targets: overall clip accuracy at least 0.85, false triggers
      under 1 per minute, algorithmic latency median under 600 ms.
- [ ] `reports/eval_stability.md` shows no accuracy decrease for earlier gestures as later
      ones are added (any decrease explained).
- [ ] `docs/evaluation/user_test.md` results filled for the same participants.
- [ ] `docs/phases/phase-9.md` summarizes results against O1 to O6.

Verification: `python scripts/eval_recognition.py`, `python scripts/eval_stability.py`,
`python scripts/make_figures.py`.

### Phase 10: Packaging, documentation, demo, final report material

Goal: the final submission state: anyone can clone, install, run, and reproduce the numbers;
the report has everything it needs.

Depends on: Phase 9.

Deliverables:
- `README.md` complete: overview, architecture figure, install, quick start, how to enroll,
  how to evaluate, results summary table, limitations, team, references.
- `docs/architecture.md` with a Mermaid diagram of the pipeline and the data flow of
  enrollment vs recognition; `docs/final_report_notes.md` collecting every metric, figure
  path, and design decision with a pointer to the phase notes, organized by the document's
  sections (methodology, evaluation, results, limitations, future work).
- `docs/demo_script.md`: a 3 minute demonstration script (enroll two gestures, recognize,
  reject a random pose, flag and refine, show stability).
- Version bump to 1.0.0, git tag `v1.0.0` on main after merge.
- Repo hygiene: no dead scripts, every script has `--help`, `requirements.txt` matches the env.

Definition of Done:
- [ ] Fresh clone into a temp dir, `pip install -e .` in `pgvb`, `download_models.py`,
      `pytest -q`, `pgvb app` all succeed following only the README.
- [ ] `docs/final_report_notes.md` cites a number or figure for every item in document 5.6.
- [ ] Demo script rehearsed once with timing noted (HUMAN).
- [ ] Tag `v1.0.0` exists on the merge commit.
- [ ] `docs/phases/phase-10.md` written; status table shows Phases 0 to 10 Done.

Verification: fresh clone procedure above.

### Phase 11 (stretch): MobileNetV2 image stream

Only after Phase 10. Goal: document 5.1 "cropped image as a second stream". Add a hand crop
from the landmark bounding box, a frozen MobileNetV2 (ImageNet weights, Keras Applications)
global-average feature reduced to 64-d by a trained projection, concatenated with the landmark
embedding and re-normalized. Must show a measurable gain on the Phase 9 evaluation to be kept;
otherwise document the result and leave it disabled by config. DoD: config flag
`image_stream.enabled`, same tests pass with it on and off, `reports/eval_image_stream.md`.

### Phase 12 (stretch): dynamic gestures

Only after Phase 10. Goal: document 1.5 future extension. Represent a gesture as a fixed-length
sequence of landmark features (resampled to 16 steps), embed with a small temporal model
trained the same episodic way, and enroll by prototype exactly as static gestures. DoD:
`gesture.kind` in the profile, a wave and a circle enrolled and recognized live, static
behavior unchanged (all earlier tests pass).

## 6. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Backbone does not reach 0.90 held-out 5-shot | Add pairwise fingertip distances and joint angles to the feature vector (`features.py` gains `extended=True`), widen the MLP, collect a third recorder. Record what was tried. |
| Threshold that rejects unknowns also rejects real gestures | Enable `min_margin`; raise `window` and `min_votes`; ask for 10 samples with more angle variety at enrollment. |
| Frame rate under 20 fps | Lower capture resolution to 640x480; run the tracker in a worker thread; skip every second frame for embedding while still drawing every frame. |
| pyttsx3 blocks or fails on macOS | Fallback `Speaker` backend using the `say` command via subprocess; select by config `tts.backend`. |
| Tkinter video is choppy | Downscale the preview to 640 wide; keep processing at capture resolution. |
| Not enough proxy participants | Minimum 4; the two team members count, plus 2 volunteers. |

## 7. Backlog

Ideas that came up but are out of scope for the current phase. Add a line, do not build.

- Per-gesture custom threshold.
- Two-handed gestures (`num_hands=2`, concatenated features).
- Symbol images shown on the board next to the message.
- Export profile as a printable gesture card for caregivers.
