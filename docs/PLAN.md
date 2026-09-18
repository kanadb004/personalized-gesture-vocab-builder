# Build Plan: Personalized Gesture Vocabulary Builder

Source of truth for what gets built, in what order, and when a phase counts as done. Derived
from `docs/DA-1.pdf` (topic statement, objectives, methodology, evaluation plan, milestones).
Read `CLAUDE.md` first for the session protocol and conventions; this file assumes them.

Revised 2026-09-16 for a 1.5 to 2 day build. The target is a working prototype plus two
deliverables: a results document with tables (`reports/results.md`) and a recorded video
demonstration (made by the team by hand, following `docs/demo_script.md`). Nothing here needs
to be industry grade. Unit tests exist only for the pure math that is cheap to test and easy to
get subtly wrong; everything else is verified by running the scripts and the app.

Each phase is one GitHub issue, one branch, one PR. Phases are ordered by dependency and sized
to a time budget (Section 5). A phase is `Done` when its Definition of Done (DoD) has been
verified by running the listed checks, not by inspection.

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
image stream are out of scope (Backlog).

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
| Distance | Cosine distance `d = 1 - dot(e, p)` on unit vectors. Accept when `d <= max_distance` (default 0.15, calibrated in Phase 3) and, optionally, `d2 - d1 >= min_margin` (default 0.0). |
| Prototype | Unit-normalized mean of the stored example embeddings. Examples are kept in the profile so refinement is an append plus recompute. |
| Smoothing | Sliding window of `window` frames (default 12). A label is confirmed when it has at least `min_votes` (default 8) votes. State machine with rising-edge firing and a refractory period so a held gesture fires once. |
| Storage | One JSON file per user profile in `profiles/<user>.json`, schema version field, backbone id field. |
| Output | pyttsx3 in a worker thread fed by a queue, with the macOS `say` command as the fallback backend (`output.tts_backend`); message board in the GUI showing the last 5 messages. Mute toggle. |
| GUI | Tkinter (ships with the env), camera frame drawn into a Label via Pillow `ImageTk`. Enrollment wizard is a Toplevel. |
| Config | `configs/default.yaml` loaded by `pgvb.config.load()`. Every tunable lives there. |
| Testing | pytest for the pure numpy math only: features, prototypes, smoothing, embedding parity, stability (O4). No camera, display, speaker, or TensorFlow in tests (the embedding parity test skips if TensorFlow is absent). Everything else is checked by running scripts and the app, with numbers written into the phase notes. |
| Datasets | Kaggle ASL Alphabet only, already on disk: `data/raw/asl_alphabet/` (87k images, 29 classes, signer A) and `data/raw/asl_alphabet_test/` (1,740 images, signer B), run through `scripts/extract_landmarks.py` with a per-class cap. `nothing` is dropped (no hand). Five letters are held out of training entirely (fixed in `configs/train_v1.yaml`) and serve as the never-seen poses for the few-shot test, cross-signer: support from signer A, query from signer B. No pose catalogue and no multi-recorder collection session. Personal gestures for the live evaluation are recorded by the team in Phase 5. Raw images are never committed. |

## 3. Target repository layout

```
personalized-gesture-vocab-builder/
  CLAUDE.md, README.md, LICENSE, pyproject.toml, requirements.txt, .gitignore
  .claude/settings.json          attribution off
  .githooks/commit-msg           rejects attribution trailers and dashes
  configs/default.yaml, configs/train_v1.yaml
  docs/PLAN.md, docs/DA-1.pdf, docs/phases/phase-N.md, docs/architecture.md,
  docs/demo_script.md, docs/evaluation/protocol.md, docs/img/
  models/hand_landmarker.task    (gitignored, already present)
  models/backbone_v1.keras       committed, training artifact
  models/backbone_v1.npz         committed, runtime weights
  models/backbone_v1.json        metadata: id, feature config, train date, held-out accuracy
  data/landmarks/asl_v1.npz      extracted and normalized ASL landmarks, committed (under 20 MB)
  data/sessions/*.npz            recorded landmark streams for offline eval, committed
  data/raw/                      gitignored
  profiles/                      gitignored except profiles/example.json
  reports/                       results.md plus the csv and png it cites
  scripts/                       download_models.py, check_env.py, demo_landmarks.py,
                                 record_session.py, extract_landmarks.py, train_backbone.py,
                                 eval_backbone.py, export_backbone.py, calibrate_threshold.py,
                                 demo_recognize.py, eval_recognition.py, eval_stability.py,
                                 make_figures.py
  src/pgvb/
    __init__.py, __main__.py, config.py
    camera.py        webcam capture loop, frame timestamps, fps counter
    landmarks.py     HandLandmarker wrapper, returns HandFrame or None
    features.py      normalization and feature vector
    replay.py        record and play back landmark streams
    embedding.py     numpy backbone forward pass, load weights
    prototypes.py    prototype math: mean, normalize, distances
    profile.py       Profile, Gesture dataclasses, JSON load/save, validation
    enroll.py        Enroller (collect samples, quality checks) and Refiner (flag a miss)
    recognize.py     Recognizer: per-frame decision (label or None, distance, margin)
    smoothing.py     VoteWindow and TriggerStateMachine
    output.py        Speaker (TTS thread), MessageBoard
    pipeline.py      glue: frame -> landmarks -> features -> embedding -> decision -> trigger
    gui/             app.py, enroll_wizard.py, dialogs.py
    train/           dataset.py, episodes.py, model.py, losses.py (Keras, training only)
  tests/             test_config.py, test_smoke.py, test_features.py, test_embedding.py,
                     test_prototypes.py, test_smoothing.py, test_stability.py
```

## 4. Phase status

Update this table in the PR that completes each phase. Status values: `Todo`, `In progress`,
`Done`. The `Notes` cell links to `docs/phases/phase-N.md`. Issue number = phase number + 1.

| Phase | Title | Budget | Status | Notes |
|---|---|---|---|---|
| 0 | Project scaffold and environment | done | Done | [phase-0](phases/phase-0.md) |
| 1 | Camera and landmark pipeline | 1.5 h | Done | [phase-1](phases/phase-1.md) |
| 2 | Backbone from ASL landmarks | 2.5 h | Done | [phase-2](phases/phase-2.md) |
| 3 | Recognition core, smoothing, and speech | 2.5 h | Done | [phase-3](phases/phase-3.md) |
| 4 | Desktop app: live view, enrollment, refinement | 3.5 h | Done | [phase-4](phases/phase-4.md) |
| 5 | Evaluation, results, and demo material | 3 h + 1 h recording | In progress | [phase-5](phases/phase-5.md) |

Suggested schedule: Day 1 is Phases 1 to 3 (about 6.5 h of building, the training run in Phase 2
can overlap with starting Phase 3). Day 2 is Phase 4 in the morning, Phase 5 in the afternoon
(recording clips takes about an hour of the team's time), then the video (1 to 2 h, by hand).

## 5. Phases

`HUMAN` marks a task that needs a person at the webcam or filling in a document; the tooling
is built first, then the session stops and states exactly what the team must do. Each phase
lists what to skip if it runs over budget.

### Phase 1: Camera and landmark pipeline (1.5 h)

Goal: O1. Live webcam frames become normalized landmark feature vectors at interactive frame
rates, and any landmark stream can be recorded and replayed offline.

Depends on: Phase 0.

Deliverables:
- `pgvb/camera.py`: `Camera` context manager over `cv2.VideoCapture`, yields
  `(frame_bgr, timestamp_ms)`, configurable index, width, height, mirror flag; `FpsMeter`.
- `pgvb/landmarks.py`: `HandTracker` wrapping `HandLandmarker` in VIDEO mode. `track(frame,
  ts_ms) -> HandFrame | None` where `HandFrame` has `landmarks` (21x3 float32), `handedness`
  ("Left"/"Right"), `score`, `ts_ms`. Returns None when no hand or score below
  `min_hand_score`. Also `track_image(image_rgb)` for IMAGE mode, used by Phase 2.
- `pgvb/features.py`: `normalize_landmarks(lm, handedness, rotate=False) -> (63,) float32`
  exactly as in Section 2; `feature_dim()`; pure numpy.
- `pgvb/replay.py`: `SessionRecorder` (append HandFrame or None per frame with timestamps,
  save to `.npz` with a label string) and `SessionPlayer` (iterate frames back). Format in
  the module docstring.
- `scripts/demo_landmarks.py`: live overlay of the 21 points and connections, handedness,
  fps; key `r` toggles recording to `data/sessions/<name>.npz`, `q` quits.
- `scripts/record_session.py --label <text> --seconds N [--out <file>]`: headless recorder
  with a 3 second countdown printed to the terminal.
- `tests/test_features.py`: output shape and dtype, translation invariance, scale invariance,
  left and right mirror to the same vector.

Definition of Done:
- [ ] `pytest -q` passes.
- [ ] `python scripts/demo_landmarks.py` runs on the built-in camera at 20 fps or more with a
      hand tracked, and shows "no hand" when the hand leaves (HUMAN, fps in the notes).
- [ ] `python scripts/record_session.py --label open_palm --seconds 5` produces
      `data/sessions/sample_open_palm.npz` that `SessionPlayer` reads back with the same frame
      count; committed (under 1 MB).
- [ ] `docs/phases/phase-1.md` records fps and camera resolution.

If over budget: skip the fps meter overlay and print fps to the terminal instead.

### Phase 2: Backbone from ASL landmarks (2.5 h including compute)

Goal: the frozen embedding space (document 5.1). Five never-seen poses must be separable using
only 5 examples each, across signers, without any weight update.

Depends on: Phase 1.

Deliverables:
- `scripts/extract_landmarks.py --images <dir> --out <file> --limit-per-class N --source
  <name>`: runs an image folder (class per subfolder) through `HandTracker.track_image`,
  stores the normalized 63-d feature, class, and source per detected image, skips undetected
  images, prints the detection rate per class, and appends to an existing output so a run can
  be resumed. Run on signer A with `--limit-per-class 500` (about 15 minutes) and on signer B
  with no cap. `nothing` is skipped. Output: `data/landmarks/asl_v1.npz` with arrays `x`
  (N, 63), `y` (N,) class names, `source` (N,) in {`asl_train`, `asl_test`}.
- `configs/train_v1.yaml`: held-out letters (five visually distinct, well detected letters,
  for example B, C, L, V, Y), episode shape (5-way, 5-shot, 10-query), number of episodes,
  learning rate, seed, augmentation ranges (rotation plus or minus 15 degrees about the wrist,
  scale 0.9 to 1.1, Gaussian jitter sigma 0.01).
- `pgvb/train/dataset.py` (`load_split(path, heldout) -> train, heldout` arrays plus
  `augment(x, rng)`), `pgvb/train/episodes.py` (episodic sampler, source-aware: support and
  query from different signers when both are available), `pgvb/train/model.py`
  (`build_backbone`), `pgvb/train/losses.py` (prototypical loss and episode accuracy).
- `scripts/train_backbone.py --config configs/train_v1.yaml`: seeds, trains for the configured
  episodes with early stopping on held-out episode accuracy, saves `models/backbone_v1.keras`
  and `reports/train_v1_history.csv`.
- `scripts/eval_backbone.py`: 5-way 5-shot and 5-way 1-shot accuracy on the held-out letters
  over 500 episodes, support from signer A and query from signer B; mean intra-pose and
  inter-pose cosine distance and the 95th percentile of intra-pose; writes
  `reports/backbone_v1_eval.md` and `reports/backbone_v1_distances.png`.
- `scripts/export_backbone.py`: writes `models/backbone_v1.npz` (BatchNorm folded) and
  `models/backbone_v1.json` (id, dims, held-out accuracies, train date, git commit).
- `pgvb/embedding.py`: `Backbone.load(path)`, `embed(x) -> (64,) unit vector`, `embed_batch`.
- `tests/test_embedding.py`: numpy forward equals Keras forward within 1e-5 on 100 random
  inputs (skipped if TensorFlow is absent); output is unit norm.

Definition of Done:
- [ ] `pytest -q` passes.
- [ ] Held-out 5-way 5-shot accuracy at least 0.85 cross-signer (target 0.90) and 5-way 1-shot
      at least 0.70, over 500 episodes, in `reports/backbone_v1_eval.md`. If 5-shot is below
      0.80, apply the fallback in Section 6 before moving on.
- [ ] Mean intra-pose cosine distance is clearly below mean inter-pose distance on held-out
      letters (both means and the 95th percentile of intra-pose in the report).
- [ ] `models/backbone_v1.npz` embeds one vector in under 1 ms (mean over 1000 calls).
- [ ] `data/landmarks/asl_v1.npz`, the three model files, and the report are committed.
- [ ] `docs/phases/phase-2.md` records detection rates, sample counts, training time, metrics.

If over budget: cap signer A at 300 per class and train for fewer episodes; the numbers still
go in the report as measured.

### Phase 3: Recognition core, smoothing, and speech (2.5 h)

Goal: O2, O3, O4, O5 as a library with no GUI, exercised through one command line script on
recorded sessions and the live camera.

Depends on: Phase 2.

Deliverables:
- `pgvb/profile.py`: `Gesture` (id, name, message, examples list of 64-d vectors, prototype,
  created_at, n_refinements) and `Profile` (schema_version, user_id, backbone_id, gestures);
  `load`, `save` (write temp file then rename), `validate` (rejects mismatched backbone id or
  dimension), `add_gesture`, `remove_gesture`, `rename`, `set_message`.
  `profiles/example.json` committed.
- `pgvb/prototypes.py`: `make_prototype(examples)`, `cosine_distances(e, protos)`,
  `nearest(e, protos) -> (index, d1, d2)`.
- `pgvb/enroll.py`: `Enroller(backbone, profile, cfg)` with `start(name, message)`,
  `add_frame(hand_frame) -> SampleStatus` (rejects no-hand frames and frames whose landmarks
  are still moving over the last `stability_frames`), `accept_sample()`, `progress`,
  `quality()` (intra-sample mean distance, closest existing gesture and its distance,
  collision warning below `collision_distance`), `commit() -> Gesture`, `cancel()`.
  `Refiner(backbone, profile)` with `refine(gesture_id, hand_frame)` (append one example,
  recompute only that prototype, bump `n_refinements`) and `undo_last(gesture_id)`.
- `pgvb/recognize.py`: `Recognizer(backbone, profile, cfg)` with `decide(hand_frame_or_none)
  -> Decision(label or None, distance, margin, embedding)` applying `max_distance` and
  `min_margin`; `reload_profile()`.
- `pgvb/smoothing.py`: `VoteWindow(window, min_votes)` and
  `TriggerStateMachine(release_frames, cooldown_ms)` emitting `Trigger(label, ts_ms)` on the
  rising edge only; a held gesture fires once; a label can fire again after it has been absent
  for `release_frames` frames or `cooldown_ms` has passed.
- `pgvb/output.py`: `Speaker` (worker thread, queue, `say(text)` returns immediately, `mute`,
  `on_started` and `on_finished` callbacks with timestamps, backends `pyttsx3` and macOS
  `say` chosen by `output.tts_backend`), `MessageBoard` (bounded list, `latest(n)`, `clear`).
- `pgvb/pipeline.py`: `Pipeline(cfg, profile)` composing tracker, features, backbone,
  recognizer, vote window, trigger; `process(frame, ts) -> FrameResult` (hand frame,
  decision, confirmed label, trigger or None, per-stage timings) and `process_hand_frame` for
  replay.
- `scripts/calibrate_threshold.py`: over 200 random trials on the held-out letters, enroll 3
  letters from 8 signer A samples each, test on signer B samples of those letters plus the 2
  unenrolled letters; sweep `max_distance` from 0.02 to 0.40; report enrolled accuracy, false
  accept rate, and pick the largest value with false accept rate at most 5 percent. Writes
  `reports/threshold_sweep.csv`, `reports/threshold_sweep.png`, `reports/threshold_sweep.md`,
  and the chosen value into `configs/default.yaml` in the same PR.
- `scripts/demo_recognize.py --profile <file>`: `--enroll <name> --message <text> --session
  <npz>` enrolls from a recording; `--session <npz>` prints per-frame decisions; `--smooth`
  prints triggers instead; `--speak` speaks them; no `--session` means live camera. Prints
  mean per-stage timing at exit.
- Tests: `tests/test_prototypes.py`, `tests/test_smoothing.py` (needs min_votes; flicker does
  not fire; held gesture fires once; release then repeat fires again), `tests/test_stability.py`
  (enroll A, B, C from synthetic clusters, snapshot prototypes, enroll D, assert A, B, C are
  bit-identical and their decisions unchanged, O4).

Definition of Done:
- [ ] `pytest -q` passes including the stability test.
- [ ] `reports/threshold_sweep.md` states the chosen `max_distance` with its enrolled accuracy
      and false accept rate; `configs/default.yaml` matches.
- [ ] HUMAN, about 10 minutes: record 4 sessions of each of 3 personal poses plus 1 session
      of an unrelated pose and 1 minute of idle hands with `record_session.py`. Enrolling from
      the first 3 recordings of each pose and replaying the fourth gives the correct label on
      at least 90 percent of frames with a hand; the unrelated pose yields None on at least 90
      percent; replaying the idle recording with `--smooth` yields 0 triggers. Numbers in the
      notes, recordings committed under `data/sessions/`.
- [ ] `--speak` on the live camera produces audible speech and the fps during speech stays
      within 10 percent of silent fps (HUMAN, note the backend and voice).
- [ ] Per-frame pipeline time (excluding capture) under 25 ms mean.
- [ ] `docs/phases/phase-3.md` written.

If over budget: drop the `min_margin` rule and the `on_finished` callback; keep everything
else.

### Phase 4: Desktop app: live view, enrollment, refinement (3.5 h)

Goal: O2, O5, and the adaptation loop (document 5.5) as a product a caregiver can operate
without the terminal.

Depends on: Phase 3.

Deliverables:
- `pgvb/gui/app.py`: main window. Left: camera view with landmark overlay and a status strip
  (hand found or not, current confirmed label, distance). Right: message board in large text
  and the gesture list (name, message, example count). Buttons: Enroll, Flag miss, Manage,
  Mute. Refresh via `after()` at the camera rate; if the pipeline cannot keep under one frame
  period in the main loop, run it in a worker thread.
- `pgvb/gui/enroll_wizard.py`: Toplevel with three steps. (1) Name and message. (2) Capture:
  camera view, 3-2-1 countdown, automatic capture when the hand is stable, progress "n of 8",
  a prompt every third sample to change the angle slightly, space to capture manually. (3)
  Review: consistency score, collision warning naming the closest existing gesture, Save or
  Redo. Cancel at any step leaves the profile untouched.
- `pgvb/gui/dialogs.py`: Flag miss (pick the intended gesture, default the closest, capture one
  stable sample with countdown, confirm; calls `Refiner`), Manage gestures (rename, edit
  message, delete, undo last refinement), Settings (profile name, camera index).
- `pgvb/__main__.py`: `pgvb app [--profile name] [--camera idx]`.
- No GUI unit tests. Verification is a manual walkthrough recorded in the notes with two
  screenshots in `docs/img/` (main window, wizard step 2).

Definition of Done:
- [ ] `pgvb app` starts in under 10 seconds from a warm cache and shows live video.
- [ ] HUMAN: one team member enrolls a gesture end to end using only the mouse in under
      2 minutes while the other times it; the time goes in the notes.
- [ ] The enrolled gesture is recognized live, spoken, and shown on the board; an unrelated
      pose shows "no gesture".
- [ ] Flag flow works live: a pose that was missed is recognized after one refinement (HUMAN).
- [ ] Closing the window releases the camera and stops the TTS thread (process exits).
- [ ] `docs/phases/phase-4.md` with the two screenshots.

If over budget: Manage dialog reduces to delete only; Settings dialog is dropped (profile and
camera come from the command line).

### Phase 5: Evaluation, results, and demo material (3 h plus 1 h recording)

Goal: document 5.6, reproduced by scripts from committed session files, written up as tables in
`reports/results.md`, plus everything the team needs to record the video.

Depends on: Phase 4.

Deliverables:
- `docs/evaluation/protocol.md`: the recording protocol. One participant (a team member):
  invent 3 to 5 personal gestures, enroll each with 8
  samples through the app, then record for each gesture 3 clips of 5 seconds
  (`data/sessions/eval/<participant>/<gesture>_<k>.npz`) and one 60 second non-gesture clip
  (resting, typing, scratching head, drinking). Time each enrollment with a stopwatch and note
  it in `data/sessions/eval/<participant>/enrollment_times.csv`.
  `scripts/record_session.py` gains `--participant` and `--gesture` naming.
- `scripts/eval_recognition.py`: replays every clip through the full pipeline with the
  participant's profile; per-gesture clip accuracy (first trigger is the correct label and no
  wrong trigger occurs), confusion matrix, false triggers per minute on non-gesture clips,
  algorithmic latency (first frame with the correct raw decision to trigger); writes
  `reports/eval_recognition.csv` and `.md`.
- `scripts/eval_stability.py`: per participant, enroll gestures one at a time; after each
  addition re-evaluate all earlier gestures on their clips; accuracy matrix (rows: number
  enrolled, columns: gesture) and an assertion that earlier prototypes are unchanged; writes
  `reports/eval_stability.md`.
- `scripts/make_figures.py`: confusion matrix, latency histogram, stability heatmap to
  `reports/img/`.
- `reports/results.md`: every table in one place, each produced by a script: (1) backbone
  few-shot accuracy and distance gap (Phase 2), (2) threshold sweep summary (Phase 3),
  (3) runtime: camera fps, per-stage ms, embed time (Phases 1 to 3), (4) per-gesture and
  overall clip accuracy, false triggers per minute, latency medians (algorithmic, and trigger
  to speech start from the Phase 3 callbacks measured live over 10 triggers),
  (5) stability matrix, (6) usability: enrollment time per gesture per participant and a short
  observations list (what confused the caregiver, what was fixed). Targets from the plan:
  clip accuracy at least 0.85, false triggers under 1 per minute, algorithmic latency median
  under 600 ms; each target is marked met or explained.
- `README.md` completed: overview, architecture, install, quick start, how to enroll, how to
  evaluate, results summary table, limitations, team. `docs/architecture.md` with a Mermaid
  diagram of the pipeline and the enrollment vs recognition data flow.
- `docs/demo_script.md`: a 3 minute video script (enroll two gestures, recognize them, reject
  an unrelated pose, flag and refine a miss, show that the first gesture still works after the
  second was added) plus the recording steps: what to open, screen capture tool, camera
  framing, audio, what to say over each step, and how to show the results tables at the end.
- Version 1.0.0 in `pyproject.toml` and `pgvb/__init__.py`; tag `v1.0.0` on main after merge.

Definition of Done (code):
- [ ] The three evaluation scripts run end to end on the Phase 3 sample sessions.
- [ ] Every number in `reports/results.md` is produced by a script from committed files.

Definition of Done (data, HUMAN, about 1 hour):
- [ ] At least 1 participant, at least 3 gestures, all clips and the profile committed.
- [ ] `reports/results.md` complete with all six tables and the targets marked.
- [ ] `docs/demo_script.md` written; fresh clone plus `pip install --no-build-isolation -e .`,
      `pytest -q`, and `pgvb app` succeed following only the README.
- [ ] `docs/phases/phase-5.md` summarizes results against O1 to O6; tag `v1.0.0` exists.

If over budget: drop `make_figures.py` (tables only), and the trigger-to-speech latency row.

## 6. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Backbone below 0.80 held-out 5-shot | Add pairwise fingertip distances and joint angles to the feature vector (`features.py` gains `extended=True`), widen the MLP, raise the per-class cap. Record what was tried. If it still fails, the demo can run on raw normalized features with cosine distance (set `backbone.id: identity`), which is still training-free enrollment; say so in the results. |
| Threshold that rejects unknowns also rejects real gestures | Enable `min_margin`; raise `window` and `min_votes`; ask for 10 samples with more angle variety at enrollment. |
| Frame rate under 20 fps | Lower capture resolution to 640x480; run the tracker in a worker thread; skip every second frame for embedding while still drawing every frame. |
| pyttsx3 blocks or fails on macOS | `output.tts_backend: say` (subprocess), already planned as the fallback. |
| Tkinter video is choppy | Downscale the preview to 640 wide; keep processing at capture resolution. |
| Not enough participants | One team member is the minimum; report with 1 and say so. |

## 7. Backlog

Ideas that are out of scope. Add a line, do not build.

- MobileNetV2 cropped-image second stream (document 5.1, was stretch Phase 11).
- Dynamic gestures via a 16-step landmark sequence embedding (document 1.5, was stretch Phase 12).
- Self-recorded pose catalogue with two or more recorders for a cross-person train split.
- Nielsen heuristic evaluation table and SUS questionnaire (document 5.6 item 2).
- Per-gesture custom threshold.
- Enrollment stability gate tuned for closed-hand poses: Phase 5 found `fist` and `thumbs_up`
  took 10 to 20 times longer to enroll than open-hand poses, and the resulting less-consistent
  `thumbs_up` prototype then false-rejected on later recordings (`reports/results.md` table 4).
- Two-handed gestures (`num_hands=2`, concatenated features).
- Symbol images shown on the board next to the message.
- Export profile as a printable gesture card for caregivers.
