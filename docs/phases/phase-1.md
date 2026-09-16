# Phase 1: Camera and landmark pipeline

## What was built

- `pgvb/camera.py`: `Camera` context manager over `cv2.VideoCapture` yielding
  `(frame_bgr, timestamp_ms)` from `read()`, configurable index, width, height, mirror; `FpsMeter`
  with a rolling window average.
- `pgvb/landmarks.py`: `HandTracker` wrapping MediaPipe Tasks `HandLandmarker` with two internal
  landmarkers, one built in VIDEO mode (`track(frame_bgr, ts_ms)`) and one in IMAGE mode
  (`track_image(image_rgb)`, for Phase 2's offline extraction). Both return a `HandFrame`
  (`landmarks` 21x3 float32, `handedness`, `score`, `ts_ms`) or `None` when no hand scores above
  `min_hand_score`.
- `pgvb/features.py`: `normalize_landmarks(lm, handedness, rotate=False)` exactly per PLAN.md
  Section 2 (translate wrist to origin, scale by wrist-to-middle-MCP distance, mirror x for left
  hands, optional rotation about the wrist), returns a (63,) float32 vector; `feature_dim()`.
- `pgvb/replay.py`: `SessionRecorder` (append `HandFrame | None` with timestamps, save to a
  documented `.npz` layout) and `SessionPlayer` (iterate frames back as `(HandFrame | None,
  ts_ms)` pairs), format documented in the module docstring.
- `scripts/demo_landmarks.py`: live overlay of the 21 points, the skeleton connections,
  handedness plus score, and fps. Hand tracking runs in a background thread (`_AsyncTracker`)
  against the most recently submitted frame so a slow tracker call never blocks the capture and
  display loop; `r` toggles recording to `data/sessions/<name>_<n>.npz`, `q` quits.
- `scripts/record_session.py --label <text> --seconds N [--out <file>]`: headless recorder
  with a 3-2-1 countdown printed to the terminal; also accepts `--participant`/`--gesture` for
  Phase 5's naming convention.
- `tests/test_features.py`: shape and dtype, translation invariance, scale invariance, left and
  right mirror to the same vector.

## Verification

- `pytest -q`: 9 passed (5 from Phase 0 plus 4 new feature tests).
- `HandTracker.track_image` and `HandTracker.track` were smoke-tested against a still image from
  `data/raw/asl_alphabet/asl_alphabet_test/asl_alphabet_test/F_test.jpg`: both modes detected one
  right hand (score 0.92), `normalize_landmarks` produced a (63,) float32 vector.
- `SessionRecorder`/`SessionPlayer` were round-trip tested with a synthetic 3-frame stream
  (one hand frame, one no-hand frame, one hand frame): `SessionPlayer` reproduced the same frame
  count, labels, handedness, and scores.
- `python scripts/demo_landmarks.py` on the built-in camera (1280x720 native, FaceTime HD on an
  M2 MacBook; a Continuity Camera handoff from a nearby iPhone was briefly grabbing index 0
  instead and had to be disconnected first): with the original synchronous single-threaded
  tracker at the default 1280x720 capture, fps measured 15.0 with no hand and 14.6 to 14.8
  while a hand was in frame, below the 20 fps target. Isolated profiling (see Deviations) found
  the bottleneck was the `HandLandmarker.detect_for_video` call itself (about 27 ms), not
  capture (about 11 ms) or `cv2.imshow` (about 18 ms), and that resolution barely mattered
  since MediaPipe resizes internally for its model input. Moving hand tracking to a background
  thread (Section 6 fallback) so the capture and display loop no longer blocks on it fixed this:
  retested at 640x480 (also lowered per Section 6) and measured 30.0 fps with no hand, left
  hand, and right hand in frame. "no hand" displayed correctly when the hand left frame.
- `python scripts/record_session.py --label open_palm --seconds 5` produced
  `data/sessions/sample_open_palm.npz` (144 frames, 34,523 bytes, well under the 1 MB limit).
  `SessionPlayer` read it back with the same frame count (144); 143 of 144 frames had a hand
  detected. A first attempt only had the hand up for the first 1.2 seconds of the 5 (35 of 146
  frames), caught by inspecting the recorded `has_hand` timeline rather than trusting the frame
  count alone, and was redone.

## Deviations

- `configs/default.yaml` and the `CameraConfig`/`Camera` defaults were changed from 1280x720 to
  640x480 (Section 6 fallback for frame rate under 20 fps). Kept even though resolution alone did
  not fix the fps (see below), because it is a reasonable default for interactive use.
- `scripts/demo_landmarks.py` gained a background thread for hand tracking (`_AsyncTracker`,
  Section 6 fallback: "run the tracker in a worker thread"), needed because the tracker call,
  not capture or display, was the actual bottleneck. `updated tests/test_config.py` for the new
  640x480 default. No change to `pgvb/landmarks.py` or `pgvb/camera.py` themselves; threading is
  local to the demo script since `record_session.py` and Phase 3's pipeline need a synchronous,
  deterministic per-frame result rather than an async one.
