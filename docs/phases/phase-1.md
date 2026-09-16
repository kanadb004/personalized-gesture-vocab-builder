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
  handedness plus score, and fps; `r` toggles recording to `data/sessions/<name>_<n>.npz`,
  `q` quits.
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
- `python scripts/demo_landmarks.py` and `python scripts/record_session.py --label open_palm
  --seconds 5` could not be run against the live camera in this session: the sandboxed terminal
  this session runs in is not authorized for camera capture on macOS
  (`OpenCV: not authorized to capture video`), and the user chose to skip granting that
  permission for this session rather than run the commands themselves. Both scripts compile
  cleanly (`python -m py_compile`) and their camera and tracker calls are exercised indirectly by
  the smoke tests above; only the live-camera fps number and the actual `sample_open_palm.npz`
  recording are outstanding.

## Deviations

None to the plan itself.

## Remaining before this phase can close

Two DoD items need a person at the webcam, from a terminal or app with camera permission
granted (System Settings > Privacy & Security > Camera):

1. `python scripts/demo_landmarks.py` at the built-in camera: confirm 20 fps or more with a hand
   tracked, and "no hand" shown when the hand leaves frame. Record the fps and camera resolution
   here.
2. `python scripts/record_session.py --label open_palm --seconds 5`: confirm it produces
   `data/sessions/sample_open_palm.npz`, that `SessionPlayer` reads it back with the same frame
   count as recorded, and commit the file (it must stay under 1 MB).

This branch is pushed and the PR is left open as a draft until both are done.
