# Phase 4: Desktop app: live view, enrollment, refinement

## What was built

`src/pgvb/gui/`:

- `_worker.py`: `CameraWorker`, a background thread owning the camera, `HandTracker`,
  `Pipeline`, `Enroller`, and `Refiner` for the app's lifetime. It runs in three modes (`live`,
  `enroll`, `flag`), driven by commands from the GUI thread over a queue, and reports frames,
  decisions, and enrollment or refinement progress back over a second queue. All mutation of the
  shared `Profile` happens on this thread only, so the GUI thread never races the worker; it only
  reads `profile.gestures` for display, refreshed after a `gestures_changed` message.
- `app.py`: the main window. Camera view with the landmark overlay, a status strip (hand found,
  confirmed label, distance, fps), a message board, and a gesture list (name, message, example
  count), plus Enroll, Flag miss, Manage, and Mute buttons. Polls the worker's output queue every
  30 ms via `after()`.
- `enroll_wizard.py`: three-step `Toplevel` wizard (name and message, capture with automatic
  accept on pose stability and a prompt to vary the angle every third sample, review with the
  intra-sample distance and a collision warning against the closest existing gesture). Cancel at
  any step leaves the profile untouched.
- `dialogs.py`: `FlagMissDialog` (captures one stable sample, defaults the gesture picker to the
  nearest existing prototype, confirms into `Refiner.refine`) and `ManageDialog` (rename, edit
  message, delete, undo last refinement).
- `pgvb app [--profile NAME] [--camera IDX]` wired up in `__main__.py`.

## Verification

Run manually (no GUI unit tests, per the plan): `pgvb app --profile phase4_test`, profile
gitignored.

- Startup: log showed the camera, `HandTracker`, and pipeline initializing with no errors;
  live video with the landmark overlay appeared well under 10 s from a warm cache (mediapipe and
  tensorflow import alone took about 6.3 s in isolation).
- Enrollment: one team member enrolled a gesture ("hello", 10 examples) end to end with the
  mouse in about 1 minute (name and message, hold pose, auto-capture, review, Save).
  `docs/img/phase4_enroll_wizard.png` shows the capture step with the 21-point overlay and the
  "n of 10 (need 5)" progress readout.
- Recognition: holding the enrolled pose showed the label and distance in the status strip,
  spoke the message ("Hello"), and appended it to the message board;
  `docs/img/phase4_main_window.png` shows this state (5 "Hello" entries on the board, gesture
  list showing 10 examples). Before enrollment and with no hand in frame the status strip showed
  "no hand | no gesture".
- Flag flow: with only one gesture enrolled there was no natural miss, so the flow was exercised
  directly: Flag miss, hold the "hello" pose, confirm against the suggested (nearest) gesture.
  The example count went from 10 to 11 and the gesture kept recognizing correctly afterward.
- Shutdown: closing the window released the camera and stopped the TTS thread; the process
  exited on its own (confirmed with `ps` after closing, no leftover process).

## Deviations from the plan

- Settings dialog (profile name, camera index) was dropped, using the over-budget fallback the
  plan already allows; profile and camera come from `pgvb app --profile --camera`. Manage still
  has rename, edit message, delete, and undo, not just delete, since implementing it fully cost
  little once rename and edit message already needed a value-editing dialog.
- The plan's per-file layout under `gui/` did not list a fourth file; the camera/pipeline worker
  thread lives in `gui/_worker.py` (leading underscore, package-private) instead of inside
  `app.py`, to keep the widget code and the threading/queue glue separate. No behavior change.
