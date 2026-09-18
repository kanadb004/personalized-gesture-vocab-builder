# Phase 5: Evaluation, results, and demo material

## What was built

- `docs/evaluation/protocol.md`: the recording protocol for each participant (invent 3 to 5
  gestures, enroll through the app, record 3 five-second clips per gesture plus one 60 second
  non-gesture clip, time enrollment with a stopwatch).
- `scripts/record_session.py` gained `--clip`, so `--participant P --gesture G --clip K`
  writes `data/sessions/eval/<P>/<G>_<K>.npz`, matching the protocol's naming.
- `scripts/_eval_common.py`: shared clip-replay helper (`ClipResult`, `replay_clip`,
  `discover_gesture_clips`, `discover_pattern_clips`) used by both eval scripts and
  `make_figures.py`, so the "first trigger correct and no wrong trigger" accuracy rule and the
  algorithmic-latency definition live in one place.
- `scripts/eval_recognition.py --profile P --dir D --nongesture PATTERN...`: replays every
  `<gesture>_*.npz` clip (gestures taken from the profile) and every non-gesture clip matching
  the given glob patterns through a fresh `Pipeline` per clip; writes per-gesture and overall
  clip accuracy, a confusion matrix, false triggers per minute on the non-gesture clips, and
  median algorithmic latency to `<out>.md` and a per-clip `<out>.csv`.
- `scripts/eval_stability.py --profile P --dir D`: reconstructs the profile's gestures in
  `created_at` order, adds them back one at a time, and after each addition replays every
  already-enrolled gesture's clips against that partial profile; writes an accuracy matrix
  (rows: gestures enrolled so far) plus an explicit assertion (exit code 1 on failure) that
  every gesture's prototype stays bit-identical to the fully enrolled profile's, no matter how
  many later gestures were added (O4, participant-level analogue of `tests/test_stability.py`).
- `scripts/make_figures.py`: renders the confusion matrix, an algorithmic-latency histogram, and
  the stability matrix as heatmaps to `reports/img/<prefix>_*.png`, recomputed from the same
  clips so the images match the tables.
- `demo_recognize.py --speak` (live mode) now measures trigger-to-speech latency itself: each
  trigger's wall-clock time is queued, `Speaker.on_started` pops it and records the gap, and the
  median over all triggers prints at exit ("trigger to speech start: median ... ms over N
  triggers"), using the `on_started` callback Phase 3 already built.
- `docs/architecture.md`: two Mermaid diagrams (the per-frame pipeline, and the enrollment vs.
  recognition vs. refinement data flow around the frozen backbone).
- `docs/demo_script.md`: a 3 minute recording script and the recording setup (screen capture
  tool, camera framing, audio).
- `README.md` completed: architecture pointer, quick start, how to enroll, how to evaluate,
  results pointer, limitations, team.
- `reports/results.md`: tables 1 to 3 (backbone accuracy, threshold sweep, runtime) filled from
  the numbers already in `reports/backbone_v1_eval.md`, `reports/threshold_sweep.md`, and
  `docs/phases/phase-1.md`/`phase-2.md`/`phase-3.md`. Tables 4 to 6 (per-participant recognition
  accuracy, stability, enrollment usability) are marked pending, see below.
- Version bumped to `1.0.0` in `pyproject.toml` and `src/pgvb/__init__.py`.

## Verification (code DoD)

- `pytest -q`: 25 passed (unchanged from Phase 4, no new pure-math logic was added that needed
  its own unit tests per the plan's testing policy).
- The three evaluation scripts run end to end against the Phase 3 sample sessions
  (`data/sessions/phase3/`), using a scratch profile enrolled from those same recordings
  (`demo_recognize.py --enroll`, 10 examples per gesture from clips 1 to 3 of each): with
  `--dir data/sessions/phase3`, `eval_recognition.py` reported 12/12 gesture clips correct
  (open_palm, peace, thumbs_up, 4 clips each, since clips 1 to 3 were also used for
  enrollment), 0 false triggers over 64.9 s of `idle.npz` + `unrelated.npz`, and a median
  algorithmic latency of 345.8 ms; `eval_stability.py` reported a 1.000 accuracy matrix at every
  enrollment step and passed the prototype-stability assertion; `make_figures.py` wrote all
  three PNGs without error. This exercises the same code path real participant data will use;
  the numbers themselves are not meaningful evaluation results, since the smoke-test profile
  used its own enrollment clips as part of the query set.
- `demo_recognize.py --profile profiles/example.json --session
  data/sessions/phase3/open_palm_4.npz --smooth` still runs correctly after the trigger-latency
  instrumentation was added (unchanged trigger output, `pending_trigger_ts` is only populated
  in live mode).

## What remains (data DoD, HUMAN, about 1 hour)

Scope changed to one participant (Section 6 risk table fallback), reflected in
`docs/PLAN.md` and `docs/evaluation/protocol.md`. Still needed, per
`docs/evaluation/protocol.md`:

- One participant, at least 3 gestures: enroll through the
  app, record 3 clips per gesture plus one non-gesture clip, commit
  `profiles/<participant>.json` and `data/sessions/eval/<participant>/`.
- Run `eval_recognition.py`, `eval_stability.py`, and `make_figures.py` per participant and fill
  `reports/results.md` tables 4, 5, and 6 (currently marked pending) with the real numbers,
  including the targets-met column.
- Record the trigger-to-speech latency over 10 live triggers (`demo_recognize.py --speak`).
- Record and edit the demo video per `docs/demo_script.md`.
- Fresh-clone check: clone the repo elsewhere, `pip install --no-build-isolation -e .`,
  `pytest -q`, `pgvb app`, following only the README.
- Tag `v1.0.0` on `main` once the above is merged.

This phase's branch stays open (not squash-merged) until the data DoD above is complete, per
`CLAUDE.md`'s "phase cannot be completed in one session" fallback: push the branch, PR as a
draft, resume in the next session.

## Deviations

- `reports/results.md` tables 4 to 6 are placeholders pending real participant recordings,
  rather than filled with fabricated or Phase 3 team-recording numbers, since those recordings
  used different gestures and don't match the data DoD.
- Plan changed from at least 2 participants to 1: only one team member was available to record
  in this session. `docs/PLAN.md` Section 5 (Phase 5 deliverables and data DoD) and Section 6
  (risk table) were updated in this same PR to match; `docs/evaluation/protocol.md` reflects the
  one-participant protocol. The results and limitations sections should say this plainly rather
  than imply a broader evaluation than what was actually run.
- No other deviations from the plan.
