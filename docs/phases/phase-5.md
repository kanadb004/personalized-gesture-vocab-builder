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
- `reports/results.md`: all six tables filled. Tables 1 to 3 from the numbers already in
  `reports/backbone_v1_eval.md`, `reports/threshold_sweep.md`, and
  `docs/phases/phase-1.md`/`phase-2.md`/`phase-3.md`. Tables 4 to 6 from the one-participant data
  collection below.
- Version bumped to `1.0.0` in `pyproject.toml` and `src/pgvb/__init__.py`.
- One participant (`kanadb`), 4 gestures enrolled through the app (`open_palm` = "Hello",
  `peace_sign` = "Thank you", `fist` = "I need help", `thumbs_up` = "Okay"), `profiles/kanadb.json`
  and `data/sessions/eval/kanadb/` (3 clips per gesture, one 60 s non-gesture clip,
  `enrollment_times.csv`) committed.
- `reports/eval_recognition_kanadb.md/.csv`, `reports/eval_stability_kanadb.md`,
  `reports/img/kanadb_confusion.png`, `reports/img/kanadb_latency.png`,
  `reports/img/kanadb_stability.png`.

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

## Verification (data DoD, HUMAN, kanadb at the webcam)

- Enrolled all 4 gestures through `pgvb app --profile kanadb` with the mouse; enrollment times
  in `data/sessions/eval/kanadb/enrollment_times.csv` and discussed in `reports/results.md`
  table 6 (open-hand poses 5 to 10 s, closed-hand poses 90 to 120 s).
- Recorded 3 five-second clips per gesture plus one 60 second non-gesture clip with
  `scripts/record_session.py --participant kanadb --gesture <g> --clip <k>`. Several clips had
  to be re-recorded because the pose was not actually held during the take (caught by asking
  after each command whether the pose was held, not by the script, which cannot tell); the final
  committed set was individually confirmed.
- `scripts/eval_recognition.py --profile profiles/kanadb.json --dir data/sessions/eval/kanadb
  --nongesture "nongesture*.npz"`: overall clip accuracy 0.750 (9/12); `open_palm`, `peace_sign`,
  `fist` all 1.000, `thumbs_up` 0.000 (false-rejected, not misclassified, see table 4); 0 false
  triggers over the 60 s non-gesture clip; median algorithmic latency 729.6 ms.
- `scripts/eval_stability.py --profile profiles/kanadb.json --dir data/sessions/eval/kanadb`:
  accuracy matrix and prototype-stability assertion both passed (O4); `thumbs_up`'s 0.000 entry
  reflects the same threshold issue, not a stability regression.
- `scripts/make_figures.py`: wrote the three PNGs under `reports/img/kanadb_*.png`.
- `demo_recognize.py --profile profiles/kanadb.json --speak`, live: median trigger-to-speech
  (queue dequeue) latency 0.1 ms over 50 triggers; see the caveat about what this callback
  actually measures in `reports/results.md` table 4.
- Two DoD targets were not met with this profile: overall clip accuracy (0.750 < 0.85, entirely
  `thumbs_up`) and algorithmic latency (729.6 ms > 600 ms). Both are explained, not silently
  missed, in `reports/results.md` table 4, and both point to the same root cause noted in table 6
  and added to the Backlog: the enrollment stability gate is not well tuned for closed-hand
  poses.

## Demo video and DA-2 document

- The demo video was recorded by kanadb, by hand, following `docs/demo_script.md` (updated in
  this session to walk through the real `kanadb` profile: enroll a new gesture live, recognize
  `open_palm` and `fist`, reject an unrelated pose, flag and refine a miss, a stability check,
  and the results tables). During recording, a new gesture (`rock_on` -> "Rock On") was enrolled
  live and `open_palm` was refined once (the flag-and-refine step used `open_palm`, not
  `thumbs_up`, a judgment call made during filming); `profiles/kanadb.json` was committed in
  this post-video state at the user's request.
- **Deviation, by explicit user instruction:** `reports/results.md` and the eval reports
  (`reports/eval_recognition_kanadb.*`, `reports/eval_stability_kanadb.md`,
  `reports/img/kanadb_*.png`) were computed from the profile *before* the video's live changes,
  and were not re-run against the post-video profile. The committed `profiles/kanadb.json` is
  therefore one snapshot ahead of the numbers cited in `reports/results.md`; the eval clips
  under `data/sessions/eval/kanadb/` still exactly reproduce those numbers if replayed against
  the profile version at the commit tagged for this phase's data collection (before "Add
  finalized DA-2 document and post-demo profile state"). This is a deliberate exception to the
  general "every number in results.md is produced by a script from committed files" DoD wording,
  made because the video was the priority once the numbers were already measured and written up.
- `docs/DA-2_results-tabulation-&-workind-demo-video_23BCE1265.pdf`: the course-submission
  document, drafted with a matching VIT front page to `docs/DA-1.pdf` (title changed to
  "DIGITAL ASSIGNMENT-2") and a full results write-up sourced from `reports/results.md`, then
  manually reviewed, edited, and finalized by the team before committing.

## Fresh-clone check

Cloned the `phase-5-evaluation-results` branch to a scratch directory, copied
`models/hand_landmarker.task` in as a machine-local asset (gitignored, not part of the clone,
per `docs/OFFLINE.md`), and confirmed:
- `pip install --no-build-isolation -e .` succeeds.
- `pytest -q`: 25 passed.
- `pgvb app --profile freshclone_check` starts and stays running (verified over 8 s) with no
  errors, following only the README's Setup/Install/Download models/Quick start steps.

The scratch clone and its throwaway profile were deleted afterward; the real repo's editable
install was reinstalled pointing back at this working copy.

## Remaining before merge

- Tag `v1.0.0` on `main` once this PR is merged.

This phase's branch stays open as a draft PR until that tag is created, per `CLAUDE.md`'s
"phase cannot be completed in one session" fallback, though everything else in the data and
code DoD is now done.

## Results against O1 to O6

- **O1 (real-time landmark extraction):** met since Phase 1; runtime table (results.md table 3)
  restates the measured fps and per-stage timing.
- **O2 (few-shot enrollment, no retraining):** met. All 4 gestures enrolled from 8 examples each
  through the app with no weight update; `thumbs_up`'s recognition problem is a threshold/
  enrollment-consistency issue, not a retraining or few-shot-capacity issue.
- **O3 (open-set rejection):** met. 0 false triggers over the 60 s non-gesture clip.
- **O4 (stability, enrolling N never disturbs 1..N-1):** met, both at the unit-test level
  (`tests/test_stability.py`) and now at the participant level (`eval_stability.py`'s explicit
  bit-identical-prototype assertion passed for all 4 gestures at every enrollment step).
- **O5 (spoken + on-screen output):** met; Phase 3 and 4 verified audible speech and the message
  board, unchanged here.
- **O6 (real-time responsiveness and enrollment usability):** partially met. Responsiveness:
  algorithmic latency (729.6 ms) missed the 600 ms target at this webcam's tracker-bound frame
  rate, explained in results.md table 4. Usability: enrollment for open-hand gestures was fast
  and easy (5 to 10 s), but closed-hand gestures were slow and inconsistent to enroll (90 to
  120 s), which is exactly the kind of caregiver-usability problem O6 is meant to catch; it is
  recorded honestly rather than hidden, with a Backlog line for the fix.

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
