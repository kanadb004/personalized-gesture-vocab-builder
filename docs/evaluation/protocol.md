# Phase 5 evaluation protocol

Per document section 5.6. Two team members are the minimum participant count (Section 6 risk
table); any volunteer who is available may be added as a third.

## Per participant

1. Pick a profile name for the participant, for example `aadit` or `kanad`. Start the app:
   `pgvb app --profile <name>`.
2. Invent 3 to 5 personal gestures with a short spoken message each (for example
   `wave` = "Hi there", `fist` = "I need help", `point_up` = "Yes please").
3. Enroll each gesture through the app's Enroll wizard, 8 samples each. Time each enrollment
   with a stopwatch (wizard open to Save) and record it in
   `data/sessions/eval/<participant>/enrollment_times.csv` with columns
   `gesture,seconds`.
4. Copy the saved profile to `profiles/<participant>.json` (the app already writes it there if
   `--profile <name>` was used with no path separator).
5. For each enrolled gesture, record 3 clips of 5 seconds with the recognizer not running
   (`record_session.py`, not the live app), holding the gesture for the clip's duration:

   ```
   python scripts/record_session.py --participant <participant> --gesture <gesture> --clip 1 --seconds 5
   python scripts/record_session.py --participant <participant> --gesture <gesture> --clip 2 --seconds 5
   python scripts/record_session.py --participant <participant> --gesture <gesture> --clip 3 --seconds 5
   ```

   This writes `data/sessions/eval/<participant>/<gesture>_<clip>.npz`.
6. Record one 60 second non-gesture clip: rest the hand, type, scratch your head, drink from a
   cup, and otherwise move naturally without holding any enrolled pose:

   ```
   python scripts/record_session.py --participant <participant> --gesture nongesture --seconds 60
   ```

   This writes `data/sessions/eval/<participant>/nongesture.npz`.
7. Commit `profiles/<participant>.json` and everything under
   `data/sessions/eval/<participant>/`.

## Running the evaluation

```
python scripts/eval_recognition.py --profile profiles/<participant>.json \
    --dir data/sessions/eval/<participant> --nongesture "nongesture*.npz" \
    --out reports/eval_recognition_<participant>

python scripts/eval_stability.py --profile profiles/<participant>.json \
    --dir data/sessions/eval/<participant> --out reports/eval_stability_<participant>

python scripts/make_figures.py --profile profiles/<participant>.json \
    --dir data/sessions/eval/<participant> --nongesture "nongesture*.npz" \
    --out-prefix reports/img/<participant>
```

`reports/results.md` cites the numbers from every participant's `.md` output plus a combined
usability table built from each `enrollment_times.csv`.

## Trigger-to-speech latency (results.md table 4)

Live only, not from a recording:

```
python scripts/demo_recognize.py --profile profiles/<participant>.json --speak
```

Hold each enrolled gesture in front of the camera long enough to trigger, release, repeat until
at least 10 triggers have fired, then press `q`. `demo_recognize.py` times every trigger to
speech start itself (the `Speaker.on_started` callback timestamp minus the wall-clock time the
trigger fired) and prints the median at exit: `trigger to speech start: median ... ms over N
triggers`. Record that line in the results table.
