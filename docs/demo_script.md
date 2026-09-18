# Demo video script (about 3 minutes)

## Recording setup

- Screen capture: QuickTime Player "New Screen Recording" (built in, no download), or the
  macOS Control Center screen recorder. Record the whole display so both the terminal and the
  app window are visible when needed.
- Camera framing: laptop webcam, presenter seated about arm's length away, hand clearly in
  frame and well lit (face a window or lamp, avoid backlight).
- Audio: record system audio too (QuickTime: click the arrow next to the record button, pick
  the built-in microphone) so `pyttsx3`/`say` speech is captured, plus narrate over it.
- Before recording: `conda activate pgvb`, `cd` to the repo, run `pgvb app --profile kanadb`
  once to confirm the camera and speech both work, then quit and clear the terminal.

## Cast

Uses the real `profiles/kanadb.json` from the Phase 5 evaluation, already enrolled with 4
gestures: `open_palm` -> "Hello", `peace_sign` -> "Thank you", `fist` -> "I need help",
`thumbs_up` -> "Okay". `thumbs_up` genuinely misses at the calibrated threshold
(`reports/results.md` table 4), which the script uses as a real flag-and-refine moment instead
of staging one.

## Script

1. **Intro (10 s).** Show the terminal. Say: "This is the Personalized Gesture Vocabulary
   Builder, a training-free AAC tool. It lets someone teach a personal hand gesture from a few
   examples, then speaks a message when it sees that gesture again."
2. **Enroll a new gesture live (30 s).** Run `pgvb app --profile kanadb`. Click Enroll. Pick a
   gesture not already in the profile (for example a "point up" pose), give it a message (for
   example "Yes please"). Hold the pose through the countdown and automatic captures, show the
   review step's consistency score, click Save. Narrate that this is 5 to 10 examples, no
   retraining, added straight into the same profile as the other four.
3. **Recognize the existing gestures (25 s).** Hold `open_palm`: show the status strip's label
   and distance, the spoken "Hello", and the message board entry. Release, hold `fist`, same
   thing for "I need help".
4. **Reject an unrelated pose (15 s).** Show a pose that was never enrolled (something distinct
   from all five). Status strip shows "no gesture", nothing is spoken.
5. **Flag and refine a real miss (30 s).** Hold `thumbs_up`. It does not trigger (this is the
   real, measured miss from `reports/results.md`, not staged). Say so on camera, then click
   Flag miss, pick `thumbs_up` from the picker, hold the pose again for the one-sample capture.
   Show the example count go up by one, then hold `thumbs_up` again and show whether it now
   recognizes (if the refined prototype still misses, say that on camera too and point to the
   results.md explanation rather than re-recording until it works).
6. **Stability check (20 s).** Say: "Enrolling that new example, or the new gesture from step 2,
   never touches the others." Hold `open_palm` or `fist` again to show it still recognizes
   exactly as before.
7. **Results (20 s).** Cut to (or screen-share) `reports/results.md`: point at the clip accuracy
   table, the false-triggers-per-minute number, and the stability matrix, noting whether each
   target (0.85 accuracy, under 1 false trigger per minute, under 600 ms latency) was met, and
   that the `thumbs_up` miss just shown on camera is exactly the one explained there.
8. **Close (10 s).** Say: "Everything in that table comes from committed recordings replayed
   through scripts, so the numbers are reproducible from the repo." Quit the app on camera to
   show the process exits cleanly.

## After recording

- Trim to about 3 minutes total, export as .mp4.
- Do not commit the raw video (over the repo's 20 MB file limit and out of scope for git);
  keep it wherever the team shares deliverables for the course submission.
