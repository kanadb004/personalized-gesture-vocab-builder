# Demo video script (about 3 minutes)

## Recording setup

- Screen capture: QuickTime Player "New Screen Recording" (built in, no download), or the
  macOS Control Center screen recorder. Record the whole display so both the terminal and the
  app window are visible when needed.
- Camera framing: laptop webcam, presenter seated about arm's length away, hand clearly in
  frame and well lit (face a window or lamp, avoid backlight).
- Audio: record system audio too (QuickTime: click the arrow next to the record button, pick
  the built-in microphone) so `pyttsx3`/`say` speech is captured, plus narrate over it.
- Before recording: `conda activate pgvb`, `cd` to the repo, run `pgvb app --profile demo` once
  to confirm the camera and speech both work, then quit and clear the terminal.

## Script

1. **Intro (10 s).** Show the terminal. Say: "This is the Personalized Gesture Vocabulary
   Builder, a training-free AAC tool. It lets someone teach a personal hand gesture from a few
   examples, then speaks a message when it sees that gesture again."
2. **Enroll gesture 1 (30 s).** Run `pgvb app --profile demo`. Click Enroll. Name it, give it a
   message (for example `wave` -> "Hello!"). Hold the pose through the countdown and automatic
   captures. Show the review step's consistency score, click Save.
3. **Enroll gesture 2 (25 s).** Repeat with a second, visually distinct gesture (for example
   `fist` -> "I need help"). Narrate that this is still 5 to 10 examples, no retraining.
4. **Recognize both (20 s).** Hold gesture 1: show the status strip's label and distance, the
   spoken message, and the message board entry. Release, hold gesture 2, same thing.
5. **Reject an unrelated pose (15 s).** Show a pose that was never enrolled (for example a peace
   sign, if not already used). Status strip shows "no gesture", nothing is spoken.
6. **Flag and refine a miss (30 s).** Hold gesture 1 at an unusual angle so it is missed. Click
   Flag miss, confirm it is gesture 1, hold the pose for the one-sample capture. Show the
   example count go up by one, then hold that same angle again and show it now recognizes
   correctly.
7. **Stability check (20 s).** Say: "Enrolling that new example, or a whole new gesture, never
   touches the others." Hold gesture 2 again to show it still recognizes exactly as before.
8. **Results (20 s).** Cut to (or screen-share) `reports/results.md`: point at the clip accuracy
   table, the false-triggers-per-minute number, and the stability matrix, noting whether each
   target (0.85 accuracy, under 1 false trigger per minute, under 600 ms latency) was met.
9. **Close (10 s).** Say: "Everything in that table comes from committed recordings replayed
   through scripts, so the numbers are reproducible from the repo." Quit the app on camera to
   show the process exits cleanly.

## After recording

- Trim to about 3 minutes total, export as .mp4.
- Do not commit the raw video (over the repo's 20 MB file limit and out of scope for git);
  keep it wherever the team shares deliverables for the course submission.
