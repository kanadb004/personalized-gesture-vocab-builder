# CLAUDE.md

Personalized Gesture Vocabulary Builder (PGVB): a few-shot, training-free AAC tool that lets a
nonverbal user or caregiver teach a webcam system a personal hand gesture from 5 to 10 examples,
then speaks or displays a message when that gesture is recognized. Course project for VIT BCSE415L
(Human Computer Interaction). Team: Aadit Singh (23BCE1413), Kanad Bhattacharya (23BCE1234).

The full build plan with per-phase definitions of done lives in `docs/PLAN.md`. Read it at the
start of every session. The original project document is `docs/DA-1.pdf`.

## Session protocol (one phase per session)

1. `git checkout main && git pull --ff-only origin main` and confirm `git status` is clean.
2. Open `docs/PLAN.md`, find the first phase whose status is not `Done`. That is the phase for
   this session. Do not skip ahead and do not start a second phase in the same session.
3. Create the GitHub issue for the phase (skip if it already exists, check with `gh issue list`):
   `gh issue create --title "Phase N: <title>" --label phase --body-file <tmpfile>`
   The body is the phase's Goal plus its Definition of Done checklist copied from the plan.
4. Branch: `git checkout -b phase-N-<slug>` (example: `phase-1-landmark-pipeline`).
5. Build the phase. Commit in small, logical steps (see commit rules below).
6. Before opening the PR, every DoD box must be verifiably true. Run the verification commands
   listed in the phase. Write `docs/phases/phase-N.md` (what was built, how it was verified,
   numbers measured, any deviation from the plan and why). Update the status table in
   `docs/PLAN.md` to `Done` for this phase.
7. Push and open the PR:
   `git push -u origin phase-N-<slug>`
   `gh pr create --title "Phase N: <title>" --body "Closes #<issue>. <one or two sentences>"`
8. Merge and clean up:
   `gh pr merge <pr> --squash --delete-branch --subject "Phase N: <title>" --body "Closes #<issue>"`
   `git checkout main && git pull --ff-only origin main && git branch -D phase-N-<slug>` (if left)
9. Verify nothing attributes Claude: `git log -3 --format='%an <%ae>%n%B'` must show only the
   team author and no trailers.

If a phase cannot be completed in one session, push the branch, leave the PR open as a draft
(`gh pr create --draft`), and record what remains in `docs/phases/phase-N.md`. The next session
resumes on that branch.

## Git and GitHub conventions

- Author on every commit: the configured git user (`kanadb <kanadb004@gmail.com>`). Never change it.
- No Claude attribution anywhere: no `Co-Authored-By`, no `Claude-Session`, no
  "Generated with Claude Code", no links to claude.ai, in commits, PR bodies, issues, or code.
  `.claude/settings.json` disables it and `.githooks/commit-msg` rejects it. After cloning,
  run `git config core.hooksPath .githooks` once.
- Commit subject: imperative mood, lower case after the first word, max 60 characters, no
  trailing period. Examples: `Add landmark normalizer`, `Fix threshold off by one`,
  `Train backbone v1 and export weights`.
- Commit body: optional, only when the subject is not enough. One to three plain lines, wrapped
  at 72 characters. No bullet lists, no headings, no emoji.
- Never use an em dash or en dash anywhere: not in commits, PRs, issues, docs, comments, or
  strings. Use a comma, colon, or plain hyphen instead. This applies to all text in the repo.
- Branches: `phase-N-<slug>`, lower case, hyphens, slug of two to four words.
- Issues: title `Phase N: <title>`, label `phase`, body = goal + DoD checklist.
- PRs: title identical to the issue title, body starts with `Closes #<issue>`, one or two
  sentences after it. Squash merge with the same subject. Delete the branch after merge.
- One phase per branch per PR. Fix-ups discovered later get their own small branch and PR named
  `fix-<slug>` with a short description, still squash merged.
- Never force push to `main`. Never commit secrets, raw images or videos, or files over 20 MB.

## Python environment

- Use the conda env `pgvb` (Python 3.11.13): `/opt/anaconda3/envs/pgvb/bin/python`, or
  `conda activate pgvb`. It was cloned from the existing `tf_env` so TensorFlow was not
  re-downloaded. Never create another env, never install into `base`, `tf_env`, or `ar_env`.
- Verified working set (2026-09-13): tensorflow 2.16.2 + tensorflow-metal 1.2.0, keras 3.10.0,
  numpy 1.26.4, mediapipe 0.10.14, opencv-python 4.11.0.86, opencv-contrib-python 4.11.0.86,
  jax 0.4.25, jaxlib 0.4.25, ml-dtypes 0.3.2, protobuf 4.25.8, pillow 11.2.1, pyttsx3 2.99,
  pytest 9.1.1, PyYAML 6.0.3, scikit-learn 1.7.0, matplotlib 3.10.3, tkinter 8.6.
- Hard pins, do not break them:
  `numpy<2` and `ml-dtypes~=0.3.1` (tensorflow 2.16 requirements),
  `opencv-python<5` and `opencv-contrib-python<5` (opencv 5 needs numpy 2),
  `jax==0.4.25` and `jaxlib==0.4.25` (mediapipe 0.10.14 pulls jax; newer jax needs ml-dtypes 0.5),
  `mediapipe==0.10.14`. Do not upgrade mediapipe: 1.0.1 installs but `HandLandmarker`
  aborts on this Mac with `DrishtiMetalHelper ... Service is unavailable` on both GPU and
  CPU delegates. 0.10.14 was verified end to end (2 hands detected on a real photo).
- `pip check` prints one cosmetic line, `mediapipe 0.10.14 is not supported on this platform`;
  ignore it. Any other `pip check` output is a real problem.
- New dependency: `pip install` it in `pgvb`, then add the exact pinned version to
  `requirements.txt` in the same commit, then run `pip check`.
- Hand tracking: use the MediaPipe Tasks API only:
  `from mediapipe.tasks.python import vision; vision.HandLandmarker` with the model file
  `models/hand_landmarker.task` (7.8 MB, downloaded by `scripts/download_models.py` from
  `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
  gitignored). `mp.solutions.hands` exists in 0.10.14 but is deprecated; do not use it.
- `import mediapipe` also imports tensorflow (optional dependency), so first import costs
  several seconds (20 s or more when the machine is loaded). Keep TensorFlow out of the
  runtime path anyway: training uses Keras, inference uses exported numpy weights (Phase 3).

## Code conventions

- Package: `src/pgvb/`, installed editable (`pip install -e .`). Scripts in `scripts/`,
  tests in `tests/`, configs in `configs/`, models in `models/`, data in `data/`.
- Python 3.11, type hints on public functions, docstrings only where the name is not enough.
- Config lives in `configs/default.yaml` and is loaded once; no magic numbers in modules.
- Unit tests must not need a camera, a display, a speaker, or TensorFlow. Camera and GUI
  checks are manual and documented in the phase notes. Tests that need the real backbone
  weights load `models/backbone_v1.npz`.
- Run `pytest -q` before every commit that touches `src/` or `tests/`.
- Keep the app runnable at all times: `python -m pgvb` must start after every merged phase
  from Phase 7 onward.
- Match the existing style of the file you are editing. Do not reformat unrelated code.

## Do not

- Do not start a phase without its issue and branch.
- Do not mark a DoD item done without running its verification.
- Do not add features outside the current phase; note ideas in `docs/PLAN.md` under Backlog.
- Do not re-plan the project; if the plan is wrong, say so in the phase notes and make the
  smallest change to the plan that fixes it, in the same PR.
