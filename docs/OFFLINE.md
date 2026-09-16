# Offline and low-bandwidth operation

Everything that needs a fast connection was fetched on 2026-09-13. After that date the
project must build with no downloads at all, and with git and GitHub traffic kept small.

## What is already on this machine

| Item | Location | Purpose |
|---|---|---|
| Conda env `pgvb` | `/opt/anaconda3/envs/pgvb` | Complete runtime and training env, see CLAUDE.md pins |
| Wheelhouse (231 wheels, 545 MB) | `~/Work/pgvb-wheelhouse/` | Every package in `requirements.txt` plus transitive deps, `setuptools`, `wheel`. Repairs the env offline. |
| `hand_landmarker.task` (7.8 MB) | `models/hand_landmarker.task` | MediaPipe hand landmark model, Phase 1 onward |
| `gesture_recognizer.task` (8.4 MB) | `models/gesture_recognizer.task` | MediaPipe built-in gesture recognizer. Not used by the pipeline; not needed by the 2 day plan. |
| MobileNetV2 ImageNet weights, no top | `~/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5` | Backlog only (MobileNetV2 stream is out of scope) |
| GitHub issues #1 to #6 | github.com/kanadb004/personalized-gesture-vocab-builder/issues | One per phase, issue number = phase number + 1 (#7 to #11 closed, superseded by the 2 day plan) |
| `phase` label | GitHub | Used by the issue workflow |
| Kaggle ASL Alphabet (1.2 GB, 87,000 images, 29 classes, 200x200) | `data/raw/asl_alphabet/asl_alphabet_train/asl_alphabet_train/<class>/` (gitignored) | Phase 2, backbone training data (signer A) |
| Kaggle ASL Alphabet Test (27 MB, 1,740 images, different signer) | `data/raw/asl_alphabet_test/<class>/` (gitignored) | Phase 2, cross-signer split (signer B) |
| Kaggle CLI 2.2.4 | `uv tool install kaggle`, token in `~/.kaggle/access_token` | Only needed if more datasets are wanted while online |

The raw image folders are large and gitignored. If they are ever lost they can be re-fetched
only while online: `kaggle datasets download grassknoted/asl-alphabet -p data/raw/asl_alphabet --unzip`
and `kaggle datasets download danrasband/asl-alphabet-test -p data/raw/asl_alphabet_test --unzip`.
The extracted landmark files that Phase 2 produces from them are small and committed, so the
raw images are not needed after Phase 2.

## Installing with no network

Any `pip` command in `pgvb` must use the wheelhouse:

```
export PIP_NO_INDEX=1
export PIP_FIND_LINKS=~/Work/pgvb-wheelhouse
pip install -r requirements.txt          # repair or reinstall the env
pip install --no-build-isolation -e .    # Phase 0 editable install, always use this form
```

`--no-build-isolation` matters: without it pip creates a temporary build env and tries to
download `setuptools`, even though it is already installed. `setuptools` 78.1.1 and `wheel`
0.45.1 are in the env, which is enough for the `src` layout `pyproject.toml`.

If the env is destroyed, recreate it without downloads:
`conda create -n pgvb --clone tf_env`, then the two pip commands above.

Adding a dependency that is not in the wheelhouse is not possible offline. Do not try; find a
way with what is installed (numpy, opencv, mediapipe, tensorflow, keras, scikit-learn,
matplotlib, pandas, tqdm, pillow, pyttsx3, PyYAML, pytest, tkinter, and the stdlib) and note
the wish in `docs/PLAN.md` under Backlog.

`scripts/download_models.py` (Phase 0) must check for the file first and never re-download an
existing model of the right size.

## Git workflow when GitHub is unreachable

Detect it with `git ls-remote --exit-code origin -h refs/heads/main` (times out or fails).
The normal per-phase workflow in CLAUDE.md needs the network at three points: the issue
(already done for Phases 0 to 5), the push, and the PR merge. Offline, do this instead:

1. Do not touch local `main`. Start the phase branch from the tip of the newest phase
   branch that has not been merged on GitHub (stacked), or from `main` if every earlier phase
   is merged. Name it exactly as usual: `phase-N-<slug>`.
2. Build and commit as usual. Update `docs/PLAN.md` status and write `docs/phases/phase-N.md`
   as usual.
3. Append one line to `docs/sync_queue.md` (create it if missing):
   `phase-N-<slug> | Phase N: <title> | #<issue> | base: <branch it was started from>`
4. Stop at the point where the PR would be opened. Do not merge locally.

When the network is back, sync in queue order, one phase at a time:

```
git checkout main && git pull --ff-only origin main
git push -u origin phase-N-<slug>
gh pr create --title "Phase N: <title>" --body "Closes #<issue>. <one or two sentences>" \
  --base main --head phase-N-<slug>
gh pr merge <pr> --squash --subject "Phase N: <title>" --body "Closes #<issue>"
git checkout main && git pull --ff-only origin main
# If the next queued branch was stacked on phase-N-<slug>, move it onto the new main:
git rebase --onto main phase-N-<slug> phase-M-<slug>
# Only now delete the merged branch, locally and on GitHub:
git branch -D phase-N-<slug> && git push origin --delete phase-N-<slug>
```

Remove the synced line from `docs/sync_queue.md` in the next commit. The rebase applies
cleanly because the squash commit on `main` has the same tree as the tip of the merged
branch; if it does not, stop and report rather than resolving conflicts by guesswork.

Slow but working network: use the normal workflow. Pushes are a few kilobytes per phase.
