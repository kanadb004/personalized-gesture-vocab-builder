# Phase 2: Backbone from ASL landmarks

## What was built

- `scripts/extract_landmarks.py --images <dir> --out <file> --limit-per-class N --source
  <name>`: runs an image folder (class per subfolder) through `HandTracker.track_image`,
  normalizes each detection, appends to an existing `.npz`, and skips classes already present
  for the same `--source` so a run can be resumed. Skips the `nothing` class and any subfolder
  with no images directly inside it (the Kaggle test set ships a duplicate nested copy).
- `configs/train_v1.yaml`: held-out letters B, C, L, V, Y; 5-way 5-shot 10-query episodes, 200
  episodes per epoch, up to 60 epochs, patience 8; augmentation of plus or minus 15 degrees
  rotation about the wrist, 0.9 to 1.1 scale, 0.01 sigma jitter.
- `pgvb/train/dataset.py` (`load_split`, source-aware `augment`), `pgvb/train/episodes.py`
  (`EpisodeSampler`, draws support from signer A and query from signer B when both are present
  for a class), `pgvb/train/model.py` (`build_backbone`, 63 -> 128 -> 128 -> 64 with BatchNorm
  and Dropout 0.2, L2-normalized output), `pgvb/train/losses.py` (prototypical loss and episode
  accuracy on cosine similarity logits).
- `scripts/train_backbone.py --config configs/train_v1.yaml`: episodic training with early
  stopping on held-out episode accuracy, saves `models/backbone_v1.keras` and
  `reports/train_v1_history.csv`.
- `scripts/eval_backbone.py --config configs/train_v1.yaml`: 500-episode 5-way 5-shot and
  5-way 1-shot cross-signer accuracy, intra/inter-pose cosine distance stats; writes
  `reports/backbone_v1_eval.md`, `reports/backbone_v1_eval.json` (consumed by
  `export_backbone.py`), and `reports/backbone_v1_distances.png`.
- `scripts/export_backbone.py`: folds each Dense+BatchNorm pair into one affine layer, writes
  `models/backbone_v1.npz` (numpy inference weights) and `models/backbone_v1.json` (id, dims,
  held-out accuracies, train date, git commit, measured embed time).
- `pgvb/embedding.py`: `Backbone.load(path)`, `embed(x)`, `embed_batch(x)`, pure numpy, no
  TensorFlow import.
- `tests/test_embedding.py`: numpy forward pass matches the Keras model within 1e-5 on 100
  random inputs, output is unit norm. Skips without TensorFlow/Keras or an exported backbone.

## Verification

- `pytest -q`: 11 passed (9 from Phases 0 to 1 plus 2 new embedding tests).
- Landmark extraction: signer A (`data/raw/asl_alphabet/asl_alphabet_train/asl_alphabet_train`,
  capped at 500/class) detected 12,497 of 14,500 attempted images (86.2% overall); worst classes
  were `N` (33.2%), `M` (58.0%), `space` and `del` (around 75%), all other letters 82% to 100%.
  Signer B (`data/raw/asl_alphabet_test`, no cap, 870 images) detected 786 (90.3%); worst was `I`
  and `space` (66.7%). `data/landmarks/asl_v1.npz` holds 13,283 examples total (6.4 MB).
- All five held-out letters (B, C, L, V, Y) have at least 166 signer A and 20 signer B examples
  each after detection loss, enough for the 5-shot support pool and 10-query queries with margin.
- Training: seed 0, `configs/train_v1.yaml`. Episodic validation accuracy on the 5 held-out
  letters (support signer A, query signer B, drawn by the same source-aware sampler used for
  training) reached 0.994 by epoch 7 and early stopped at epoch 15 (461.9 s total, about 30 s
  per epoch on an M2 MacBook CPU).
- `scripts/eval_backbone.py`, 500 held-out episodes: 5-way 5-shot accuracy 0.988 (target >= 0.85,
  stretch 0.90) and 5-way 1-shot accuracy 0.957 (target >= 0.70), both comfortably above target.
  Mean intra-pose cosine distance 0.0960, mean inter-pose distance 1.0405 (95th percentile
  intra-pose 0.9570, pulled up by a small tail of harder pairs; the bulk of the intra-pose mass
  sits near zero, see `reports/backbone_v1_distances.png`), clearly separated as required.
- `models/backbone_v1.npz` mean embed time: 0.033 ms over 1000 calls (target < 1 ms).
- `data/landmarks/asl_v1.npz`, `models/backbone_v1.{keras,npz,json}`, and
  `reports/backbone_v1_eval.md` are committed; all files are well under the 20 MB limit (the
  largest, the landmarks file, is 6.4 MB).

## Deviations

- None from the plan's fixed technical decisions. The 5-shot and 1-shot targets were both met on
  the first successful training run (after fixing a Keras custom-layer registration bug, see
  below), so the Section 6 fallback (extended features, wider MLP, higher per-class cap) was not
  needed.
- A first training run failed to load in `eval_backbone.py` and `export_backbone.py`:
  `keras.models.load_model` could not deserialize the custom `L2Normalize` layer because
  `pgvb.train.model` (which registers it via `@keras.saving.register_keras_serializable`) was
  never imported by those scripts, only by `train_backbone.py`. Fixed by importing
  `pgvb.train.model.L2Normalize` for its registration side effect in every script and test that
  loads a `.keras` file, and retraining once (the first, unloadable checkpoint was discarded).
- `scripts/eval_backbone.py` and the parity test print a benign Keras warning
  ("The structure of `inputs` doesn't match the expected structure... Expected: ['landmarks']")
  when calling the loaded model on a raw numpy array instead of a named input; this does not
  affect the computed values (confirmed by the parity test passing within 1e-5) and was left
  alone rather than adding input-naming plumbing that nothing else needs.
