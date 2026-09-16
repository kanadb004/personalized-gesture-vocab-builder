"""Episodic N-way K-shot sampler for prototypical-network training. Source-aware: when a
class has examples from both signers, support is drawn from asl_train and query from
asl_test so training episodes mirror the cross-signer few-shot eval. Training-only module."""

from __future__ import annotations

import numpy as np

from pgvb.train.dataset import SplitData, augment


class EpisodeSampler:
    def __init__(
        self,
        split: SplitData,
        n_way: int,
        k_shot: int,
        n_query: int,
        aug_cfg: dict | None,
        rng: np.random.Generator,
    ) -> None:
        self.split = split
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.aug_cfg = aug_cfg
        self.rng = rng

        self.classes = sorted(set(split.y.tolist()))
        if len(self.classes) < n_way:
            raise ValueError(f"need at least {n_way} classes, got {len(self.classes)}")

        self._by_class_source: dict[str, dict[str, np.ndarray]] = {}
        for c in self.classes:
            mask_c = split.y == c
            per_source = {}
            for s in sorted(set(split.source[mask_c].tolist())):
                per_source[s] = np.where(mask_c & (split.source == s))[0]
            self._by_class_source[c] = per_source

    def _pick_support_query(self, class_name: str) -> tuple[np.ndarray, np.ndarray]:
        sources = self._by_class_source[class_name]
        if "asl_train" in sources and "asl_test" in sources:
            support_pool, query_pool = sources["asl_train"], sources["asl_test"]
        else:
            pool = next(iter(sources.values()))
            support_pool = query_pool = pool

        support_replace = len(support_pool) < self.k_shot
        support_idx = self.rng.choice(support_pool, size=self.k_shot, replace=support_replace)

        remaining = np.setdiff1d(query_pool, support_idx) if query_pool is support_pool else query_pool
        if len(remaining) == 0:
            remaining = query_pool
        query_replace = len(remaining) < self.n_query
        query_idx = self.rng.choice(remaining, size=self.n_query, replace=query_replace)
        return support_idx, query_idx

    def sample_episode(
        self, apply_augment: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (support_x, support_y, query_x, query_y). Labels are episode-local ints
        0..n_way-1, in the order the classes were sampled."""
        episode_classes = self.rng.choice(self.classes, size=self.n_way, replace=False)

        support_x_parts, support_y_parts = [], []
        query_x_parts, query_y_parts = [], []
        for label, class_name in enumerate(episode_classes):
            support_idx, query_idx = self._pick_support_query(class_name)
            support_x_parts.append(self.split.x[support_idx])
            support_y_parts.append(np.full(len(support_idx), label, dtype=np.int64))
            query_x_parts.append(self.split.x[query_idx])
            query_y_parts.append(np.full(len(query_idx), label, dtype=np.int64))

        support_x = np.concatenate(support_x_parts, axis=0)
        support_y = np.concatenate(support_y_parts, axis=0)
        query_x = np.concatenate(query_x_parts, axis=0)
        query_y = np.concatenate(query_y_parts, axis=0)

        if apply_augment and self.aug_cfg is not None:
            support_x = augment(support_x, self.rng, **self.aug_cfg)
            query_x = augment(query_x, self.rng, **self.aug_cfg)

        return support_x, support_y, query_x, query_y
