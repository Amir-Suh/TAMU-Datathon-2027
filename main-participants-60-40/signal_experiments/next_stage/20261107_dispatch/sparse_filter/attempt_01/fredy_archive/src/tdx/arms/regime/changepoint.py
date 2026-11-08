"""Episode-pooled categorical change detection with an episode-held-out threshold."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from tdx.data import Split


def _log_likelihood(counts: np.ndarray) -> np.ndarray:
    """Multinomial MLE log likelihood, with unknown observations omitted."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum(axis=-1, keepdims=True)
    # The zero-count convention is 0 log 0 = 0.
    return (counts * np.log(np.maximum(counts, 1) / np.maximum(total, 1))).sum(-1)


def _threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Maximise binary accuracy; ties choose the most conservative threshold."""
    finite = np.unique(scores[np.isfinite(scores)])
    if not len(finite):
        return 0.0
    candidates = np.r_[np.nextafter(finite[0], -np.inf), finite]
    correct = ((scores[:, None] > candidates) == labels[:, None]).sum(axis=0)
    return float(candidates[np.flatnonzero(correct == correct.max())[-1]])


@dataclass
class ChangeModel:
    """Threshold training reads conditions only from explicitly supplied splits.

    A candidate pools all observed machines before and after its tick. Its score
    is the two-distribution log-likelihood gain minus the BIC cost of four extra
    parameters and a log(number of candidates) search penalty.
    """

    window: int = 120
    step: int = 4
    min_segment_ticks: int = 8
    min_segment_obs: int = 64
    n_folds: int = 5
    threshold: float = 0.0
    report: dict = field(default_factory=dict)

    def __post_init__(self):
        if min(self.window, self.step, self.min_segment_ticks, self.min_segment_obs) < 1:
            raise ValueError("Change detector dimensions must be positive")
        if self.n_folds < 2:
            raise ValueError("At least two CV folds are required")

    def evidence(self, split: Split) -> tuple[np.ndarray, np.ndarray]:
        """Return best candidate tick and penalised gain, before thresholding."""
        ticks = np.full(len(split.ids), -1, dtype=np.int64)
        scores = np.full(len(split.ids), -np.inf, dtype=np.float64)
        for e, raw_cutoff in enumerate(split.cutoff):
            cutoff = min(int(raw_cutoff), split.y.shape[-1])
            # Slicing here is intentional: future storage and n_obs are untrusted.
            history = split.y[e, :, :max(cutoff, 0)]
            counts = np.stack([(history == k).sum(axis=0) for k in range(5)], axis=-1)
            prefix = np.vstack([np.zeros((1, 5)), np.cumsum(counts, axis=0)])
            candidates = np.arange(cutoff - self.window, cutoff, self.step)
            candidates = candidates[(candidates >= self.min_segment_ticks)
                                    & (candidates <= cutoff - self.min_segment_ticks)]
            if not len(candidates):
                continue
            before, after = prefix[candidates], prefix[-1] - prefix[candidates]
            valid = ((before.sum(-1) >= self.min_segment_obs)
                     & (after.sum(-1) >= self.min_segment_obs))
            candidates, before, after = candidates[valid], before[valid], after[valid]
            if not len(candidates):
                continue
            penalty = 2.0 * np.log(prefix[-1].sum()) + np.log(len(candidates))
            gains = (_log_likelihood(before) + _log_likelihood(after)
                     - _log_likelihood(prefix[-1]) - penalty)
            best = int(np.argmax(gains))
            ticks[e], scores[e] = candidates[best], gains[best]
        return ticks, scores

    def detect(self, split: Split) -> tuple[np.ndarray, np.ndarray]:
        """Return change ticks (−1 for no trigger) and the episode trigger mask."""
        ticks, scores = self.evidence(split)
        fires = np.isfinite(scores) & (scores > self.threshold)
        return np.where(fires, ticks, -1), fires

    def fit(self, splits: Iterable[Split]) -> ChangeModel:
        splits = list(splits)
        if not splits or any(s.condition is None for s in splits):
            raise ValueError("Threshold fitting needs explicitly labelled episode conditions")
        scores = np.concatenate([self.evidence(s)[1] for s in splits])
        conditions = np.concatenate([s.condition for s in splits]).astype(str)
        ids = [str(ep) for s in splits for ep in s.ids]
        if not ids:
            raise ValueError("Threshold fitting needs at least one episode")
        # Hash ordering is stable across input order; identical episodes cannot
        # occur on both sides of a fold even if supplied in multiple splits.
        ordered = sorted(set(ids), key=lambda ep: (hashlib.sha256(ep.encode()).hexdigest(), ep))
        n_folds = min(self.n_folds, len(ordered))
        fold_map = {ep: i % n_folds for i, ep in enumerate(ordered)}
        folds = np.array([fold_map[ep] for ep in ids])
        labels = conditions == "shared_change"
        held_out = np.zeros(len(ids), dtype=bool)
        fold_thresholds = []
        for fold in range(n_folds):
            held = folds == fold
            train = ~held
            # With one episode there is no honest CV estimate.
            threshold = _threshold(scores[train], labels[train]) if train.any() else 0.0
            held_out[held] = scores[held] > threshold
            fold_thresholds.append(threshold)
        self.threshold = _threshold(scores, labels)
        fires = scores > self.threshold
        rates = lambda mask: {c: float(fires[mask & (conditions == c)].mean())
                              for c in sorted(set(conditions[mask]))}
        train_mask = np.concatenate([np.full(len(s.ids), s.name == "train") for s in splits])
        self.report = {
            "threshold": self.threshold,
            "cv_accuracy": float(np.mean(held_out == labels)) if n_folds > 1 else None,
            "cv_folds": n_folds,
            "cv_fold_thresholds": fold_thresholds,
            "cv_fire_rates": {c: float(held_out[conditions == c].mean()) for c in sorted(set(conditions))},
            "fire_rates": rates(np.ones(len(ids), dtype=bool)),
            "train_fire_rates": rates(train_mask),
            "fit_accuracy": float(np.mean(fires == labels)),
            "n_episodes": len(ids),
            "penalty": "2*log(n_observations)+log(n_valid_candidates)",
        }
        return self

    def to_dict(self) -> dict:
        return {"window": self.window, "step": self.step,
                "min_segment_ticks": self.min_segment_ticks,
                "min_segment_obs": self.min_segment_obs, "n_folds": self.n_folds,
                "threshold": self.threshold, "report": self.report}

    @classmethod
    def from_dict(cls, state: dict) -> ChangeModel:
        return cls(**state)
