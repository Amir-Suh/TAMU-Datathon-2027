"""Frequency pooling with the kit's priors and recency weights."""

import json
from pathlib import Path

import numpy as np

from tdx.arm import Arm


class PooledArm(Arm):
    """Match FrequencyForecaster; transductive history is deliberately ignored.

    Histories and hard labels from label_splits define the fitted population.
    This gives the kit's train-only fit in dev and includes validation in final.
    Contexts use the cutoff signal, never a constructed tick-level signal.
    """

    def __init__(self, method="pooled", prior_strength=20.0, half_life=24.0,
                 transductive=False):
        if method not in ("pooled", "pooled_recent", "recent", "context"):
            raise ValueError(f"Unknown pooled method: {method}")
        self.method = method
        self.prior_strength = float(prior_strength)
        self.half_life = float(half_life)
        if (not np.isfinite(self.prior_strength) or self.prior_strength <= 0
                or not np.isfinite(self.half_life) or self.half_life <= 0):
            raise ValueError("prior_strength and half_life must be finite and positive")
        self.global_prior = None
        self.context_priors = None

    @staticmethod
    def _contexts(split):
        return np.asarray(split.mtype, dtype=np.int64) * 3 + split.signal + 1

    @staticmethod
    def _observed(split):
        y = np.asarray(split.y)
        ticks = np.arange(y.shape[-1])
        return (y >= 0) & (ticks[None, None, :] < split.cutoff[:, None, None])

    def fit(self, rnd, spec):
        counts = np.zeros((9, 5), dtype=np.float64)
        present = np.zeros(9, dtype=bool)
        for name in spec.label_splits:
            split = rnd.splits[name]
            if split.label is None:
                raise ValueError(f"Missing labels for fitted split {name!r}")
            context = self._contexts(split)
            present[context.ravel()] = True
            observed = self._observed(split)
            for c in range(5):
                machine_counts = np.sum(observed & (split.y == c), axis=-1)
                machine_counts = machine_counts + (split.label == c)
                counts[:, c] += np.bincount(context.ravel(),
                                           weights=machine_counts.ravel(), minlength=9)
        total = counts.sum()
        if total == 0:
            raise ValueError("Training needs at least one observed payout or label")
        self.global_prior = (counts.sum(axis=0) + 0.5) / (total + 2.5)
        self.context_priors = np.broadcast_to(self.global_prior, (9, 5)).copy()
        self.context_priors[present] = (
            (counts[present] + 12.0 * self.global_prior)
            / (counts[present].sum(axis=-1, keepdims=True) + 12.0)
        )

    def predict(self, split):
        if self.global_prior is None:
            raise ValueError("Fit or load the model before predicting")
        context = self.context_priors[self._contexts(split)]
        if self.method == "context":
            result = context.copy()
        else:
            pooled = self.method in ("pooled", "pooled_recent")
            prior = context if pooled else self.global_prior
            strength = self.prior_strength if pooled else 1.0
            shape = (*split.y.shape[:2], 5)
            result = np.broadcast_to(strength * prior, shape).copy()
            observed = self._observed(split)
            if self.method in ("recent", "pooled_recent"):
                age = split.cutoff[:, None] - 1 - np.arange(split.y.shape[-1])
                weight = np.exp2(-np.maximum(age, 0).astype(np.float64) / self.half_life)
                weights = observed * weight[:, None, :]
            else:
                weights = observed
            for c in range(5):
                result[..., c] += np.sum(weights * (split.y == c), axis=-1)
            result /= result.sum(axis=-1, keepdims=True)
        result = np.clip(result, 1e-6, 1.0)
        result /= result.sum(axis=-1, keepdims=True)
        return result

    def save(self, directory):
        if self.global_prior is None:
            raise ValueError("Cannot save an unfitted model")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        value = {"model_version": 1, "method": self.method,
                 "prior_strength": self.prior_strength, "half_life": self.half_life,
                 "global_prior": self.global_prior.tolist(),
                 "context_priors": self.context_priors.tolist()}
        (directory / "model.json").write_text(json.dumps(value, sort_keys=True) + "\n",
                                              encoding="utf-8")

    @classmethod
    def load(cls, directory):
        value = json.loads((Path(directory) / "model.json").read_text(encoding="utf-8"))
        if value.get("model_version") != 1:
            raise ValueError("Unsupported saved pooled model")
        model = cls(value["method"], value["prior_strength"], value["half_life"])
        model.global_prior = np.asarray(value["global_prior"], dtype=np.float64)
        model.context_priors = np.asarray(value["context_priors"], dtype=np.float64)
        if model.global_prior.shape != (5,) or model.context_priors.shape != (9, 5):
            raise ValueError("Malformed saved pooled model")
        for probabilities in (model.global_prior, model.context_priors):
            if (not np.isfinite(probabilities).all() or (probabilities <= 0).any()
                    or not np.allclose(probabilities.sum(axis=-1), 1, atol=1e-12, rtol=0)):
                raise ValueError("Malformed saved pooled probabilities")
        return model
