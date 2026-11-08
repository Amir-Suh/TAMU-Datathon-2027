"""A causal posterior over signed log-odds effects from local lag motifs."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit

from .baseline import FrequencyForecaster
from .causal import LAGS, LAG_PAIRS, MISSING, lagged_spins, tick_arrays
from .schema import PAYOUTS, ValidationError


AMPLITUDES = (-2, -1, 1, 2)
PAYOUT_CLASSES = np.arange(len(PAYOUTS))
RECENCY_STRENGTH = 8
RECENCY_HALF_LIFE = 12
RECENCY_DECAY = 2 ** (-1 / RECENCY_HALF_LIFE)


@dataclass(frozen=True)
class MotifConfig:
    family: str = "singleton_pair"
    evidence_half_life: float = None

    def __post_init__(self):
        if self.family not in ("null", "singleton", "singleton_pair"):
            raise ValueError("Unknown motif family.")
        if self.evidence_half_life is not None and self.evidence_half_life <= 0:
            raise ValueError("Evidence half-life must be positive.")


def motif_design(spins, config):
    """Return motif labels and signed features, with zero reserved for missing."""
    if config.family == "null":
        return (), np.zeros((*spins.shape, 0), dtype=np.int8)
    lags = lagged_spins(spins)
    signs = np.where(lags == MISSING, 0, np.where(lags > 0, 1, -1)).astype(np.int8)
    singletons = tuple((lag,) for lag in LAGS)
    if config.family == "singleton":
        return singletons, signs
    left, right = np.array(LAG_PAIRS).T - 1
    return singletons + LAG_PAIRS, np.concatenate((signs, signs[..., left] * signs[..., right]), axis=-1)


def hypothesis_prior(config):
    if config.family == "null":
        return np.array([1.])
    n_single = len(LAGS) * len(AMPLITUDES)
    if config.family == "singleton":
        return np.r_[.5, np.full(n_single, .5 / n_single)]
    n_pair = len(LAG_PAIRS) * len(AMPLITUDES)
    return np.r_[.5, np.full(n_single, .25 / n_single), np.full(n_pair, .25 / n_pair)]


def motif_rollout(spins, priors, config, *, diagnostic_ticks=None):
    """Forecast each tick before updating. Optionally capture one posterior per machine."""
    motifs, features = motif_design(spins, config)
    log_prior = np.log(hypothesis_prior(config))
    log_evidence = np.zeros((len(spins), len(log_prior)))
    forecasts = np.empty((*spins.shape, len(PAYOUTS)))
    diagnostics = None
    if diagnostic_ticks is not None:
        diagnostics = {
            "motifs": motifs,
            "posterior": np.empty_like(log_evidence),
            "observed_support": (
                (features != 0) & (spins[..., None] != MISSING)
                & (np.arange(spins.shape[1])[None, :, None] < diagnostic_ticks[:, None, None])
            ).sum(axis=1),
            "available_at_cutoff": features[np.arange(len(spins)), diagnostic_ticks] != 0,
        }

    prior_counts = RECENCY_STRENGTH * priors
    counts = prior_counts.copy()
    decay = 1. if config.evidence_half_life is None else 2 ** (-1 / config.evidence_half_life)
    for tick in range(spins.shape[1]):
        positive = counts[:, 1:]
        positive_total = positive.sum(axis=1)
        baseline = positive_total / counts.sum(axis=1)
        if motifs:
            offset = logit(np.clip(baseline, 1e-12, 1 - 1e-12))
            contrast = features[:, tick, :, None] * AMPLITUDES
            alternatives = np.where(contrast != 0, expit(offset[:, None, None] + contrast),
                                    baseline[:, None, None])
            win = np.column_stack([baseline, alternatives.reshape(len(spins), -1)])
        else:
            win = baseline[:, None]

        log_weights = log_prior + log_evidence
        weights = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        if diagnostics is not None:
            at_endpoint = diagnostic_ticks == tick
            diagnostics["posterior"][at_endpoint] = weights[at_endpoint]
        mixture = (weights * win).sum(axis=1)
        forecasts[:, tick, 0] = 1 - mixture
        forecasts[:, tick, 1:] = mixture[:, None] * (positive / positive_total[:, None])

        outcome = spins[:, tick]
        likelihood = np.where(outcome[:, None] > 0, win, 1 - win)
        log_likelihood = np.log(np.maximum(likelihood, 1e-300))
        relative = log_likelihood - log_likelihood[:, :1]
        log_evidence = decay * log_evidence + (outcome != MISSING)[:, None] * relative
        counts = prior_counts + RECENCY_DECAY * (counts - prior_counts) + (outcome[:, None] == PAYOUT_CLASSES)

    return forecasts, diagnostics


class SignedMotifForecaster:
    def __init__(self, config=MotifConfig()):
        self.config = config
        self.prior = FrequencyForecaster("context")

    def fit(self, queries, history, labels=None):
        self.prior.fit(queries, history, labels)
        return self

    def prior_rows(self, queries):
        if self.prior.global_prior is None:
            raise ValidationError("Fit the signed motif model before predicting.")
        return np.array([self.prior.context_priors.get(self.prior._key(q), self.prior.global_prior)
                         for q in queries])

    def predict(self, queries, history):
        queries = tuple(queries)
        spins, cutoffs = tick_arrays(queries, history)
        forecast, _ = motif_rollout(spins, self.prior_rows(queries), self.config)
        return {q.query_id: tuple(row) for q, row in zip(queries, forecast[np.arange(len(queries)), cutoffs])}
