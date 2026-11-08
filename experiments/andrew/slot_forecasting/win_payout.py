"""Regularized forecasts with separate win and conditional payout heads."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .causal import MISSING, feature_rows, tick_arrays
from .io import require_ids
from .schema import PAYOUTS, ValidationError, payout


def _contexts(queries):
    return np.array([f"{q.machine_type}|{q.signal}" for q in queries])


def training_rows(queries, history, labels, *, history_samples=32, peers=True):
    require_ids(labels, {q.query_id for q in queries})
    spins, cutoffs = tick_arrays(queries, history)
    observed = spins != MISSING
    reverse_rank = np.cumsum(observed[:, ::-1], axis=1)[:, ::-1]
    machines, ticks = np.nonzero(observed & (reverse_rank <= history_samples))
    historical_labels = spins[machines, ticks]
    endpoints = np.arange(len(queries))
    machines = np.r_[machines, endpoints]
    ticks = np.r_[ticks, cutoffs]
    outcomes = np.r_[historical_labels, [PAYOUTS.index(payout(labels[q.query_id])) for q in queries]]
    features = feature_rows(queries, spins, machines, ticks, peers=peers)
    return features, outcomes, machines, _contexts(queries)[machines]


def _fit_head(features, outcomes, strength):
    classes = np.unique(outcomes)
    if len(classes) < 2:
        return int(classes[0]) if len(classes) else None
    return LogisticRegression(C=strength, solver="lbfgs", max_iter=400).fit(features, outcomes)


def _head_probabilities(head, features, classes):
    result = np.zeros((len(features), classes))
    if head is None:
        result.fill(1 / classes)
    elif isinstance(head, int):
        result[:, head] = 1
    else:
        result[:, head.classes_] = head.predict_proba(features)
    return result


class WinPayoutForecaster:
    def __init__(self, *, strength=.1, peers=True, direct=False, positive_prior_weight=.15):
        self.strength = strength
        self.peers = peers
        self.direct = direct
        self.positive_prior_weight = positive_prior_weight
        self.center = None

    def fit(self, queries, history, labels):
        queries = tuple(queries)
        features, outcomes, _, contexts = training_rows(queries, history, labels, peers=self.peers)
        return self.fit_rows(features, outcomes, contexts)

    def fit_rows(self, features, outcomes, contexts):
        self.center = features.mean(axis=0)
        self.scale = np.maximum(features.std(axis=0), .1)
        features = (features - self.center) / self.scale
        if self.direct:
            self.payout_head = _fit_head(features, outcomes, self.strength)
            return self
        positive = outcomes > 0
        self.win_head = _fit_head(features, positive.astype(int), self.strength)
        self.payout_head = _fit_head(features[positive], outcomes[positive] - 1, self.strength)
        n_positive_classes = len(PAYOUTS) - 1
        counts = np.bincount(outcomes[positive] - 1, minlength=n_positive_classes) + .5
        self.global_positive_prior = counts / counts.sum()
        self.positive_priors = {}
        for context in np.unique(contexts):
            labels = outcomes[positive & (contexts == context)] - 1
            counts = np.bincount(labels, minlength=n_positive_classes) + 12 * self.global_positive_prior
            self.positive_priors[context] = counts / counts.sum()
        return self

    def predict_rows(self, features, contexts, *, positive_prior_weight=None):
        if self.center is None:
            raise ValidationError("Fit the win/payout model before predicting.")
        features = (features - self.center) / self.scale
        if self.direct:
            return _head_probabilities(self.payout_head, features, len(PAYOUTS))
        win = _head_probabilities(self.win_head, features, 2)[:, 1]
        sizes = _head_probabilities(self.payout_head, features, len(PAYOUTS) - 1)
        weight = self.positive_prior_weight if positive_prior_weight is None else positive_prior_weight
        if weight:
            prior = np.array([self.positive_priors.get(key, self.global_positive_prior) for key in contexts])
            sizes = (1 - weight) * sizes + weight * prior
        return np.column_stack([1 - win, win[:, None] * sizes])

    def predict(self, queries, history, *, positive_prior_weight=None):
        queries = tuple(queries)
        spins, cutoffs = tick_arrays(queries, history)
        features = feature_rows(queries, spins, np.arange(len(queries)), cutoffs, peers=self.peers)
        forecasts = self.predict_rows(features, _contexts(queries), positive_prior_weight=positive_prior_weight)
        return {q.query_id: tuple(row) for q, row in zip(queries, forecasts)}
