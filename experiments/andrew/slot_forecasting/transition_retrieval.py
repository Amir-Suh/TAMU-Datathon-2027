"""Causal retrieval of same-machine transitions by masked binary lag agreement."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from .baseline import FrequencyForecaster
from .causal import LAGS, MISSING, lagged_spins, tick_arrays
from .schema import PAYOUTS, ValidationError


SUPPORT_CAP = 64
MATCH_HALF_LIFE = 24
MISMATCH_PENALTY = 2
SHRINKAGE = 4
PRIOR_HALF_LIFE = 12
PRIOR_STRENGTH = 8
LEARNED_HISTORY = 8
SIMILARITY_WIDTH = 16
SIMILARITY_EPOCHS = 3
SIMILARITY_BATCH = 64
SIMILARITY_STEP = .01


@dataclass(frozen=True)
class RetrievalRows:
    machines: np.ndarray
    ticks: np.ndarray
    outcomes: np.ndarray
    prediction: np.ndarray
    recency: np.ndarray
    match_mass: np.ndarray
    effective_support: np.ndarray
    recent_observations: np.ndarray


def causal_support(observed_ticks, observed_values, lags, endpoints):
    """Return the most recent observed transitions strictly before each endpoint."""
    stops = np.searchsorted(observed_ticks, endpoints)
    offsets = stops[:, None] - np.arange(SUPPORT_CAP, 0, -1)
    valid = offsets >= 0
    if not len(observed_ticks):
        return (np.full((*valid.shape, len(LAGS)), MISSING),
                np.zeros_like(valid), np.zeros_like(offsets))
    slots = np.maximum(offsets, 0)
    ticks = observed_ticks[slots]
    keys = np.where(valid[..., None], lags[ticks], MISSING)
    wins = valid & (observed_values[slots] > 0)
    ages = np.where(valid, endpoints[:, None] - ticks, 0)
    return keys, wins, ages


def _comparison(query, keys, ages):
    shared = (query[:, None, :] != MISSING) & (keys != MISSING)
    overlap = shared.sum(axis=-1)
    matches = shared & ((query[:, None, :] > 0) == (keys > 0))
    decay_overlap = np.exp2(-ages / MATCH_HALF_LIFE) * (overlap / len(LAGS))
    return shared, matches, overlap, decay_overlap


def retrieval_weights(query, keys, ages):
    _, matches, overlap, decay_overlap = _comparison(query, keys, ages)
    mismatches = overlap - matches.sum(axis=-1)
    return decay_overlap * np.exp(-MISMATCH_PENALTY * mismatches / np.maximum(overlap, 1))


def _similarity_inputs(query, keys, ages, machine_type, signal):
    shared, matches, _, decay_overlap = _comparison(query, keys, ages)
    static = np.r_[np.array(("steady", "balanced", "jackpot")) == machine_type,
                   np.array((-1, 0, 1)) == signal]
    age_feature = np.log1p(ages) / np.log1p(SUPPORT_CAP)
    features = np.concatenate([shared, matches, shared & (query[:, None, :] == keys),
                               age_feature[..., None], np.broadcast_to(static, (*ages.shape, len(static)))],
                              axis=-1, dtype=np.float32)
    return features, decay_overlap.astype(np.float32)


def _weighted_forecast(weights, wins, baseline):
    mass = weights.sum(axis=1)
    win = (np.sum(weights * wins, axis=1) + SHRINKAGE * baseline) / (mass + SHRINKAGE)
    squared = np.sum(weights ** 2, axis=1)
    effective = np.divide(mass ** 2, squared, out=np.zeros_like(mass), where=squared > 0)
    return win, mass, effective


def retrieve(query, keys, wins, ages, baseline):
    return _weighted_forecast(retrieval_weights(query, keys, ages), wins, baseline)


def _machine_context(spins, lags, endpoints, prior):
    observed_ticks = np.flatnonzero(spins != MISSING)
    observed_values = spins[observed_ticks]
    ages = endpoints[:, None] - observed_ticks - 1
    decay = np.exp2(-np.maximum(ages, 0) / PRIOR_HALF_LIFE) * (ages >= 0)
    counts = PRIOR_STRENGTH * prior + decay @ (observed_values[:, None] == np.arange(len(PAYOUTS)))
    recency = counts / counts.sum(axis=1, keepdims=True)
    support = causal_support(observed_ticks, observed_values, lags, endpoints)
    recent = np.count_nonzero((ages >= 0) & (ages < SUPPORT_CAP), axis=1)
    return recency, support, recent


def machine_rows(spins, lags, endpoints, prior, *, similarity=None, context=None):
    recency, (keys, wins, ages), recent = _machine_context(spins, lags, endpoints, prior)
    sizes = recency[:, 1:] / recency[:, 1:].sum(axis=1, keepdims=True)
    if similarity is None:
        weights = retrieval_weights(lags[endpoints], keys, ages)
    else:
        features, decay_overlap = _similarity_inputs(lags[endpoints], keys, ages, *context)
        weights = decay_overlap * similarity.predict(features)
    win, mass, effective = _weighted_forecast(weights, wins, 1 - recency[:, 0])
    return np.column_stack([1 - win, win[:, None] * sizes]), recency, mass, effective, recent


class LearnedSimilarity:
    """Two dense layers trained through the normalized retrieval forecast."""

    def __init__(self, seed):
        self.seed = seed

    def _forward(self, features):
        hidden_weights, hidden_bias, output_weights, output_bias = self.parameters
        hidden = np.tanh(features @ hidden_weights + hidden_bias)
        return hidden, expit(hidden @ output_weights + output_bias)

    def predict(self, features):
        return self._forward(features)[1]

    def _gradients(self, features, decay_overlap, support_wins, baseline, outcome):
        hidden, similarity = self._forward(features)
        weights = decay_overlap * similarity
        denominator = weights.sum(axis=1) + SHRINKAGE
        win = ((weights * support_wins).sum(axis=1) + SHRINKAGE * baseline) / denominator
        d_loss = (win - outcome) / (len(win) * win * (1 - win))
        d_logit = ((d_loss / denominator)[:, None] * (support_wins - win[:, None])
                   * weights * (1 - similarity)).astype(np.float32)
        hidden_gradient = d_logit[..., None] * self.parameters[2] * (1 - hidden ** 2)
        return (features.reshape(-1, features.shape[-1]).T @ hidden_gradient.reshape(-1, hidden.shape[-1]),
                hidden_gradient.sum(axis=(0, 1)), hidden.reshape(-1, hidden.shape[-1]).T @ d_logit.ravel(),
                np.array(d_logit.sum()))

    def fit(self, features, decay_overlap, support_wins, baseline, outcomes):
        rng = np.random.default_rng(self.seed)
        n_features = features.shape[-1]
        self.parameters = [rng.normal(0, n_features ** -.5, (n_features, SIMILARITY_WIDTH)).astype(np.float32),
                           np.zeros(SIMILARITY_WIDTH, dtype=np.float32),
                           rng.normal(0, SIMILARITY_WIDTH ** -.5, SIMILARITY_WIDTH).astype(np.float32),
                           np.zeros((), dtype=np.float32)]
        first_moments = [np.zeros_like(parameter) for parameter in self.parameters]
        second_moments = [np.zeros_like(parameter) for parameter in self.parameters]
        step = 0
        for _ in range(SIMILARITY_EPOCHS):
            order = rng.permutation(len(outcomes))
            for start in range(0, len(order), SIMILARITY_BATCH):
                batch = order[start:start + SIMILARITY_BATCH]
                step += 1
                gradients = self._gradients(features[batch], decay_overlap[batch], support_wins[batch],
                                            baseline[batch], outcomes[batch])
                for parameter, first, second, gradient in zip(self.parameters, first_moments, second_moments, gradients):
                    first[...] = .9 * first + .1 * gradient
                    second[...] = .999 * second + .001 * gradient ** 2
                    parameter[...] -= (SIMILARITY_STEP * first / (1 - .9 ** step)
                                       / (np.sqrt(second / (1 - .999 ** step)) + 1e-8))
        return self


class TransitionRetrievalForecaster:
    def __init__(self):
        self.prior = FrequencyForecaster("context")
        self.similarity = None

    def fit(self, queries, history, labels=None):
        self.prior.fit(queries, history, labels)
        self.similarity = None
        return self

    def _endpoint_batches(self, queries, history, historical_endpoints):
        if self.prior.global_prior is None:
            raise ValidationError("Fit the retrieval model before predicting.")
        spins, cutoffs = tick_arrays(queries, history)
        lags = lagged_spins(spins)
        for i, (query, cutoff) in enumerate(zip(queries, cutoffs)):
            prior = np.array(self.prior.context_priors.get(self.prior._key(query), self.prior.global_prior))
            observed = np.flatnonzero(spins[i] != MISSING)
            sample = observed[max(0, len(observed) - historical_endpoints):]
            endpoints = np.r_[sample, cutoff]
            yield i, query, spins[i], lags[i], endpoints, prior

    def fit_similarity(self, queries, history, labels, *, seed):
        batches = []
        for _, query, spins, lags, endpoints, prior in self._endpoint_batches(tuple(queries), history, LEARNED_HISTORY):
            recency, (keys, wins, ages), _ = _machine_context(spins, lags, endpoints, prior)
            features, factors = _similarity_inputs(lags[endpoints], keys, ages, query.machine_type, query.signal)
            outcomes = np.r_[spins[endpoints[:-1]] > 0, labels[query.query_id] > 0]
            batches.append((features, factors, wins, 1 - recency[:, 0], outcomes))
        self.similarity = LearnedSimilarity(seed).fit(*map(np.concatenate, zip(*batches)))
        return self

    def predict_rows(self, queries, history, *, historical_endpoints=0):
        rows, machines, all_ticks, outcomes = [], [], [], []
        for i, query, spins, lags, endpoints, prior in self._endpoint_batches(tuple(queries), history, historical_endpoints):
            rows.append(machine_rows(spins, lags, endpoints, prior, similarity=self.similarity,
                                     context=(query.machine_type, query.signal)))
            machines.extend([i] * len(endpoints))
            all_ticks.extend(endpoints)
            outcomes.extend(spins[endpoints])
        return RetrievalRows(np.array(machines), np.array(all_ticks), np.array(outcomes),
                             *(np.concatenate(values) for values in zip(*rows)))

    def predict(self, queries, history):
        queries = tuple(queries)
        rows = self.predict_rows(queries, history)
        return {query.query_id: tuple(row) for query, row in zip(queries, rows.prediction)}
