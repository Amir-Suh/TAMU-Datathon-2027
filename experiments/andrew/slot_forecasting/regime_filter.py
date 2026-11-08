"""Causal finite-state filtering of local win-rate level and lag direction."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit

from .adaptive import AdaptiveLagForecaster
from .causal import MISSING, tick_arrays
from .peer_lags import adaptive_trace


REGIME_STATES = {
    "base": ((1, 0),),
    "level": ((1, 0), (1, -1), (1, 1)),
    "direction": ((1, 0), (1, -1), (1, 1), (0, 0), (-1, 0)),
}
REGIME_PRIORS = {
    "base": (1.,),
    "level": (.6, .2, .2),
    "direction": (.5, .15, .15, .1, .1),
}


@dataclass(frozen=True)
class RegimeConfig:
    family: str = "direction"
    hazard: float = 1 / 64
    window: int = 40

    def __post_init__(self):
        if self.family not in REGIME_STATES:
            raise ValueError("Unknown regime family.")
        if not 0 <= self.hazard <= 1 or not isinstance(self.window, int) or self.window < 0:
            raise ValueError("Use a hazard in [0, 1] and a nonnegative integer window.")


def _state_probabilities(base_win, no_lag_win, states):
    lag_scale, level_shift = np.asarray(states).T
    no_lag_logit = logit(np.clip(no_lag_win, 1e-12, 1 - 1e-12))
    adaptive_logit = logit(np.clip(base_win, 1e-12, 1 - 1e-12))
    lag_delta = adaptive_logit - no_lag_logit
    return expit(no_lag_logit[..., None] + lag_delta[..., None] * lag_scale + level_shift)


def filter_win_windows(base_win, no_lag_win, outcomes, states, prior, hazard):
    """Return pre-outcome win forecasts and state probabilities for aligned windows."""
    state_win = _state_probabilities(base_win, no_lag_win, states)
    prior = np.asarray(prior, dtype=float)
    posterior = np.broadcast_to(prior, (len(outcomes), len(prior)))
    hazards = np.broadcast_to(hazard, outcomes.shape)
    prediction = np.empty(outcomes.shape)
    for tick in range(outcomes.shape[1]):
        h = hazards[:, tick, None]
        pre = (1 - h) * posterior + h * prior
        p = state_win[:, tick]
        prediction[:, tick] = (pre * p).sum(axis=1)
        outcome = outcomes[:, tick, None]
        likelihood = np.where(outcome == MISSING, 1., np.where(outcome > 0, p, 1 - p))
        posterior = pre * likelihood
        posterior /= posterior.sum(axis=1, keepdims=True)
    return prediction, pre


def filter_endpoints(base, no_lag_win, spins, machines, ticks, config, *, hazards=None):
    """Start each endpoint's filter 40 ticks earlier and retain the base payout mix."""
    selected = base[machines, ticks]
    if config.family == "base":
        return selected, np.ones((len(machines), 1))
    raw_ticks = ticks[:, None] + np.arange(-config.window, 1)
    times = np.maximum(raw_ticks, 0)
    row = machines[:, None]
    outcomes = np.where((raw_ticks >= 0) & (raw_ticks < ticks[:, None]), spins[row, times], MISSING)
    h = config.hazard if hazards is None else hazards[row, times]
    win, state = filter_win_windows(1 - base[row, times, 0], no_lag_win[row, times], outcomes,
                                    REGIME_STATES[config.family], REGIME_PRIORS[config.family], h)
    probabilities = np.empty_like(selected)
    probabilities[:, 0] = 1 - win[:, -1]
    positive = selected[:, 1:]
    probabilities[:, 1:] = win[:, -1, None] * positive / positive.sum(axis=1, keepdims=True)
    return probabilities, state


class OnlineRegimeForecaster:
    def __init__(self, config=RegimeConfig(), *, weight_rate=48):
        self.config = config
        self.adaptive = AdaptiveLagForecaster(weight_rate=weight_rate)

    def fit(self, queries, history, labels=None):
        self.adaptive.fit(queries, history, labels)
        return self

    def predict(self, queries, history):
        queries = tuple(queries)
        spins, cutoffs = tick_arrays(queries, history)
        trace = adaptive_trace(queries, spins, self.adaptive)
        probabilities, _ = filter_endpoints(trace.base, trace.no_lag_win, spins,
                                             np.arange(len(queries)), cutoffs, self.config)
        return {q.query_id: tuple(row) for q, row in zip(queries, probabilities)}
