"""Fit pair priors and predict conditional payout sizes from causal histories."""

from dataclasses import dataclass

import numpy as np

from .bayesian_lag_pairs import BayesianLagPairForecaster, LagPairConfig, lag_pair_rollout
from .causal import LAG_PAIRS, MISSING, feature_rows, lagged_spins, tick_arrays
from .combine_forecasts import conditional_positive, with_positive_head
from .peer_lags import endpoint_rows
from .schema import PAYOUTS
from .win_payout import WinPayoutForecaster, _contexts


BASE_HAZARD = 1 / 96
BASE_CONFIG = LagPairConfig(evidence_half_life=None, hazard=BASE_HAZARD)
EXPANDED_GRID = ((-2, -1, 1, 2, -3, 3), (.225, .225, .225, .225, .05, .05))
UNIFORM = np.full(len(LAG_PAIRS), 1 / len(LAG_PAIRS))
PAYOUT_WEIGHT = 20
PAYOUT_HALF_LIFE = 24
HEADS = ("logistic", "recent", "matched")


@dataclass(frozen=True)
class TrainingData:
    queries: tuple
    history: tuple
    labels: dict


@dataclass(frozen=True)
class WinSettings:
    rho: float = 0.
    hazard: float = BASE_HAZARD
    expanded: bool = False


@dataclass(frozen=True)
class PairTuningFit:
    model: BayesianLagPairForecaster
    prevalence: np.ndarray
    logistic: WinPayoutForecaster

    def prior(self, settings):
        if settings.rho == 0:
            return UNIFORM.copy()
        return (1 - settings.rho) * UNIFORM + settings.rho * self.prevalence

    def rollout(self, queries, spins, settings, *, record_trace=False):
        config = LagPairConfig(evidence_half_life=None, hazard=settings.hazard)
        return lag_pair_rollout(
            spins, self.model.prior_rows(queries), config,
            pair_prior=self.prior(settings) if settings.rho else None,
            pair_grid=EXPANDED_GRID if settings.expanded else None, record_trace=record_trace,
        )


def fit_pair_tuning(train, *, logistic=None):
    """Fit shared state using only the supplied training episodes."""
    queries, history, labels = train.queries, train.history, train.labels
    model = BayesianLagPairForecaster(BASE_CONFIG).fit(queries, history, labels)
    auxiliary = BayesianLagPairForecaster(BASE_CONFIG).fit(queries, history)
    spins, cutoffs = tick_arrays(queries, history)
    _, details = lag_pair_rollout(spins, auxiliary.prior_rows(queries), BASE_CONFIG, diagnostic_ticks=cutoffs)
    mass = details["pair_posterior"].sum(axis=0)
    prevalence = mass / mass.sum() if mass.sum() > 0 else UNIFORM.copy()
    if logistic is None:
        logistic = WinPayoutForecaster(strength=.02, peers=False, positive_prior_weight=.15)
        logistic.fit(queries, history, labels)
    return PairTuningFit(model, prevalence, logistic)


@dataclass(frozen=True)
class Endpoints:
    queries: tuple
    spins: np.ndarray
    cutoffs: np.ndarray
    machines: np.ndarray
    ticks: np.ndarray

    @classmethod
    def build(cls, queries, history, *, include_history):
        spins, cutoffs = tick_arrays(queries, history)
        machines, ticks = (endpoint_rows(spins, cutoffs) if include_history
                           else (np.arange(len(queries)), cutoffs))
        return cls(queries, spins, cutoffs, machines, ticks)

    @property
    def is_cutoff(self):
        return self.ticks == self.cutoffs[self.machines]

    def outcomes(self, labels):
        values = self.spins[self.machines, self.ticks].copy()
        values[self.is_cutoff] = [PAYOUTS.index(labels[q.query_id]) for q in self.queries]
        return values

    def episode_ids(self):
        return np.array([q.episode_id for q in self.queries])[self.machines]

    def positive_prior(self, fit):
        features = feature_rows(self.queries, self.spins, self.machines, self.ticks, peers=False)
        contexts = _contexts(self.queries)[self.machines]
        payout = fit.logistic.predict_rows(features, contexts)
        return conditional_positive(payout, fit.logistic.global_positive_prior)

    def win(self, fit, settings, *, record_trace=False):
        forecasts, diagnostics = fit.rollout(self.queries, self.spins, settings, record_trace=record_trace)
        return 1 - forecasts[self.machines, self.ticks, 0], diagnostics


def payout_heads(endpoints, leading_pair, prior):
    """Smooth the logistic positive mix using observed earlier positive payouts."""
    spins, machines, ticks = endpoints.spins, endpoints.machines, endpoints.ticks
    lagged = lagged_spins(spins)
    pairs = np.asarray(LAG_PAIRS)[leading_pair] - 1
    left, right = pairs.T
    current_left = lagged[machines, ticks, left]
    current_right = lagged[machines, ticks, right]
    current_known = (current_left != MISSING) & (current_right != MISSING)
    current_parity = (current_left > 0) == (current_right > 0)

    time = np.arange(spins.shape[1])
    outcomes = spins[machines]
    age = ticks[:, None] - 1 - time
    weights = 2 ** (-np.maximum(age, 0) / PAYOUT_HALF_LIFE)
    weights *= (age >= 0) & (outcomes > 0)
    past_left = lagged[machines[:, None], time, left[:, None]]
    past_right = lagged[machines[:, None], time, right[:, None]]
    matches = ((past_left > 0) == (past_right > 0)) == current_parity[:, None]
    matches &= (past_left != MISSING) & (past_right != MISSING) & current_known[:, None]
    match_weights = weights * matches

    def counts(local_weights):
        return np.column_stack([(local_weights * (outcomes == k)).sum(axis=1)
                                for k in range(1, len(PAYOUTS))])

    def smooth(values):
        return (PAYOUT_WEIGHT * prior + values) / (PAYOUT_WEIGHT + values.sum(axis=1, keepdims=True))

    recent, matched = counts(weights), counts(match_weights)
    recent_mass = recent.sum(axis=1, keepdims=True)
    matched_mass = matched.sum(axis=1, keepdims=True)
    equal_mass = recent * np.divide(matched_mass, recent_mass, out=np.zeros_like(matched_mass), where=recent_mass > 0)
    return {
        "logistic": prior, "recent": smooth(recent), "matched": smooth(matched),
        "recent_equal_mass": smooth(equal_mass),
    }


def joint_predictions(endpoints, fit, settings):
    win, details = endpoints.win(fit, settings, record_trace=True)
    trace = details["trace"]
    top = trace["leading_pair"][endpoints.machines, endpoints.ticks]
    positive = payout_heads(endpoints, top, endpoints.positive_prior(fit))
    return {name: with_positive_head(1 - win, head) for name, head in positive.items()}, trace
