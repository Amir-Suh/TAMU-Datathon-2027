"""Causal Bayesian forecasts over signed and categorical lag pairs."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit, logsumexp

from .baseline import FrequencyForecaster
from .causal import LAGS, LAG_PAIRS, MISSING, lagged_spins, tick_arrays
from .signed_motifs import AMPLITUDES, PAYOUT_CLASSES, RECENCY_DECAY, RECENCY_STRENGTH


WIDE_AMPLITUDES = (*AMPLITUDES, -3, 3, -4, 4)
WIDE_MASS = (.20, .20, .20, .20, .075, .075, .025, .025)
FAMILY_MASS = (.50, .25, .25)
SINGLE_END = 1 + len(LAGS) * len(AMPLITUDES)
TABLE_STRENGTH = 8.


@dataclass(frozen=True)
class LagPairConfig:
    family: str = "signed"
    evidence_half_life: float = 48
    hazard: float = 0.
    wide: bool = False

    def __post_init__(self):
        if self.family not in ("signed", "parity_table", "four_cell_table"):
            raise ValueError("Unknown lag-pair family.")
        if self.evidence_half_life is not None and not 0 < self.evidence_half_life < np.inf:
            raise ValueError("Evidence half-life must be positive and finite.")
        if not 0 <= self.hazard <= 1:
            raise ValueError("Reset hazard must be in [0, 1].")
        if self.hazard and self.evidence_half_life is not None:
            raise ValueError("Choose within-pair switching or discounted evidence.")
        if self.family != "signed" and (self.evidence_half_life is not None or self.hazard or self.wide):
            raise ValueError("Conjugate tables use static, undiscounted evidence.")


def _lag_states(spins):
    lags = lagged_spins(spins)
    binary = np.where(lags == MISSING, MISSING, lags > 0).astype(np.int8)
    a, b = np.asarray(LAG_PAIRS).T - 1
    left, right = binary[..., a], binary[..., b]
    known = (left != MISSING) & (right != MISSING)
    joint = np.where(known, 2 * left + right, MISSING)
    return binary, joint


def _log_pair_transition(log_pair, hazard, log_amplitude_prior):
    """Redistribute amplitude mass without changing a pair's total mass."""
    if hazard == 0:
        return log_pair
    reset = logsumexp(log_pair, axis=-1, keepdims=True) + log_amplitude_prior
    if hazard == 1:
        return reset
    return np.logaddexp(np.log1p(-hazard) + log_pair, np.log(hazard) + reset)


def _log_evidence(win, outcome):
    likelihood = np.where(outcome[:, None] > 0, win, 1 - win)
    log_likelihood = np.log(np.maximum(likelihood, 1e-300))
    return np.where(outcome[:, None] != MISSING, log_likelihood - log_likelihood[:, :1], 0.)


def _new_diagnostics(spins, joint, ticks, amplitudes=None):
    if ticks is None:
        return None
    rows = np.arange(len(spins))
    before = np.arange(spins.shape[1])[None, :] < ticks[:, None]
    observed = before & (spins != MISSING)
    diagnostics = {
        "pair_posterior": np.empty((len(rows), len(LAG_PAIRS))),
        "pair_support": ((joint != MISSING) & observed[..., None]).sum(axis=1),
        "available_at_cutoff": joint[rows, ticks] != MISSING,
    }
    if amplitudes is not None:
        diagnostics["amplitudes"] = np.asarray(amplitudes)
        diagnostics["amplitude_posterior"] = np.empty((len(rows), len(amplitudes)))
    return diagnostics


def _emit(forecast, tick, win, weights, payout_counts):
    nonzero = (weights * win).sum(axis=1)
    positive = payout_counts[:, 1:]
    forecast[:, tick, 0] = 1 - nonzero
    forecast[:, tick, 1:] = nonzero[:, None] * (positive / positive.sum(axis=1, keepdims=True))


def _signed_rollout(spins, priors, binary, joint, config, diagnostic_ticks, pair_prior, pair_grid, record_trace):
    amplitudes = np.array(WIDE_AMPLITUDES if config.wide else AMPLITUDES)
    amplitude_mass = np.array(WIDE_MASS if config.wide else [1 / len(amplitudes)] * len(amplitudes))
    if pair_grid is not None:
        amplitudes, amplitude_mass = (np.asarray(part) for part in pair_grid)
        if (amplitudes.ndim != 1 or len(amplitudes) == 0 or amplitudes.shape != amplitude_mass.shape
                or not np.isfinite(amplitudes).all() or len(np.unique(amplitudes)) != len(amplitudes)
                or np.any(amplitude_mass <= 0) or not np.isclose(amplitude_mass.sum(), 1, rtol=0, atol=1e-12)):
            raise ValueError("Pair grid needs distinct finite amplitudes and positive normalized masses.")
    if pair_prior is not None:
        pair_prior = np.asarray(pair_prior)
        if (pair_prior.shape != (len(LAG_PAIRS),) or np.any(pair_prior <= 0)
                or not np.isclose(pair_prior.sum(), 1, rtol=0, atol=1e-12)):
            raise ValueError("Pair prior needs positive normalized mass for every lag pair.")
    log_amplitude_prior = np.log(amplitude_mass)
    n_single = SINGLE_END - 1
    pair_mass = (np.tile(FAMILY_MASS[2] * amplitude_mass / len(LAG_PAIRS), len(LAG_PAIRS))
                 if pair_prior is None else (FAMILY_MASS[2] * pair_prior[:, None] * amplitude_mass).ravel())
    log_prior = np.log(np.r_[FAMILY_MASS[0], np.full(n_single, FAMILY_MASS[1] / n_single), pair_mass])
    single_sign = np.where(binary == MISSING, 0, 2 * binary - 1)
    pair_sign = np.where(joint == MISSING, 0, np.where((joint == 0) | (joint == 3), 1, -1))
    n = len(spins)
    forecast = np.empty((*spins.shape, len(PAYOUT_CLASSES)))
    diagnostics = _new_diagnostics(spins, joint, diagnostic_ticks, amplitudes)
    if record_trace:
        diagnostics = {} if diagnostics is None else diagnostics
        trace = {
            "amplitudes": amplitudes,
            "leading_pair": np.empty(spins.shape, dtype=np.int8),
            "pair_mass": np.empty(spins.shape),
            "leading_pair_mass": np.empty(spins.shape),
            "pair_entropy": np.empty(spins.shape),
            "leading_pair_support": np.empty(spins.shape, dtype=np.int32),
            "leading_pair_amplitude": np.empty((*spins.shape, len(amplitudes))),
        }
        diagnostics["trace"] = trace
        support = np.zeros((n, len(LAG_PAIRS)), dtype=np.int32)
        machines = np.arange(n)
    log_weights = np.broadcast_to(log_prior, (n, len(log_prior))).copy()
    payout_prior = RECENCY_STRENGTH * priors
    payout_counts = payout_prior.copy()
    half_life = config.evidence_half_life
    decay = 2 ** (-1 / half_life) if half_life is not None else 1.

    for tick, outcome in enumerate(spins.T):
        log_pair = log_weights[:, SINGLE_END:].reshape(n, len(LAG_PAIRS), len(amplitudes))
        if config.hazard:
            log_pair[:] = _log_pair_transition(log_pair, config.hazard, log_amplitude_prior)
        log_weights -= logsumexp(log_weights, axis=1, keepdims=True)
        weights = np.exp(log_weights)
        baseline = payout_counts[:, 1:].sum(axis=1) / payout_counts.sum(axis=1)
        offset = logit(np.clip(baseline, 1e-12, 1 - 1e-12))
        singleton = (single_sign[:, tick, :, None] * AMPLITUDES).reshape(n, -1)
        pair = (pair_sign[:, tick, :, None] * amplitudes).reshape(n, -1)
        contrast = np.concatenate((singleton, pair), axis=1)
        alternatives = np.where(contrast != 0, expit(offset[:, None] + contrast), baseline[:, None])
        win = np.column_stack((baseline, alternatives))
        _emit(forecast, tick, win, weights, payout_counts)
        if diagnostic_ticks is not None:
            at_cutoff = diagnostic_ticks == tick
            pair_weights = weights[at_cutoff, SINGLE_END:].reshape(-1, len(LAG_PAIRS), len(amplitudes))
            diagnostics["pair_posterior"][at_cutoff] = pair_weights.sum(axis=2)
            diagnostics["amplitude_posterior"][at_cutoff] = pair_weights.sum(axis=1)

        if record_trace:
            # Conditional distributions stay normalized even when total pair mass underflows.
            pair_log_mass = logsumexp(log_pair, axis=2)
            top = pair_log_mass.argmax(axis=1)
            total_pair = logsumexp(pair_log_mass, axis=1)
            pair_log_conditional = pair_log_mass - total_pair[:, None]
            trace["leading_pair"][:, tick] = top
            trace["pair_mass"][:, tick] = np.exp(total_pair)
            trace["leading_pair_mass"][:, tick] = np.exp(pair_log_mass[machines, top])
            trace["pair_entropy"][:, tick] = -np.sum(np.exp(pair_log_conditional) * pair_log_conditional, axis=1)
            trace["leading_pair_support"][:, tick] = support[machines, top]
            trace["leading_pair_amplitude"][:, tick] = np.exp(
                log_pair[machines, top] - pair_log_mass[machines, top, None])
            support += (joint[:, tick] != MISSING) & (outcome[:, None] != MISSING)

        relative = _log_evidence(win, outcome)
        if decay != 1:
            log_weights = log_prior + decay * (log_weights - log_prior) + relative
        else:
            log_weights += relative
        payout_counts = payout_prior + RECENCY_DECAY * (payout_counts - payout_prior) + (
            outcome[:, None] == PAYOUT_CLASSES)

    return forecast, diagnostics


def _table_rollout(spins, priors, binary, joint, config, diagnostic_ticks):
    if config.family == "parity_table":
        pair_states = np.where(joint == MISSING, MISSING, (joint == 0) | (joint == 3))
        cells = 2
    else:
        pair_states, cells = joint, 4
    states = np.concatenate((np.zeros_like(binary[..., :1]), binary, pair_states), axis=2)
    n, _, n_hypotheses = states.shape
    pair_start = 1 + len(LAGS)
    log_prior = np.log(np.r_[FAMILY_MASS[0], np.full(len(LAGS), FAMILY_MASS[1] / len(LAGS)),
                             np.full(len(LAG_PAIRS), FAMILY_MASS[2] / len(LAG_PAIRS))])
    log_weights = np.broadcast_to(log_prior, (n, n_hypotheses)).copy()
    totals = np.zeros((n, n_hypotheses, cells))
    wins = np.zeros_like(totals)
    machine, hypothesis = np.ogrid[:n, :n_hypotheses]
    prior_wins = TABLE_STRENGTH * priors[:, 1:].sum(axis=1, keepdims=True)
    forecast = np.empty((*spins.shape, len(PAYOUT_CLASSES)))
    payout_prior = RECENCY_STRENGTH * priors
    payout_counts = payout_prior.copy()
    diagnostics = _new_diagnostics(spins, joint, diagnostic_ticks)

    for tick, outcome in enumerate(spins.T):
        log_weights -= logsumexp(log_weights, axis=1, keepdims=True)
        weights = np.exp(log_weights)
        valid = states[:, tick] != MISSING
        cell = np.maximum(states[:, tick], 0)
        win = (wins[machine, hypothesis, cell] + prior_wins) / (
            totals[machine, hypothesis, cell] + TABLE_STRENGTH)
        win = np.where(valid, win, win[:, :1])
        _emit(forecast, tick, win, weights, payout_counts)
        if diagnostics is not None:
            at_cutoff = diagnostic_ticks == tick
            diagnostics["pair_posterior"][at_cutoff] = weights[at_cutoff, pair_start:]

        log_weights += _log_evidence(win, outcome)
        update = valid & (outcome[:, None] != MISSING)
        totals[machine, hypothesis, cell] += update
        wins[machine, hypothesis, cell] += update & (outcome[:, None] > 0)
        payout_counts = payout_prior + RECENCY_DECAY * (payout_counts - payout_prior) + (
            outcome[:, None] == PAYOUT_CLASSES)

    return forecast, diagnostics


def lag_pair_rollout(spins, priors, config, *, diagnostic_ticks=None, pair_prior=None,
                     pair_grid=None, record_trace=False):
    """Forecast each tick from the observed prefix, then update local state."""
    binary, joint = _lag_states(spins)
    if config.family == "signed":
        return _signed_rollout(spins, priors, binary, joint, config, diagnostic_ticks,
                               pair_prior, pair_grid, record_trace)
    if pair_prior is not None or pair_grid is not None or record_trace:
        raise ValueError("Pair tuning and traces require the signed model.")
    return _table_rollout(spins, priors, binary, joint, config, diagnostic_ticks)


class BayesianLagPairForecaster:
    def __init__(self, config=LagPairConfig()):
        self.config = config
        self.prior = FrequencyForecaster("context")

    def fit(self, queries, history, labels=None):
        self.prior.fit(queries, history, labels)
        return self

    def prior_rows(self, queries):
        queries = tuple(queries)
        priors = self.prior.predict(queries, ())
        return np.array([priors[q.query_id] for q in queries])

    def predict(self, queries, history):
        queries = tuple(queries)
        spins, cutoffs = tick_arrays(queries, history)
        forecast, _ = lag_pair_rollout(spins, self.prior_rows(queries), self.config)
        return {q.query_id: tuple(row)
                for q, row in zip(queries, forecast[np.arange(len(queries)), cutoffs])}
