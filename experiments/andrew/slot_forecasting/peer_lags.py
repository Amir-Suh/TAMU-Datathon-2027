"""Causal adaptive forecasts and prospective corrections from lagged peers."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit

from .causal import LAG_PAIRS, LAGS, MISSING, lagged_spins
from .schema import PAYOUTS, ValidationError


@dataclass(frozen=True)
class AdaptiveTrace:
    base: np.ndarray
    no_lag_win: np.ndarray
    context_support: np.ndarray
    no_lag_support: np.ndarray
    population_win: np.ndarray


def adaptive_trace(queries, spins, model):
    """Record the adaptive model's forecast at every tick, before that tick's update."""
    if model.prior.global_prior is None:
        raise ValidationError("Fit the adaptive population prior before forecasting.")
    priors = np.array([model.prior.context_priors.get(model.prior._key(q), model.prior.global_prior)
                       for q in queries])
    population_win = 1 - priors[:, 0]
    lagged = lagged_spins(spins)
    binary = np.where(lagged == MISSING, MISSING, lagged > 0)
    pair_a, pair_b = np.array(LAG_PAIRS).T - 1
    a, b = binary[..., pair_a], binary[..., pair_b]
    pairs = np.where((a == MISSING) | (b == MISSING), MISSING, 2 * a + b)
    contexts = np.concatenate([np.zeros_like(binary[..., :1]), binary, pairs], axis=-1)
    decays = 2 ** (-1 / np.array(model.half_lives, dtype=float))
    n_decay, n_machine, n_expert = len(decays), len(queries), contexts.shape[-1]
    totals = np.zeros((n_decay, n_machine, n_expert, 4))
    wins = np.zeros_like(totals)
    excess = np.zeros(totals.shape[:3])
    evidence = np.zeros(n_machine)
    payout_prior = model.strength * priors
    payout_counts = payout_prior.copy()
    d, m, e = np.ogrid[:n_decay, :n_machine, :n_expert]
    one_end = 1 + len(LAGS)
    family_mass = np.r_[.6, np.full(len(LAGS), .2 / len(LAGS)),
                        np.full(len(LAG_PAIRS), .2 / len(LAG_PAIRS))]
    fixed_weights = (.4 ** np.arange(n_decay))[:, None, None] * family_mass[None, None, :]
    fixed_weights /= fixed_weights.sum()
    outcome_classes = np.arange(len(PAYOUTS))
    base = np.empty((*spins.shape, len(PAYOUTS)))
    no_lag_win = np.empty(spins.shape)
    context_support = np.empty(spins.shape)
    no_lag_support = np.empty(spins.shape)

    for tick in range(spins.shape[1]):
        context = contexts[:, tick]
        available = context != MISSING
        slot = np.maximum(context, 0)
        support = totals[d, m, e, slot]
        positive = wins[d, m, e, slot]
        no_lag = (positive[..., 0] + model.strength * population_win) / (support[..., 0] + model.strength)
        singles = (positive[..., 1:one_end] + 4 * no_lag[..., None]) / (support[..., 1:one_end] + 4)
        singles = np.where(available[:, 1:one_end], singles, no_lag[..., None])
        pair_prior = .5 * no_lag[..., None] + .25 * (singles[..., pair_a] + singles[..., pair_b])
        pair_pred = (positive[..., one_end:] + 6 * pair_prior) / (support[..., one_end:] + 6)
        pair_pred = np.where(available[:, one_end:], pair_pred, no_lag[..., None])
        forecasts = np.concatenate([no_lag[..., None], singles, pair_pred], axis=-1)

        weights = fixed_weights * np.exp(-model.weight_rate * excess / (evidence[:, None] + 8))
        mixture = (weights * forecasts).sum(axis=(0, 2)) / weights.sum(axis=(0, 2))
        sizes = payout_counts[:, 1:]
        sizes = sizes / sizes.sum(axis=1, keepdims=True)
        base[:, tick] = np.column_stack([1 - mixture, mixture[:, None] * sizes])
        no_lag_win[:, tick] = no_lag[0]
        no_lag_support[:, tick] = support[0, :, 0]
        context_weights = weights[..., 1:] * available[:, 1:]
        context_support[:, tick] = (context_weights * support[..., 1:]).sum(axis=(0, 2)) / np.maximum(
            context_weights.sum(axis=(0, 2)), 1e-15)

        outcome = spins[:, tick]
        observed = outcome != MISSING
        won = outcome > 0
        error = (forecasts - won[:, None]) ** 2
        excess = decays[0] * excess + (error - error[0, :, :1]) * observed[:, None]
        evidence = decays[0] * evidence + observed
        update = observed[:, None] & available
        totals *= decays[:, None, None, None]
        wins *= decays[:, None, None, None]
        totals[d, m, e, slot] += update
        wins[d, m, e, slot] += update & won[:, None]
        payout_counts = payout_prior + decays[0] * (payout_counts - payout_prior) + (
            outcome[:, None] == outcome_classes)

    return AdaptiveTrace(base, no_lag_win, context_support, no_lag_support, population_win)


def endpoint_rows(spins, cutoffs, *, history_samples=32):
    """Indices of the balanced historical sample followed by all actual cutoffs."""
    observed = spins != MISSING
    reverse_rank = np.cumsum(observed[:, ::-1], axis=1)[:, ::-1]
    machines, ticks = np.nonzero(observed & (reverse_rank <= history_samples))
    return np.r_[machines, np.arange(len(cutoffs))], np.r_[ticks, cutoffs]


def adjust_win(base, correction):
    """Apply a log-odds correction while preserving the positive payout mix."""
    forecast = base.copy()
    changed = correction != 0
    p = 1 - base[changed, 0]
    corrected = expit(logit(np.clip(p, 1e-12, 1 - 1e-12)) + correction[changed])
    sizes = base[changed, 1:]
    forecast[changed, 0] = 1 - corrected
    forecast[changed, 1:] = corrected[:, None] * sizes / sizes.sum(axis=1, keepdims=True)
    return forecast


@dataclass(frozen=True)
class PeerSettings:
    top: int
    shrink: float
    min_gain: float = .002


@dataclass(frozen=True)
class PeerFeatures:
    values: np.ndarray
    observed: np.ndarray
    lags: np.ndarray


def episode_peer_features(queries, spins, population_win, *, shift=11, prior_strength=8):
    """Return peer × lag candidates and pooled type × lag controls for one episode."""
    observed = spins != MISSING
    wins = spins > 0
    prefix_wins = np.cumsum(wins, axis=1) - wins
    prefix_count = np.cumsum(observed, axis=1) - observed
    rate = (prefix_wins + prior_strength * population_win[:, None]) / (prefix_count + prior_strength)
    centered = np.where(observed, wins - rate, 0)
    n_machine, n_tick = spins.shape
    peers = np.nonzero(~np.eye(n_machine, dtype=bool))[1].reshape(n_machine, n_machine - 1)

    def lag_candidates(offset):
        tick = np.arange(n_tick)[:, None] - np.asarray(LAGS) - offset
        indices = np.maximum(tick, 0)
        values = centered[peers[:, :, None, None], indices]
        valid = observed[peers[:, :, None, None], indices] & (tick >= 0)
        return values, valid

    values, valid = lag_candidates(0)
    shifted, shifted_valid = lag_candidates(shift)
    common = valid & shifted_valid
    individual_lags = np.tile(LAGS, n_machine - 1)

    def individual(x, available):
        shape = (n_machine, n_tick, len(individual_lags))
        return PeerFeatures(x.transpose(0, 2, 1, 3).reshape(shape),
                            available.transpose(0, 2, 1, 3).reshape(shape), individual_lags)

    machine_types = np.array([q.machine_type for q in queries])
    pooled_x, pooled_valid = [], []
    for machine_type in np.unique(machine_types):
        of_type = (machine_types[peers] == machine_type)[:, :, None, None]
        contributors = valid & of_type
        count = contributors.sum(axis=1)
        peer_count = of_type.sum(axis=1)
        mean = np.where(contributors, values, 0).sum(axis=1) / np.maximum(count, 1)
        pooled_x.append(mean * np.sqrt(count / np.maximum(peer_count, 1)))
        pooled_valid.append(count > 0)
    return {
        "individual": individual(values, valid),
        "pooled": PeerFeatures(np.concatenate(pooled_x, axis=2), np.concatenate(pooled_valid, axis=2),
                               np.tile(LAGS, len(pooled_x))),
        "aligned_matched": individual(values, common),
        "shifted_matched": individual(shifted, common),
    }


@dataclass(frozen=True)
class PeerEvidence:
    coefficients: np.ndarray
    pairs: np.ndarray
    gain: np.ndarray


def prospective_evidence(features, spins, base_win, *, ridge=2, half_life=24,
                         min_pairs=8, evidence_ticks=12):
    """Estimate each coefficient and its prequential Brier gain before every tick."""
    values, available = features.values, features.observed
    n_machine, n_tick, n_candidates = values.shape
    shape = (n_machine, n_candidates)
    numerator, denominator = np.zeros(shape), np.zeros(shape)
    pairs = np.zeros(shape, dtype=np.int32)
    evaluated = np.zeros_like(pairs)
    gains = np.zeros(shape)
    recent = np.zeros((evidence_ticks, *shape))
    m, c = np.ogrid[:n_machine, :n_candidates]
    coefficients = np.empty_like(values)
    support = np.empty(values.shape, dtype=np.int32)
    gain_trace = np.empty_like(values)
    base_logit = logit(np.clip(base_win, 1e-12, 1 - 1e-12))
    decay = 2 ** (-1 / half_life)
    for tick in range(n_tick):
        x = values[:, tick]
        eligible = available[:, tick] & (pairs >= min_pairs)
        coefficient = np.where(eligible, np.clip(numerator / (ridge + denominator), -2, 2), 0)
        coefficients[:, tick] = coefficient
        support[:, tick] = pairs
        gain_trace[:, tick] = np.where(eligible & (evaluated >= evidence_ticks), gains / evidence_ticks, 0)

        observed = spins[:, tick, None] != MISSING
        y = spins[:, tick, None] > 0
        p = base_win[:, tick, None]
        prediction = expit(base_logit[:, tick, None] + coefficient * x)
        delta = (y - p) ** 2 - (y - prediction) ** 2
        score = eligible & observed
        slot = evaluated % evidence_ticks
        old = recent[slot, m, c]
        gains += np.where(score, delta - old, 0)
        recent[slot, m, c] = np.where(score, delta, old)
        evaluated += score

        update = observed & available[:, tick]
        numerator = decay * numerator + np.where(update, x * (y - p), 0)
        denominator = decay * denominator + np.where(update, x ** 2 * p * (1 - p), 0)
        pairs += update

    return PeerEvidence(coefficients, support, gain_trace)


def selected_correction(features, evidence, machines, ticks, settings):
    """Average the best eligible peer corrections, sharing the last place across ties."""
    gain = evidence.gain[machines, ticks]
    supported = gain > settings.min_gain
    if gain.shape[-1] == 0:
        return np.zeros(len(machines)), np.zeros_like(gain)
    top = min(settings.top, gain.shape[-1])
    ranks = np.where(supported, gain, -np.inf)
    threshold = np.partition(ranks, -top, axis=1)[:, -top, None]
    above = supported & (gain > threshold)
    tied = supported & (gain == threshold)
    tie_weight = (top - above.sum(axis=1, keepdims=True)) / np.maximum(tied.sum(axis=1, keepdims=True), 1)
    weights = above + tied * tie_weight
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1)
    logit_steps = evidence.coefficients[machines, ticks] * features.values[machines, ticks]
    return settings.shrink * (weights * logit_steps).sum(axis=1), weights


def peer_predictions(queries, spins, trace, machines, ticks, settings, *, shift=11):
    """Return each configured comparison and the chosen lag/sign/support at cutoff."""
    episodes = defaultdict(list)
    for i, query in enumerate(queries):
        episodes[query.episode_id].append(i)
    variants = ("individual", "pooled", "aligned_matched", "shifted_matched")
    base = trace.base[machines, ticks]
    predictions = {config: {variant: base.copy() for variant in variants} for config in settings}
    diagnostics = {config: {variant: [] for variant in variants} for config in settings}
    query_cutoffs = np.array([q.cutoff for q in queries])
    for members in episodes.values():
        rows = np.flatnonzero(np.isin(machines, members))
        local = np.empty(len(queries), dtype=int)
        local[members] = np.arange(len(members))
        row_machines, row_ticks = local[machines[rows]], ticks[rows]
        cutoffs = np.flatnonzero(row_ticks == query_cutoffs[machines[rows]])
        episode_spins = spins[members, :query_cutoffs[members].max() + 1]
        features = episode_peer_features([queries[i] for i in members], episode_spins,
                                         trace.population_win[members], shift=shift)
        episode_win = 1 - trace.base[members, :episode_spins.shape[1], 0]
        for variant, variant_features in features.items():
            evidence = prospective_evidence(variant_features, episode_spins, episode_win)
            for config in settings:
                steps, weights = selected_correction(variant_features, evidence, row_machines, row_ticks, config)
                predictions[config][variant][rows] = adjust_win(base[rows], steps)
                for row in cutoffs:
                    m, t = row_machines[row], row_ticks[row]
                    supported = evidence.gain[m, t] > config.min_gain
                    selected = weights[row] > 0
                    diagnostics[config][variant].append({
                        "query_id": queries[machines[rows[row]]].query_id,
                        "matched_candidates": int(variant_features.observed[m, t].sum()),
                        "supported_candidates": int(supported.sum()),
                        "selected_lags": variant_features.lags[selected].tolist(),
                        "selected_signs": np.sign(evidence.coefficients[m, t, selected]).astype(int).tolist(),
                        "selected_pairs": evidence.pairs[m, t, selected].tolist(),
                        "selected_weights": weights[row, selected].tolist(),
                        "logit_step": float(steps[row]),
                    })
    return predictions, diagnostics
