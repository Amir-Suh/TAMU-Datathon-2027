"""Machine-local lag experts scored before each historical update."""

import numpy as np

from .baseline import FrequencyForecaster
from .causal import LAGS, LAG_PAIRS, MISSING, lagged_spins, tick_arrays
from .schema import ValidationError


class AdaptiveLagForecaster:
    def __init__(self, *, half_lives=(12, 48), strength=8, weight_rate=48):
        self.half_lives = tuple(half_lives)
        if not self.half_lives or any(h <= 0 for h in self.half_lives):
            raise ValueError("Use positive decay half-lives.")
        if strength <= 0 or weight_rate < 0:
            raise ValueError("Use positive prior strength and nonnegative weight rate.")
        self.strength = strength
        self.weight_rate = weight_rate
        self.prior = FrequencyForecaster("context")

    def fit(self, queries, history, labels=None):
        self.prior.fit(queries, history, labels)
        return self

    def predict(self, queries, history):
        return self.predict_variants(queries, history)["adaptive_lag"]

    def predict_variants(self, queries, history):
        if self.prior.global_prior is None:
            raise ValidationError("Fit the lag model before predicting.")
        queries = tuple(queries)
        spins, cutoffs = tick_arrays(queries, history)
        lagged = lagged_spins(spins)
        binary = np.where(lagged == MISSING, MISSING, lagged > 0)
        pair_a, pair_b = np.array(LAG_PAIRS).T - 1
        a, b = binary[..., pair_a], binary[..., pair_b]
        pairs = np.where((a == MISSING) | (b == MISSING), MISSING, 2 * a + b)
        contexts = np.concatenate([np.zeros_like(binary[..., :1]), binary, pairs], axis=-1)
        priors = np.array([
            self.prior.context_priors.get(self.prior._key(q), self.prior.global_prior)
            for q in queries
        ])

        decays = 2 ** (-1 / np.array(self.half_lives, dtype=float))
        n_decay, n_machine, n_expert = len(decays), len(queries), contexts.shape[-1]
        totals = np.zeros((n_decay, n_machine, n_expert, 4))
        wins = np.zeros_like(totals)
        excess = np.zeros(totals.shape[:3])
        evidence = np.zeros(n_machine)
        payout_counts = self.strength * priors
        d, m, e = np.ogrid[:n_decay, :n_machine, :n_expert]
        one_end = 1 + len(LAGS)
        # Give the no-lag, singleton, and pair families fixed prior mass.
        family_mass = np.r_[.6, np.full(len(LAGS), .2 / len(LAGS)),
                            np.full(len(LAG_PAIRS), .2 / len(LAG_PAIRS))]
        decay_mass = .4 ** np.arange(n_decay)
        fixed_weights = decay_mass[:, None, None] * family_mass[None, None, :]
        fixed_weights /= fixed_weights.sum()
        outputs = {name: np.empty_like(priors) for name in ("lag_no_context", "lag_fixed", "adaptive_lag")}
        outcome_classes = np.arange(priors.shape[-1])

        for tick in range(spins.shape[1]):
            context = contexts[:, tick]
            available = context != MISSING
            slot = np.maximum(context, 0)
            support = totals[d, m, e, slot]
            positive = wins[d, m, e, slot]
            no_lag = (positive[..., 0] + self.strength * (1 - priors[:, 0])) / (
                support[..., 0] + self.strength)
            singles = (positive[..., 1:one_end] + 4 * no_lag[..., None]) / (
                support[..., 1:one_end] + 4)
            singles = np.where(available[:, 1:one_end], singles, no_lag[..., None])
            pair_prior = .5 * no_lag[..., None] + .25 * (singles[..., pair_a] + singles[..., pair_b])
            pair_pred = (positive[..., one_end:] + 6 * pair_prior) / (support[..., one_end:] + 6)
            pair_pred = np.where(available[:, one_end:], pair_pred, no_lag[..., None])
            forecasts = np.concatenate([no_lag[..., None], singles, pair_pred], axis=-1)

            finished = cutoffs == tick
            if finished.any():
                # payout_counts was updated after the last observed tick.
                sizes = payout_counts[finished, 1:]
                sizes = sizes / sizes.sum(axis=1, keepdims=True)
                selected = forecasts[:, finished]
                weights = fixed_weights * np.exp(
                    -self.weight_rate * excess[:, finished] / (evidence[finished, None] + 8))
                mixture = (weights * selected).sum(axis=(0, 2)) / weights.sum(axis=(0, 2))
                for name, win in (
                    ("lag_no_context", no_lag[0, finished]),
                    ("lag_fixed", (fixed_weights * selected).sum(axis=(0, 2))),
                    ("adaptive_lag", mixture),
                ):
                    outputs[name][finished] = np.column_stack([1 - win, win[:, None] * sizes])

            outcome = spins[:, tick]
            observed = outcome != MISSING
            won = outcome > 0
            error = (forecasts - won[:, None]) ** 2
            excess = decays[0] * excess + (error - error[0, :, :1]) * observed[:, None]
            evidence = decays[0] * evidence + observed
            update = observed[:, None] & available
            # After scoring, age the old counts and add this tick's outcome.
            totals *= decays[:, None, None, None]
            wins *= decays[:, None, None, None]
            totals[d, m, e, slot] += update
            wins[d, m, e, slot] += update & won[:, None]
            payout_counts = self.strength * priors + decays[0] * (
                payout_counts - self.strength * priors) + (outcome[:, None] == outcome_classes)

        return {name: {q.query_id: tuple(row) for q, row in zip(queries, probabilities)}
                for name, probabilities in outputs.items()}
