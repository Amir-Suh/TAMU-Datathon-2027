"""Checks for prequential Bayesian updates and the tuning boundary."""

from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.special import betaln, expit, logit

from run_model_comparison import tuning_data
from slot_forecasting.bayesian_lag_pairs import (
    FAMILY_MASS, LagPairConfig, _log_pair_transition, lag_pair_rollout,
)
from slot_forecasting.causal import LAGS, LAG_PAIRS, MISSING
from slot_forecasting.experiments import episode_folds
from slot_forecasting.io import write_rows
from slot_forecasting.schema import (
    CONDITIONS, HISTORY_COLUMNS, PROBABILITY_COLUMNS, QUERY_COLUMNS, SUBMISSION_COLUMNS, TARGET_COLUMNS, Query,
)
from slot_forecasting.signed_motifs import MotifConfig, RECENCY_DECAY, motif_rollout

PRIOR = np.array([.55, .31, .09, .04, .01])
STATIC = LagPairConfig(evidence_half_life=None)
SWITCH = LagPairConfig(evidence_half_life=None, hazard=1/32)
PARITY = LagPairConfig("parity_table", None)
FOUR_CELL = LagPairConfig("four_cell_table", None)


def _signed_simulation(amplitudes, n_ticks=256):
    """Generate outcomes with effects on lags (2, 5) and the model's recency offset."""
    rng = np.random.default_rng(271828)
    spins = np.zeros((len(amplitudes), n_ticks), dtype=np.int8)
    wins = np.zeros(len(amplitudes))
    total = 0.
    baseline = np.empty_like(spins, dtype=float)
    parity = np.ones_like(spins)
    for tick in range(n_ticks):
        baseline[:, tick] = (8 * (1 - PRIOR[0]) + wins) / (8 + total)
        if tick >= 5:
            parity[:, tick] = (2 * spins[:, tick - 2] - 1) * (2 * spins[:, tick - 5] - 1)
        effect = amplitudes[:, tick] * parity[:, tick] if tick >= 5 else 0.
        probability = expit(logit(baseline[:, tick]) + effect)
        spins[:, tick] = rng.random(len(amplitudes)) < probability
        wins = RECENCY_DECAY * wins + spins[:, tick]
        total = RECENCY_DECAY * total + 1
    return spins, baseline, parity


def reversal_diagnostics():
    switch_at = 160
    effects = np.full((32, 256), 2.)
    effects[:, switch_at:] = -2.
    spins, baseline, parity = _signed_simulation(effects)
    priors = np.broadcast_to(PRIOR, (len(spins), len(PRIOR)))
    results = {}
    for name, config in (("static", STATIC), ("switch", SWITCH)):
        forecast, _ = lag_pair_rollout(spins, priors, config)
        win = 1 - forecast[:, switch_at:, 0]
        error = np.mean((win - spins[:, switch_at:]) ** 2)
        implied = (logit(win) - logit(baseline[:, switch_at:])) * parity[:, switch_at:]
        correct = implied < -.5
        sustained = correct[:, :-2] & correct[:, 1:-1] & correct[:, 2:]
        delays = [np.flatnonzero(row)[0] for row in sustained if row.any()]
        results[name] = {
            "post_change_brier": float(error), "recovered": len(delays),
            "median_ticks_to_sustained_recovery": float(np.median(delays)) if delays else None,
        }
    return results


class BayesianLagPairTest(unittest.TestCase):
    def test_signed_and_zero_hazard_limits(self):
        rng = np.random.default_rng(5)
        spins = rng.integers(-1, 5, size=(3, 90), dtype=np.int8)
        priors = np.broadcast_to(PRIOR, (3, len(PRIOR)))
        for config in (LagPairConfig(), STATIC):
            expected, _ = motif_rollout(spins, priors, MotifConfig("singleton_pair", config.evidence_half_life))
            observed, _ = lag_pair_rollout(spins, priors, config)
            np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)

    def test_switch_filter_against_exact_state_paths(self):
        hazard, prior = .24, np.array([.3, .7])
        emission = np.array([[.2, .9], [.7, .3], [.1, .6]])
        observed = [1, MISSING, 0]
        transition = (1 - hazard) * np.eye(len(prior)) + hazard * prior
        log_mass = np.log(prior[None, None, :])
        for tick, label in enumerate(observed):
            log_mass = _log_pair_transition(log_mass, hazard, np.log(prior))
            predicted = np.exp(log_mass)[0, 0]
            exact = np.zeros(2)
            for path in product(range(2), repeat=tick + 1):
                weight = prior[path[0]]
                for t in range(tick):
                    if observed[t] != MISSING:
                        w = emission[t, path[t]]
                        weight *= w if observed[t] else 1 - w
                    weight *= transition[path[t], path[t + 1]]
                exact[path[-1]] += weight
            np.testing.assert_allclose(predicted, exact / exact.sum(), rtol=0, atol=1e-14)
            if label != MISSING:
                likelihood = emission[tick] if label else 1 - emission[tick]
                posterior = predicted * likelihood
                log_mass = np.log(posterior[None, None, :] / posterior.sum())
        two_pairs = np.log(np.array([[[.1, .3], [.4, .2]]]))
        predicted = _log_pair_transition(two_pairs, hazard, np.log(prior))
        np.testing.assert_allclose(np.exp(predicted).sum(axis=-1), np.exp(two_pairs).sum(axis=-1))

    def test_table_against_collapsed_beta_evidence(self):
        spins = np.array([[0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0]], dtype=np.int8)
        priors = PRIOR[None, :]
        forecast, _ = lag_pair_rollout(spins, priors, FOUR_CELL)
        alpha, beta = 8 * (1 - PRIOR[0]), 8 * PRIOR[0]
        hypotheses = [(), *((lag,) for lag in LAGS), *LAG_PAIRS]
        prior = np.r_[FAMILY_MASS[0], np.full(len(LAGS), FAMILY_MASS[1] / len(LAGS)),
                      np.full(len(LAG_PAIRS), FAMILY_MASS[2] / len(LAG_PAIRS))]
        counts = [np.zeros((2 ** len(lags), 2)) for lags in hypotheses]
        fallback_evidence = np.zeros(len(hypotheses))
        for tick, y in enumerate(spins[0]):
            likelihood = []
            probabilities = []
            baseline = (alpha + counts[0][0, 1]) / (alpha + beta + counts[0].sum())
            for i, lags in enumerate(hypotheses):
                likelihood.append(fallback_evidence[i] + np.sum(
                    betaln(alpha + counts[i][:, 1], beta + counts[i][:, 0]) - betaln(alpha, beta)))
                if lags and tick < max(lags):
                    probabilities.append(baseline)
                    fallback_evidence[i] += np.log(baseline if y else 1 - baseline)
                    continue
                cell = sum(int(spins[0, tick - lag]) * 2 ** (len(lags) - j - 1) for j, lag in enumerate(lags))
                row = counts[i][cell]
                probabilities.append((alpha + row[1]) / (alpha + beta + row.sum()))
                row[y] += 1
            weights = prior * np.exp(likelihood)
            self.assertAlmostEqual(1 - forecast[0, tick, 0], weights @ probabilities / weights.sum(), places=13)

    def test_future_missing_and_normalization(self):
        rng = np.random.default_rng(7)
        spins = rng.integers(0, 5, size=(3, 100), dtype=np.int8)
        # Match the short-history pattern before a synthetic tick-80 cutoff.
        spins[:, :40] = MISSING
        spins[:, rng.choice(40, size=24, replace=False)] = 0
        priors = np.broadcast_to(PRIOR, (len(spins), len(PRIOR)))
        modified = spins.copy()
        modified[:, 80:] = rng.integers(0, 5, size=(3, 20))
        missing_lag = spins.copy()
        missing_lag[:, 79] = MISSING
        observed_zero = missing_lag.copy()
        observed_zero[:, 79] = 0
        for config in (LagPairConfig(), LagPairConfig(wide=True), SWITCH, PARITY, FOUR_CELL):
            forecast, detail = lag_pair_rollout(spins, priors, config, diagnostic_ticks=np.full(3, 80))
            other, _ = lag_pair_rollout(modified, priors, config)
            np.testing.assert_array_equal(forecast[:, :81], other[:, :81])
            absent, absent_detail = lag_pair_rollout(missing_lag, priors, config, diagnostic_ticks=np.full(3, 80))
            zero, zero_detail = lag_pair_rollout(observed_zero, priors, config, diagnostic_ticks=np.full(3, 80))
            self.assertFalse(np.array_equal(absent[:, 80], zero[:, 80]))
            self.assertGreater(zero_detail["available_at_cutoff"].sum(), absent_detail["available_at_cutoff"].sum())
            self.assertTrue(np.all((forecast >= 0) & (forecast <= 1)))
            np.testing.assert_allclose(forecast.sum(axis=2), 1, rtol=0, atol=1e-14)
            self.assertTrue(detail["available_at_cutoff"].all())
            empty = np.full_like(spins, MISSING)
            no_data, prior = lag_pair_rollout(empty, priors, config, diagnostic_ticks=np.full(3, 80))
            np.testing.assert_allclose(no_data, np.broadcast_to(PRIOR, forecast.shape), rtol=0, atol=1e-14)
            np.testing.assert_allclose(prior["pair_posterior"].sum(axis=1), FAMILY_MASS[2], atol=1e-14)
            self.assertFalse(prior["pair_support"].any())

    def test_seeded_opposite_signs_and_reversal(self):
        effects = np.repeat(np.repeat([[2.], [-2.], [0.]], 8, axis=0), 256, axis=1)
        spins, _, _ = _signed_simulation(effects)
        priors = np.broadcast_to(PRIOR, (len(spins), len(PRIOR)))
        spins = np.pad(spins, ((0, 0), (0, 1)), constant_values=MISSING)
        _, info = lag_pair_rollout(spins, priors, STATIC, diagnostic_ticks=np.full(len(spins), 256))
        index = LAG_PAIRS.index((2, 5))
        planted_mass = info["pair_posterior"][:, index]
        mean_amplitude = info["amplitude_posterior"] @ info["amplitudes"]
        self.assertGreater(np.median(planted_mass[:16]), .9)
        self.assertGreater(np.median(mean_amplitude[:8]), 1.)
        self.assertLess(np.median(mean_amplitude[8:16]), -1.)
        self.assertLess(np.median(info["pair_posterior"][16:].sum(axis=1)), .25)
        reversal = reversal_diagnostics()
        self.assertLess(reversal["switch"]["post_change_brier"], reversal["static"]["post_change_brier"])
        self.assertGreater(reversal["switch"]["recovered"], reversal["static"]["recovered"])

    def test_asymmetric_table(self):
        rng = np.random.default_rng(112)
        spins = np.zeros((24, 256), dtype=np.int8)
        truth = np.empty_like(spins, dtype=float)
        levels = np.array([.11, .86, .19, .45])
        for tick in range(spins.shape[1]):
            truth[:, tick] = levels[2 * spins[:, tick - 2] + spins[:, tick - 5]] if tick >= 5 else .35
            spins[:, tick] = rng.random(len(spins)) < truth[:, tick]
        priors = np.broadcast_to(PRIOR, (len(spins), len(PRIOR)))
        forecasts = [lag_pair_rollout(spins, priors, config)[0] for config in (FOUR_CELL, PARITY)]
        losses = [np.mean((1 - p[:, -80:, 0] - truth[:, -80:]) ** 2) for p in forecasts]
        self.assertLess(*losses)

    def test_tuning_loader_skips_invalid_audit_values(self):
        queries = tuple(Query(f"q-{c}-{i}", f"e-{c}-{i}", "m", "steady", 0, 2)
                        for c in CONDITIONS for i in range(2))
        groups = {q.query_id: c for q, c in zip(queries, np.repeat(CONDITIONS, 2))}
        tuning, _ = episode_folds(queries, groups)
        def value(q, valid):
            return valid if q.episode_id in tuning else "invalid-audit-value"
        with TemporaryDirectory() as directory:
            d = Path(directory)
            write_rows(d / "queries.csv", QUERY_COLUMNS, map(vars, queries))
            write_rows(d / "conditions.csv", ("query_id", "condition"),
                       ({"query_id": key, "condition": c} for key, c in groups.items()))
            write_rows(d / "history.csv", HISTORY_COLUMNS,
                       ({"episode_id": q.episode_id, "machine_id": "m", "tick": 1,
                         "payout": value(q, 0)} for q in queries))
            probs = {q.query_id: {col: value(q, .2) for col in PROBABILITY_COLUMNS} for q in queries}
            write_rows(d / "targets.csv", TARGET_COLUMNS,
                       ({"query_id": q.query_id, "episode_id": q.episode_id,
                         "outcome": value(q, 0), **probs[q.query_id]} for q in queries))
            write_rows(d / "reference.csv", SUBMISSION_COLUMNS,
                       ({"query_id": key, **row} for key, row in probs.items()))
            selected, history, targets, conditions, reference = tuning_data(d)
        self.assertEqual({q.episode_id for q in selected}, tuning)
        self.assertEqual({row.episode_id for row in history}, tuning)
        self.assertEqual({row.episode_id for row in targets}, tuning)
        self.assertEqual(set(reference), set(conditions))


if __name__ == "__main__":
    unittest.main()
