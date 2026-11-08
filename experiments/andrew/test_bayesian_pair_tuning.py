"""Leakage and replay checks for lag-pair fine tuning."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from run_bayesian_lag_pair_tuning import _tuning_inputs, evaluate_outer_fold, grouped_folds
from slot_forecasting.bayesian_lag_pairs import lag_pair_rollout
from slot_forecasting.bayesian_pair_tuning import (
    BASE_CONFIG, BASE_HAZARD, EXPANDED_GRID, UNIFORM, Endpoints, TrainingData, WinSettings,
    fit_pair_tuning, joint_predictions,
)
from slot_forecasting.causal import LAG_PAIRS
from slot_forecasting.experiments import episode_folds
from slot_forecasting.io import write_rows
from slot_forecasting.schema import CONDITIONS, HISTORY_COLUMNS, QUERY_COLUMNS, Play, Query
from slot_forecasting.signed_motifs import RECENCY_DECAY, RECENCY_STRENGTH
from test_bayesian_lag_pairs import PRIOR, _signed_simulation


def training_fixture():
    rng = np.random.default_rng(13)
    queries = tuple(Query(f"q-{condition}-{i}", f"e-{condition}-{i}", "m", "steady", 0, 44)
                    for condition in CONDITIONS for i in range(4))
    groups = {q.query_id: c for q, c in zip(queries, np.repeat(CONDITIONS, 4))}
    history = tuple(Play(q.episode_id, q.machine_id, t, int(rng.choice([0, 1, 2, 5, 20])))
                    for q in queries for t in range(q.cutoff))
    labels = {q.query_id: int(rng.choice([0, 1, 2, 5, 20])) for q in queries}
    return TrainingData(queries, history, labels), groups


class PairTuningTest(unittest.TestCase):
    def test_outer_label_and_history_isolation(self):
        data, groups = training_fixture()
        held = grouped_folds(data.queries, groups, 4)[0]
        original, forecasts = evaluate_outer_fold(data, groups, held)
        changed_labels = {q.query_id: (1 if data.labels[q.query_id] == 0 else 0)
                          if q.episode_id in held else data.labels[q.query_id] for q in data.queries}
        changed, label_forecasts = evaluate_outer_fold(replace(data, labels=changed_labels), groups, held)
        self.assertEqual(original["selection"], changed["selection"])
        for name in forecasts:
            np.testing.assert_array_equal(label_forecasts[name], forecasts[name])

        altered_episode = min(held)
        changed_history = tuple(replace(row, payout=0 if row.payout else 20)
                                if row.episode_id == altered_episode else row for row in data.history)
        changed, history_forecasts = evaluate_outer_fold(replace(data, history=changed_history), groups, held)
        for field in ("selection", "population_prior", "learned_pair_prior", "configuration"):
            self.assertEqual(original[field], changed[field])
        held_queries = [q for q in data.queries if q.episode_id in held]
        unaffected = np.array([q.episode_id != altered_episode for q in held_queries])
        for name in forecasts:
            np.testing.assert_array_equal(history_forecasts[name][-len(held):][unaffected],
                                          forecasts[name][-len(held):][unaffected])

    def test_historical_payout_features_are_causal(self):
        data, _ = training_fixture()
        fit = fit_pair_tuning(data)
        endpoints = Endpoints.build(data.queries, data.history, include_history=False)
        endpoints = replace(endpoints, ticks=np.arange(len(data.queries)) + 20)
        settings = WinSettings(.8, 1/144, True)
        baseline, trace = joint_predictions(endpoints, fit, settings)
        changed = endpoints.spins.copy()
        for machine, tick in zip(endpoints.machines, endpoints.ticks):
            changed[machine, tick:] = (changed[machine, tick:] + 1) % 5
        forecast, other_trace = joint_predictions(replace(endpoints, spins=changed), fit, settings)
        for name in baseline:
            np.testing.assert_array_equal(forecast[name], baseline[name])
            np.testing.assert_allclose(forecast[name].sum(axis=1), 1, atol=1e-12)
        for key in ("leading_pair", "leading_pair_amplitude", "leading_pair_support"):
            index = (endpoints.machines, endpoints.ticks)
            np.testing.assert_array_equal(trace[key][index], other_trace[key][index])

    def test_prior_trace_and_absolute_tick_gaps(self):
        prior = np.array([[.55, .31, .09, .04, .01]])
        spins = np.array([[0, 1, 0, 1, 1, 0, 3, 1, 0, 2, -1, -1, -1, -1]], dtype=np.int8)
        params = .2 * UNIFORM
        params[4] += .8
        forecast, details = lag_pair_rollout(spins, prior, BASE_CONFIG, pair_prior=params,
                                             pair_grid=EXPANDED_GRID, record_trace=True)
        trace = details["trace"]
        self.assertEqual(trace["leading_pair"][0, 0], 4)
        self.assertAlmostEqual(trace["pair_mass"][0, 0], .25, places=13)
        np.testing.assert_allclose(trace["leading_pair_amplitude"].sum(axis=2), 1, atol=1e-12)
        self.assertTrue((trace["leading_pair"][0, 10:] == trace["leading_pair"][0, 10]).all())
        for t in range(11, 14):
            self.assertAlmostEqual(trace["pair_mass"][0, t], trace["pair_mass"][0, 10], places=13)
            decay = (1 - BASE_HAZARD) ** (t - 10)
            expected = decay * trace["leading_pair_amplitude"][0, 10] + (1 - decay) * np.array(EXPANDED_GRID[1])
            np.testing.assert_allclose(trace["leading_pair_amplitude"][0, t], expected, atol=1e-12)
            counts = RECENCY_STRENGTH * prior[0].copy()
            for s, value in enumerate(spins[0, :t]):
                if value >= 0:
                    counts[value] += RECENCY_DECAY ** (t - 1 - s)
            positive = forecast[0, t, 1:]
            np.testing.assert_allclose(positive / positive.sum(), counts[1:] / counts[1:].sum(), atol=1e-12)

    def test_history_can_overcome_a_prior_favoring_the_wrong_pair(self):
        spins, _, _ = _signed_simulation(np.full((8, 256), 2.))
        spins = np.pad(spins, ((0, 0), (0, 1)), constant_values=-1)
        priors = np.broadcast_to(PRIOR, (len(spins), len(PRIOR)))
        params = .2 * UNIFORM
        params[LAG_PAIRS.index((1, 3))] += .8
        _, details = lag_pair_rollout(spins, priors, BASE_CONFIG, pair_prior=params,
                                     diagnostic_ticks=np.full(len(spins), 256))
        planted = details["pair_posterior"][:, LAG_PAIRS.index((2, 5))]
        self.assertGreater(np.median(planted), .9)

    def test_prepare_loader_has_no_answer_inputs(self):
        queries = tuple(Query(f"q-{c}-{e}-{m}", f"e-{c}-{e}", f"m{m}", "steady", 0, 40)
                        for c in CONDITIONS for e in range(24) for m in range(8))
        groups = {q.query_id: c for q, c in zip(queries, np.repeat(CONDITIONS, 24 * 8))}
        tuning, _ = episode_folds(queries, groups)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            d = root / "validation"
            write_rows(d / "queries.csv", QUERY_COLUMNS, map(vars, queries))
            write_rows(d / "conditions.csv", ("query_id", "condition"),
                       ({"query_id": key, "condition": c} for key, c in groups.items()))
            write_rows(d / "history.csv", HISTORY_COLUMNS,
                       ({"episode_id": q.episode_id, "machine_id": q.machine_id, "tick": 1,
                         "payout": 0 if q.episode_id in tuning else "invalid-audit-payout"} for q in queries))
            selected, histories, conditions = _tuning_inputs(root)
        self.assertEqual({q.episode_id for q in selected}, tuning)
        self.assertEqual({row.episode_id for row in histories}, tuning)
        self.assertEqual(set(conditions), {q.query_id for q in selected})


if __name__ == "__main__":
    unittest.main()
