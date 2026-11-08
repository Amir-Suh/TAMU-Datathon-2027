"""Unit tests for tail_eval and short_backtest tools."""

import numpy as np
import pytest

from tdx.tools.tail_eval import (
    compute_parts,
    compute_metrics,
    paired_bootstrap,
    compute_reliability_pwin,
    compute_reliability_size,
    build_model,
)
from tdx.tools.short_backtest import prepare_short_backtest_round, collect_short_episodes


def test_perfect_oracle_size_part_zero():
    """A perfect-oracle forecast has size part 0 on a one-hot case."""
    # Outcomes: 0 (non-win), 1 (win payout 1), 2 (win payout 2), 3 (win payout 5), 4 (win payout 20)
    outs = np.array([1, 2, 3, 4], dtype=np.int8)  # all wins
    # Perfect one-hot forecasts for the respective win classes
    p = np.zeros((4, 5), dtype=np.float64)
    for i, o in enumerate(outs):
        p[i, o] = 1.0

    wl, sl, tl = compute_parts(p, outs)

    # For one-hot win predictions:
    # pw = 1 - p0 = 1.0, po = 1.0 => po / pw = 1.0 => -log(1.0) = 0.0
    assert np.allclose(sl, 0.0, atol=1e-12), f"Expected size part 0, got {sl}"

    met = compute_metrics(p, outs)
    assert np.isclose(met['size_part'], 0.0, atol=1e-12)
    assert met['n_wins'] == 4
    assert met['n_outcomes'] == 4


def test_win_size_total_decomposition_exact():
    """The decomposition win part + size part equals total loss to 1e-12 for all wins."""
    rng = np.random.RandomState(42)
    # Generate random probability distributions [100, 5]
    logits = rng.randn(100, 5)
    e = np.exp(logits - logits.max(-1, keepdims=True))
    p = e / e.sum(-1, keepdims=True)

    # Generate random outcomes in {0, 1, 2, 3, 4}
    outs = rng.randint(0, 5, size=100).astype(np.int8)

    wl, sl, tl = compute_parts(p, outs)

    # For wins (outs > 0): win part + size part == total loss to 1e-12
    win_mask = outs > 0
    diff_win = np.abs((wl[win_mask] + sl[win_mask]) - tl[win_mask])
    assert np.all(diff_win < 1e-12), f"Max difference on wins: {diff_win.max():.3e}"

    # For non-wins (outs == 0): win part == total loss to 1e-12
    non_win_mask = outs == 0
    diff_non_win = np.abs(wl[non_win_mask] - tl[non_win_mask])
    assert np.all(diff_non_win < 1e-12), f"Max difference on non-wins: {diff_non_win.max():.3e}"


def test_paired_bootstrap_interval_contains_point_estimate():
    """The paired episode bootstrap returns an interval containing the point estimate."""
    rng = np.random.RandomState(123)
    W, E, M = 30, 20, 8
    outs = rng.randint(0, 5, size=(W, E, M)).astype(np.int8)

    # Model a
    logits_a = rng.randn(W, E, M, 5)
    p_a = np.exp(logits_a) / np.exp(logits_a).sum(-1, keepdims=True)

    # Model b (slightly different probabilities)
    logits_b = logits_a + 0.3 * rng.randn(W, E, M, 5)
    p_b = np.exp(logits_b) / np.exp(logits_b).sum(-1, keepdims=True)

    boot = paired_bootstrap(p_a, p_b, outs, B=1000, seed=0)

    # Interval must contain the mean difference
    assert boot['size_ci'][0] <= boot['size_mean'] <= boot['size_ci'][1], (
        f"Size CI [{boot['size_ci'][0]}, {boot['size_ci'][1]}] does not contain mean {boot['size_mean']}"
    )
    assert boot['total_ci'][0] <= boot['total_mean'] <= boot['total_ci'][1], (
        f"Total CI [{boot['total_ci'][0]}, {boot['total_ci'][1]}] does not contain mean {boot['total_mean']}"
    )
    # Interval must have positive width
    assert boot['size_ci'][1] > boot['size_ci'][0]
    assert boot['total_ci'][1] > boot['total_ci'][0]


def test_brier_score_known_cases():
    """Verify Brier score on known exact synthetic cases."""
    outs = np.array([0, 1, 2, 3, 4], dtype=np.int8)
    # Perfect oracle
    p_perfect = np.eye(5)
    met_perf = compute_metrics(p_perfect, outs)
    assert np.isclose(met_perf['brier'], 0.0)

    # Completely wrong (predicted class (o+1)%5 with prob 1.0)
    p_wrong = np.roll(np.eye(5), 1, axis=1)
    met_wrong = compute_metrics(p_wrong, outs)
    # For each sample, (1 - 0)^2 + (0 - 1)^2 = 2.0
    assert np.isclose(met_wrong['brier'], 2.0)


def test_reliability_tables():
    """Verify reliability table bin counts and value ranges."""
    rng = np.random.RandomState(99)
    N = 1000
    p = rng.dirichlet(np.ones(5), size=N)
    outs = rng.randint(0, 5, size=N).astype(np.int8)

    bins_win = compute_reliability_pwin(p, outs, n_bins=10)
    assert len(bins_win) == 10
    total_count_win = sum(b['count'] for b in bins_win)
    assert total_count_win == N
    for b in bins_win:
        assert 0.0 <= b['pred_pwin'] <= 1.0
        assert 0.0 <= b['realized_win_rate'] <= 1.0

    bins_sz = compute_reliability_size(p, outs, n_bins=10)
    assert len(bins_sz) == 10
    total_count_sz = sum(b['count'] for b in bins_sz)
    assert total_count_sz == (outs > 0).sum()
    for b in bins_sz:
        assert 0.0 <= b['pred_1x_share'] <= 1.0
        assert 0.0 <= b['realized_1x_share'] <= 1.0


def test_build_model_factory():
    """Verify build_model builds valid arms and specs for all required names."""
    for name in ('pooled_recent', 'm3', 'gbdt', 'gru_fac', 'gru_plain', 'tfm'):
        arm, spec = build_model(name, seed=1)
        assert arm is not None
        assert spec is not None
