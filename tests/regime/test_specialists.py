from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from tdx.arms.regime.specialists import fit_priors, forecast


def make_split():
    y = np.random.default_rng(4).integers(0, 5, (3, 8, 256), dtype=np.int8)
    y[..., 50:] = -1
    y[0, :, 9:20] = -1
    return SimpleNamespace(y=y, cutoff=np.array([50, 50, 50]), mtype=np.tile(np.arange(8) % 3, (3, 1)))


def test_forecast_is_causal_and_normalised():
    s = make_split()
    priors = fit_priors([s])
    expected = forecast(s, priors)
    s.y[..., 50:] = 4
    actual = forecast(s, priors)
    assert np.array_equal(actual, expected)
    assert np.max(np.abs(actual.sum(-1) - 1)) < 1e-12
    assert actual.dtype == np.float64
    assert np.all((actual > 0) & (actual < 1))


def test_unknown_lag_falls_back_and_postchange_ignores_old_context():
    s = make_split()
    priors = fit_priors([s])
    s.y[..., 49] = -1
    assert np.array_equal(forecast(s, priors), forecast(s, priors, transition_strength=None))
    s = make_split()
    starts = np.full(3, 40)
    expected = forecast(s, priors, starts=starts)
    s.y[..., :40] = 0
    assert np.array_equal(expected, forecast(s, priors, starts=starts))
