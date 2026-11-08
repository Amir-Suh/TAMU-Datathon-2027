from dataclasses import replace
from pathlib import Path

import numpy as np

from tdx.data import Round, Split
from tdx.fit import FitSpec, dev_spec, final_spec
from tdx.arms.glm import GLMArm, build
from tdx.arms.glm.features import all_features, cutoff_features, feature_names
from tdx.arms.glm.model import objective
from tdx.run import build_arm


def make_split(E=2, T=48, seed=761):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 5, (E, 8, T), dtype=np.int8)
    y[..., -4:] = -1
    return Split('train', [f'e{i}' for i in range(E)], y,
                 np.tile(np.arange(8) % 3, (E, 1)).astype(np.int8),
                 np.zeros((E, 8), dtype=np.int8), np.full(E, T - 4, dtype=np.int16),
                 (y >= 0).sum(-1), np.array([[f'q{e}_{m}' for m in range(8)] for e in range(E)], object),
                 label=rng.integers(0, 5, (E, 8), dtype=np.int8))


def rnd_of(s):
    return Round('synthetic', Path('.'), {'train': s}, 60)


def test_shared_harness_device_and_smoke_arguments():
    import tdx.arms.glm as module
    model = build_arm(module, {'L': 2}, device='auto', smoke=False)
    assert isinstance(model, GLMArm)
    assert model.params['L'] == 2
    assert build_arm(module, {}, device='cpu', smoke=True).params == build().params


def test_feature_causality_and_cutoff_storage():
    s = make_split()
    original = all_features(s, L=8)
    rng = np.random.default_rng(12)
    y = s.y.copy()
    y[..., 19:] = rng.integers(0, 5, y[..., 19:].shape)
    changed = all_features(replace(s, y=y), L=8)
    assert np.array_equal(original[..., :20, :], changed[..., :20, :])
    y = s.y.copy()
    y[..., int(s.cutoff[0]):] = 3
    assert np.array_equal(original, all_features(replace(s, y=y), L=8))
    assert np.array_equal(cutoff_features(s, L=8), original[:, :, int(s.cutoff[0])])


def test_missing_neighbours_rates_and_observed_window():
    s = make_split(E=1, T=36)
    y = np.full_like(s.y, -1)
    y[:, :, 0] = 0
    y[:, :, 3] = 3
    y[0, 7, 3] = -1
    s = replace(s, y=y)
    names = feature_names(L=2)
    x = all_features(s, L=2)
    assert x[0, 0, 4, names.index('others_lag1_wins')] == 0
    assert x[0, 0, 4, names.index('others_lag1_missing')] == 1
    assert x[0, 7, 4, names.index('others_lag1_wins')] == 7
    assert x[0, 7, 4, names.index('others_lag1_high')] == 7
    rate0 = x[0, 0, 4, names.index('ema4_class0')]
    assert np.isclose(rate0, np.exp2(-3 / 4) / (1 + np.exp2(-3 / 4)))
    assert x[0, 0, 4, names.index('win_last20_observed')] == .5
    assert x[0, 0, 0, names.index('ema4_class0')] == 0


def test_probabilities_save_load_and_prediction_causality(tmp_path):
    s = make_split()
    model = build(pooled_maxiter=30, local_maxiter=15)
    model.fit(rnd_of(s), dev_spec())
    assert set(model.fit_diagnostics) == {'success', 'status', 'nit', 'objective', 'max_abs_grad', 'message'}
    assert np.isfinite(model.fit_diagnostics['objective'])
    assert np.isfinite(model.fit_diagnostics['max_abs_grad'])
    assert 0 < model.fit_diagnostics['nit'] <= model.params['pooled_maxiter']
    p = model.predict(s)
    assert p.dtype == np.float64
    assert np.all((p > 0) & (p < 1))
    np.testing.assert_allclose(p.sum(-1), 1, atol=1e-15)
    model.save(tmp_path)
    restored = GLMArm.load(tmp_path)
    assert np.array_equal(p, restored.predict(s))
    assert model.fit_diagnostics == restored.fit_diagnostics
    y = s.y.copy()
    y[..., int(s.cutoff[0]):] = 4
    assert np.array_equal(p, model.predict(replace(s, y=y)))


def test_gradient():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(9, 4))
    target = rng.dirichlet(np.ones(5), size=9)
    row_weight = rng.uniform(.1, 1, 9)
    center = rng.normal(size=(4, 5))
    flat = rng.normal(size=20)
    args = (x, target, row_weight, 2.3, center)
    _, analytic = objective(flat, *args)
    numeric = np.empty(20)
    for j in range(20):
        step = np.zeros(20)
        step[j] = 1e-5
        numeric[j] = (objective(flat + step, *args)[0] - objective(flat - step, *args)[0]) / 2e-5
    np.testing.assert_allclose(analytic, numeric, rtol=1e-7, atol=1e-8)


def test_fitspec_hard_and_soft_supervision(monkeypatch):
    s = make_split(E=1, T=12)
    v = replace(s, name='validation', target=np.full((1, 8, 5), .2))
    rnd = Round('synthetic', Path('.'), {'train': s, 'validation': v}, 60)
    captured = []
    def capture(x, y, weights, penalty, center, maxiter, diagnostics=None):
        captured.append((x.copy(), y.copy(), weights.copy()))
        return center.copy()
    monkeypatch.setattr('tdx.arms.glm.arm.fit_weights', capture)
    model = build(w_cut=20)
    model.fit(rnd, FitSpec((), ('train',), ()))
    assert captured[-1][1].shape == (8, 5)
    assert np.all(captured[-1][2] == 20)
    model.fit(rnd, FitSpec((), ('train', 'validation'), ('validation',)))
    assert captured[-1][1].shape == (24, 5)
    assert np.array_equal(captured[-1][1][-8:], v.target.reshape(8, 5))
    assert np.array_equal(captured[-1][1][:8], np.eye(5)[s.label.ravel()])


def test_planted_lag2():
    s = make_split(E=12, T=256)
    rng = np.random.default_rng(20261107)
    y = np.empty_like(s.y)
    y[..., :2] = rng.integers(0, 2, y[..., :2].shape)
    for t in range(2, 256):
        p0 = np.where(y[..., t - 2] == 0, .9, .1)
        y[..., t] = (rng.random(p0.shape) >= p0).astype(np.int8)
    s = replace(s, y=y, cutoff=np.full(12, 256, dtype=np.int16), label=None)
    model = build(L=2, neighbours=False, rates=False, lambda_=100,
                  pooled_maxiter=100, local_maxiter=30)
    model.fit(rnd_of(s), FitSpec(('train',), (), ()))
    p0 = model.predict(s)[..., 0]
    planted = np.where(y[..., 254] == 0, .9, .1)
    assert np.max(np.abs(p0 - planted)) < .1
