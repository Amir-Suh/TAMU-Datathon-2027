from dataclasses import replace

import numpy as np

from tdx.arms.glm.factorized import FactorizedGLMArm, objective, probabilities
from tdx.arms.glm import GLMArm, build
from tdx.fit import FitSpec, dev_spec
from test_glm import make_split, rnd_of


def test_factorized_soft_and_hard_gradient():
    rng = np.random.default_rng(910)
    x = rng.normal(size=(12, 4))
    target = rng.dirichlet(np.ones(5), size=12)
    target[:5] = np.eye(5)
    row_weight = rng.uniform(.1, 2, 12)
    center = rng.normal(size=(4, 6))
    flat = rng.normal(size=24)
    args = (x, target, row_weight, 1.7, center)
    _, analytic = objective(flat, *args)
    numeric = np.empty_like(flat)
    for j in range(len(flat)):
        step = np.zeros_like(flat)
        step[j] = 1e-5
        numeric[j] = (objective(flat + step, *args)[0] - objective(flat - step, *args)[0]) / 2e-5
    np.testing.assert_allclose(analytic, numeric, rtol=1e-7, atol=1e-8)


def test_losses_provide_no_size_gradient():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(9, 3))
    target = np.tile([1., 0, 0, 0, 0], (9, 1))
    center = rng.normal(size=(3, 6))
    _, gradient = objective(center.ravel(), x, target, np.ones(9), 10., center)
    assert np.array_equal(gradient.reshape(3, 6)[:, 2:], np.zeros((3, 4)))


def test_factorized_probabilities_save_load_and_causality(tmp_path):
    s = make_split()
    model = build(two_stage=True, pooled_maxiter=60, local_maxiter=40)
    model.fit(rnd_of(s), dev_spec())
    p = model.predict(s)
    assert p.dtype == np.float64
    assert np.all((p > 0) & (p < 1))
    np.testing.assert_allclose(p.sum(-1), 1, atol=1e-15)
    model.save(tmp_path)
    restored = FactorizedGLMArm.load(tmp_path)
    assert np.array_equal(p, restored.predict(s))
    assert isinstance(GLMArm.load(tmp_path), FactorizedGLMArm)
    assert np.array_equal(p, GLMArm.load(tmp_path).predict(s))
    assert restored.fit_diagnostics == model.fit_diagnostics
    y = s.y.copy()
    y[..., int(s.cutoff[0]):] = 4
    assert np.array_equal(p, model.predict(replace(s, y=y)))


def test_factorized_planted_lag2():
    s = make_split(E=8, T=256)
    rng = np.random.default_rng(20261107)
    y = np.empty_like(s.y)
    y[..., :2] = rng.integers(0, 5, y[..., :2].shape)
    size_probability = np.array([.4, .3, .2, .1])
    for t in range(2, 256):
        p0 = np.where(y[..., t - 2] == 0, .9, .1)
        positive = rng.choice(np.arange(1, 5), size=p0.shape, p=size_probability)
        y[..., t] = np.where(rng.random(p0.shape) < p0, 0, positive)
    s = replace(s, y=y, cutoff=np.full(8, 256, dtype=np.int16), label=None)
    model = FactorizedGLMArm(L=2, neighbours=False, rates=False, lambda_=100,
                            pooled_maxiter=200, local_maxiter=80)
    model.fit(rnd_of(s), FitSpec(('train',), (), ()))
    p = model.predict(s)
    planted = np.where(y[..., 254] == 0, .9, .1)
    assert np.max(np.abs(p[..., 0] - planted)) < .1
    expected = np.concatenate((planted[..., None], (1 - planted[..., None]) * size_probability), axis=-1)
    assert np.max(np.abs(p - expected)) < .1


def test_default_dispatch_is_exactly_unchanged():
    s = make_split(E=1, T=24)
    default = build(pooled_maxiter=30, local_maxiter=20)
    explicit = build(two_stage=False, pooled_maxiter=30, local_maxiter=20)
    assert type(default) is GLMArm
    assert default.params == explicit.params
    default.fit(rnd_of(s), dev_spec())
    explicit.fit(rnd_of(s), dev_spec())
    assert np.array_equal(default.W0, explicit.W0)
    assert np.array_equal(default.predict(s), explicit.predict(s))
