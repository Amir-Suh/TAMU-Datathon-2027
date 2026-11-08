import numpy as np
import pytest
from tdx.arms.seq import build as build_seq
from tdx.arms.seqloss import VARIANTS, build
from tdx.data import load_round
from tdx.fit import dev_spec
from tdx.run import fit_round

TINY = dict(h=16, epochs=1, batch=64, seed=3)


def _fit(arm):
    rnd = load_round('data/main', answers=False)
    spec = dev_spec(True)
    arm.fit(fit_round(rnd, spec), spec)
    return rnd, arm


def test_ce_variant_matches_seq_factor():
    rnd, a = _fit(build(loss='ce', factor=True, **TINY))
    _, b = _fit(build_seq(factor=True, **TINY))
    s = rnd.splits['validation']
    # float summation order differs (masked zeros are summed here), which compounds over a fit; 2e-3 still catches a real loss difference
    assert np.abs(a.predict(s) - b.predict(s)).max() < 2e-3


def test_size_w_one_equals_ce():
    rnd, a = _fit(build(size_w=1.0, factor=True, **TINY))
    _, b = _fit(build(loss='ce', factor=True, **TINY))
    s = rnd.splits['validation']
    assert np.allclose(a.predict(s), b.predict(s), atol=1e-6)


@pytest.mark.parametrize('name', sorted(VARIANTS))
def test_every_variant_trains_and_predicts(name):
    params = dict(VARIANTS[name]); params.update(TINY)
    rnd, a = _fit(build(**params))
    p = a.predict(rnd.splits['validation'])
    assert p.shape == (96, 8, 5) and np.isfinite(p).all() and np.allclose(p.sum(-1), 1) and a.parity < 1e-9


def test_size_w_needs_factor():
    with pytest.raises(ValueError):
        build(size_w=2.0, factor=False)


def test_save_load_roundtrip_is_a_plain_seq_export(tmp_path):
    from tdx.arms.seq.arm import SeqArm
    rnd, a = _fit(build(loss='brier', factor=True, **TINY))
    a.save(tmp_path)
    s = rnd.splits['validation']
    assert np.array_equal(a.predict(s), SeqArm.load(tmp_path).predict(s))
