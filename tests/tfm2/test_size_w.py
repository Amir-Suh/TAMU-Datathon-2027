import numpy as np
from tdx.arms.tfm2 import build
from tdx.data import load_round
from tdx.fit import dev_spec
from tdx.run import fit_round

TINY = dict(d=16, L=1, heads=2, epochs=1, batch=64, seed=3, conv=3, alibi=True)


def _fit(**kw):
    rnd = load_round('data/main', answers=False)
    spec = dev_spec(True)
    arm = build(**TINY, **kw)
    arm.fit(fit_round(rnd, spec), spec)
    return rnd, arm


def test_size_w_one_is_the_default_and_two_changes_the_fit(tmp_path):
    rnd, a = _fit()
    _, b = _fit(size_w=1.0)
    _, c = _fit(size_w=2.0)
    s = rnd.splits['validation']
    # GPU attention kernels are not bit-reproducible between two identical fits (differences ~1e-8)
    assert np.allclose(a.predict(s), b.predict(s), atol=1e-6, rtol=0)
    pc = c.predict(s)
    assert not np.allclose(a.predict(s), pc) and np.allclose(pc.sum(-1), 1) and c.parity < 1e-6
    assert len(c.history) == 1 and np.isfinite(c.history[0])
    c.save(tmp_path)
    from tdx.arms.tfm2.arm import Tfm2Arm
    assert np.array_equal(Tfm2Arm.load(tmp_path).predict(s), pc) and (tmp_path / 'history.json').exists()
