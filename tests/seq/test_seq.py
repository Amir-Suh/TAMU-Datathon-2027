import numpy as np
from tdx.arms.seq import build
from tdx.arms.seq.arm import SeqArm
from tdx.data import load_round
from tdx.fit import dev_spec
from tdx.run import fit_round


def test_numpy_matches_torch_and_roundtrips(tmp_path):
    rnd = load_round('data/main', answers=False)
    arm = build(h=16, msg=True, epochs=1, batch=64)
    arm.fit(fit_round(rnd, dev_spec(False)), dev_spec(False))  # raises if parity > 1e-9
    assert arm.parity < 1e-9
    s = rnd.splits['validation']
    p = arm.predict(s)
    assert p.shape == (96, 8, 5) and np.allclose(p.sum(-1), 1)
    arm.save(tmp_path)
    q = SeqArm.load(tmp_path).predict(s)
    assert np.array_equal(p, q)


def test_prediction_ignores_ticks_at_or_after_cutoff():
    rnd = load_round('data/main', answers=False)
    arm = build(h=8, msg=False, epochs=1, batch=64)
    arm.fit(fit_round(rnd, dev_spec(False)), dev_spec(False))
    s = rnd.splits['validation']
    y = s.y.copy(); y[np.broadcast_to(np.arange(y.shape[-1])[None, None, :] >= s.cutoff[:, None, None], y.shape)] = 3
    assert (y != s.y).any()
    from dataclasses import replace
    assert np.array_equal(arm.predict(s), arm.predict(replace(s, y=y)))
