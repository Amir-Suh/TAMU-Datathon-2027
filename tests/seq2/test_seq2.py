import numpy as np
from tdx.arms.seq2 import build
from tdx.arms.seq2.arm import Seq2Arm
from tdx.data import load_round
from tdx.fit import dev_spec
from tdx.run import fit_round


def test_parity_and_roundtrip(tmp_path):
    rnd = load_round('data/main', answers=False)
    arm = build(h=16, short_flag=True, factor=True, epochs=1, batch=64)
    arm.fit(fit_round(rnd, dev_spec(False)), dev_spec(False))
    assert arm.parity < 1e-9
    s = rnd.splits['validation']
    p = arm.predict(s)
    assert p.shape == (96, 8, 5) and np.allclose(p.sum(-1), 1)
    arm.save(tmp_path)
    q = Seq2Arm.load(tmp_path).predict(s)
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


def test_defaults_match_seq():
    rnd = load_round('data/main', answers=False)
    spec = dev_spec(False)
    fitted_rnd = fit_round(rnd, spec)

    from tdx.arms.seq import build as build_seq
    arm_seq = build_seq(h=16, epochs=1, batch=64, seed=42)
    arm_seq.fit(fitted_rnd, spec)
    p_seq = arm_seq.predict(rnd.splits['validation'])

    arm_seq2 = build(h=16, epochs=1, batch=64, seed=42)
    arm_seq2.fit(fitted_rnd, spec)
    p_seq2 = arm_seq2.predict(rnd.splits['validation'])

    diff = float(np.abs(p_seq - p_seq2).max())
    assert diff < 1e-5, f"Expected predictions within 1e-5, got max diff {diff}"
