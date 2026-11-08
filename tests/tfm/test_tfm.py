import numpy as np
from tdx.arms.tfm import build
from tdx.arms.tfm.arm import TfmArm, forward_np, _masked
from tdx.data import load_round
from tdx.fit import dev_spec
from tdx.run import fit_round


def test_numpy_matches_torch_and_roundtrips(tmp_path):
    rnd = load_round('data/main', answers=False)
    arm = build(d=16, L=1, heads=2, epochs=1, batch=64, factor=True)
    arm.fit(fit_round(rnd, dev_spec(False)), dev_spec(False))
    assert arm.parity < 1e-6
    s = rnd.splits['validation']
    p = arm.predict(s)
    assert p.shape == (96, 8, 5) and np.allclose(p.sum(-1), 1)
    arm.save(tmp_path)
    q = TfmArm.load(tmp_path).predict(s)
    assert np.array_equal(p, q)


def test_prediction_ignores_ticks_at_or_after_cutoff():
    rnd = load_round('data/main', answers=False)
    arm = build(d=16, L=1, heads=2, epochs=1, batch=64)
    arm.fit(fit_round(rnd, dev_spec(False)), dev_spec(False))
    s = rnd.splits['validation']
    y = s.y.copy()
    y[np.broadcast_to(np.arange(y.shape[-1])[None, None, :] >= s.cutoff[:, None, None], y.shape)] = 3
    assert (y != s.y).any()
    from dataclasses import replace
    assert np.array_equal(arm.predict(s), arm.predict(replace(s, y=y)))


def test_causality():
    rnd = load_round('data/main', answers=False)
    arm = build(d=16, L=2, heads=2, epochs=1, batch=64)
    arm.fit(fit_round(rnd, dev_spec(False)), dev_spec(False))
    s = rnd.splits['validation']
    y1 = _masked(s)
    y2 = y1.copy()
    t_pert = 50
    valid = t_pert < s.cutoff[:, None]
    y2[:, :, t_pert] = np.where(valid, (y1[:, :, t_pert] + 1) % 5, y2[:, :, t_pert])

    steps = int(s.cutoff.max())
    out1 = forward_np(arm.w, y1, s.mtype.astype(np.int64), steps, arm.p['d'], arm.p['L'], arm.p['heads'], arm.p['factor'])
    out2 = forward_np(arm.w, y2, s.mtype.astype(np.int64), steps, arm.p['d'], arm.p['L'], arm.p['heads'], arm.p['factor'])

    diff_before = np.max(np.abs(out1[:, :, :t_pert] - out2[:, :, :t_pert]))
    assert diff_before == 0.0, f"Causality violated: max diff before tick {t_pert} is {diff_before}"

    diff_at = np.max(np.abs(out1[:, :, t_pert] - out2[:, :, t_pert]))
    assert diff_at > 0.0, "Perturbation did not change output at tick t"
