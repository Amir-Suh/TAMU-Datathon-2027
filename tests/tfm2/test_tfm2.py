import subprocess
import sys
import numpy as np
from dataclasses import replace
from tdx.arms.tfm2 import build, Tfm2Arm
from tdx.arms.tfm2.arm import forward_np, _masked
from tdx.data import load_round
from tdx.fit import dev_spec
from tdx.run import fit_round


def test_numpy_matches_torch_and_roundtrips(tmp_path):
    rnd = load_round('data/main', answers=False)
    arm = build(d=16, L=1, heads=2, epochs=1, batch=64, conv=9, alibi=True)
    arm.fit(fit_round(rnd, dev_spec(False)), dev_spec(False))
    assert arm.parity < 1e-6, f"Parity failed: {arm.parity}"
    s = rnd.splits['validation']
    p = arm.predict(s)
    assert p.shape == (96, 8, 5) and np.allclose(p.sum(-1), 1)
    arm.save(tmp_path)
    q = Tfm2Arm.load(tmp_path).predict(s)
    assert np.array_equal(p, q)


def test_prediction_ignores_ticks_at_or_after_cutoff():
    rnd = load_round('data/main', answers=False)
    arm = build(d=16, L=1, heads=2, epochs=1, batch=64, conv=9, alibi=True)
    arm.fit(fit_round(rnd, dev_spec(False)), dev_spec(False))
    s = rnd.splits['validation']
    y = s.y.copy()
    y[np.broadcast_to(np.arange(y.shape[-1])[None, None, :] >= s.cutoff[:, None, None], y.shape)] = 3
    assert (y != s.y).any()
    assert np.array_equal(arm.predict(s), arm.predict(replace(s, y=y)))


def test_causality():
    rnd = load_round('data/main', answers=False)
    arm = build(d=16, L=1, heads=2, epochs=1, batch=64, conv=9, alibi=True)
    arm.fit(fit_round(rnd, dev_spec(False)), dev_spec(False))
    s = rnd.splits['validation']
    y1 = _masked(s)
    y2 = y1.copy()
    t_pert = 50
    valid = t_pert < s.cutoff[:, None]
    y2[:, :, t_pert] = np.where(valid, (y1[:, :, t_pert] + 1) % 5, y2[:, :, t_pert])

    steps = int(s.cutoff.max())
    out1 = forward_np(arm.w, y1, s.mtype.astype(np.int64), steps,
                      arm.p['d'], arm.p['L'], arm.p['heads'], arm.p['conv'], arm.p['alibi'])
    out2 = forward_np(arm.w, y2, s.mtype.astype(np.int64), steps,
                      arm.p['d'], arm.p['L'], arm.p['heads'], arm.p['conv'], arm.p['alibi'])

    diff_before = np.max(np.abs(out1[:, :, :t_pert] - out2[:, :, :t_pert]))
    assert diff_before == 0.0, f"Causality violated: max diff before tick {t_pert} is {diff_before}"

    diff_at = np.max(np.abs(out1[:, :, t_pert] - out2[:, :, t_pert]))
    assert diff_at > 0.0, "Perturbation did not change output at tick t"


def test_no_torch_on_load_and_predict(tmp_path):
    rnd = load_round('data/main', answers=False)
    arm = build(d=16, L=1, heads=2, epochs=1, batch=64, conv=9, alibi=True)
    arm.fit(fit_round(rnd, dev_spec(False)), dev_spec(False))
    arm.save(tmp_path)

    code = f"""
import sys
from tdx.data import load_round
from tdx.arms.tfm2 import Tfm2Arm

rnd = load_round('data/main', answers=False)
s = rnd.splits['validation']
arm = Tfm2Arm.load({repr(str(tmp_path))})
p = arm.predict(s)
assert p.shape == (96, 8, 5)
assert 'torch' not in sys.modules, f'torch imported: {{[m for m in sys.modules if "torch" in m]}}'
print('OK')
"""
    res = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert 'OK' in res.stdout
