"""Unit tests for moegate4 arm (K=4 mixture of experts with auxiliary condition loss)."""
import subprocess
import sys
import tempfile
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
import json

from tdx.data import load_round
from tdx.paths import data_root
from tdx.fit import dev_spec
from tdx.arms.moegate.arm import MoeGateArm
from tdx.arms.moegate4.arm import MoeGate4Arm
from tdx.tools.leak_test import assert_step_causality


@pytest.fixture(scope="module")
def rnd():
    return load_round(data_root(None, 'main'), 'main', answers=False)


def test_a_k3_ref_equals_moegate(rnd):
    """(a) k3_ref equals the moegate arm for the same seed and tiny settings (within 1e-5)."""
    spec = dev_spec(transductive=False)
    arm1 = MoeGateArm(K=3, h=16, d_expert=8, epochs=2, batch=16, seed=42, transductive=False, device='cpu', smoke=True)
    arm2 = MoeGate4Arm(K=3, lam_cond=0.0, h=16, d_expert=8, epochs=2, batch=16, seed=42, transductive=False, device='cpu', smoke=True)

    arm1.fit(rnd, spec)
    arm2.fit(rnd, spec)

    val_s = rnd.splits['validation']
    p1 = arm1.predict(val_s)
    p2 = arm2.predict(val_s)

    diff = float(np.max(np.abs(p1 - p2)))
    assert diff < 1e-5, f"Diff {diff} >= 1e-5"


def test_b_gates_sum_and_usage(rnd):
    """(b) gate rows sum to 1 and no expert has < 1% usage on a tiny fit."""
    spec = dev_spec(transductive=False)
    arm = MoeGate4Arm(K=4, lam_cond=0.3, h=16, d_expert=8, epochs=2, batch=16, seed=42, transductive=False, device='cpu', smoke=True)
    arm.fit(rnd, spec)

    val_s = rnd.splits['validation']
    u = arm.usage(val_s)
    assert np.allclose(u.sum(axis=-1), 1.0, atol=1e-6)
    mean_u = u.mean(axis=0)
    assert np.all(mean_u >= 0.01), f"Expert usage too low: {mean_u}"


def test_c_predictions_ignore_future(rnd):
    """(c) predictions ignore ticks at or after the cutoff."""
    spec = dev_spec(transductive=False)
    arm = MoeGate4Arm(K=4, lam_cond=0.3, h=16, d_expert=8, epochs=2, batch=16, seed=42, transductive=False, device='cpu', smoke=True)
    arm.fit(rnd, spec)

    val_s = rnd.splits['validation']
    pred1 = arm.predict(val_s)

    dirty_y = val_s.y.copy()
    for e in range(len(val_s.ids)):
        c = int(val_s.cutoff[e])
        dirty_y[e, :, c:] = 4
    dirty_s = replace(val_s, y=dirty_y)
    pred2 = arm.predict(dirty_s)

    diff = float(np.max(np.abs(pred1 - pred2)))
    assert diff == 0.0, f"Future history affected predictions: diff={diff}"


def test_d_parity_and_save_load(rnd):
    """(d) parity + save/load round trip."""
    spec = dev_spec(transductive=False)
    arm = MoeGate4Arm(K=4, lam_cond=0.3, h=16, d_expert=8, epochs=1, batch=16, seed=42, transductive=False, device='cpu')
    arm.fit(rnd, spec)

    assert arm.parity is not None and arm.parity < 1e-9, f"Parity too high: {arm.parity}"

    val_s = rnd.splits['validation']
    pred1 = arm.predict(val_s)

    with tempfile.TemporaryDirectory() as td:
        arm.save(td)
        assert json.loads((Path(td) / 'meta.json').read_text())['flag'] == 'causal_v1'
        loaded = MoeGate4Arm.load(td)
        pred2 = loaded.predict(val_s)

    diff = float(np.max(np.abs(pred1 - pred2)))
    assert diff == 0.0, f"Save/load mismatch: diff={diff}"


def test_g_initialized_weights_are_step_causal_and_float64():
    rng = np.random.RandomState(101)
    H, K, D = 16, 4, 8
    y = np.zeros((1, 8, 128), dtype=np.int64)
    y[:, :, 64:] = -1
    mt = np.zeros((1, 8), dtype=np.int64)
    a = lambda *shape: rng.normal(0, .2, shape)
    w = dict(tok=a(6, 8), typ=a(4, 4), wih=a(3 * H, 12), whh=a(3 * H, H),
             bih=a(3 * H), bhh=a(3 * H), gw=a(K, H + 7), gb=a(K))
    for k in range(K):
        w.update({f'e{k}_w1': a(D, H), f'e{k}_b1': a(D),
                  f'e{k}_w2': a(5, D), f'e{k}_b2': a(5)})
    arm = MoeGate4Arm(K=K, h=H, d_expert=D)
    arm.w = w
    deltas = assert_step_causality(arm, y, mt, k=32)
    print('moegate4', ' '.join(f'{key} {value:.9f}' for key, value in deltas.items()))
    assert all(value == 0.0 for value in deltas.values())


def test_h_legacy_export_rejected(tmp_path):
    arm = MoeGate4Arm(h=16)
    arm.w = {'tok': np.zeros((6, 8))}
    arm.save(tmp_path)
    meta_path = tmp_path / 'meta.json'
    meta = json.loads(meta_path.read_text())
    del meta['flag']
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='trained with the non-causal flag'):
        MoeGate4Arm.load(tmp_path)


def test_e_lam_cond_without_conditions(rnd):
    """(e) with lam_cond>0 and no condition labels (condition None) the fit still runs."""
    spec = dev_spec(transductive=False)
    splits = {name: replace(s, condition=None) for name, s in rnd.splits.items()}
    rnd_nocond = replace(rnd, splits=splits)

    arm = MoeGate4Arm(K=4, lam_cond=0.5, h=16, d_expert=8, epochs=2, batch=16, seed=42, transductive=False, device='cpu', smoke=True)
    arm.fit(rnd_nocond, spec)
    p = arm.predict(rnd_nocond.splits['validation'])
    assert p.shape == (96, 8, 5)
    assert np.all(np.isfinite(p))


def test_f_no_torch_after_load_predict(rnd):
    """(f) 'torch' not in sys.modules after load + predict in a subprocess."""
    spec = dev_spec(transductive=False)
    arm = MoeGate4Arm(K=4, lam_cond=0.3, h=16, d_expert=8, epochs=2, batch=16, seed=42, transductive=False, device='cpu', smoke=True)
    arm.fit(rnd, spec)

    with tempfile.TemporaryDirectory() as td:
        arm.save(td)
        script = f"""
import sys
from tdx.arms.moegate4.arm import MoeGate4Arm
arm = MoeGate4Arm.load({repr(td)})
assert "torch" not in sys.modules, f"torch in modules: {{[m for m in sys.modules if 'torch' in m]}}"
from tdx.data import load_round
from tdx.paths import data_root
rnd = load_round(data_root(None, 'main'), 'main', answers=False)
p = arm.predict(rnd.splits["validation"])
assert "torch" not in sys.modules, f"torch in modules: {{[m for m in sys.modules if 'torch' in m]}}"
assert p.shape == (96, 8, 5)
print("SUCCESS")
"""
        res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
        assert "SUCCESS" in res.stdout
