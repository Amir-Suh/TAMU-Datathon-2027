import subprocess
import sys
from pathlib import Path
import numpy as np
import json
from tdx.arms.moegate import build as build_moe
from tdx.arms.moegate.arm import MoeGateArm
from tdx.arms.seq import build as build_seq
from tdx.data import load_round
from tdx.fit import dev_spec
from tdx.run import fit_round
from tdx.tools.leak_test import assert_step_causality
from tdx.arms.moegate.arm import forward_np


def test_a_k1_equals_seq_factored():
    rnd = load_round('data/main', answers=False)
    spec = dev_spec(False)
    seed = 42
    h = 16

    arm_seq = build_seq(h=h, factor=True, epochs=1, batch=64, seed=seed)
    arm_seq.fit(fit_round(rnd, spec), spec)
    p_seq = arm_seq.predict(rnd.splits['validation'])

    arm_moe = build_moe(K=1, d_expert=0, h=h, factor=True, epochs=1, batch=64, seed=seed)
    arm_moe.fit(fit_round(rnd, spec), spec)
    p_moe = arm_moe.predict(rnd.splits['validation'])

    diff = np.abs(p_seq - p_moe).max()
    assert diff < 1e-5, f"K=1 vs seq diff {diff} >= 1e-5"


def test_b_gate_sums_to_1_and_non_degenerate():
    rnd = load_round('data/main', answers=False)
    spec = dev_spec(False)
    arm = build_moe(K=3, d_expert=32, h=16, epochs=2, batch=64, seed=7)
    arm.fit(fit_round(rnd, spec), spec)
    s = rnd.splits['validation']
    u = arm.usage(s)
    assert np.allclose(u.sum(axis=-1), 1.0, atol=1e-6)
    mean_u = u.mean(axis=0)
    for k in range(3):
        assert mean_u[k] > 0.01, f"Expert {k} usage {mean_u[k]} is below 1%"


def test_c_predictions_ignore_ticks_at_or_after_cutoff():
    rnd = load_round('data/main', answers=False)
    spec = dev_spec(False)
    arm = build_moe(K=3, d_expert=32, h=16, epochs=1, batch=64, seed=7)
    arm.fit(fit_round(rnd, spec), spec)
    s = rnd.splits['validation']
    y_corrupt = s.y.copy()
    y_corrupt[np.broadcast_to(np.arange(y_corrupt.shape[-1])[None, None, :] >= s.cutoff[:, None, None], y_corrupt.shape)] = 3
    assert (y_corrupt != s.y).any()
    from dataclasses import replace
    p_orig = arm.predict(s)
    p_corrupt = arm.predict(replace(s, y=y_corrupt))
    assert np.allclose(p_orig, p_corrupt, atol=1e-12)


def test_d_parity_and_roundtrip(tmp_path):
    rnd = load_round('data/main', answers=False)
    spec = dev_spec(False)
    arm = build_moe(K=3, d_expert=32, h=16, epochs=1, batch=64, seed=7)
    arm.fit(fit_round(rnd, spec), spec)
    assert arm.parity < 1e-9
    s = rnd.splits['validation']
    p = arm.predict(s)
    assert p.shape == (96, 8, 5) and np.allclose(p.sum(-1), 1.0)

    save_dir = tmp_path / 'moe_model'
    arm.save(save_dir)
    loaded = MoeGateArm.load(save_dir)
    assert json.loads((save_dir / 'meta.json').read_text())['flag'] == 'causal_v1'
    p_loaded = loaded.predict(s)
    assert np.allclose(p, p_loaded, atol=1e-12)


def test_f_initialized_weights_are_step_causal_and_float64():
    rng = np.random.RandomState(101)
    H, K, D = 16, 3, 8
    y = np.zeros((1, 8, 128), dtype=np.int64)
    y[:, :, 64:] = -1
    mt = np.zeros((1, 8), dtype=np.int64)
    a = lambda *shape: rng.normal(0, .2, shape)
    w = dict(tok=a(6, 8), typ=a(4, 4), wih=a(3 * H, 12), whh=a(3 * H, H),
             bih=a(3 * H), bhh=a(3 * H), gw=a(K, H + 7), gb=a(K))
    for k in range(K):
        w.update({f'e{k}_w1': a(D, H), f'e{k}_b1': a(D),
                  f'e{k}_w2': a(5, D), f'e{k}_b2': a(5)})
    arm = MoeGateArm(K=K, h=H, d_expert=D)
    arm.w = w
    arm.step_outputs(y, mt, 32)
    deltas = assert_step_causality(arm, y, mt, k=32)
    print('moegate', ' '.join(f'{key} {value:.9f}' for key, value in deltas.items()))
    assert all(value == 0.0 for value in deltas.values())


def test_g_legacy_export_rejected(tmp_path):
    arm = MoeGateArm(h=16)
    arm.w = {'tok': np.zeros((6, 8))}
    arm.save(tmp_path)
    meta_path = tmp_path / 'meta.json'
    meta = json.loads(meta_path.read_text())
    del meta['flag']
    meta_path.write_text(json.dumps(meta))
    import pytest
    with pytest.raises(ValueError, match='trained with the non-causal flag'):
        MoeGateArm.load(tmp_path)


def test_e_no_torch_in_subprocess(tmp_path):
    rnd = load_round('data/main', answers=False)
    spec = dev_spec(False)
    arm = build_moe(K=3, d_expert=32, h=16, epochs=1, batch=64, seed=7)
    arm.fit(fit_round(rnd, spec), spec)
    save_dir = tmp_path / 'subp_model'
    arm.save(save_dir)

    code = (
        f"import sys\n"
        f"sys.path.insert(0, 'src')\n"
        f"from tdx.arms.moegate.arm import MoeGateArm\n"
        f"from tdx.data import load_round\n"
        f"arm = MoeGateArm.load(r'{save_dir}')\n"
        f"rnd = load_round('data/main', answers=False)\n"
        f"p = arm.predict(rnd.splits['validation'])\n"
        f"assert p.shape == (96, 8, 5)\n"
        f"torch_mods = [m for m in sys.modules if 'torch' in m]\n"
        f"assert len(torch_mods) == 0, f'Torch modules found: {{torch_mods}}'\n"
        f"print('ALL_PASSED')\n"
    )
    res = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert 'ALL_PASSED' in res.stdout
