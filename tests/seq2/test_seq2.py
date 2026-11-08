import numpy as np
import json
import pytest
from tdx.arms.seq2 import build
from tdx.arms.seq2.arm import Seq2Arm
from tdx.data import load_round
from tdx.fit import dev_spec
from tdx.run import fit_round
from tdx.tools.leak_test import assert_step_causality


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
    assert json.loads((tmp_path / 'meta.json').read_text())['flag'] == 'causal_v1'


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


def test_step_flag_is_causal_and_legacy_flagged_export_rejected(tmp_path):
    rng = np.random.RandomState(101)
    H = 16
    y = np.zeros((1, 8, 128), dtype=np.int64)
    y[:, :, 64:] = -1
    mt = np.zeros((1, 8), dtype=np.int64)
    a = lambda *shape: rng.normal(0, .2, shape)
    arm = Seq2Arm(h=H, short_flag=True)
    arm.w = dict(tok=a(6, 8), typ=a(4, 4), flg=a(2, 4), wih=a(3 * H, 16),
                 whh=a(3 * H, H), bih=a(3 * H), bhh=a(3 * H), hw=a(5, H), hb=a(5))
    deltas = assert_step_causality(arm, y, mt, k=32)
    print('seq2', ' '.join(f'{key} {value:.9f}' for key, value in deltas.items()))
    assert all(value == 0.0 for value in deltas.values())

    arm.save(tmp_path)
    meta_path = tmp_path / 'meta.json'
    meta = json.loads(meta_path.read_text())
    del meta['flag']
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='trained with the non-causal flag'):
        Seq2Arm.load(tmp_path)

    safe_dir = tmp_path / 'unflagged_legacy'
    safe_dir.mkdir()
    safe = Seq2Arm(h=H, short_flag=False)
    safe.w = {}
    safe.save(safe_dir)
    safe_meta_path = safe_dir / 'meta.json'
    safe_meta = json.loads(safe_meta_path.read_text())
    del safe_meta['flag']
    safe_meta_path.write_text(json.dumps(safe_meta))
    with pytest.raises(ValueError, match='trained with the non-causal flag'):
        Seq2Arm.load(safe_dir)
