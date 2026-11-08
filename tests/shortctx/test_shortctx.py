import subprocess
import sys
from dataclasses import replace
import numpy as np

from tdx.arms.shortctx import build, ShortCtxArm
from tdx.arms.shortctx import arm as shortctx_arm
from tdx.arms.shortctx.arm import forward_np
from tdx.arms.shortctx.sampler import sample_layout
from tdx.data import load_round
from tdx.fit import dev_spec
from tdx.run import fit_round


def test_sampler_layout():
    # (a) sampler: exactly 64 observed items, last 40 consecutive ending at cutoff_prime-1,
    # 24 older distinct and < cutoff_prime-40, no tick >= cutoff_prime, deterministic for fixed seed.
    rng_init = np.random.RandomState(0)
    y_row = rng_init.randint(0, 5, size=256).astype(np.int8)

    cutoff_prime = 100
    res = sample_layout(y_row, cutoff_prime, rng=42)
    assert res is not None, "sample_layout returned None for valid input"
    tokens, offsets = res

    # Exactly 64 observed items
    assert tokens.shape == (64,) and tokens.dtype == np.int8
    assert offsets.shape == (64,) and offsets.dtype == np.int16
    assert (tokens >= 0).all() and (tokens <= 4).all()

    # Reconstruct ticks from offsets: offsets = cutoff_prime - tick => tick = cutoff_prime - offsets
    ticks = cutoff_prime - offsets
    assert (ticks < cutoff_prime).all(), "Found tick >= cutoff_prime"
    assert (ticks >= 0).all(), "Found negative tick"

    # The last 40 items are consecutive ticks ending at cutoff_prime - 1
    recent_ticks = ticks[24:]
    assert len(recent_ticks) == 40
    assert recent_ticks[-1] == cutoff_prime - 1
    assert np.array_equal(recent_ticks, np.arange(cutoff_prime - 40, cutoff_prime))
    assert np.all(np.diff(recent_ticks) == 1), "Recent 40 ticks are not consecutive"

    # The 24 older items are distinct and < cutoff_prime - 40
    older_ticks = ticks[:24]
    assert len(older_ticks) == 24
    assert (older_ticks < cutoff_prime - 40).all(), "Older tick >= cutoff_prime - 40"
    assert len(np.unique(older_ticks)) == 24, "Older ticks not distinct"

    # Deterministic for a fixed seed
    tokens2, offsets2 = sample_layout(y_row, cutoff_prime, rng=42)
    assert np.array_equal(tokens, tokens2)
    assert np.array_equal(offsets, offsets2)

    # Different seed draws different older items
    tokens3, offsets3 = sample_layout(y_row, cutoff_prime, rng=999)
    assert not np.array_equal(offsets[:24], offsets3[:24])

    # Returns None when machine does not have 64 observed ticks before cutoff_prime
    # 1. cutoff_prime < 64
    assert sample_layout(y_row, 50, rng=42) is None
    # 2. unobserved tick in recent 40
    y_missing_recent = y_row.copy()
    y_missing_recent[85] = -1
    assert sample_layout(y_missing_recent, cutoff_prime, rng=42) is None
    # 3. fewer than 24 older observed ticks
    y_missing_older = y_row.copy()
    y_missing_older[:40] = -1  # only 20 older observed ticks left (< 60)
    assert sample_layout(y_missing_older, cutoff_prime, rng=42) is None


def test_pad_token_is_masked_and_token_bounds_are_checked():
    rng = np.random.RandomState(12)
    H = 4
    shapes = {
        'tok.weight': (6, H), 'off.weight': (64, H), 'typ.weight': (4, H),
        'flg.weight': (2, H), 'cell.weight_ih': (3 * H, H),
        'cell.weight_hh': (3 * H, H), 'cell.bias_ih': (3 * H,),
        'cell.bias_hh': (3 * H,), 'fc1.weight': (H, 2 * H),
        'fc1.bias': (H,), 'fc2.weight': (5, H), 'fc2.bias': (5,),
    }
    w = {k: rng.normal(0, .2, shape) for k, shape in shapes.items()}
    tokens = rng.randint(0, 5, (1, 64))
    tokens[:, :3] = shortctx_arm.PAD_TOKEN
    tokens[:, 30] = shortctx_arm.PAD_TOKEN
    offsets = np.arange(64, 0, -1, dtype=np.int16)[None]
    mtype = np.array([1])
    flag = np.array([0])

    expected = forward_np(w, tokens, offsets, mtype, flag, H=H)
    changed_offsets = offsets.copy()
    changed_offsets[tokens == shortctx_arm.PAD_TOKEN] = 1000
    actual = forward_np(w, tokens, changed_offsets, mtype, flag, H=H)
    assert np.array_equal(expected, actual)

    jackpots = tokens.copy()
    jackpots[tokens == shortctx_arm.PAD_TOKEN] = 4
    assert not np.array_equal(expected, forward_np(w, jackpots, offsets, mtype, flag, H=H))

    for bad_token in (-1, 6):
        invalid = tokens.copy()
        invalid[0, 0] = bad_token
        with np.testing.assert_raises(ValueError):
            forward_np(w, invalid, offsets, mtype, flag, H=H)


def test_prediction_uses_observed_ticks_and_pads_short_layout(monkeypatch):
    from types import SimpleNamespace

    seen = {}

    def capture(_w, tokens, offsets, mtype, flag, H=64):
        seen['tokens'] = tokens.copy()
        seen['offsets'] = offsets.copy()
        return np.zeros((tokens.shape[0], 5), dtype=np.float64)

    monkeypatch.setattr(shortctx_arm, 'forward_np', capture)
    y = np.full((1, 1, 100), -1, dtype=np.int8)
    obs_ticks = np.array([10, 20, 30, 40, 50, 60, 61, 63, 64, 66, 69, 72, 75, 78, 81, 84, 87])
    y[0, 0, obs_ticks] = np.arange(len(obs_ticks)) % 5
    split = SimpleNamespace(y=y, cutoff=np.array([90]), mtype=np.array([[2]]))
    arm = ShortCtxArm(h=4)
    arm.w = {}
    arm.predict(split)

    tokens = seen['tokens'][0]
    offsets = seen['offsets'][0]
    real = tokens != shortctx_arm.PAD_TOKEN
    assert np.all((tokens[real] >= 0) & (tokens[real] <= 4))
    assert np.count_nonzero(real) == len(obs_ticks)
    assert np.array_equal(np.sort(90 - offsets[real]), obs_ticks)
    assert np.count_nonzero(tokens == shortctx_arm.PAD_TOKEN) == 64 - len(obs_ticks)


def test_predictions_ignore_ticks_at_or_after_cutoff():
    # (b) predictions ignore ticks at or after the cutoff
    rnd = load_round('data/main', answers=False)
    arm = build(d=16, h=16, mlp_h=16, epochs=1, smoke=True)
    arm.fit(fit_round(rnd, dev_spec(False)), dev_spec(False))

    s = rnd.splits['validation']
    y_mod = s.y.copy()
    mask = np.broadcast_to(np.arange(y_mod.shape[-1])[None, None, :] >= s.cutoff[:, None, None], y_mod.shape)
    y_mod[mask] = 3

    assert (y_mod != s.y).any(), "Mask did not alter any values"
    p_orig = arm.predict(s)
    p_mod = arm.predict(replace(s, y=y_mod))
    assert np.array_equal(p_orig, p_mod)


def test_parity_and_save_load_roundtrip(tmp_path):
    # (c) parity + save/load round trip on a tiny model, 1 epoch
    rnd = load_round('data/main', answers=False)
    arm = build(d=16, h=16, mlp_h=16, epochs=1, smoke=True)
    arm.fit(fit_round(rnd, dev_spec(False)), dev_spec(False))

    assert arm.parity is not None
    assert arm.parity < 1e-6, f"Parity check failed: {arm.parity:.3e}"

    s = rnd.splits['validation']
    p = arm.predict(s)
    assert p.shape == (96, 8, 5)
    assert np.allclose(p.sum(axis=-1), 1.0)

    arm.save(tmp_path)
    arm_loaded = ShortCtxArm.load(tmp_path)
    p_loaded = arm_loaded.predict(s)
    assert np.array_equal(p, p_loaded)

    # Legacy exports have only the five payout embeddings. Loading adds the PAD
    # row so sparse observed histories can still be predicted.
    with np.load(tmp_path / 'model.npz') as z:
        legacy_w = {k: z[k] for k in z.files}
    legacy_w['tok.weight'] = legacy_w['tok.weight'][:5]
    np.savez(tmp_path / 'model.npz', **legacy_w)
    arm_legacy = ShortCtxArm.load(tmp_path)
    assert arm_legacy.w['tok.weight'].shape[0] == 6

    from types import SimpleNamespace
    sparse_y = np.full((1, 1, 100), -1, dtype=np.int8)
    sparse_ticks = np.array([10, 20, 30, 40, 50, 60, 70, 80, 85, 89])
    sparse_y[0, 0, sparse_ticks] = np.arange(len(sparse_ticks)) % 5
    sparse_split = SimpleNamespace(y=sparse_y, cutoff=np.array([90]), mtype=np.array([[2]]))
    sparse_p = arm_legacy.predict(sparse_split)
    assert sparse_p.shape == (1, 1, 5)
    assert np.allclose(sparse_p.sum(axis=-1), 1.0)


def test_no_torch_on_load_and_predict(tmp_path):
    # (d) 'torch' not in sys.modules after load + predict in a subprocess
    rnd = load_round('data/main', answers=False)
    arm = build(d=16, h=16, mlp_h=16, epochs=1, smoke=True)
    arm.fit(fit_round(rnd, dev_spec(False)), dev_spec(False))
    arm.save(tmp_path)

    code = f"""
import sys
from tdx.data import load_round
from tdx.arms.shortctx import ShortCtxArm

rnd = load_round('data/main', answers=False)
s = rnd.splits['validation']
arm = ShortCtxArm.load({repr(str(tmp_path))})
p = arm.predict(s)
assert p.shape == (96, 8, 5)
assert 'torch' not in sys.modules, f'torch imported: {{[m for m in sys.modules if "torch" in m]}}'
print('OK')
"""
    res = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert 'OK' in res.stdout
