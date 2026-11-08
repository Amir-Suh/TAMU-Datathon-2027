import subprocess
import sys
from dataclasses import replace
import numpy as np

from tdx.arms.shortctx import build, ShortCtxArm
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
