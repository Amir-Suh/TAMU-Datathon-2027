import numpy as np
from tdx.arms.seqloss import build
from tdx.arms.seq.arm import forward_np
from tdx.data import load_round
from tdx.fit import dev_spec
from tdx.run import fit_round


def test_history_and_checkpoints(tmp_path):
    rnd = load_round('data/main', answers=False)
    spec = dev_spec(True)
    a = build(h=16, epochs=2, batch=64, seed=3, factor=True, ckpt_every=1)
    a.fit(fit_round(rnd, spec), spec)
    assert len(a.history) == 2 and sorted(a.checkpoints) == [1, 2]
    last = a.checkpoints[2]
    assert all(np.allclose(last[k], a.w[k]) for k in a.w)          # final checkpoint == final weights
    a.save(tmp_path)
    assert (tmp_path / 'history.json').exists() and (tmp_path / 'ckpt_001.npz').exists()
