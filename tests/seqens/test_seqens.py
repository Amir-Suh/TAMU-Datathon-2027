import json
import numpy as np
from tdx.arms.seq import build
from tdx.arms.seqens.arm import SeqEnsArm, assemble
from tdx.data import load_round
from tdx.fit import dev_spec
from tdx.run import fit_round


def test_ensemble_matches_seq_arm_and_has_no_torch(tmp_path):
    import sys
    rnd = load_round('data/main', answers=False)
    spec = dev_spec(False)
    dirs = []
    preds = []
    for seed, factor in ((1, False), (2, True)):
        arm = build(h=16, msg=bool(seed == 1), epochs=1, batch=64, seed=seed, factor=factor)
        arm.fit(fit_round(rnd, spec), spec)
        d = tmp_path / f'm{seed}'
        arm.save(d)
        dirs.append(d)
        preds.append(arm.predict(rnd.splits['validation']))
    assemble(dirs, tmp_path / 'ens')
    ens = SeqEnsArm.load(tmp_path / 'ens')
    got = ens.predict(rnd.splits['validation'])
    assert np.allclose(got, np.mean(preds, axis=0), atol=1e-12, rtol=0)
    assert got.shape == (96, 8, 5) and np.allclose(got.sum(-1), 1)
