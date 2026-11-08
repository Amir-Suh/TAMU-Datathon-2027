import glob
import numpy as np
import pytest
from tdx.arms.tfm2.arm import Tfm2Arm
from tdx.arms.tfmens.arm import TfmEnsArm
from tdx.arms.seqens.arm import assemble
from tdx.data import load_round


def test_tfmens_matches_tfm2_and_averages(tmp_path):
    dirs = sorted(glob.glob('runs/tfm2/d128l3_seed_2[12]/model'))
    if len(dirs) < 2:
        pytest.skip('tfm2 exports not present')
    rnd = load_round('data/main', answers=False)
    s = rnd.splits['validation']
    want = np.mean([Tfm2Arm.load(d).predict(s) for d in dirs], axis=0)
    assemble(dirs, tmp_path / 'ens')
    got = TfmEnsArm.load(tmp_path / 'ens').predict(s)
    assert np.allclose(got, want, atol=1e-12, rtol=0)
