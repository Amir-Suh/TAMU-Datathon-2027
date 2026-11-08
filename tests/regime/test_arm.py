from dataclasses import replace
import numpy as np
import pytest
from tdx.data import load_round
from tdx.fit import dev_spec, final_spec
from tdx.run import fit_round
from tdx.arms.regime import build
from tdx.arms.regime.arm import RegimeArm


@pytest.fixture(scope='module')
def fitted():
    rnd = load_round('data/main', answers=False)
    model = build().fit(fit_round(rnd, dev_spec()), dev_spec())
    return rnd, model


def test_model_roundtrip_deterministic_and_causal(fitted, tmp_path):
    rnd, model = fitted
    split = rnd.splits['validation']
    expected = model.predict(split)
    dirty = split.y.copy()
    for e, cutoff in enumerate(split.cutoff):
        dirty[e, :, cutoff:] = 4
    assert np.array_equal(expected, model.predict(replace(split, y=dirty)))
    model.save(tmp_path / 'model')
    loaded = RegimeArm.load(tmp_path / 'model')
    assert np.array_equal(expected, loaded.predict(split))
    second = build().fit(fit_round(rnd, dev_spec()), dev_spec())
    assert np.array_equal(expected, second.predict(split))
    assert np.array_equal(np.load(tmp_path / 'cond_posterior_val.npy'), model.condition_posterior(split))
    for method in ('d3_changepoint', 'd2_d3_mix'):
        loaded.method = method
        output = loaded.predict(split)
        assert output.dtype == np.float64
        assert output.shape == (96, 8, 5)
        assert np.all((output > 0) & (output < 1))
        assert np.max(np.abs(output.sum(-1) - 1)) < 1e-12


def test_dev_ignores_attached_evaluation_answers(fitted):
    rnd, model = fitted
    # fit receives a Round with poisoned unauthorized answers: direct arm use
    # must still use FitSpec rather than discovering available supervision.
    poison = replace(rnd.splits['validation'], condition=np.full(96, 'invalid'), target=np.full((96, 8, 5), np.nan))
    poisoned = replace(rnd, splits={**rnd.splits, 'validation': poison})
    actual = build().fit(poisoned, dev_spec()).predict(rnd.splits['validation'])
    assert np.array_equal(actual, model.predict(rnd.splits['validation']))
