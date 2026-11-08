import numpy as np
import pytest
from tdx.data import load_round, PAYOUTS
from tdx.fit import dev_spec, final_spec
from tdx.run import fit_round
from tdx.score import score, episode_values
from tdx.arms.pooled import build
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import load_inputs, read_labels


def test_reference_score():
    rnd = load_round()
    s = rnd.splits['validation']
    result = score(rnd, s.reference)
    assert result['points'] == 30.0
    assert set(result['by_condition'].values()) == {1.0}
    assert result['tuning']['points'] == result['audit']['points'] == 30.0
    assert len(episode_values(rnd, s.reference)) == 96


@pytest.mark.parametrize('method', ['pooled', 'pooled_recent', 'recent', 'context'])
def test_kit_parity_and_determinism(method, tmp_path):
    rnd = load_round()
    spec = dev_spec(True)
    model = build(method=method)
    model.fit(fit_round(rnd, spec), spec)
    pred = model.predict(rnd.splits['validation'])
    q, h = load_inputs(rnd.root / 'train')
    kit = FrequencyForecaster(method).fit(q, h, read_labels(rnd.root / 'train/labels.csv', q))
    q, h = load_inputs(rnd.root / 'validation')
    mapped = kit.predict(q, h)
    expected = np.array([mapped[q] for q in rnd.splits['validation'].query_ids.flat]).reshape(96,8,5)
    np.testing.assert_allclose(pred, expected, rtol=0, atol=5e-16)
    model2 = build(method=method)
    model2.fit(fit_round(rnd, spec), spec)
    np.testing.assert_array_equal(pred, model2.predict(rnd.splits['validation']))
    model.save(tmp_path / method)
    loaded = type(model).load(tmp_path / method)
    np.testing.assert_array_equal(pred, loaded.predict(rnd.splits['validation']))
    if method in ('pooled', 'pooled_recent'):
        assert score(rnd, pred)['points'] == pytest.approx(30 if method == 'pooled' else 30.825854, abs=1e-6)


def test_fit_permissions():
    rnd = load_round()
    dev = fit_round(rnd, dev_spec(True))
    assert dev.splits['validation'].label is dev.splits['validation'].target is dev.splits['validation'].condition is None
    assert dev.splits['train'].label is not None
    final = fit_round(rnd, final_spec())
    assert final.splits['validation'].label is not None
    assert final.splits['validation'].target is not None
    assert final.splits['validation'].reference is None


def test_final_execution_scores_nothing(tmp_path, monkeypatch):
    from tdx.run import execute
    from slot_forecasting.io import read_submission
    monkeypatch.setattr('tdx.run.append', lambda *args: None)
    rnd = load_round()
    result = execute(rnd, 'pooled', 'pooled_recent', 'final', out=tmp_path)
    assert result is None
    assert not (tmp_path / 'score.json').exists()
    assert not (tmp_path / 'val_pred.npy').exists()
    assert len(read_submission(tmp_path / 'test_pred.csv')) == 1536
    assert (tmp_path / 'model' / 'model.json').exists()
