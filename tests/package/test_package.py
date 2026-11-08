"""Organizer-facing integration and failure-mode checks."""
import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from tdx.data import load_round
from tdx.package.build import build_entry
from tdx.package import blend, predict
from tdx.package.repro import compare_csv
from tdx.paths import ROOT


@pytest.fixture(scope='module')
def packaged(tmp_path_factory):
    folder = tmp_path_factory.mktemp('entry')
    result = build_entry(out=folder / 'entry', zip_path=folder / 'entry.zip')
    return result


def test_entry_root_precision_and_no_unneeded_modules(packaged):
    with zipfile.ZipFile(packaged['zip']) as archive:
        names = set(archive.namelist())
        assert {'README.md', 'reproduce.json', 'predict.py', 'requirements.txt',
                'blend.json', 'config.json', 'models/pooled/pooled_recent/model.json'} <= names
        assert not any('torch' in n or 'tdx/run.py' in n or '__pycache__' in n for n in names)
        assert archive.read('requirements.txt').decode() == f'numpy=={np.__version__}\n'
    compare_csv(packaged['csv'], packaged['csv'], ROOT / 'runs/pooled/pooled_recent/test_pred.csv')


def test_inference_reads_only_test_and_no_repo_cache(packaged, tmp_path, monkeypatch):
    calls = []
    original = predict._read_split
    def read(root, name):
        calls.append(name)
        assert name == 'test'
        return original(root, name)
    monkeypatch.setattr(predict, '_read_split', read)
    predict.run(packaged['entry'], ROOT / 'data/main', tmp_path / 'out.csv')
    assert calls == ['test']
    compare_csv(packaged['csv'], tmp_path / 'out.csv', packaged['csv'])
    assert not (Path(packaged['entry']) / '.cache').exists()


def test_failed_model_falls_back_and_is_reported(packaged, tmp_path, monkeypatch, capsys):
    import shutil
    root = tmp_path / 'broken'
    shutil.copytree(packaged['entry'], root)
    (root / 'models/pooled/pooled_recent/model.json').write_text('{}')
    sentinel = tmp_path / 'fallback.json'
    def record(errors, selected):
        sentinel.write_text(json.dumps(dict(errors=errors, selected=selected)))
    monkeypatch.setattr(predict, 'fallback_record', record)
    predict.run(root, ROOT / 'data/main', tmp_path / 'fallback.csv')
    value = json.loads(sentinel.read_text())
    assert value['selected'] == 'pooled_recent'
    assert value['errors'][0]['entry'] == 'primary'
    assert 'Primary forecast failed' in capsys.readouterr().err
    # Fallback vs primary is an upload-tolerance comparison. Exact equality is
    # required for two repetitions of the same path, not different reductions.
    compare_csv(tmp_path / 'fallback.csv', tmp_path / 'fallback.csv', packaged['csv'])


def test_wrong_round_fails_before_forecast(packaged, tmp_path):
    import shutil
    data = tmp_path / 'data'
    data.mkdir()
    meta = json.loads((ROOT / 'data/main/metadata.json').read_text())
    meta['round'] = 'adaptation'
    (data / 'metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='another round'):
        predict.run(packaged['entry'], data, tmp_path / 'out.csv')
    assert not (tmp_path / 'out.csv').exists()


def test_builder_preserves_unowned_existing_folder(tmp_path):
    entry = tmp_path / 'entry'
    entry.mkdir()
    user_file = entry / 'important.txt'
    user_file.write_text('keep')
    with pytest.raises(ValueError, match='Refusing to replace'):
        build_entry(out=entry, zip_path=tmp_path / 'entry.zip')
    assert user_file.read_text() == 'keep'


def test_builder_rejects_dev_export_and_preserves_zip(tmp_path):
    archive = tmp_path / 'entry.zip'
    archive.write_bytes(b'old artifact')
    with pytest.raises(ValueError, match='final-mode exports'):
        build_entry(arms='pooled/pooled_recent', out=tmp_path / 'entry', zip_path=archive)
    assert archive.read_bytes() == b'old artifact'


def test_no_torch_assertion(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch.fake', object())
    with pytest.raises(RuntimeError, match='PyTorch imported'):
        predict.no_torch()


def test_dry_run_does_not_load_data_or_make_output(tmp_path, capsys):
    from tdx.pipeline import pipeline
    config = tmp_path / 'pipeline.json'
    config.write_text(json.dumps({'arms': [{'arm': 'pooled', 'variant': 'recent'}]}))
    out = tmp_path / 'output'
    pipeline(round_name='adaptation', data=tmp_path / 'missing', config=config, out=out, dry_run=True)
    text = capsys.readouterr().out
    assert 'dev pooled/recent' in text and 'final pooled/recent' in text
    assert 'pooled/pooled_recent' in text and 'entry_adaptation.zip' in text
    assert not out.exists()


@pytest.mark.parametrize('config', [
    {'weights': [1.1, -0.1]}, {'weights': [0.1, 0.1]},
    {'weights': [[0.5, 0.5]]}, {'weights': [float('nan'), 1]},
    {'arms': ['b/y', 'a/x'], 'weights': [0.5, 0.5]},
    {'weights': [0.5, 0.5], 'calibrator': {'weights': []}},
])
def test_bad_blend_rejected(config):
    with pytest.raises(ValueError):
        blend.validate(config, ['a/x', 'b/y'])


def test_condition_routing_uses_observations():
    s = load_round(answers=False).splits['test']
    short = np.all(s.n_obs == 64, axis=1)
    a = np.broadcast_to([0.6, 0.1, 0.1, 0.1, 0.1], (*s.query_ids.shape, 5)).copy()
    b = np.broadcast_to([0.2] * 5, a.shape).copy()
    p = blend.apply({'weights': [[1, 0], [0, 1], [1, 0], [1, 0]]}, [a, b], s)
    np.testing.assert_allclose(p[short], b[short], rtol=0, atol=1e-15)
    np.testing.assert_allclose(p[~short], a[~short], rtol=0, atol=1e-15)
    explicit = blend.apply({'weights': [[1, 0], [0, 1], [1, 0], [1, 0]]}, [a, b], s, blend.posterior(s))
    np.testing.assert_array_equal(p, explicit)


def test_refit_four_similar_pooled_variants():
    rnd = load_round()
    split = rnd.splits['validation']
    predictions = [np.load(ROOT / f'runs/pooled/{name}/val_pred.npy')
                   for name in ('pooled_recent', 'pooled', 'recent', 'context')]
    weights = np.asarray(blend.fit(predictions, split, split.reference))
    assert weights.shape == (4, 4)
    assert np.all(weights >= 0)
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-12, rtol=0)
    fitted = blend.apply({'weights': weights.tolist()}, predictions, split)
    uniform = blend.apply({'weights': [0.25] * 4}, predictions, split)
    def loss(p):
        error = ((p - split.target) ** 2).sum(axis=(1, 2)) / 8
        reference = ((split.reference - split.target) ** 2).sum(axis=(1, 2)) / 8
        return np.mean([error[split.condition == c].mean() / reference[split.condition == c].mean()
                        for c in blend.CONDITIONS])
    assert loss(fitted) <= loss(uniform) + 1e-12


def test_final_blend_learns_predictive_arm():
    rnd = load_round()
    split = rnd.splits['validation']
    good = blend.probabilities(split.target)
    bad = np.broadcast_to([0.2] * 5, good.shape).copy()
    weights = blend.fit([bad, good], split, split.reference)
    result = blend.apply({'weights': weights}, [bad, good], replace(split, target=None, condition=None))
    assert np.max(np.abs(result - good)) < 2e-6


def test_reproduction_rejects_nan_duplicate_and_nonidentical(tmp_path):
    clean = tmp_path / 'clean.csv'
    clean.write_text('query_id,p_0,p_1,p_2,p_5,p_20\na,0.2,0.2,0.2,0.2,0.2\n')
    other = tmp_path / 'other.csv'
    other.write_text(clean.read_text().replace('0.2,0.2', '0.200000000001,0.199999999999', 1))
    with pytest.raises(ValueError, match='exactly equal'):
        compare_csv(clean, other, clean)
    other.write_text(clean.read_text().replace('0.2', 'nan', 1))
    with pytest.raises(ValueError, match='Invalid probabilities'):
        compare_csv(clean, other, clean)
    other.write_text(clean.read_text() + clean.read_text().splitlines()[1] + '\n')
    with pytest.raises(ValueError, match='Duplicate'):
        compare_csv(clean, other, clean)
