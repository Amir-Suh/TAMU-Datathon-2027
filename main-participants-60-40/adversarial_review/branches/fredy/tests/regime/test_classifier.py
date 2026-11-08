import json
from dataclasses import replace

import numpy as np
import pytest

from tdx.data import Split
from tdx.arms.regime.classifier import CONDITIONS, ConditionModel, episode_features, _folds


def sample_split():
    rng = np.random.default_rng(42)
    n = 24
    y = rng.integers(0, 5, (n, 8, 120), dtype=np.int8)
    cutoff = np.full(n, 100)
    conditions = np.array([CONDITIONS[i % 4] for i in range(n)], dtype=object)
    for e in range(n):
        y[e, :, 100:] = -1
        if conditions[e] == 'short_history':
            y[e, :, :36] = -1
    return Split('train', [f'ep_{i}' for i in range(n)], y,
                 np.tile(np.arange(8) % 3, (n, 1)), np.zeros((n, 8)), cutoff,
                 (y >= 0).sum(axis=-1), np.full((n, 8), 'q'), condition=conditions)


def test_history_features_ignore_future_and_answers():
    split = sample_split()
    expected = episode_features(split)
    changed = split.y.copy()
    changed[:, :, 100:] = 4
    perturbed = replace(split, y=changed, n_obs=np.zeros_like(split.n_obs),
                        signal=np.ones_like(split.signal), label=np.ones((24, 8)),
                        condition=None, target=np.ones((24, 8, 5)), reference=None)
    np.testing.assert_array_equal(expected, episode_features(perturbed))
    assert expected.shape == (24, 34)
    assert np.isfinite(expected).all()
    np.testing.assert_array_equal(expected[:, 0], split.condition == 'short_history')


def test_probability_serialization_and_short_override():
    split = sample_split()
    model = ConditionModel(c_grid=(0.01, 1.)).fit([split])
    p = model.predict_proba(split)
    np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-14)
    assert (p >= 0).all()
    np.testing.assert_array_equal(p[split.condition == 'short_history'],
                                  np.tile([0., 1., 0., 0.], (6, 1)))
    restored = ConditionModel.from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_array_equal(restored.predict_proba(split), p)
    assert sum(map(sum, model.report['cv_confusion'])) == len(split.ids)
    assert 0 <= model.report['cv_accuracy'] <= 1
    # Answers and cutoff signals cannot affect inference.
    np.testing.assert_array_equal(p, model.predict_proba(replace(split, condition=None,
                                                                signal=split.signal + 99)))


def test_explicit_labelled_training_and_deterministic_group_folds():
    split = sample_split()
    with pytest.raises(ValueError, match='explicit'):
        ConditionModel().fit({'train': split})
    with pytest.raises(ValueError, match='Only explicitly'):
        ConditionModel().fit([replace(split, name='test')])
    with pytest.raises(ValueError, match='Only explicitly'):
        ConditionModel().fit([replace(split, condition=None)])
    with pytest.raises(ValueError, match='Duplicate'):
        ConditionModel().fit([split, split])
    labels = np.array([CONDITIONS.index(c) for c in split.condition])
    folds = _folds(split.ids, labels)
    order = np.random.default_rng(1).permutation(len(labels))
    shuffled = _folds([split.ids[i] for i in order], labels[order])
    np.testing.assert_array_equal(folds[order], shuffled)
    assert set(folds) == set(range(5))
