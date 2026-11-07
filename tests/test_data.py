import csv
import numpy as np
from tdx.data import load_round, PAYOUTS
from tdx.paths import data_root


def test_data_shapes_and_missing():
    rnd = load_round()
    for name, size in [('train', 96), ('validation', 96), ('test', 192)]:
        s = rnd.splits[name]
        assert s.y.shape == (size, 8, 256)
        assert s.y.dtype == np.int8
        assert np.all(s.y[np.broadcast_to(np.arange(256) >= s.cutoff[:, None, None], s.y.shape)] == -1)
        np.testing.assert_array_equal(s.n_obs, (s.y >= 0).sum(-1))
    assert (rnd.splits['train'].n_obs == 64).sum() == 192
    assert rnd.splits['validation'].target.shape == (96, 8, 5)
    assert rnd.splits['test'].target is None


def test_round_trip_history():
    s = load_round(answers=False).splits['train']
    ei = {ep: i for i, ep in enumerate(s.ids)}
    with (data_root() / 'train/queries.csv').open() as f:
        qs = list(csv.DictReader(f))
    machines = {ep: sorted(q['machine_id'] for q in qs if q['episode_id'] == ep) for ep in s.ids}
    with (data_root() / 'train/history.csv').open() as f:
        rows = list(csv.DictReader(f))
    for i in np.random.default_rng(77).choice(len(rows), 1000, replace=False):
        r = rows[i]
        assert PAYOUTS[s.y[ei[r['episode_id']], machines[r['episode_id']].index(r['machine_id']), int(r['tick'])]] == int(r['payout'])


def test_input_cache_excludes_validation_answers():
    rnd = load_round(answers=False)
    s = rnd.splits['validation']
    assert s.label is s.target is s.condition is s.reference is None
    assert rnd.splits['train'].label is not None
