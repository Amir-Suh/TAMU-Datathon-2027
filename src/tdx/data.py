"""Dense, deterministic round data. Cached inputs never contain validation answers."""
from dataclasses import dataclass, replace, fields
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .paths import ROOT, data_root

PAYOUTS = (0, 1, 2, 5, 20)
_TYPES = {'steady': 0, 'balanced': 1, 'jackpot': 2}
_CODES = {str(p): i for i, p in enumerate(PAYOUTS)}

@dataclass
class Split:
    name: str
    ids: list[str]
    y: np.ndarray
    mtype: np.ndarray
    signal: np.ndarray
    cutoff: np.ndarray
    n_obs: np.ndarray
    query_ids: np.ndarray
    label: np.ndarray | None = None
    condition: np.ndarray | None = None
    target: np.ndarray | None = None
    reference: np.ndarray | None = None

@dataclass
class Round:
    name: str
    root: Path
    splits: dict[str, Split]
    max_points: float

def _rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def _answers(split, root, *, validation=False):
    folder = root / split.name
    qi = {q: (e, m) for (e, m), q in np.ndenumerate(split.query_ids)}
    label = np.full(split.query_ids.shape, -1, dtype=np.int8)
    for r in _rows(folder / 'labels.csv'):
        label[qi[r['query_id']]] = _CODES[r['outcome']]
    condition = np.full(len(split.ids), '', dtype=object)
    for r in _rows(folder / 'conditions.csv'):
        e, _ = qi[r['query_id']]
        if condition[e] and condition[e] != r['condition']:
            raise ValueError('Episode spans multiple conditions')
        condition[e] = r['condition']
    kwargs = dict(label=label, condition=condition)
    if validation:
        for name, filename in [('target', 'targets.csv'), ('reference', 'reference.csv')]:
            arr = np.empty((*split.query_ids.shape, 5), dtype=np.float64)
            seen = set()
            for r in _rows(folder / filename):
                arr[qi[r['query_id']]] = [float(r[f'p_{p}']) for p in PAYOUTS]
                seen.add(r['query_id'])
            if seen != set(qi):
                raise ValueError(f'{filename}: query IDs differ')
            kwargs[name] = arr
    if np.any(label < 0) or np.any(condition == ''):
        raise ValueError('Incomplete answers')
    return replace(split, **kwargs)

def with_answers(rnd):
    """Attach released validation answers only in scoring/final orchestration."""
    splits = dict(rnd.splits)
    if splits['validation'].target is None:
        splits['validation'] = _answers(splits['validation'], rnd.root, validation=True)
    return replace(rnd, splits=splits)

def _read_split(root, name):
    qs = _rows(root / name / 'queries.csv')
    ids = sorted({r['episode_id'] for r in qs})
    ei = {v: i for i, v in enumerate(ids)}
    machines = {ep: sorted(r['machine_id'] for r in qs if r['episode_id'] == ep) for ep in ids}
    if any(len(ms) != 8 or len(set(ms)) != 8 for ms in machines.values()):
        raise ValueError('Expected eight unique machines per episode')
    mi = {(ep, m): j for ep, ms in machines.items() for j, m in enumerate(ms)}
    shape = (len(ids), 8)
    y = np.full((*shape, 256), -1, dtype=np.int8)
    mt, sig = np.empty(shape, np.int8), np.empty(shape, np.int8)
    cutoff = np.full(len(ids), -1, np.int16)
    qids = np.empty(shape, object)
    for r in qs:
        e, m = ei[r['episode_id']], mi[r['episode_id'], r['machine_id']]
        c = int(r['cutoff'])
        if not 0 <= c <= 256 or cutoff[e] not in (-1, c):
            raise ValueError('Invalid/inconsistent cutoff')
        cutoff[e] = c
        mt[e, m], sig[e, m] = _TYPES[r['machine_type']], int(r['signal'])
        qids[e, m] = r['query_id']
    for r in _rows(root / name / 'history.csv'):
        e, m = ei[r['episode_id']], mi[r['episode_id'], r['machine_id']]
        t = int(r['tick'])
        if not 0 <= t < cutoff[e] or y[e, m, t] != -1:
            raise ValueError('Future/duplicate/out of range history')
        y[e, m, t] = _CODES[r['payout']]
    split = Split(name, ids, y, mt, sig, cutoff, (y >= 0).sum(-1).astype(np.int16), qids)
    return _answers(split, root) if name == 'train' else split

def load_round(root=None, name=None, *, answers=True):
    root = data_root(root, name or 'main')
    raw = (root / 'metadata.json').read_bytes()
    meta = json.loads(raw)
    name = name or meta['round']
    # Cache identity includes root; same metadata at another data root must not alias.
    digest = hashlib.sha256(raw).hexdigest()[:16]
    cache = ROOT / '.cache' / f'{name}_{digest}.npz'
    splits = {}
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            valid = str(z['root']) == str(root) and int(z['version']) == 1
            if valid:
                for sn in ('train', 'validation', 'test'):
                    kw = {}
                    for f in fields(Split):
                        key = f'{sn}_{f.name}'
                        if key in z:
                            v = z[key]
                            kw[f.name] = v.tolist() if f.name == 'ids' else (v.astype(object) if f.name in ('query_ids', 'condition') else v)
                    splits[sn] = Split(name=sn, **kw)
    if not splits:
        splits = {sn: _read_split(root, sn) for sn in ('train', 'validation', 'test')}
        payload = dict(root=np.array(str(root)), version=np.array(1))
        for sn, split in splits.items():
            for f in fields(Split):
                value = getattr(split, f.name)
                if f.name != 'name' and value is not None:
                    payload[f'{sn}_{f.name}'] = np.asarray(value, dtype=str) if f.name in ('ids', 'query_ids', 'condition') else value
        cache.parent.mkdir(exist_ok=True)
        import os, tempfile
        with tempfile.NamedTemporaryFile(dir=cache.parent, suffix='.npz', delete=False) as f:
            tmp = Path(f.name)
        try:
            np.savez_compressed(tmp, **payload)
            os.replace(tmp, cache)
        finally:
            tmp.unlink(missing_ok=True)
    rnd = Round(name, root, splits, float(meta['max_points']))
    return with_answers(rnd) if answers else rnd
