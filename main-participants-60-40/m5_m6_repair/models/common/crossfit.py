"""Episode-heldout rows and session-local model fits for stacked models.

The fingerprint includes every training field a model may use. In particular,
changing an outcome or a history invalidates cached fits, even if the episode
IDs and the requested model name stay the same.
"""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, fields, replace

import numpy as np

from .data import MISSING, Split
from .selection import PSEUDO_SHIFTS, pseudo_splits


def subset_split(split: Split, rows: np.ndarray) -> Split:
    rows = np.asarray(rows, dtype=int)
    if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= split.n):
        raise ValueError("rows must be valid one-dimensional indices")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("duplicate rows")
    values = {}
    for f in fields(Split):
        value = getattr(split, f.name)
        if f.name in ("name",):
            continue
        if value is None:
            values[f.name] = None
        elif isinstance(value, list):
            values[f.name] = [value[i] for i in rows]
        else:
            values[f.name] = value[rows].copy()
    return replace(split, **values)


def episode_folds(split: Split, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stable folds: all queries from one episode occupy one heldout fold."""
    episodes = sorted(set(str(x) for x in split.episode_ids))
    if not 2 <= n_splits <= len(episodes):
        raise ValueError(f"need 2..{len(episodes)} episode folds, got {n_splits}")
    buckets = [set(episodes[i::n_splits]) for i in range(n_splits)]
    names = np.asarray([str(x) for x in split.episode_ids])
    folds = []
    for bucket in buckets:
        held = np.flatnonzero(np.isin(names, list(bucket)))
        train = np.flatnonzero(~np.isin(names, list(bucket)))
        folds.append((train, held))
    return folds


@dataclass
class PseudoFrame:
    splits: list[Split]
    valid: list[np.ndarray]
    row_ids: list[tuple[str, str, int]]
    episode_ids: np.ndarray
    y: np.ndarray


def pseudo_frame(train: Split) -> PseudoFrame:
    splits = pseudo_splits(train)
    valid = [np.flatnonzero(s.labels != MISSING) for s in splits]
    ids = [(str(s.episode_ids[i]), str(s.query_ids[i]), shift)
           for shift, s, indices in zip(PSEUDO_SHIFTS, splits, valid) for i in indices]
    if len(set(ids)) != len(ids):
        raise ValueError("pseudo-query row IDs are not unique")
    groups = np.concatenate([s.episode_ids[idx] for s, idx in zip(splits, valid)])
    y = np.concatenate([(s.labels[idx] > 0).astype(float) for s, idx in zip(splits, valid)])
    return PseudoFrame(splits, valid, ids, groups, y)


def frame_values(model, frame: PseudoFrame, method: str) -> np.ndarray:
    """Predict or extract features in the frame's stable shift/row order."""
    return np.concatenate([getattr(model, method)(s)[idx] for s, idx in
                           zip(frame.splits, frame.valid)], axis=0)


def align_rows(source: PseudoFrame, target: PseudoFrame) -> np.ndarray:
    pos = {key: i for i, key in enumerate(target.row_ids)}
    try:
        return np.asarray([pos[key] for key in source.row_ids], dtype=int)
    except KeyError as exc:
        raise ValueError(f"pseudo row missing from target: {exc}") from exc


def split_fingerprint(split: Split) -> str:
    h = hashlib.sha256()
    for f in fields(Split):
        if f.name in ("name", "targets", "reference", "conditions"):
            continue  # never train on validation truth or condition labels
        value = getattr(split, f.name)
        h.update(f.name.encode())
        if value is None:
            h.update(b"none")
            continue
        arr = np.asarray(value)
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        if arr.dtype.kind in "OUS":
            for item in arr.flat:
                b = str(item).encode()
                h.update(len(b).to_bytes(8, "little"))
                h.update(b)
        else:
            h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


class FitCache:
    """No global state: safe to share only within one fit/evaluation session."""

    def __init__(self):
        self._models = {}

    def fit(self, kind: str, train: Split, factory, config=()):
        fingerprint = split_fingerprint(train)
        key = (kind, repr(config), fingerprint)
        if key not in self._models:
            model = factory().fit(train)
            model._fit_fingerprint = fingerprint
            self._models[key] = model
        return self._models[key]


def factory_config(factory):
    """Constructor and class-level tuning settings for a cache key."""
    if inspect.isclass(factory):
        return (id(factory), str(inspect.signature(factory.__init__)),
                repr(getattr(factory, "GRID", None)), repr(getattr(factory, "ITER_GRID", None)),
                repr(getattr(factory, "GBM", None)))
    return (id(factory), repr(getattr(factory, "__dict__", {})))
