"""Load one split of the participant kit into dense numpy arrays.

Every query is one machine, so arrays are indexed by query row ``m``. The
history becomes a matrix ``X[m, t]`` holding the payout *index* (0..4 for
0x, 1x, 2x, 5x, 20x) observed at tick ``t``, or ``MISSING`` when the tick was
not observed. Columns at or after the machine's cutoff are always MISSING, so a
model that reads ``X`` cannot see the future.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

PAYOUTS = (0, 1, 2, 5, 20)
PAYOUT_INDEX = {p: i for i, p in enumerate(PAYOUTS)}
PROB_COLUMNS = tuple(f"p_{p}" for p in PAYOUTS)
TYPES = ("steady", "balanced", "jackpot")
SIGNALS = (-1, 0, 1)
CONDITIONS = ("stable", "short_history", "drift", "shared_change")
N_CONTEXTS = len(TYPES) * len(SIGNALS)
MISSING = -1


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


@dataclass
class Split:
    name: str
    query_ids: list
    episode_ids: np.ndarray
    machine_ids: list
    machine_type: np.ndarray      # index into TYPES
    signal: np.ndarray            # -1, 0, 1
    cutoff: np.ndarray            # tick to predict
    X: np.ndarray                 # int8 [M, T] payout index or MISSING
    labels: np.ndarray | None = None      # payout index observed at the cutoff
    conditions: np.ndarray | None = None  # train / validation only
    targets: np.ndarray | None = None     # [M, 5] true probabilities (validation)
    reference: np.ndarray | None = None   # [M, 5] frozen pooled forecast (validation)

    @property
    def n(self) -> int:
        return len(self.query_ids)

    @property
    def context(self) -> np.ndarray:
        """Type x signal cell, 0..8."""
        return self.machine_type * len(SIGNALS) + (self.signal + 1)

    def X_with_labels(self) -> np.ndarray:
        """History plus the labelled outcome at the cutoff (for training/selection only)."""
        if self.labels is None:
            raise ValueError(f"{self.name} has no labels")
        X = self.X.copy()
        X[np.arange(self.n), self.cutoff] = self.labels
        return X


def truncate(split: Split, shift: int) -> Split:
    """Pretend each machine's cutoff was ``shift`` ticks earlier (train/selection only).

    The observed payout at the new cutoff becomes the label (MISSING when that
    tick was not observed) and every later tick is hidden. Truth columns are
    dropped because they refer to the real cutoff. Earlier queries use neutral signal.
    """
    X = split.X_with_labels()
    cut = split.cutoff - shift
    rows = np.arange(split.n)
    labels = X[rows, cut].copy()
    X = X[:, : int(cut.max()) + 1].copy()
    X[np.arange(X.shape[1])[None, :] >= cut[:, None]] = MISSING
    signal = np.where(cut == split.cutoff, split.signal, 0)
    return replace(split, name=f"{split.name}-shift{shift}", cutoff=cut, X=X, labels=labels,
                   signal=signal, targets=None, reference=None)


def load_split(public_dir: str | Path, name: str) -> Split:
    d = Path(public_dir) / name
    queries = _rows(d / "queries.csv")
    index = {(q["episode_id"], q["machine_id"]): i for i, q in enumerate(queries)}
    cutoff = np.array([int(q["cutoff"]) for q in queries])
    X = np.full((len(queries), int(cutoff.max()) + 1), MISSING, dtype=np.int8)
    for h in _rows(d / "history.csv"):
        m, t = index[(h["episode_id"], h["machine_id"])], int(h["tick"])
        if t >= cutoff[m]:
            raise ValueError(f"history tick {t} at/after cutoff for {h['machine_id']}")
        X[m, t] = PAYOUT_INDEX[int(h["payout"])]

    qids = [q["query_id"] for q in queries]
    pos = {qid: i for i, qid in enumerate(qids)}
    split = Split(
        name=name,
        query_ids=qids,
        episode_ids=np.array([q["episode_id"] for q in queries]),
        machine_ids=[q["machine_id"] for q in queries],
        machine_type=np.array([TYPES.index(q["machine_type"]) for q in queries]),
        signal=np.array([int(q["signal"]) for q in queries]),
        cutoff=cutoff,
        X=X,
    )
    if (d / "labels.csv").exists():
        split.labels = np.empty(split.n, dtype=np.int8)
        for r in _rows(d / "labels.csv"):
            split.labels[pos[r["query_id"]]] = PAYOUT_INDEX[int(r["outcome"])]
    if (d / "conditions.csv").exists():
        split.conditions = np.empty(split.n, dtype=object)
        for r in _rows(d / "conditions.csv"):
            split.conditions[pos[r["query_id"]]] = r["condition"]
    for attr, fname in (("targets", "targets.csv"), ("reference", "reference.csv")):
        if (d / fname).exists():
            arr = np.empty((split.n, len(PAYOUTS)))
            for r in _rows(d / fname):
                arr[pos[r["query_id"]]] = [float(r[c]) for c in PROB_COLUMNS]
            setattr(split, attr, arr)
    return split
