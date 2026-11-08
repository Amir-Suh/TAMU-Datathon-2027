"""Scoring with the participant kit's own scorer, plus a few extra diagnostics.

``contest_points`` uses ``slot_forecasting.contest.contest_score`` from the kit,
so validation points here are exactly what ``check_submission.py --split
validation`` reports.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .data import CONDITIONS, PAYOUTS, TYPES, Split

REPO = Path(__file__).resolve().parents[2]
KIT_DIR = REPO / "main-participants-60-40"
DEFAULT_DATA = KIT_DIR / "public"

if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

from slot_forecasting.contest import score_contest_files  # noqa: E402
from slot_forecasting.io import read_queries, read_submission, write_submission  # noqa: E402


def to_dict(split: Split, probs: np.ndarray) -> dict:
    probs = probs / probs.sum(axis=1, keepdims=True)
    return {qid: tuple(float(v) for v in row) for qid, row in zip(split.query_ids, probs)}


def write_predictions(path: Path, predictions: dict) -> None:
    write_submission(path, predictions)


def check_test_file(path: Path, data_dir: Path) -> int:
    """Same checks as the kit's checker for a test upload: every query once, valid rows."""
    expected = {q.query_id for q in read_queries(Path(data_dir) / "test" / "queries.csv")}
    return len(read_submission(path, expected))


def contest_points(prediction_csv: Path, data_dir: Path, max_points: float = 60) -> dict:
    v = Path(data_dir) / "validation"
    return score_contest_files(prediction_csv, v / "targets.csv", v / "queries.csv", v / "conditions.csv",
                               v / "reference.csv", max_points=max_points)


def extra_metrics(split: Split, probs: np.ndarray) -> dict:
    """Mean squared error by machine type and by condition, and log-loss on observed outcomes."""
    err = ((probs - split.targets) ** 2).sum(axis=1)
    out = {"mean_error": float(err.mean())}
    out["error_by_type"] = {t: float(err[split.machine_type == i].mean()) for i, t in enumerate(TYPES)}
    out["error_by_condition"] = {c: float(err[split.conditions == c].mean()) for c in CONDITIONS}
    p_obs = probs[np.arange(split.n), split.labels.astype(int)]
    out["outcome_logloss"] = float(-np.mean(np.log(np.clip(p_obs, 1e-9, 1))))
    out["payout_order"] = list(PAYOUTS)
    return out
