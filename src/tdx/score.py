"""Kit-backed contest scores and episode errors."""
import numpy as np
from . import paths
from .data import PAYOUTS
from slot_forecasting.schema import Target
from slot_forecasting.contest import contest_score
from slot_forecasting.scoring import score_predictions
from slot_forecasting.experiments import episode_folds, subset_score

def prediction_map(split, pred):
    pred = np.asarray(pred)
    if pred.shape != (*split.query_ids.shape, 5) or pred.dtype != np.float64:
        raise ValueError('Expected float64 probabilities with shape (E,8,5)')
    if not np.isfinite(pred).all() or np.any(pred <= 0) or np.any(pred >= 1) or np.max(np.abs(pred.sum(-1) - 1)) > 1e-12:
        raise ValueError('Probabilities must be finite, strictly between zero and one, and sum to one')
    return {str(q): tuple(p) for q, p in zip(split.query_ids.flat, pred.reshape(-1, 5))}

def _inputs(rnd, pred_val):
    s = rnd.splits['validation']
    if any(v is None for v in (s.target, s.reference, s.condition, s.label)):
        raise ValueError('Scoring requires released validation answers')
    targets = tuple(Target(str(s.query_ids[e, m]), ep, tuple(s.target[e, m]), PAYOUTS[int(s.label[e, m])]) for e, ep in enumerate(s.ids) for m in range(8))
    groups = {str(s.query_ids[e, m]): str(s.condition[e]) for e in range(len(s.ids)) for m in range(8)}
    return prediction_map(s, pred_val), targets, groups, prediction_map(s, s.reference)

def _compact(result):
    return dict(points=result['technical_points'], mean_loss_ratio=result['mean_loss_ratio'],
                by_condition={c: r['loss_ratio'] for c, r in result['by_condition'].items()})

def score(rnd, pred_val):
    pred, targets, groups, ref = _inputs(rnd, pred_val)
    result = contest_score(pred, targets, groups, ref, max_points=rnd.max_points)
    tuning, audit = episode_folds(targets, groups)
    out = _compact(result)
    out['contest'] = result
    for key, eps in [('tuning', tuning), ('audit', audit)]:
        out[key] = _compact(subset_score(pred, targets, groups, ref, eps, rnd.max_points))
    return out

def episode_values(rnd, pred_val):
    pred, targets, groups, ref = _inputs(rnd, pred_val)
    err = score_predictions(pred, targets)['episode_errors']
    ref_err = score_predictions(ref, targets)['episode_errors']
    cond = {t.episode_id: groups[t.query_id] for t in targets}
    return {ep: dict(error=err[ep], reference_error=ref_err[ep], condition=cond[ep]) for ep in sorted(err)}
