"""History-only episode condition classification with reproducible grouped CV."""
from __future__ import annotations

import hashlib
import numpy as np

CONDITIONS = ('stable', 'short_history', 'drift', 'shared_change')
PAYOUTS = np.array([0., 1., 2., 5., 20.])


def _mean(values):
    return float(np.mean(values)) if len(values) else 0.


def _synchrony(y):
    """Mean pair correlation on jointly observed real ticks; constants give zero."""
    correlations = []
    for m in range(len(y)):
        for n in range(m):
            valid = (y[m] >= 0) & (y[n] >= 0)
            a, b = (y[m, valid] > 0).astype(float), (y[n, valid] > 0).astype(float)
            if len(a) > 1 and a.std() > 0 and b.std() > 0:
                correlations.append(float(np.corrcoef(a, b)[0, 1]))
            else:
                correlations.append(0.)
    return _mean(correlations)


def episode_features(split):
    """Build one row per episode. Never trust cached counts or read future ticks.

    Window statistics use each machine's last observed rows, while synchrony
    uses actual aligned ticks. Missing observations are never treated as losses.
    """
    rows = []
    for e in range(len(split.ids)):
        past = np.asarray(split.y[e, :, :int(split.cutoff[e])])
        observed = [v[v >= 0] for v in past]
        row = [float(all(len(v) == 64 for v in observed))]
        for window in (10, 20, 40, 80):
            recent_p, earlier_p, recent_w, earlier_w = [], [], [], []
            for values in observed:
                recent, earlier = values[-window:], values[-2 * window:-window]
                recent_p.append(_mean(PAYOUTS[recent]))
                earlier_p.append(_mean(PAYOUTS[earlier]))
                recent_w.append(_mean(recent > 0))
                earlier_w.append(_mean(earlier > 0))
            rp, ep, rw, ew = map(_mean, (recent_p, earlier_p, recent_w, earlier_w))
            row.extend((rp, ep, rp - ep, rw, ew, rw - ew))
        row.extend((_synchrony(past[:, -40:]), _synchrony(past[:, :-40])))
        counts = (past >= 0).sum(axis=0)
        mean_win = np.divide((past > 0).sum(axis=0), counts,
                             out=np.zeros(len(counts), dtype=float), where=counts > 0)
        # Only fully observed time points contribute to the rolling statistic.
        complete = counts == past.shape[0]
        rolling = [mean_win[t-19:t+1].mean() for t in range(19, len(counts))
                   if complete[t-19:t+1].all()]
        row.append(float(np.var(rolling)) if rolling else 0.)
        for kind in range(3):
            selected = [v for m, v in enumerate(observed) if split.mtype[e, m] == kind]
            values = np.concatenate(selected) if selected else np.empty(0, dtype=int)
            row.extend((_mean(PAYOUTS[values]), _mean(values > 0)))
        rows.append(row)
    return np.asarray(rows, dtype=float).reshape(len(split.ids), 34)


def _folds(ids, labels):
    """Extend kit episode_folds' per-condition SHA256 ordering to five folds."""
    result = np.empty(len(ids), dtype=int)
    for condition in range(4):
        indices = np.flatnonzero(labels == condition)
        ordered = sorted(indices, key=lambda i: hashlib.sha256(ids[i].encode()).hexdigest())
        for j, index in enumerate(ordered):
            result[index] = j % 5
    return result


def _probabilities(x, mean, scale, coef, intercept, classes):
    logits = ((x - mean) / scale) @ coef.T + intercept
    # sklearn stores one logit for a binary fit.
    if logits.shape[1] == 1:
        logits = np.column_stack((np.zeros(len(x)), logits[:, 0]))
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    out = np.zeros((len(x), 4))
    out[:, classes] = probabilities
    short = x[:, 0] == 1
    out[short] = (0., 1., 0., 0.)
    return out


class ConditionModel:
    def __init__(self, c_grid=(0.01, 0.1, 1., 10., 100.)):
        self.c_grid = tuple(float(c) for c in c_grid)
        if not self.c_grid or any(c <= 0 for c in self.c_grid):
            raise ValueError('C grid must contain positive values')
        self.report = {}

    @staticmethod
    def _fit_arrays(x, labels, c):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(C=c, solver='lbfgs', max_iter=2000, random_state=0)
        model.fit(scaler.transform(x), labels)
        return scaler.mean_, scaler.scale_, model.coef_, model.intercept_, model.classes_

    def fit(self, splits):
        """Train only on explicitly supplied splits, e.g. [train] or [train, val].

        Supplying a labelled validation Split in this sequence deliberately opts
        into final fitting. Inference never inspects condition labels.
        """
        if not isinstance(splits, (list, tuple)) or not splits:
            raise ValueError('Pass an explicit nonempty sequence of labelled splits')
        ids, labels, features = [], [], []
        for split in splits:
            if split.name not in ('train', 'validation') or split.condition is None:
                raise ValueError('Only explicitly supplied train/validation conditions are allowed')
            if len(split.condition) != len(split.ids):
                raise ValueError('One condition is required per episode')
            ids.extend(split.ids)
            try:
                labels.extend(CONDITIONS.index(str(c)) for c in split.condition)
            except ValueError as error:
                raise ValueError('Unknown condition') from error
            features.append(episode_features(split))
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate episodes across training splits')
        labels, x = np.asarray(labels), np.concatenate(features)
        if set(labels.tolist()) != set(range(4)) or min(np.bincount(labels)) < 5:
            raise ValueError('Five-fold CV requires at least five episodes per condition')
        folds = _folds(ids, labels)
        candidates = []
        for c in sorted(set(self.c_grid)):
            prediction = np.zeros((len(labels), 4))
            for fold in range(5):
                train, held = folds != fold, folds == fold
                params = self._fit_arrays(x[train], labels[train], c)
                prediction[held] = _probabilities(x[held], *params)
            candidates.append((float(np.mean(prediction.argmax(axis=1) == labels)), c, prediction))
        accuracy, self.c, prediction = max(candidates, key=lambda value: (value[0], -value[1]))
        selection_accuracy = accuracy
        nested_prediction = np.zeros_like(prediction)
        outer_cs = []
        for outer in range(5):
            training, held = folds != outer, folds == outer
            inner_x, inner_labels = x[training], labels[training]
            inner_ids = [episode for i, episode in enumerate(ids) if training[i]]
            inner_folds = _folds(inner_ids, inner_labels)
            inner_scores = []
            for c in sorted(set(self.c_grid)):
                inner_prediction = np.zeros((len(inner_labels), 4))
                for inner in range(5):
                    use, check = inner_folds != inner, inner_folds == inner
                    params = self._fit_arrays(inner_x[use], inner_labels[use], c)
                    inner_prediction[check] = _probabilities(inner_x[check], *params)
                inner_scores.append((float(np.mean(inner_prediction.argmax(axis=1) == inner_labels)), c))
            outer_c = max(inner_scores, key=lambda item: (item[0], -item[1]))[1]
            outer_cs.append(outer_c)
            params = self._fit_arrays(inner_x, inner_labels, outer_c)
            nested_prediction[held] = _probabilities(x[held], *params)
        accuracy = float(np.mean(nested_prediction.argmax(axis=1) == labels))
        confusion = np.zeros((4, 4), dtype=int)
        np.add.at(confusion, (labels, nested_prediction.argmax(axis=1)), 1)
        self.mean, self.scale, self.coef, self.intercept, self.classes = self._fit_arrays(x, labels, self.c)
        self.report = dict(conditions=list(CONDITIONS), cv_accuracy=accuracy,
                           cv_confusion=confusion.tolist(), selected_c=self.c,
                           selected_c_cv_accuracy=selection_accuracy, outer_selected_cs=outer_cs,
                           c_scores=[dict(c=c, accuracy=a) for a, c, _ in candidates],
                           n_episodes=len(ids), fitted_splits=[s.name for s in splits],
                           cv_note='Accuracy/confusion use nested five-fold episode CV; selected_c_cv_accuracy is optimistic selection CV')
        return self

    def predict_proba(self, split):
        return _probabilities(episode_features(split), self.mean, self.scale,
                              self.coef, self.intercept, self.classes)

    def to_dict(self):
        return dict(version=1, c_grid=list(self.c_grid), c=self.c, report=self.report,
                    **{name: getattr(self, name).tolist()
                       for name in ('mean', 'scale', 'coef', 'intercept', 'classes')})

    @classmethod
    def from_dict(cls, state):
        if state['version'] != 1:
            raise ValueError('Unsupported condition model version')
        model = cls(state['c_grid'])
        model.c, model.report = state['c'], state['report']
        for name in ('mean', 'scale', 'coef', 'intercept', 'classes'):
            setattr(model, name, np.asarray(state[name], dtype=int if name == 'classes' else float))
        return model
