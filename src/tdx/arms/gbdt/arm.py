"""Global deterministic LightGBM with whole-episode early stopping."""
import json
from pathlib import Path
import hashlib
import numpy as np
import lightgbm as lgb
from tdx.arm import Arm
from tdx.feat import tick_rows
from .features import features, query_features


class GBDTArm(Arm):
    def __init__(self, tau=None, last_n=None, leaves=15, transductive=False,
                 drop_neighbour=False, drop_divergence=False, drop_pair=True, w_cut=20.,
                 rounds=400, patience=35, threads=1, smoke=False):
        self.params = dict(tau=tau, last_n=last_n, leaves=leaves, transductive=transductive,
                           drop_neighbour=drop_neighbour, drop_divergence=drop_divergence, drop_pair=drop_pair,
                           w_cut=w_cut, rounds=rounds, patience=patience, threads=threads, smoke=smoke)
        if not 1 <= threads <= 8:
            raise ValueError('threads must be in 1..8')

    def fit(self, rnd, spec):
        p = self.params
        xs, ys, ws, gs = [], [], [], []
        for name in spec.tick_splits:
            s = rnd.splits[name]
            y = np.where(np.arange(s.y.shape[-1])[None, None, :] < s.cutoff[:, None, None], s.y, -1)
            x, names, groups = features(y, s.mtype)
            self.names = names
            self.keep = np.array([not ((g == 'neighbour' and p['drop_neighbour']) or (g == 'divergence' and p['drop_divergence']) or (g == 'pair' and p['drop_pair'])) for g in groups])
            e, m, t = tick_rows(s)
            n_before = x[e, m, t, names.index('n_obs_before')]
            end = np.max(np.where(y >= 0, np.arange(y.shape[-1]), -1), axis=2)
            ok = n_before >= 8
            if p['last_n'] is not None:
                ok &= t > end[e, m] - p['last_n']
            e, m, t = e[ok], m[ok], t[ok]
            xs.append(x[e, m, t][:, self.keep])
            ys.append(y[e, m, t])
            ws.append(np.ones(len(t)) if p['tau'] is None else np.exp2(-(end[e, m]-t)/float(p['tau'])))
            gs.extend(f'{name}:{s.ids[i]}' for i in e)
        # W2 uses cutoff supervision only in final mode. Sort for stable row order.
        cutoff_splits = sorted(set(spec.label_splits) | set(spec.soft_splits)) if spec.soft_splits else []
        for name in cutoff_splits:
            s = rnd.splits[name]
            q, _, _ = query_features(s)
            q = q.reshape(-1, q.shape[-1])[:, self.keep]
            ids = [f'{name}:{ep}' for ep in s.ids for _ in range(8)]
            if name in spec.label_splits and s.label is not None:
                labels = s.label.ravel()
                valid = labels >= 0
                xs.append(q[valid]); ys.append(labels[valid]); ws.append(np.full(valid.sum(), p['w_cut']))
                gs.extend(i for i, v in zip(ids, valid) if v)
            if name in spec.soft_splits and s.target is not None:
                xs.append(np.repeat(q, 5, axis=0)); ys.append(np.tile(np.arange(5), len(q)))
                ws.append(p['w_cut'] * s.target.ravel()); gs.extend(i for i in ids for _ in range(5))
        x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
        # Hash train episodes only; validation/test histories never form early-stop targets.
        held_ids = sorted(rnd.splits['train'].ids, key=lambda ep: hashlib.sha256(ep.encode()).hexdigest())[:max(1, len(rnd.splits['train'].ids)//5)]
        held = np.isin(gs, [f'train:{ep}' for ep in held_ids])
        params = dict(objective='multiclass', num_class=5, metric='multi_logloss',
                      num_leaves=p['leaves'], min_data_in_leaf=200, learning_rate=.05,
                      feature_fraction=.8, bagging_fraction=.8, bagging_freq=1,
                      lambda_l2=10., num_threads=p['threads'], deterministic=True,
                      force_row_wise=True, seed=7, bagging_seed=7, feature_fraction_seed=7,
                      data_random_seed=7, verbosity=-1)
        feature_names = list(np.asarray(self.names)[self.keep])
        def dataset(mask):
            return lgb.Dataset(x[mask], label=y[mask], weight=w[mask], feature_name=feature_names, free_raw_data=True)
        rounds = min(p['rounds'], 12) if p['smoke'] else p['rounds']
        pilot = lgb.train(params, dataset(~held), num_boost_round=rounds,
                          valid_sets=[dataset(held)], callbacks=[lgb.early_stopping(p['patience'], verbose=False)])
        self.best_iteration = pilot.best_iteration or rounds
        self.model = lgb.train(params, lgb.Dataset(x, label=y, weight=w, feature_name=feature_names), num_boost_round=self.best_iteration)
        self.fit_rows = len(y)
        return self

    def predict(self, split):
        x, names, _ = query_features(split)
        if names != self.names:
            # Models saved before a feature group was added: align by saved column names.
            missing = [n for n in self.names if n not in names]
            if missing:
                raise ValueError(f'Feature schema differs: {missing[:3]}')
            x = x[..., [names.index(n) for n in self.names]]
        pred = self.model.predict(x.reshape(-1, x.shape[-1])[:, self.keep], num_threads=self.params['threads'])
        pred = np.clip(np.asarray(pred, dtype=np.float64), 1e-6, 1-1e-6)
        pred /= pred.sum(axis=-1, keepdims=True)
        return pred.reshape(*split.y.shape[:2], 5)

    def save(self, directory):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(directory/'model.txt'))
        (directory/'features.json').write_text(json.dumps(dict(params=self.params, names=self.names,
             keep=self.keep.tolist(), best_iteration=self.best_iteration, fit_rows=self.fit_rows), indent=2)+'\n')

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory/'features.json').read_text())
        arm = cls(**meta['params'])
        arm.names, arm.keep = meta['names'], np.asarray(meta['keep'], bool)
        arm.best_iteration, arm.fit_rows = meta['best_iteration'], meta['fit_rows']
        arm.model = lgb.Booster(model_file=str(directory/'model.txt'))
        return arm
