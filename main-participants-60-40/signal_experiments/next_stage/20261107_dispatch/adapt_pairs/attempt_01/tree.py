"""Matched causal pair features and 31-leaf win heads."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from filtering import PAIRS, discover, lag_inputs
from fixtures import COMMON_PAIRS, MASKS, masked


def ordering(head):
    return np.argsort(-head.prior, kind="stable")


ORIGINAL = np.array([np.flatnonzero((PAIRS == pair).all(axis=1))[0] for pair in COMMON_PAIRS])
PSEUDO = (48, 64, 80, 96, 112, 128, 144, 160, 176, 192)


def feature_arrays(split, ticks, top):
    """Each record uses observations strictly before its pseudo or final cutoff."""
    n = split.n
    h = split.X
    total = np.zeros((n, len(PAIRS)), dtype=np.int16)
    correct = total.copy()
    recent_count = np.zeros((40, *total.shape), dtype=np.int8)
    recent_correct = recent_count.copy()
    recent_n, recent_k = total.copy(), total.copy()
    n_obs = np.zeros(n)
    n_win = np.zeros(n)
    answer = {}
    for t in range(max(ticks) + 1):
        parity, valid = lag_inputs(h, t)
        idx = t % 40
        if t in ticks:
            def summary(c, z):
                # Posterior mean accuracy of the best fixed polarity, shrunk toward chance.
                accuracy = (np.maximum(z, c - z) + 4) / (c + 8)
                direction = z >= c - z
                prediction = np.where(valid & (c > 0), (parity == direction).astype(float), .5)
                return accuracy, prediction

            full_acc, full_pred = summary(total, correct)
            recent_acc, recent_pred = summary(recent_n, recent_k)
            best = ((full_acc - .5) * np.sqrt(total)).argmax(axis=1)
            pair_fields = [np.where(valid, 1 - parity, -1), full_acc, recent_acc,
                           full_pred, recent_pred, total, recent_n]
            # Signal is neutral on synthetic training pseudo-cutoffs, as it is unavailable
            # in the analogous contest history. The true cutoff carries its recorded signal.
            signal = split.signal if t == split.cutoff.max() else np.zeros(n)
            base = np.column_stack((split.machine_type, signal, np.repeat(t, n) / 256,
                                    n_obs, (n_win + 2) / (n_obs + 4)))
            orig = np.column_stack((base, *(value[:, ORIGINAL] for value in pair_fields)))
            dynamic = np.column_stack((base, *(value[:, top] for value in pair_fields),
                                       *(value[np.arange(n), best] for value in pair_fields)))
            answer[t] = (orig, dynamic)
        if t == max(ticks):
            break
        observed = h[:, t] >= 0
        win = h[:, t] > 0
        count = observed[:, None] & valid
        matches = count & (parity == win[:, None])
        total += count
        correct += matches
        recent_n += count.astype(np.int16) - recent_count[idx]
        recent_k += matches.astype(np.int16) - recent_correct[idx]
        recent_count[idx] = count
        recent_correct[idx] = matches
        n_obs += observed
        n_win += win
    return answer


def train_trees(split, main, subset):
    """Rank on other whole episodes and train both trees on identical labelled rows."""
    fold_number = np.arange(split.n) // 8 % 4
    inputs = [[], []]
    labels = []
    rankings = {}
    for fold in range(4):
        is_group = fold_number == fold
        fitted, _ = discover(subset(split, ~is_group), main)
        top = ordering(fitted)[:6]
        rankings[str(fold)] = {"train_episodes": sorted(set(split.episode_ids[~is_group])),
                               "held_episodes": sorted(set(split.episode_ids[is_group])),
                               "pair_ranking": PAIRS[ordering(fitted)].tolist()}
        group = subset(split, is_group)
        # Apply the same mask family to both feature sets at actual training pseudo-cutoffs.
        for t in PSEUDO:
            pseudo = subset(group, np.ones(group.n, dtype=bool))
            pseudo.cutoff[:] = t
            pseudo.X[:, t:] = -1
            target = group.X[:, t] > 0
            for tag in MASKS:
                original, dynamic = feature_arrays(masked(pseudo, tag), (t,), top)[t]
                # feature_arrays treats the pseudo split's final timestamp as actual.
                original[:, 1] = 0
                dynamic[:, 1] = 0
                inputs[0].append(original)
                inputs[1].append(dynamic)
                labels.append(target)
    y = np.concatenate(labels)
    models = [HistGradientBoostingClassifier(max_leaf_nodes=31, max_iter=80,
                                             l2_regularization=2, random_state=8724).fit(np.concatenate(x), y)
              for x in inputs]
    return models, rankings
