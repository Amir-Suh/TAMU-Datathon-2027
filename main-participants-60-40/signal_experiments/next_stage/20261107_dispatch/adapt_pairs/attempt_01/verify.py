"""Independent enumeration of run-length states on a short masked stream."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

from filtering import HAZARD, PAIRS, STRENGTH, Head, bocpd
from fixtures import FIT_SEED, generate


def exact(split, head, swap, gaps, single):
    h = split.X[0]
    end = split.cutoff[0]
    state = {(0, i, d): (head.prior[i] / 2, np.zeros((2, 2)))
             for i in range(len(PAIRS)) for d in range(2)}
    for t in range(end + 1):
        ctx = split.context[0] if t == end else split.machine_type[0] * 3 + 1
        y = h[t]
        next_state = {}
        for (start, i, d), (mass, stats) in state.items():
            switch = t > 0 and (gaps or y >= 0 or t == end) and (not single or start == 0)
            hz = HAZARD if switch else 0
            next_state[start, i, d] = (mass * (1 - hz), stats)
            if not switch:
                continue
            destinations = [(i, 1 - d, 1 - swap)]
            if swap:
                destinations += [(j, d, swap * head.prior[j] / (1 - head.prior[i]))
                                 for j in range(len(PAIRS)) if j != i]
            for j, new_direction, p in destinations:
                key = (t, j, new_direction)
                prior_mass, new_stats = next_state.get(key, (0, np.zeros((2, 2))))
                next_state[key] = (prior_mass + mass * hz * p, new_stats)
        state = {}
        win = 0
        rule_posterior = np.zeros(2 * len(PAIRS))
        for (start, i, d), (mass, stats) in next_state.items():
            a, b = PAIRS[i]
            valid = t >= b and h[t - a] >= 0 and h[t - b] >= 0
            if valid:
                cell = int(((h[t - a] > 0) != (h[t - b] > 0)) == d)
                alpha = STRENGTH * head.rates[ctx, cell] + stats[cell, 1]
                p_win = alpha / (STRENGTH + stats[cell].sum())
            else:
                p_win = head.default[ctx]
            if t == end:
                win += mass * p_win
                rule_posterior[2 * i + d] += mass
                continue
            if y >= 0:
                mass *= p_win if y > 0 else 1 - p_win
                if valid:
                    stats = stats.copy()
                    stats[cell, int(y > 0)] += 1
            state[start, i, d] = (mass, stats)
        if t == end:
            return win, rule_posterior
        denom = sum(mass for mass, _ in state.values())
        state = {key: (mass / denom, stats) for key, (mass, stats) in state.items()}


def main():
    base = generate("stable", FIT_SEED, 1)
    end = 13
    h = base.X[:1, :end + 1].copy()
    h[:, [3, 7, 11, end]] = -1
    row = replace(base, query_ids=base.query_ids[:1], episode_ids=base.episode_ids[:1],
                  machine_ids=base.machine_ids[:1], machine_type=base.machine_type[:1],
                  signal=base.signal[:1], cutoff=np.array([end]), X=h)
    raw = np.arange(1, len(PAIRS) + 1)
    head = Head(raw / raw.sum(), np.repeat([[.12, .8]], 9, axis=0),
                np.full(9, .45), np.repeat([[.4, .3, .2, .1]], 9, axis=0))
    results = {}
    for swap, gaps, single in ((0, True, False), (.25, True, False), (.25, False, False), (0, True, True)):
        expected_win, expected_mass = exact(row, head, swap, gaps, single)
        actual_win, actual_mass, trace = bocpd(row, head, swap=swap, gaps=gaps, single=single,
                                              budget=2 * len(PAIRS) * (end + 1))
        max_diff = max(abs(expected_win - actual_win[0]), np.max(np.abs(expected_mass - actual_mass[0])))
        assert max_diff < 1e-12
        assert trace["pruned"].max() < 1e-12
        if not gaps:
            assert np.all(trace["reset"][:, [3, 7, 11]] == 0)
        results[f"swap={swap},gap_hazard={gaps},single={single}"] = float(max_diff)
    Path(sys.argv[1]).write_text(json.dumps({"fixture_cutoff": end, "missing_ticks": [3, 7, 11],
                                            "max_error_vs_enumeration": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
