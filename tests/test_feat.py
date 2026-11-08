from types import SimpleNamespace
import numpy as np
from tdx.feat import lag_classes, others_win, tick_rows


def test_features_brute_force():
    y = np.random.default_rng(123).integers(-1, 5, (3, 8, 30), dtype=np.int8)
    lagged = lag_classes(y, 8)
    other = others_win(y, [1, 2, 8, 31])
    for e in range(3):
        for m in range(8):
            for t in range(30):
                for lag in range(1, 9):
                    assert lagged[e, m, t, lag-1] == (y[e, m, t-lag] if t >= lag else -1)
                for j, lag in enumerate([1, 2, 8, 31]):
                    values = [y[e, mm, t-lag] for mm in range(8) if mm != m] if t >= lag else [-1]
                    expected = -1 if min(values) < 0 else sum(v > 0 for v in values)
                    assert other[e, m, t, j] == expected
    for t in [0, 1, 13, 29]:
        altered = y.copy()
        altered[..., t:] = 4
        np.testing.assert_array_equal(lag_classes(altered, 8)[..., :t+1, :], lagged[..., :t+1, :])
        np.testing.assert_array_equal(others_win(altered, [1, 2, 8, 31])[..., :t+1, :], other[..., :t+1, :])


def test_tick_rows():
    y = np.random.default_rng(12).integers(-1, 5, (3, 8, 30), dtype=np.int8)
    cutoff = np.array([8, 13, 30])
    actual = list(zip(*tick_rows(SimpleNamespace(y=y, cutoff=cutoff))))
    expected = [(e,m,t) for e in range(3) for m in range(8) for t in range(cutoff[e]) if y[e,m,t] >= 0]
    assert actual == expected
