from dataclasses import replace
import numpy as np
import pytest
from tdx.data import load_round
from tdx.fit import dev_spec
from tdx.run import fit_round
from tdx.arms.pooled import build


def test_future_storage_cannot_affect_pooled():
    rnd = load_round(answers=False)
    model = build(method='pooled_recent')
    model.fit(fit_round(rnd, dev_spec()), dev_spec())
    s = rnd.splits['validation']
    altered = s.y.copy()
    for e, cutoff in enumerate(s.cutoff):
        altered[e, :, cutoff:] = 4
    np.testing.assert_array_equal(model.predict(s), model.predict(replace(s, y=altered)))


def test_fit_never_opens_validation_answers():
    # Full audit hook (including pathlib/io.open) lives in the reusable CLI.
    import sys
    active = [True]
    def audit(event, args):
        if active[0] and event == 'open' and isinstance(args[0], (str, bytes)):
            p = str(args[0]).replace('\\', '/')
            assert not any(f'/validation/{name}.csv' in p for name in ('labels','targets','conditions'))
    sys.addaudithook(audit)
    try:
        rnd = load_round(answers=False)
        model = build(method='pooled_recent')
        model.fit(fit_round(rnd, dev_spec(True)), dev_spec(True))
    finally:
        active[0] = False


def test_audit_and_probe_reject_leaking_models(tmp_path):
    import pytest
    from tdx.tools.leak_test import audit_dev_files, assert_prediction_causality, truncate_for_probe
    s = truncate_for_probe(load_round(answers=False).splits['validation'])
    class FutureDependent:
        def predict(self, split):
            p = np.full((*split.y.shape[:2], 5), 0.2)
            p[..., 0] += (split.y[..., -1] == 4) * 0.01
            p[..., 1] -= (split.y[..., -1] == 4) * 0.01
            return p
    with pytest.raises(AssertionError, match='Future history changed'):
        assert_prediction_causality(FutureDependent(), s)
    forbidden = tmp_path / 'validation' / 'targets.csv'
    forbidden.parent.mkdir()
    forbidden.write_text('test')
    with audit_dev_files(tmp_path) as audit:
        with pytest.raises(PermissionError):
            forbidden.read_text()
    assert len(audit['blocked']) == 1
    assert forbidden.read_text() == 'test'


def test_bootstrap_known_difference():
    from tdx.tools.compare import paired_points_interval
    condition = np.repeat(['stable','short_history','drift','shared_change'], 3)
    baseline = np.ones(12)
    assert paired_points_interval(baseline, baseline, baseline, condition, 60) == (0.0, 0.0)
    np.testing.assert_allclose(paired_points_interval(0.8*baseline, baseline, baseline, condition, 60), [10/3, 10/3], rtol=0, atol=1e-14)


def _step_history():
    y = np.zeros((3, 8, 128), dtype=np.int8)
    y[..., 64:] = -1
    return y, np.zeros((3, 8), dtype=np.int64)


def test_step_probe_rejects_whole_row_observed_count():
    from tdx.tools.leak_test import assert_step_causality

    class WholeRowCount:
        def step_outputs(self, y, mtype, steps):
            flag = ((y >= 0).sum(-1) == 64).astype(np.float64)
            return np.broadcast_to(flag[..., None, None], (*y.shape[:2], steps, 5)).copy()

    y, mt = _step_history()
    with pytest.raises(AssertionError, match='Step causality remove'):
        assert_step_causality(WholeRowCount(), y, mt, k=32)


@pytest.mark.parametrize('k', [32, 56])
def test_step_probe_accepts_causal_prefix(k):
    from tdx.tools.leak_test import assert_step_causality

    class Prefix:
        def step_outputs(self, y, mtype, steps):
            prefix = np.cumsum(y[..., :steps] + 1, axis=-1).astype(np.float64)
            return np.repeat(prefix[..., None], 5, axis=-1)

    y, mt = _step_history()
    assert assert_step_causality(Prefix(), y, mt, k=k) == {
        'values': 0.0, 'remove': 0.0, 'add': 0.0, 'move': 0.0}


def test_probe_neutralizes_only_moved_signals_and_recounts():
    from tdx.tools.leak_test import truncate_for_probe
    split = load_round(answers=False).splits['validation']
    # Ensure this assertion cannot pass by chance because released signals are zero.
    split = replace(split, signal=np.ones_like(split.signal))
    probe = truncate_for_probe(split, episodes=3, gap=8)
    moved = probe.cutoff != split.cutoff
    assert moved.sum() == 3
    assert np.all(probe.signal[moved] == 0)
    np.testing.assert_array_equal(probe.signal[~moved], split.signal[~moved])
    np.testing.assert_array_equal(split.signal, 1)
    np.testing.assert_array_equal(probe.n_obs, (probe.y >= 0).sum(-1))


def test_full_audit_keeps_real_short_layout_for_step_probe(monkeypatch):
    import importlib
    from types import SimpleNamespace
    from tdx.tools.leak_test import check_arm

    class CountLeak:
        def fit(self, rnd, spec):
            pass

        def predict(self, split):
            return np.full((*split.y.shape[:2], 5), 0.2)

        def step_outputs(self, y, mtype, steps):
            flag = ((y >= 0).sum(-1) == 64).astype(np.float64)
            return np.broadcast_to(flag[..., None, None], (*y.shape[:2], steps, 5)).copy()

    rnd = load_round(answers=False)
    # Preserve a 64-observation layout that the cutoff audit will shorten.
    val = rnd.splits['validation']
    y = val.y.copy()
    y[0] = -1
    cutoff = int(val.cutoff[0])
    y[0, :, cutoff - 64:cutoff] = 0
    val = replace(val, y=y, n_obs=(y >= 0).sum(-1).astype(val.n_obs.dtype))
    rnd = replace(rnd, splits=dict(rnd.splits, validation=val))
    monkeypatch.setattr('tdx.data.load_round', lambda *a, **kw: rnd)
    module = SimpleNamespace(VARIANTS={'count': {}}, build=lambda **kw: CountLeak())
    original_import = importlib.import_module
    monkeypatch.setattr(importlib, 'import_module',
                        lambda name: module if name == 'tdx.arms.count_probe' else original_import(name))
    with pytest.raises(AssertionError, match='Step causality remove'):
        check_arm('count_probe', 'count')


@pytest.mark.parametrize('arm,variant,overrides', [
    ('seq', 'gru128_factor', {'epochs': 1, 'h': 16, 'device': 'cpu'}),
    ('tfm2', 'd128l3', {'epochs': 1, 'd': 16, 'L': 1, 'device': 'cpu'}),
])
def test_tiny_neural_arm_passes_full_audit(arm, variant, overrides):
    from tdx.tools.leak_test import check_arm
    result = check_arm(arm, variant, overrides=overrides)
    assert result['max_abs_difference'] == 0.0
    assert result['step_causality'] == 0.0
    assert result['forbidden_opens'] == 0
