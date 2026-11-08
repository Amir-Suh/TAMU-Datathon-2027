from dataclasses import replace
import numpy as np
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
