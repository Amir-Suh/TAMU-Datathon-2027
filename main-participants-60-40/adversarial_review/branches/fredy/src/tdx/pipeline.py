"""One-command round refit, final forecast, packaging and offline verification."""
import os
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_var] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from tdx.paths import ROOT, data_root
from tdx.package.build import build_entry, parse_arms
from tdx.package.blend import fit, validate


def _settings(path):
    value = json.loads(Path(path).read_text())
    if not value.get('arms'):
        raise ValueError('Pipeline config must list arms')
    ids = [f'{a["arm"]}/{a["variant"]}' for a in value['arms']]
    parse_arms(','.join(ids))
    return value, ids


def pipeline(*, round_name, data=None, config=None, out=None, dry_run=False, main_data=None):
    config_path = Path(config or ROOT / 'config/pipeline.json').resolve()
    config, ids = _settings(config_path)
    data = data_root(data, round_name)
    output = Path(out or ROOT / 'runs/pipeline' / round_name).resolve()
    dev, final = output / 'dev', output / 'final'
    baseline = 'pooled/pooled_recent'
    dev_ids = list(dict.fromkeys([*ids, baseline]))
    print(f'PIPELINE round={round_name} data={data} config={config_path}', flush=True)
    for name in dev_ids:
        print(f'  dev {name} -> {dev / name}', flush=True)
    print(f'  compare {dev} baseline={baseline}', flush=True)
    print(f'  refit blend on current validation -> {output / "weights.json"}', flush=True)
    for name in ids:
        print(f'  final {name} -> {final / name}', flush=True)
    print(f'  build {output / ("entry_" + round_name + ".zip")} + submission.csv', flush=True)
    print('  check submission + fresh-venv offline reproduction twice', flush=True)
    if dry_run:
        return None
    from tdx.data import load_round, with_answers
    from tdx.run import execute
    from tdx.tools.compare import compare
    from tdx.package.repro import check
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((data / 'metadata.json').read_text())
    if metadata['round'] != round_name:
        raise ValueError('Data metadata round differs from --round')
    rnd = load_round(data, round_name, answers=False)
    settings = {f'{a["arm"]}/{a["variant"]}': a for a in config['arms']}
    for name in dev_ids:
        arm, variant = name.split('/')
        execute(rnd, arm, variant, 'dev', settings.get(name, {}).get('params'), out=dev / name)
    compare(dev, baseline=baseline, data=data, round_name=round_name)
    scoring = with_answers(rnd)
    s = scoring.splits['validation']
    predictions = [np.load(dev / name / 'val_pred.npy', allow_pickle=False) for name in ids]
    weights = validate({'weights': fit(predictions, s, s.reference)}, ids)
    (output / 'weights.json').write_text(json.dumps(weights, indent=2) + '\n')
    print('BLEND REFIT ' + json.dumps(weights) + ' (final fit; no CV score claimed)', flush=True)
    for name in ids:
        arm, variant = name.split('/')
        execute(scoring, arm, variant, 'final', settings[name].get('params'), out=final / name)
    result = build_entry(arms=ids, blend=output / 'weights.json', round_name=round_name,
                         data=data, out=output / 'entry', zip_path=output / f'entry_{round_name}.zip',
                         runs=final, csv_path=output / 'submission.csv', pipeline_config=config)
    report = check(result['zip'], data, result['csv'])
    (output / 'repro_report.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'pipeline.json').write_text(json.dumps(dict(config=config, result=result), indent=2) + '\n')
    print(f'PIPELINE OK csv={result["csv"]} zip={result["zip"]} sha256={result["sha256"]}', flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--round', choices=('main', 'adaptation'), required=True)
    p.add_argument('--data')
    p.add_argument('--config')
    p.add_argument('--out')
    p.add_argument('--main-data', help='Accepted for launcher compatibility; current-round refit does not reuse it')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    pipeline(round_name=a.round, data=a.data, config=a.config, out=a.out, dry_run=a.dry_run, main_data=a.main_data)


if __name__ == '__main__':
    main()
