"""Build a portable entry and matching forecast, using the kit ZIP validator."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from tdx.paths import ROOT, data_root
from tdx.package.adapters import copy_runtime, load_arm
from tdx.package.blend import validate


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def parse_arms(value):
    result = []
    for item in value.split(','):
        parts = item.split('/')
        if len(parts) != 2 or any(not p or not p.replace('_', '').isalnum() for p in parts):
            raise ValueError('Use comma-separated arm/variant names')
        if item in result:
            raise ValueError(f'Duplicate arm: {item}')
        result.append(item)
    if not result:
        raise ValueError('At least one arm is required')
    return result


def build_entry(*, arms=None, blend=None, round_name='main', data=None,
                out='entry', zip_path='entry_main.zip', runs='runs', csv_path=None,
                pipeline_config=None, allow_dev=False):
    data = data_root(data, round_name)
    metadata = json.loads((data / 'metadata.json').read_text())
    if metadata['round'] != round_name:
        raise ValueError('Data metadata round differs from --round')
    out, zip_path, runs = Path(out).resolve(), Path(zip_path).resolve(), Path(runs).resolve()
    csv_path = Path(csv_path).resolve() if csv_path else zip_path.with_suffix('.csv')
    if zip_path.is_relative_to(out) or csv_path.is_relative_to(out):
        raise ValueError('ZIP and forecast CSV must be outside the entry directory')
    if len({out, zip_path, csv_path}) != 3:
        raise ValueError('Entry, ZIP and CSV paths must be distinct')
    if out.exists() and not (out / '.tdx-entry').is_file():
        raise ValueError(f'Refusing to replace a folder not created by this builder: {out}')
    ids = parse_arms(arms) if isinstance(arms, str) else list(arms or ['pooled/pooled_recent'])
    parse_arms(','.join(ids))
    skeleton = arms is None
    settings = pipeline_config or {}
    blend_value = json.loads(Path(blend).read_text()) if blend else {'weights': [1 / len(ids)] * len(ids)}
    blend_value = validate(blend_value, ids)
    out.parent.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.tdx-build-', dir=out.parent) as work:
        work = Path(work)
        source = work / 'code'
        source.mkdir()
        pins = copy_runtime(sorted({x.split('/')[0] for x in ids} | {'pooled'}), source)
        # The inference loader uses _read_split, never load_round's repo cache.
        # ROOT is scratch if a downstream arm explicitly asks for a cache.
        (source / 'tdx' / 'paths.py').write_text(
            'import os\nfrom pathlib import Path\n'
            'ROOT = Path("/tmp/tdx-entry-cache")\nKIT = Path(__file__).resolve().parents[1]\n'
            'def data_root(data=None, round_name="main"):\n'
            '    if data is None:\n        raise ValueError("Explicit --data is required")\n'
            '    return Path(data).resolve()\n')
        (source / 'tdx' / 'package').mkdir(exist_ok=True)
        (source / 'tdx' / 'package' / '__init__.py').write_text('')
        shutil.copyfile(Path(__file__).with_name('blend.py'), source / 'tdx' / 'package' / 'blend.py')
        shutil.copyfile(Path(__file__).with_name('predict.py'), source / 'predict.py')
        records = []
        for arm_id in ids:
            name, variant = arm_id.split('/')
            destination = source / 'models' / name / variant
            if skeleton:
                from tdx.arms.pooled import build
                from tdx.data import _read_split, Round
                from tdx.fit import dev_spec
                arm = build(method='pooled_recent')
                arm.fit(Round(round_name, data, {'train': _read_split(data, 'train')}, float(metadata['max_points'])), dev_spec())
                arm.save(destination)
                params = {'method': 'pooled_recent'}
                fit_mode = 'dev'
            elif getattr(importlib.import_module(f'tdx.arms.{name}'), 'RUNTIME_FIT', False):
                # Cheap arm refitted on the released train split at inference time; nothing to ship but a stub.
                destination.mkdir(parents=True)
                (destination / 'meta.json').write_text('{"runtime_fit": true}\n')
                params = {}
                fit_mode = 'runtime'
            else:
                folder = runs / name / variant
                meta = json.loads((folder / 'meta.json').read_text())
                if meta['mode'] != 'final' and not (allow_dev and meta['mode'] == 'dev'):
                    raise ValueError(f'{arm_id}: build selected arms from final-mode exports (or pass allow_dev for validated dev-mode models)')
                if meta['round'] != round_name or Path(meta['data']).resolve() != data:
                    raise ValueError(f'{arm_id}: saved export uses another round/data root')
                if meta.get('metadata_sha256') != hashlib.sha256((data / 'metadata.json').read_bytes()).hexdigest():
                    raise ValueError(f'{arm_id}: metadata fingerprint differs')
                shutil.copytree(folder / 'model', destination)
                params = meta['params']
                fit_mode = meta['mode']
            arm = load_arm(name, destination)
            records.append(dict(id=arm_id, arm=name, variant=variant, params=params,
                                module=type(arm).__module__, **{'class': type(arm).__name__},
                                model=destination.relative_to(source).as_posix(), fit_mode=fit_mode))
        fallback = list(settings.get('fallback', ids))
        fallback = ['pooled_recent' if item == 'pooled/pooled_recent' and item not in ids else item for item in fallback]
        if any(item not in ids and item != 'pooled_recent' for item in fallback):
            raise ValueError('Fallback arms must be included in --arms')
        fallback = [item for item in fallback if item != 'pooled_recent'] + ['pooled_recent']
        config = dict(version=1, round=round_name, season_id=metadata['season_id'],
                      seeds=settings.get('seeds', {'global': 7}), arms=records, primary=ids,
                      fallback=fallback, requirements=pins)
        _json(source / 'config.json', config)
        _json(source / 'blend.json', blend_value)
        _json(source / 'reproduce.json', {'image': 'python', 'command': ['python', 'predict.py', '--data', '/data', '--out', '/output/submission.csv']})
        (source / 'requirements.txt').write_text('\n'.join(pins) + '\n')
        (source / '.tdx-entry').write_text('W6 generated entry\n')
        expected = work / 'submission.csv'
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTHONPATH', None)
        subprocess.run([sys.executable, str(source / 'predict.py'), '--data', str(data), '--out', str(expected)],
                       cwd=source, env=env, check=True)
        sha = hashlib.sha256(expected.read_bytes()).hexdigest()
        (source / 'README.md').write_text(
            '# House Edge entry\n\n'
            f'Round: {round_name}; season: {metadata["season_id"]}. Team: supply in submission form.\n'
            f'Python 3.13; CPU only; one inference thread; seeds in config.json.\n'
            f'Dependencies: {", ".join(pins)}.\n\n'
            'Run from this folder: `python predict.py --data /data --out /output/submission.csv`.\n'
            'The data and submission folder can be read-only. Only the requested output and /tmp are written.\n'
            'Models and settings are in models/, config.json and blend.json. No main-round data is reused in adaptation.\n'
            'Saved models were fitted with tdx.run in the mode recorded in config.json; '
            'the no-arm skeleton fits pooled_recent on train history and labels only. '
            'Machine-specific inference uses released test history. The emergency fallback fits pooled_recent on train.\n'
            f'Forecast SHA-256: {sha}.\n'
            'Training/refit: `python -m tdx.pipeline --round ' + round_name + ' --data <public>` in the project checkout.\n'
            'Organizer limits: 2 CPU cores, 16 GiB RAM, 30 minutes; no GPU or network.\n'
            'Validate the separate CSV with the released kit checker and run the project fresh-folder repro_check.sh.\n'
            'Fallback use is printed to stderr and /tmp/predict_fallback.json; never upload a fallback run.\n')
        archive = work / 'entry.zip'
        subprocess.run([sys.executable, str(ROOT / 'kit' / 'build_entry.py'), '--source', str(source), '--out', str(archive)], check=True)
        subprocess.run([sys.executable, str(ROOT / 'kit' / 'check_submission.py'), str(expected), '--data', str(data)], check=True)
        if out.exists():
            shutil.rmtree(out)
        shutil.move(str(source), out)
        os.replace(archive, zip_path)
        os.replace(expected, csv_path)
    print(f'ENTRY OK zip={zip_path} bytes={zip_path.stat().st_size} csv={csv_path} sha256={sha}', flush=True)
    return dict(entry=str(out), zip=str(zip_path), csv=str(csv_path), sha256=sha, requirements=pins)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arms')
    p.add_argument('--blend')
    p.add_argument('--round', choices=('main', 'adaptation'), default='main')
    p.add_argument('--data')
    p.add_argument('--out', default='entry')
    p.add_argument('--zip', default='entry_main.zip')
    p.add_argument('--csv')
    p.add_argument('--runs', default='runs')
    p.add_argument('--allow-dev', action='store_true', help='accept dev-mode (validated, no validation targets) exports')
    a = p.parse_args(argv)
    build_entry(arms=a.arms, blend=a.blend, round_name=a.round, data=a.data, out=a.out,
                zip_path=a.zip, runs=a.runs, csv_path=a.csv, allow_dev=a.allow_dev)


if __name__ == '__main__':
    main()
