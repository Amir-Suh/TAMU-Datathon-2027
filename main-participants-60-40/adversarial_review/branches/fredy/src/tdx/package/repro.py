"""Fresh-venv, offline, two-run verification for packaged entries.

Dependency setup accepts original wheels from TDX_WHEELHOUSE, or repacks exact
cached uv wheels into scratch. It never inherits system site packages or calls
an index. Network audit hooks cover Python sockets in the inference process.
"""
import argparse
import csv
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

from tdx.paths import ROOT
from slot_forecasting.submission_contract import extract_archive


def _name(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def cached_wheels(requirements, destination, cache=None):
    """Rebuild wheels from uv's extracted wheel cache, not the active venv."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    cache = Path(cache or os.environ.get('UV_CACHE_DIR', Path.home() / '.cache' / 'uv'))
    needed = dict(line.strip().split('==') for line in requirements if line.strip() and not line.startswith('#'))
    for name, version in needed.items():
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or not re.fullmatch(r'[A-Za-z0-9_.+!-]+', version):
            raise ValueError('Reproduction requires exact name==version pins')
        sources = sorted((cache / 'wheels-v6').glob(f'**/{_name(name)}/{version}-*'))
        candidates = []
        for source in sources:
            if not source.is_dir():
                continue
            infos = list(source.glob('*.dist-info'))
            for info in infos:
                meta = Parser().parsestr((info / 'METADATA').read_text())
                if _name(meta['Name']) != _name(name) or meta['Version'] != version:
                    continue
                wheel = Parser().parsestr((info / 'WHEEL').read_text())
                for tag in wheel.get_all('Tag', []):
                    py, abi, platform = tag.split('-')
                    if py not in ('cp313', 'py3', 'py2.py3'):
                        continue
                    if platform != 'any' and 'linux' not in platform:
                        continue
                    if platform != 'any' and not platform.endswith(os.uname().machine):
                        continue
                    candidates.append((source, info, tag))
        if not candidates:
            raise FileNotFoundError(f'No cached Python 3.13 wheel for {name}=={version}; supply TDX_WHEELHOUSE')
        source, info, tag = candidates[0]
        filename = info.name.removesuffix('.dist-info') + '-' + tag + '.whl'
        target = destination / filename
        with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(source.rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                    archive.write(path, path.relative_to(source))
    return destination


_GUARD = r'''
import json, os, pathlib, runpy, sys
sys.dont_write_bytecode = True
script = pathlib.Path(sys.argv[1]).resolve()
sys.argv = sys.argv[1:]
sys.path.insert(0, str(script.parent))
allowed_output = pathlib.Path(sys.argv[sys.argv.index('--out') + 1]).resolve()

def audit(event, args):
    if event in ('socket.connect', 'socket.connect_ex', 'socket.getaddrinfo', 'socket.sendto', 'socket.bind', 'subprocess.Popen', 'os.system'):
        raise RuntimeError('Offline reproduction blocked: ' + event)
    if event == 'open':
        path, mode, flags = args
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing and isinstance(path, (str, bytes, os.PathLike)):
            target = pathlib.Path(os.fsdecode(path)).resolve()
            if target != allowed_output and not target.is_relative_to('/tmp'):
                raise RuntimeError('Write outside output or /tmp: ' + str(target))
sys.addaudithook(audit)
runpy.run_path(str(script), run_name='__main__')
assert not any(n == 'torch' or n.startswith('torch.') for n in sys.modules), 'torch imported'
'''


def read_csv(path):
    columns = ['query_id', 'p_0', 'p_1', 'p_2', 'p_5', 'p_20']
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != columns:
            raise ValueError(f'{path}: unexpected CSV columns')
        values = {}
        for row in reader:
            if row['query_id'] in values:
                raise ValueError('Duplicate query ID')
            values[row['query_id']] = [float(row[c]) for c in columns[1:]]
    import math
    if not values or any(not math.isfinite(x) or not 0 < x < 1 for row in values.values() for x in row):
        raise ValueError('Invalid probabilities')
    return values


def compare_csv(first, second, uploaded):
    a, b, u = (read_csv(path) for path in (first, second, uploaded))
    if set(a) != set(b) or set(a) != set(u):
        raise ValueError('Query ID sets differ')
    d12 = max(abs(x - y) for key in a for x, y in zip(a[key], b[key]))
    d1u = max(abs(x - y) for key in a for x, y in zip(a[key], u[key]))
    print(f'max|run1-run2|={d12:.3e} max|run1-uploaded|={d1u:.3e} rows={len(a)}', flush=True)
    if d12 != 0 or d1u > 1e-9:
        raise ValueError('Reproduction differs; repeated CPU predictions must be exactly equal')
    return d12, d1u


def check(zip_path, data, uploaded, *, python=None, wheelhouse=None):
    zip_path, data, uploaded = (Path(p).resolve() for p in (zip_path, data, uploaded))
    python = python or os.environ.get('TDX_PY', str(Path.home() / '.venvs/tdx/bin/python'))
    version = subprocess.check_output([python, '-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'], text=True).strip()
    if version != '3.13':
        raise ValueError(f'Reproduction requires Python 3.13, got {version}')
    work = Path(tempfile.mkdtemp(prefix='tdx_repro.', dir='/tmp'))
    print(f'workdir {work}', flush=True)
    command = extract_archive(zip_path.read_bytes(), work / 'code')['command']
    code, venv = work / 'code', work / 'venv'
    if command != ['python', 'predict.py', '--data', '/data', '--out', '/output/submission.csv']:
        raise ValueError('Unexpected reproduction command; refusing to alter its semantics')
    subprocess.run([python, '-m', 'venv', '--without-pip', str(venv)], check=True)
    requirements = (code / 'requirements.txt').read_text().splitlines()
    wheels = Path(wheelhouse or os.environ.get('TDX_WHEELHOUSE', '')) if (wheelhouse or os.environ.get('TDX_WHEELHOUSE')) else cached_wheels(requirements, work / 'wheels')
    subprocess.run(['uv', 'pip', 'install', '--offline', '--no-cache', '--no-index', '--find-links', str(wheels),
                    '--python', str(venv / 'bin/python'), '-r', str(code / 'requirements.txt')], check=True)
    guard = work / 'guard.py'
    guard.write_text(_GUARD)
    for path in code.rglob('*'):
        path.chmod(0o555 if path.is_dir() else 0o444)
    code.chmod(0o555)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH', None)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    outputs, metrics = [], []
    # Choose CPUs actually allowed by this process; container CPU 0 may be absent.
    cpus = sorted(os.sched_getaffinity(0))[:2] if hasattr(os, 'sched_getaffinity') else []
    fallback = Path('/tmp/predict_fallback.json')
    for i in (1, 2):
        output = work / f'out{i}' / 'submission.csv'
        output.parent.mkdir()
        fallback.unlink(missing_ok=True)
        timing = work / f'time{i}.txt'
        invocation = [str(venv / 'bin/python'), '-I', str(guard), str(code / 'predict.py'), '--data', str(data), '--out', str(output)]
        if cpus and shutil.which('taskset'):
            invocation = ['taskset', '-c', ','.join(map(str, cpus)), *invocation]
        started = time.perf_counter()
        subprocess.run(['/usr/bin/time', '-f', 'wall=%es peakRSS=%MKB', '-o', str(timing), *invocation], cwd=code, env=env, check=True, timeout=1800)
        metric = dict(run=i, elapsed=time.perf_counter() - started, timing=timing.read_text().strip())
        print(f'run {i}: {metric["timing"]}', flush=True)
        metrics.append(metric)
        if fallback.exists():
            raise RuntimeError('FAIL: fallback used: ' + fallback.read_text())
        outputs.append(output)
    d12, d1u = compare_csv(*outputs, uploaded)
    for output in [*outputs, uploaded]:
        subprocess.run([str(venv / 'bin/python'), str(ROOT / 'kit/check_submission.py'), str(output), '--data', str(data)], check=True, env=env)
    report = dict(zip=str(zip_path), uploaded=str(uploaded), sha256=hashlib.sha256(uploaded.read_bytes()).hexdigest(),
                  requirements=requirements, workdir=str(work), runs=metrics, repeated_max_difference=d12,
                  uploaded_max_difference=d1u, torch_imported=False,
                  isolation='fresh Python 3.13 venv, read-only code, two CPU affinity, Python socket/subprocess audit')
    (work / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'REPRO OK report={work / "report.json"}', flush=True)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('zip')
    p.add_argument('data')
    p.add_argument('uploaded')
    p.add_argument('--python')
    p.add_argument('--wheelhouse')
    a = p.parse_args(argv)
    check(a.zip, a.data, a.uploaded, python=a.python, wheelhouse=a.wheelhouse)


if __name__ == '__main__':
    main()
