"""Run an arm's pilot variants in one process with shared cached inputs."""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from .run import arm_module, execute, parse_set
from .data import load_round
from .paths import data_root

def _one(job):
    root, rname, answers, arm, variant, mode, over, device, smoke = job
    rnd = load_round(root, rname, answers=answers)
    execute(rnd, arm, variant, mode, over, device=device, smoke=smoke)
    return variant

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', required=True)
    p.add_argument('--variants')
    p.add_argument('--mode', choices=('dev', 'final'), default='dev')
    p.add_argument('--round', choices=('main', 'adaptation'), default='main')
    p.add_argument('--data')
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--jobs', type=int, default=1, help='parallel worker processes (variants run concurrently)')
    p.add_argument('--threads', type=int, default=4, help='BLAS/OpenMP threads per worker when --jobs > 1')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--set', action='append', default=[])
    a = p.parse_args(argv)
    module = arm_module(a.arm)
    variants = a.variants.split(',') if a.variants else list(module.VARIANTS)
    root = data_root(a.data, a.round)
    if a.jobs > 1:
        for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[var] = str(a.threads)
        jobs = [(root, a.round, a.mode == 'final', a.arm, v, a.mode, parse_set(a.set), a.device, a.smoke) for v in variants]
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for v in ex.map(_one, jobs):
                print('done', v, flush=True)
        return
    rnd = load_round(root, a.round, answers=a.mode == 'final')
    for variant in variants:
        execute(rnd, a.arm, variant, a.mode, parse_set(a.set), device=a.device, smoke=a.smoke)

if __name__ == '__main__':
    main()
