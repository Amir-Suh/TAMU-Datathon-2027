"""Assemble trained seq models into a packaged `seqens` export under runs/seqens/<variant>/.

usage: python -m tdx.tools.make_ensemble --variant ens12 --members runs/seq/a/model runs/seq/b/model ... [--mode dev|final]
The members' own meta.json files (one folder up from model/) are checked to share the data root and mode.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from tdx.arms.seqens.arm import assemble
from tdx.paths import ROOT


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', required=True)
    p.add_argument('--members', nargs='+', required=True)
    p.add_argument('--runs', default=str(ROOT / 'runs'))
    a = p.parse_args(argv)
    metas = [json.loads((Path(m).parent / 'meta.json').read_text()) for m in a.members]
    modes = {m['mode'] for m in metas}
    roots = {m['data'] for m in metas}
    shas = {m['metadata_sha256'] for m in metas}
    if len(modes) != 1 or len(roots) != 1 or len(shas) != 1:
        raise SystemExit(f'members disagree on mode/data/metadata: {modes} {roots} {shas}')
    out = Path(a.runs) / 'seqens' / a.variant
    if out.exists():
        shutil.rmtree(out)
    assemble([Path(m) for m in a.members], out / 'model')
    first = metas[0]
    meta = dict(arm='seqens', variant=a.variant, mode=modes.pop(), round=first['round'], data=first['data'],
                data_root=first['data'], metadata_sha256=first['metadata_sha256'],
                params={'members': [str(Path(m).parent) for m in a.members], 'member_params': first['params']})
    (out / 'meta.json').write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
    print(f'seqens/{a.variant}: {len(a.members)} members, mode={meta["mode"]} -> {out}')


if __name__ == '__main__':
    main()
