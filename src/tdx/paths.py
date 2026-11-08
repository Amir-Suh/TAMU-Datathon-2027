"""Repository and round paths, independent of the caller's directory."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / 'kit'
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

def data_root(data=None, round_name='main'):
    return Path(data or os.environ.get('TDX_DATA_ROOT') or ROOT / 'data' / round_name).resolve()
