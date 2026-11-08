"""Reload frozen replay checkpoints using an archived Fredy source tree."""

import gzip
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
KIT = HERE.parents[1]
sys.path.append(str(KIT))

from slot_forecasting.io import read_submission
from tdx.arms.gbdt.arm import GBDTArm
from tdx.arms.seq.arm import SeqArm
from tdx.data import load_round


def main():
    rnd = load_round(KIT / "public", "main", answers=False)
    records = {}
    for directory in sorted((HERE / "checkpoints").iterdir()):
        name = directory.name
        with tempfile.TemporaryDirectory() as temporary:
            if (directory / "model.txt.gz").exists():
                temp = Path(temporary)
                shutil.copy(directory / "features.json", temp)
                with gzip.open(directory / "model.txt.gz") as source, (temp / "model.txt").open("wb") as out:
                    shutil.copyfileobj(source, out)
                model = GBDTArm.load(temp)
                splits = (("train", "train_forecasts"),)
            else:
                model = SeqArm.load(directory)
                splits = ()
            rows = {}
            splits += (("validation", "forecasts"), ("test", "test_forecasts"))
            for split, folder in splits:
                data = rnd.splits[split]
                ids = data.query_ids.ravel()
                expected = read_submission(HERE / folder / f"{name}.csv", ids)
                result = model.predict(data).reshape(-1, 5)
                error = float(np.max(np.abs(result - np.array([expected[key] for key in ids]))))
                if not np.isfinite(error) or error > 1e-8:
                    raise ValueError(f"{name}/{split} prediction replay differs by {error}")
                rows[split] = {"queries": len(ids), "max_absolute_difference": error}
            records[name] = rows
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
