"""Apply the train-only P0 tail to each ordered-cell candidate."""
import argparse
import importlib
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path.cwd()))
p0 = importlib.import_module("signal_experiments.research_next.P0.run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sources", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    train = p0.load_split(Path("public"), "train")
    ratios = p0.fit(train.X, train.machine_type, alpha=0)
    for split in ("validation", "test"):
        queries_path = Path("public") / split / "queries.csv"
        query_ids = [q.query_id for q in p0.read_queries(queries_path)]
        for name in ("W1a", "W2"):
            source = args.sources / f"{split}_{name}.csv"
            output = args.out / f"{split}_{name}_P0.csv"
            p0.predict(queries_path, source, ratios, output)
            by_id = p0.read_submission(source, query_ids)
            predictions = p0.read_submission(output, query_ids)
            old = np.asarray([by_id[q] for q in query_ids])
            new = np.asarray([predictions[q] for q in query_ids])
            assert np.allclose(old[:, :2], new[:, :2], atol=1e-12, rtol=0)
            assert np.allclose(old[:, 2:].sum(1), new[:, 2:].sum(1), atol=1e-12, rtol=0)
            print(output, len(query_ids), flush=True)


if __name__ == "__main__":
    main()
