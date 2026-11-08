"""Summarize audited data opens and check successful runs for protected reads."""
import argparse
import json
import os
from pathlib import Path


def summarize(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["event"] == "start", path
    assert rows[-1]["event"] == "end" and rows[-1].get("exit") in (0, None), path
    assert "error" not in rows[-1], path
    stage = rows[0]["stage"]
    reads = sorted({
        r["path"] for r in rows if r["event"] == "open"
        and r["flags"] & os.O_ACCMODE != os.O_WRONLY
    })
    protected = {"targets.csv", "labels.csv", "reference.csv", "conditions.csv"}
    for read in reads:
        source = Path(read)
        if source.name not in protected:
            continue
        assert "test" not in source.parts, (path, read)
        if stage == "fit-predict":
            assert not {"validation", "lockbox"} & set(source.parts), (path, read)
    assert not [r for r in rows if r.get("allowed") is False], path
    data_reads = [s for s in reads if Path(s).name in protected | {
        "queries.csv", "history.csv", "metadata.json", "folds.csv"
    }]
    return {
        "path": str(path), "stage": stage, "script": rows[0]["script"],
        "audited_paths_read": len(reads),
        "data_reads": data_reads,
        "child_processes": sorted({r["command"] for r in rows if r["event"] == "child_process"}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("trace", type=Path, nargs="+")
    args = parser.parse_args()
    result = [summarize(path) for path in args.trace]
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified {len(result)} traces")


if __name__ == "__main__":
    main()
