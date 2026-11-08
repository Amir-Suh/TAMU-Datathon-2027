"""Verify a fresh Torch-free replay against the fit/predict freeze."""
import argparse
from hashlib import sha256
import json
from pathlib import Path


def read_trace(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    args = parser.parse_args()
    attempt = args.attempt
    fitted = attempt / "full"
    replay = attempt / "replay"
    expected = json.loads((fitted / "forecast_freeze.json").read_text())
    assert json.loads((replay / "forecast_freeze.json").read_text()) == expected
    for name, digest in expected.items():
        contents = (fitted / name).read_bytes()
        assert sha256(contents).hexdigest() == digest, name
        assert (replay / name).read_bytes() == contents, name
    protected = {"labels.csv", "reference.csv", "conditions.csv", "targets.csv"}
    for trace_name in ("smoke_access.jsonl", "full_access.jsonl", "replay_access.jsonl"):
        trace = read_trace(attempt / trace_name)
        assert trace[-1].get("exit") == 0, trace[-1]
        assert not any(r.get("allowed") is False for r in trace), trace_name
        assert not any(r["event"] in {"child_process", "socket_connect"} for r in trace), trace_name
        for row in trace:
            if row["event"] != "open":
                continue
            path = Path(row["path"])
            if path.parts[:2] in {("public", "validation"), ("public", "test")}:
                assert path.name not in protected, row
            if trace_name == "replay_access.jsonl":
                assert path.parts[:2] != ("public", "train"), row
    print(f"Bitwise replay confirmed for {len(expected) - 1} validation/test CSVs and the counterfactual archive.")


if __name__ == "__main__":
    main()
