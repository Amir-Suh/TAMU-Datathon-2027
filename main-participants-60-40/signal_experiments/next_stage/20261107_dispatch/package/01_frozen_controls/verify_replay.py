"""Check clean container predictions against the frozen candidates by query ID."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
KIT = HERE.parents[4]
DISPATCH = HERE.parents[1]
SIGNAL = KIT / "signal_experiments"
FROZEN = KIT / "m5_m6_repair/results/repaired-m5-m6"
sys.path.insert(0, str(KIT))

from slot_forecasting.io import read_queries, read_submission
from slot_forecasting.submission_contract import inspect_archive

PAIRS = {
    "m3": (HERE, FROZEN / "m3/submission.csv", FROZEN / "m3/validation_predictions.csv"),
    "hybrid": (HERE,
               SIGNAL / "combined/m3_match_split_win_pair_l31_factor_gru_size_test.csv",
               SIGNAL / "combined/m3_match_split_win_pair_l31_factor_gru_size.csv"),
    "w1a_p0": (HERE.parent / "02_w1a_p0",
               DISPATCH / "coordinator/assembly/test_W1a_P0.csv",
               DISPATCH / "coordinator/assembly/validation_W1a_P0.csv"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(expected_path, actual_path, split):
    ids = [q.query_id for q in read_queries(KIT / "public" / split / "queries.csv")]
    expected = read_submission(expected_path, ids)
    actual = read_submission(actual_path, ids)
    error = max(abs(a - b) for key in ids for a, b in zip(expected[key], actual[key]))
    assert error <= 1e-8, (split, error)
    assert actual_path.stat().st_size <= 10_000_000
    for path in (expected_path, actual_path):
        subprocess.run([sys.executable, "-B", "check_submission.py", str(path), "--split", split],
                       cwd=KIT, check=True)
    return {"expected_csv": str(expected_path.relative_to(KIT)), "expected_sha256": digest(expected_path),
            "replay_sha256": digest(actual_path), "query_count": len(ids),
            "exact_ids": True, "max_normalized_probability_difference": error}


def verify_archive(path):
    config = inspect_archive(path.read_bytes())
    with zipfile.ZipFile(path) as z:
        entries = z.infolist()
        expanded = sum(entry.file_size for entry in entries)
        assert len(entries) <= 5_000
        assert expanded <= 512 * 1024 ** 2
        assert path.stat().st_size <= 100_000_000
        assert all(not entry.filename.endswith((".csv", ".joblib", ".pkl", ".pt")) for entry in entries)
        members = {name: hashlib.sha256(z.read(name)).hexdigest() for name in z.namelist()}
    return {"path": str(path.relative_to(KIT)), "sha256": digest(path),
            "compressed_bytes": path.stat().st_size, "unpacked_bytes": expanded,
            "entries": len(entries), "config": config, "member_sha256": members}


def verify_access(path, split):
    trace = [json.loads(line) for line in path.read_text().splitlines()]
    assert trace[0]["event"] == "start" and trace[-1]["event"] == "end"
    assert all(record.get("allowed", True) for record in trace)
    observed = {record["path"] for record in trace
                if record["event"] == "open" and record["path"].startswith("/data/")}
    assert observed == {f"/data/{split}/history.csv", f"/data/{split}/queries.csv"}
    return sorted(observed)


def verify(scratch, arm):
    report_dir, test_candidate, validation_candidate = PAIRS[arm]
    archive = verify_archive(report_dir / f"{arm}_code.zip")
    run = scratch / f"container-{arm}"
    exact = run / "submission.csv"
    assert exact.read_bytes() == (run / "test_audited.csv").read_bytes()
    test = compare(test_candidate, exact, "test")
    val = compare(validation_candidate, run / "validation_audited.csv", "validation")
    test["accessed_data"] = verify_access(run / "test_trace.jsonl", "test")
    val["accessed_data"] = verify_access(run / "validation_trace.jsonl", "validation")
    runtime = json.loads((run / "runtime.json").read_text())
    assert runtime["test"]["command"] == archive["config"]["command"]
    assert runtime["test"]["elapsed_seconds"] < 1800
    assert runtime["test"]["peak_rss_bytes"] < 16 * 1024 ** 3
    for source, output in {"submission.csv": "test_replay.csv", "validation_audited.csv": "validation_replay.csv",
                           "test_trace.jsonl": "test_access.jsonl", "validation_trace.jsonl": "validation_access.jsonl"}.items():
        shutil.copyfile(run / source, report_dir / f"{arm}_{output}")
    result = {"archive": archive, "runtime": runtime, "test": test, "validation": val}
    (report_dir / f"{arm}_replay.json").write_text(json.dumps(result, indent=2) + "\n")
    print(arm, "sha256", archive["sha256"], "test error", test["max_normalized_probability_difference"],
          "validation error", val["max_normalized_probability_difference"], "wall s",
          runtime["test"]["elapsed_seconds"], "peak RSS bytes", runtime["test"]["peak_rss_bytes"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scratch", type=Path)
    parser.add_argument("arms", nargs="+", choices=PAIRS)
    args = parser.parse_args()
    for arm in args.arms:
        verify(args.scratch, arm)


if __name__ == "__main__":
    main()
