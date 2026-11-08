"""Verify immutable inputs, split boundaries, and historical cutoff boundaries."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path):
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("manifest", type=Path)
    cli.add_argument("out", type=Path)
    args = cli.parse_args()
    manifest = json.loads(args.manifest.read_text())
    bad_hashes = [
        name for name, expected in manifest["input_sha256"].items()
        if not Path(name).is_file() or sha256(Path(name)) != expected
    ]
    if bad_hashes:
        raise ValueError(f"Frozen inputs changed: {bad_hashes}")

    with Path(manifest["folds"]).open(newline="") as source:
        fold_rows = list(csv.DictReader(source))
    fold_keys = {(r["split"], r["episode_id"]) for r in fold_rows}
    assert len(fold_keys) == len(fold_rows)
    assert all(r["fold"] in "0123" and len(r["fold"]) == 1 for r in fold_rows)
    across_splits = set()
    report = {}
    for split in ("train", "validation", "test"):
        folder = Path(manifest["data"]) / split
        with (folder / "queries.csv").open(newline="") as source:
            queries = list(csv.DictReader(source))
        cutoffs = {(q["episode_id"], q["machine_id"]): int(q["cutoff"]) for q in queries}
        episodes = {key[0] for key in cutoffs}
        assert len(cutoffs) == len(queries)
        assert not across_splits & episodes
        across_splits.update(episodes)
        if split != "test":
            assert {episode for part, episode in fold_keys if part == split} == episodes
        seen = set()
        max_tick_minus_cutoff = None
        with (folder / "history.csv").open(newline="") as source:
            for row in csv.DictReader(source):
                pair = (row["episode_id"], row["machine_id"])
                tick = int(row["tick"])
                gap = tick - cutoffs[pair]
                assert 0 <= tick and gap < 0
                key = (*pair, tick)
                assert key not in seen
                seen.add(key)
                max_tick_minus_cutoff = (
                    gap if max_tick_minus_cutoff is None else max(max_tick_minus_cutoff, gap)
                )
        report[split] = {
            "episodes": len(episodes),
            "queries": len(queries),
            "history_rows": len(seen),
            "max_tick_minus_cutoff": max_tick_minus_cutoff,
        }
    args.out.write_text(json.dumps({
        "verified_input_hashes": len(manifest["input_sha256"]),
        "manifest_sha256": sha256(args.manifest),
        "episode_ids_disjoint": True,
        "history_pre_cutoff": True,
        "fold_coverage": True,
        "splits": report,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
