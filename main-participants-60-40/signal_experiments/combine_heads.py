"""Pair one model's total win mass with another's positive payout proportions."""

import argparse
import math
from pathlib import Path
import sys

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))

from slot_forecasting.io import read_queries, read_submission, write_submission


def combine(queries, win_csv, size_csv, output):
    ids = [query.query_id for query in read_queries(queries)]
    win_forecast = read_submission(win_csv, ids)
    size_forecast = read_submission(size_csv, ids)
    combined = {}
    for key in ids:
        p0 = win_forecast[key][0]
        payouts = size_forecast[key][1:]
        mass = math.fsum(payouts)
        if mass <= 0:
            raise ValueError(f"Cannot compute the conditional payout mix for {key}.")
        combined[key] = (p0, *(p / mass * (1 - p0) for p in payouts))
    write_submission(output, combined)
    saved = read_submission(output, ids)
    if max(abs(saved[key][0] - win_forecast[key][0]) for key in ids) > 1e-12:
        raise ValueError("Combined probabilities changed the win source.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--win", type=Path, required=True)
    parser.add_argument("--size", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve() in {args.win.resolve(), args.size.resolve(), args.queries.resolve()}:
        parser.error("Use a separate output file.")
    combine(args.queries, args.win, args.size, args.out)


if __name__ == "__main__":
    main()
