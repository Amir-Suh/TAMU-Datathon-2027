# W6 — Packaging, offline reproduction, round-two refit

## Role and objective
Build the submission pipeline so that a valid entry exists early and any arm set can be dropped in later: `entry/` builder, `predict.py`, `reproduce.json`, `requirements.txt`, ZIP via the kit's `build_entry.py --source`, a fresh-venv offline reproduction check, and the one-command round-two refit.

## Why
The latest Form response counts even if invalid; the organizer reruns the code offline on 2 CPU cores with a 1e-8 tolerance (`kit/docs/README.md` section 7, `kit/docs/SUBMISSION_README.md`). Review row 16: GPU/threaded outputs can differ; ship numpy. Read first: those two docs, `kit/build_entry.py`, `kit/slot_forecasting/submission_contract.py`.

## Decisions (do not reopen)
- Entry layout (ZIP root): `README.md`, `reproduce.json` (`{"image":"python","command":["python","predict.py","--data","/data","--out","/output/submission.csv"]}`), `predict.py`, `requirements.txt` (exact pins; numpy always, lightgbm/scipy/scikit-learn only if a shipped arm needs them), `tdx/` (only the modules the shipped arms import, copied by the builder, not the whole repo), `slot_forecasting/` modules needed, `models/<arm>/…` (weights/ npz/ text models), `blend.json` (stack weights, arm list), `config.json` (seeds, variants).
- `predict.py --data D --out F`: loads queries and histories from `D/test`, ALSO reads `D/train` and `D/validation` when an arm fits per-machine components at run time (the GLM Level 1 only needs test history; no training at run time beyond that), runs shipped arms in float64 numpy, applies `blend.json`, clips/renormalises, writes the six-column CSV with 17 significant digits. No network, no writes outside `--out`, `/tmp` only for scratch. Single thread (`OMP_NUM_THREADS=1` etc. set at the top of `predict.py` before imports).
- Fallback chain inside `predict.py`: if a shipped arm raises, log to stderr and fall back to the next entry in `config.json["fallback"]`, last element `pooled_recent` computed from `D/train`. A silent fallback must also be recorded in `/tmp/predict_fallback.json` and printed so `repro_check.sh` fails on it.
- Builder CLI: `python -m tdx.package.build --arms glm/best,gbdt/best --blend runs/stack/s1/weights.json --round main --out entry/ --zip entry_main.zip`. With no `--arms` it builds the `pooled_recent` skeleton (first deliverable, used for the safe entry).
- Reproduction check: `claude_output/sh/repro_check.sh` creates a fresh venv with Python 3.13 from `requirements.txt` only, unzips the ZIP to a new folder, runs the `reproduce.json` command twice with `--data <public>`, compares the two outputs and the uploaded CSV (max abs diff must be ≤ 1e-9), runs `kit/check_submission.py`, prints wall time and peak RSS. Smoke-test this script on the `pooled_recent` skeleton.
- Round-two: `claude_output/sh/refit_round.sh <new_public_dir>` calls `python -m tdx.pipeline --round adaptation --data <dir>`; the pipeline (a) runs the main-chosen variants in `dev` on the new data and prints `compare`, (b) refits the stack weights on the new validation targets, (c) runs `final` mode, (d) builds `entry_adaptation.zip` and the CSV. Every step reads a `config/pipeline.json` (arm/variant list, written by Claude after the pilots; the skeleton lists `pooled/pooled_recent`). It must run end to end on the main data with `--round main` as its own test.

## Preconditions
Hard: W0 merged. Soft: W1/W2/W3/W4 outputs absent → ship only what exists; the builder takes an arm list.

## Owned files
`src/tdx/package/*`, `src/tdx/pipeline.py`, `config/pipeline.json`, `claude_output/sh/{repro_check.sh,refit_round.sh}` (edit the stubs), `tests/package/*`, `claude_output/notes/W6_*`. Forbidden: arm directories (read-only for the builder).

## Work packages
- **P1 skeleton**: build the `pooled_recent` entry, ZIP it with the kit builder, pass `repro_check.sh`. This is the first deliverable; report its CSV SHA-256 and the exact upload files.
- **P2 arm adapters**: each arm exposes `save/load` (W1–W4 provide); the builder copies the arm code + weights; add numpy-only inference check (`import torch` must not appear in `predict.py`'s import graph — assert with `sys.modules`).
- **P3 pipeline / refit**: as above; plus a `--dry-run` that prints the plan.
- **P4 notes**: `claude_output/notes/W6_package.md`: exact commands, runtimes, the size of the ZIP, and the list of requirements pinned.

## Acceptance
```
~/.venvs/tdx/bin/python -m pytest -q tests/package
~/.venvs/tdx/bin/python -m tdx.package.build --out entry/ --zip entry_main.zip
bash claude_output/sh/repro_check.sh entry_main.zip data/main runs/pooled/pooled_recent/test_pred.csv
~/.venvs/tdx/bin/python -m tdx.pipeline --round main --data data/main --dry-run
```
Time box 75 minutes for P1+P3 (P2 re-run at the end by the user's command once arms are selected).

## Models
| Package | Model | Effort | Why |
|---|---|---|---|
| P1, P3 | gpt-6.1-sol | high | offline-reproduction details are easy to get subtly wrong |
| P2 | gpt-6.1-sol | medium | adapters |
| P4 | gpt-6-luna | low | notes |
| Review (read-only) | gpt-6-astra | high | different model; asks "would this fail the organizer rerun?" |
