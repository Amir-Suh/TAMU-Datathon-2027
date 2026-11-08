# W6 package and pipeline handoff

Status: the main pooled skeleton was built and reproduced offline. The final pipeline artifacts are in `runs/pipeline/main/`; upload the CSV and ZIP below as separate files. The prior standalone skeleton ZIP is stale and must not be uploaded.

## Exact artifacts

| File | Size | SHA-256 |
|---|---:|---|
| `runs/pipeline/main/entry_main.zip` | 18,088 bytes | `a91736b0aa973c38f98da162bcc75e34bb203a9189a0a1f9ac664bf42f65c123` |
| `runs/pipeline/main/submission.csv` | 199,404 bytes | `1fdd80f9ad7395f6bf4c511bd57129ad4dfa4ec393eb098b3f11aa3dfc18b77a` |

CSV SHA-256 from the kit checker: `1fdd80f9ad7395f6bf4c511bd57129ad4dfa4ec393eb098b3f11aa3dfc18b77a` (1,536 main/test queries). The ZIP contains `numpy==2.5.4` as its only pinned requirement; runtime was Python 3.13.15. The shipped arm is `pooled/pooled_recent`, blend weight 1.0.

## Commands and recorded runtimes

Run from the repository root:

```bash
~/.venvs/tdx/bin/python -m tdx.pipeline --round main --data data/main --dry-run
~/.venvs/tdx/bin/python -m tdx.pipeline --round main --data data/main
~/.venvs/tdx/bin/python -m pytest -q tests/package
python kit/check_submission.py runs/pipeline/main/submission.csv
bash claude_output/sh/repro_check.sh runs/pipeline/main/entry_main.zip data/main runs/pipeline/main/submission.csv
```

Recorded pipeline log: pooled recent dev prediction 0.063 s; final prediction 0.012 s; entry build and submission check completed, then fresh-venv reproduction completed twice. Fresh reproduction used a new Python 3.13.15 venv, installed only numpy 2.5.4, and reported 0.46 s / 149,248 KB peak RSS on run 1 and 0.44 s / 149,552 KB on run 2. Maximum repeated and uploaded CSV differences were both 0.000e+00; torch was not imported. Each kit check reported `VALID FORMAT: 1536 queries | main | test` and the CSV hash above. Detailed machine-readable record: `runs/pipeline/main/repro_report.json`.

The acceptance command above for tests must be rerun after the fallback comparison fix: the recorded earlier `runs/package/pytest.log` has 33 passed and one failed because the fallback output differed from the primary by 1.110e-16 and the test required bitwise equality between unlike forecast paths. This is not a fresh reproduction failure; the pipeline’s fresh run matched the uploaded CSV exactly. Use the latest test output for final status.

Round two is configured for a one-command refit, after the adaptation kit is available:

```bash
bash claude_output/sh/refit_round.sh <new_public_dir>
```

## Pilot rows (from `runs/LOG.md`)

| Variant | Points | Stable | Short | Drift | Shared | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| pooled / pooled_recent | 30.825854 | 0.958227 | 1.000561 | 0.946405 | 0.880479 | 0.091 |
| pooled / pooled | 30.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.054 |
| pooled / recent | 30.743191 | 0.985914 | 1.017846 | 0.938333 | 0.864513 | 0.054 |
| pooled / context | 29.685186 | 0.990280 | 1.079960 | 1.029656 | 0.984945 | 0.050 |

Pipeline paired episode comparison selected `pooled_recent` as the configured baseline; with only itself in the final blend, the displayed paired interval was `[+0.000, +0.000]`. No validation score is claimed for the final refit.

## Upload

Upload exactly these two matching files for the current main round:

- `runs/pipeline/main/submission.csv`
- `runs/pipeline/main/entry_main.zip`

Do not substitute `runs/pooled/pooled_recent/test_pred.csv` or the stale root `entry_main.zip`; those belong to the earlier skeleton build and have different hashes.
