# COMMON CONTRACT (prepended to every Codex stream)

You are one Codex stream in the House Edge datathon repo (tag `tdx`). Read `AGENTS.md` and `claude_output/02_BAKEOFF_PLAN.md` first. This pace is fast: run experiments directly, do not wait for approvals, do not ask questions unless blocked by a missing file or a decision this spec does not cover (then choose the simpler option and report it).

## 0. Environment
- Python: `~/.venvs/tdx/bin/python` (3.13; numpy, scipy, scikit-learn, lightgbm, torch, pytest are installed). Nothing installs in the sandbox; if a package is missing, use stdlib/numpy or report it.
- Set `PYTHONDONTWRITEBYTECODE=1`. Work only inside your clone (your cwd). Data: `data/main/` (symlink; train, validation, test with the usual CSVs). Kit code: `kit/slot_forecasting` (add `kit/` to `sys.path` via `tdx.paths`).
- CPU budget when you run things: up to 12 threads per stream (24-core host, refinement wave). Run grids in parallel with `python -m tdx.sweep --arm A --jobs 4 --threads 3`. Only the inference-timing check in the final package uses the organizer limit of 2 cores (`taskset -c 0,1`). Pilot wave W1/W2/W4/W6 started with the old cap of 8.
- Do not commit. Leave changes uncommitted and end with the list of changed paths.

## 1. Shared interfaces (W0 implements; W1–W6 import, never reimplement)
```python
# src/tdx/data.py
PAYOUTS = (0, 1, 2, 5, 20)           # class index 0..4
class Split:   # one of train / validation / test
    name: str; ids: list[str]        # episode ids, sorted
    y: np.ndarray        # (E, 8, 256) int8, class 0..4, -1 = not observed (and every tick >= cutoff)
    mtype: np.ndarray    # (E, 8) int8: 0 steady, 1 balanced, 2 jackpot
    signal: np.ndarray   # (E, 8) int8 in {-1, 0, 1}  (cutoff-level only)
    cutoff: np.ndarray   # (E,) int16
    n_obs: np.ndarray    # (E, 8) int16
    query_ids: np.ndarray  # (E, 8) object
    label: np.ndarray | None      # (E, 8) int8 class of the realised cutoff payout (train, validation)
    condition: np.ndarray | None  # (E,) object in {stable, short_history, drift, shared_change} (train, validation)
    target: np.ndarray | None     # (E, 8, 5) float64 true probabilities (validation)
    reference: np.ndarray | None  # (E, 8, 5) float64 frozen pooled forecasts (validation)
class Round: name: str; root: Path; splits: dict[str, Split]; max_points: float
def load_round(root, name=None) -> Round      # cached under .cache/ keyed by metadata hash
# src/tdx/fit.py
@dataclass(frozen=True) class FitSpec: tick_splits: tuple[str, ...]; label_splits: tuple[str, ...]; soft_splits: tuple[str, ...]
def dev_spec(transductive: bool) -> FitSpec    # tick: train (+validation,test if transductive); label: train; soft: ()
def final_spec() -> FitSpec                     # tick: train,validation,test; label: train,validation; soft: validation
# src/tdx/arm.py
class Arm:                                      # base class
    def fit(self, rnd: Round, spec: FitSpec) -> None: ...
    def predict(self, split: Split) -> np.ndarray: ...     # (E, 8, 5) float64, rows sum to 1, all > 0
    def save(self, directory: Path) -> None: raise NotImplementedError
    @classmethod
    def load(cls, directory: Path) -> "Arm": raise NotImplementedError
# src/tdx/arms/<name>/__init__.py exposes  VARIANTS: dict[str, dict]  and  build(**params) -> Arm
# src/tdx/feat.py
def lag_classes(y, L) -> (E, 8, 256, L) int8      # [.., t, k-1] = y[.., t-k] or -1; strictly past only
def others_win(y, lags) -> (E, 8, 256, len(lags)) # number of the OTHER 7 machines with payout > 0 at t-lag (-1 if any missing)
def tick_rows(split) -> (e_idx, m_idx, t_idx)      # every observed (e, m, t) usable as a label
# src/tdx/score.py
def score(rnd, pred_val) -> dict    # {"points", "mean_loss_ratio", "by_condition": {c: ratio}, "audit": {...}, "tuning": {...}}  via kit contest_score
def episode_values(rnd, pred_val) -> dict   # per-episode error and reference error, for bootstrap
# CLI
python -m tdx.run --arm NAME --variant VNAME --mode dev|final [--set key=value ...] [--round main|adaptation] [--data DIR] [--out runs/NAME/VNAME]
python -m tdx.sweep --arm NAME [--variants a,b,c] [--mode dev] [--device auto|cpu|cuda] [--smoke]   # runs the arm's VARIANTS in one process (data cached); --device/--smoke are passed to build(**params) only if the arm accepts them
python -m tdx.tools.compare runs/ [--baseline pooled_recent]            # table + paired episode bootstrap
python -m tdx.tools.leak_test --arm NAME [--variant VNAME]
```
`run` writes `runs/NAME/VNAME/{val_pred.npy, val_pred.csv, test_pred.csv, score.json, meta.json}`; in `final` mode it writes `test_pred.csv`, `test_pred.npy`, and `meta.json` only, and the model directory `model/` if `save` is implemented. It appends one line to `runs/LOG.md`: `time | arm | variant | mode | points | stable | short | drift | shared | seconds`.

## 2. Hard rules (leakage and correctness)
1. A prediction at tick `t` uses only ticks `< t` of the same episode. Missing ticks are unknown, never zero.
2. `signal` is cutoff-level: tick-level feature builders must not receive it.
3. In `dev` mode never read `validation` labels/targets/conditions or any test file other than queries and history. `transductive` means validation/test HISTORY rows only.
4. Never split an episode across folds. Never read private files, never use the network.
5. Output probabilities are float64, finite, in (0, 1), each row sums to 1 within 1e-12 (clip at 1e-6 then renormalise).
6. Deterministic: fixed seeds; the same command twice gives max abs difference 0.0 on CPU (state any exception).

## 3. Models available (paste of the current catalog; check with `codex debug models`)
`gpt-6-astra` (frontier), `gpt-6.1-sol` (workhorse), `gpt-6-sol`, `gpt-6-luna` (fast/cheap), `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`. Efforts: low, medium, high, xhigh. Use the cheapest model that can be verified by a stronger check; the reviewer is a different model from the implementer. Your final report must include the per-package delegation table (package, model, effort, why) and say which mechanism you used to pick a subagent's model.

## 4. Bounds
- If the spec has a soft slip (a file called new that exists, a path that needs a one-line fix), fix it and report it. Stop only for: scope you were not given, a missing hard precondition, a read-only tree, network/compute needs, or a decision that is the user's.
- Never edit another stream's owned paths. Never edit a test's expectation to make it pass without saying which and why in the report.
- Keep runs inside the time box in your spec; if a variant exceeds it, record the timeout and move on.

## 4b. Concise writing
Say each thing once. No restating in another format, no closing summary, no padding. Every sentence adds a fact, reason, step, number or decision.

## 5. Final report (write to the `-o` path; also copy to `claude_output/notes/<stream>_report.md`)
1. Status per work package (done / partial / not done) and the exact command that proves it with its printed output (numbers, not adjectives).
2. Files created/edited (list).
3. Pilot results table: variant, points, four ratios, seconds (copy from `runs/LOG.md`).
4. Surprises and open risks (max 8 lines).
5. Delegation table (see section 3).
6. Changed paths for the user to commit (explicit paths, no wildcards).
