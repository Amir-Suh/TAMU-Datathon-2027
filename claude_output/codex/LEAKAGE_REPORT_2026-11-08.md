**Found future-observedness leakage in `moegate`, `moegate4`, and flagged `seq2`, plus two contract violations affecting backtesting and its leakage check.** These model defects are absent from the members of `candidate3b_clean_refit`. I did not demonstrate a new forbidden-input dependency in that ZIP.

Read `AGENTS.md` and `kit/docs/RULES.md:30–40` first. All tests were read-only. No files were created or changed.

**Findings, ordered by danger**

Reproductions P1–P4 below are pasteable from the repository root. Probability/logit changes from initialized weights demonstrate architectural dependence; they are **not estimates of contest-point gains**.

| ID | Severity | File:line | Input that leaks and into what | Evidence: command and quoted output | Estimated impact | Minimal fix |
|---|---|---|---|---|---|---|
| F1 | **CONFIRMED LEAK** | [moegate/arm.py:79](/home/ganfwe/Downloads/TAMU-Datathon-2027/src/tdx/arms/moegate/arm.py:79), training lines 207, 296, 325–331 | Whether future ticks are observed determines the whole-row `count == 64` flag, which enters gates for earlier next-tick training examples. | **P1:** `"moegate ... remove 0.005350618 add 0.005350618"` for outputs at steps 0–31 after changing only tick 63 or 100. Payout-value changes and count-preserving moves: `0.0`. | Violates causality during historical training. Learned-model score effect cannot be measured without corrected refits. No direct effect on clean-refit ZIP: arm absent. | Compute flag/features from observations through the current input tick, identically in Torch and NumPy; retrain. |
| F2 | **CONFIRMED LEAK** | [moegate4/arm.py:80](/home/ganfwe/Downloads/TAMU-Datathon-2027/src/tdx/arms/moegate4/arm.py:80), training lines 221, 331, 360–365 | Same future-observedness flag enters every historical gate. | **P1:** `"moegate4 ... remove 0.002312858 add 0.002312858"`. Changing cutoff 64→63 also changes outputs at steps 0–31 by `0.002312858`. | Same scope as F1. Auxiliary condition supervision is **not** an additional demonstrated leak. Arm absent from clean ZIP. | Same prefix-based fix as F1. Keep condition supervision restricted by `label_splits`. |
| F3 | **CONFIRMED LEAK** | [seq2/arm.py:39](/home/ganfwe/Downloads/TAMU-Datathon-2027/src/tdx/arms/seq2/arm.py:39), lines 112–117, 145–155 | With `short_flag=True`, the final row’s observed count selects an embedding fed into the GRU from its first step. | **P1:** `"seq2 ... remove 0.173931807 add 0.173931807"` in earlier **logits**. Its existing cutoff audit nevertheless returns `0.0`. | Historical training violates causality for flagged variants. Unflagged `seq` exports are unaffected. No direct clean-ZIP effect. | Use a causal per-step count/flag, or remove the flag; retrain affected variants. |
| F4 | **CONFIRMED CONTRACT VIOLATION** | [shortctx/arm.py:32](/home/ganfwe/Downloads/TAMU-Datathon-2027/src/tdx/arms/shortctx/arm.py:32), lines 226–243 | Missing recent ticks reach `w['tok.weight'][tokens]` as `-1`. NumPy selects the last embedding: class 4, payout **20**. Missing observations become jackpots. | **P2:** `"missing recent tokens [1325, 1165, 989, 821, 660, 501, 331, 165] total 5957"`; replacing every `-1` token with `4` gives `"missing_vs_jackpot maxdelta 0.0"`. | Corrupts `shortctx` forecasts on the actual short-backtest prefixes: 5,957/61,440 recent-token positions, **9.70%**. Metric bias may have either sign; no defensible point estimate without a corrected trained model. Actual released query layouts had zero such missing recent slots. | Add a distinct missing token and consistent training support, or use an explicit fallback for unsupported layouts. Never index payout embeddings with negative tokens. |
| F5 | **CONFIRMED CONTRACT VIOLATION** | [tools/leak_test.py:74](/home/ganfwe/Downloads/TAMU-Datathon-2027/src/tdx/tools/leak_test.py:74) | `truncate_for_probe` moves cutoffs earlier but retains the original cutoff-only signal. The advertised “legal earlier-cutoff histories” are not legal pseudo-queries. | **P2:** `"retained nonzero signals 28 changed queries 40"`. **P1:** all three leaking architectures still return `"existing_cutoff_audit 0.0"`. | No direct score output, but the mandatory audit gives false reassurance: it tests storage after the query cutoff, not historical forward-pass causality or signal availability. | Neutralize signal on changed cutoffs. Add per-tick future-value **and future-observedness** perturbations to the audit. |
| S1 | **SUSPECTED** — isolation gap, no forbidden read demonstrated | [package/repro.py:79](/home/ganfwe/Downloads/TAMU-Datathon-2027/src/tdx/package/repro.py:79) | Reproduction’s guard restricts writes/network/subprocesses but permits arbitrary local reads, including validation answers or another data root. | Executed the guard’s `audit` function with simulated read events: `"repro guard read allowed data/main/validation/targets.csv"` and `"repro guard read allowed /main_data/validation/targets.csv"`. No private file was opened. | A successful reproduction does not prove data confinement. Current inspected clean-ZIP inference does not use these paths. | Run with filesystem isolation or add an explicit data-read policy alongside the existing write/network guard. |
| S2 | **SUSPECTED** — dormant unsafe loader | [m3flip/amir/common/data.py:107](/home/ganfwe/Downloads/TAMU-Datathon-2027/src/tdx/arms/m3flip/amir/common/data.py:107), lines 115–120 | Vendored `load_split(public_dir, name)` loads optional labels/targets/reference for **any** split name, including `test`, if files exist. | Source inspection: `if (d / "labels.csv").exists():` and `for attr, fname in (("targets", "targets.csv"), ("reference", "reference.csv")):` lack a split restriction. | No active leak found: wrapper and packaged inference use `tdx.data._read_split` plus `to_amir`; local test directory has no answers. | Refuse answer loading for `name == 'test'`; require explicit opt-in for validation answers. |
| R1 | **REFUTED** | `moegate/arm.py:25–59,132`; corresponding `moegate4` code | Alleged lookahead from pair statistics and the `t+1` index. | Literal manual pair counting at prediction tick 73: `"pair manual vs index q delta 0.0"`. Future perturbations at 32/64/96 left features through those prediction ticks identical. | No demonstrated inflation from pair alignment. | None. |
| R2 | **REFUTED** | `run.py:24–30`; `tools/short_backtest.py:30–60`; `tools/cv_blend.py:29–55`; `tools/halfval.py:44–54` | Validation-answer exposure in current dev orchestration; scored-episode overlap in repaired short backtest; held-fold targets entering blend-weight fit. | `"dev forbidden fields 0"`; `"short fit splits ['train'] scored episodes 24 overlap 0"`; **P4:** `"held-target weight delta 0.0"`; `"held supervised cells 0"`. | No mechanical score inflation demonstrated through these paths. Historical checkpoint provenance remains a limitation below. | None for the tested mechanisms. |

For F1–F3, changing only the stored `n_obs` field did nothing: each arm recomputes the dangerous count from the entire `y` row. Changing cutoff 64→63 before the same 32-step forward pass reproduced each leak. At the **actual** query cutoff, the count itself is legal; broadcasting it backward during training is the violation.

**P1 — reproduce F1–F3 and the existing audit’s blind spot**

This uses deterministic initialized weights because no trained exports for these arms are available here. It performs no fit and writes nothing.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OPENBLAS_NUM_THREADS=1 ~/.venvs/tdx/bin/python - <<'PY'
import importlib
import numpy as np
from dataclasses import replace
from tdx.data import Split
from tdx.tools.leak_test import assert_prediction_causality

rng = np.random.RandomState(101)
H = 16
def a(*shape):
    return rng.normal(0, .2, shape)

y = np.zeros((1, 8, 128), int)
y[:, :, 64:] = -1
mt = np.zeros((1, 8), int)
s = Split('validation', ['e'], y, mt, mt, np.array([64]),
          np.full((1, 8), 64), np.arange(8)[None])

for name, K in [('moegate', 3), ('moegate4', 4), ('seq2', 0)]:
    m = importlib.import_module('tdx.arms.' + name + '.arm')
    w = dict(tok=a(6, 8), typ=a(4, 4),
             wih=a(3*H, 12 if K else 16), whh=a(3*H, H),
             bih=a(3*H), bhh=a(3*H))
    if K:
        w.update(gw=a(K, H+7), gb=a(K))
        for k in range(K):
            w.update({f'e{k}_w1': a(8, H), f'e{k}_b1': a(8),
                      f'e{k}_w2': a(5, 8), f'e{k}_b2': a(5)})
        cls = m.MoeGateArm if K == 3 else m.MoeGate4Arm
        arm = cls(h=H, d_expert=8)
        fn = lambda z: m.forward_np(w, z, mt, 32, H, K, 8)[0]
    else:
        w.update(flg=a(2, 4), hw=a(5, H), hb=a(5))
        arm = m.Seq2Arm(h=H, short_flag=True)
        fn = lambda z: m.forward_np(w, z, mt, 32, H, False, False, True)
    arm.w = w

    values = y.copy(); values[:, :, 32:64] = 4
    remove = y.copy(); remove[:, :, 63] = -1
    add = y.copy(); add[:, :, 100] = 2
    move = add.copy(); move[:, :, 63] = -1
    deltas = [np.max(abs(fn(y)-fn(z))) for z in (values, remove, add, move)]
    print(name, 'value %.9f remove %.9f add %.9f move %.9f' % tuple(deltas))
    print(name, 'existing_cutoff_audit', assert_prediction_causality(arm, s),
          'n_obs_delta', np.max(abs(
              arm.predict(s)-arm.predict(replace(s, n_obs=np.zeros_like(mt))))))
    print(name, 'cutoff_delta', np.max(abs(
        fn(m._masked(s))-fn(m._masked(replace(s, cutoff=np.array([63])))))))
PY
```

Observed principal output:

```text
moegate value 0.000000000 remove 0.005350618 add 0.005350618 move 0.000000000
moegate existing_cutoff_audit 0.0 n_obs_delta 0.0
moegate4 value 0.000000000 remove 0.002312858 add 0.002312858 move 0.000000000
moegate4 existing_cutoff_audit 0.0 n_obs_delta 0.0
seq2 value 0.000000000 remove 0.173931807 add 0.173931807 move 0.000000000
seq2 existing_cutoff_audit 0.0 n_obs_delta 0.0
```

**P2 — reproduce F4–F5 on the released validation histories**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OPENBLAS_NUM_THREADS=1 ~/.venvs/tdx/bin/python - <<'PY'
import numpy as np
from pathlib import Path
from tdx.data import _read_split, Round
from tdx.tools.short_backtest import collect_short_episodes, short_windows
from tdx.tools.leak_test import truncate_for_probe
from tdx.arms.shortctx.arm import forward_np

root = Path('data/main').resolve()
v = _read_split(root, 'validation')
r = Round('main', root, {'validation': v}, 60)
short = collect_short_episodes(r)
counts = []
for p, _ in short_windows(short):
    counts.append(sum(np.count_nonzero(p.y[e, :, int(c)-40:int(c)] < 0)
                      for e, c in enumerate(p.cutoff)))
print('missing recent tokens', counts, 'total', sum(counts))

probe = truncate_for_probe(v)
changed = probe.cutoff != v.cutoff
print('retained nonzero signals', np.count_nonzero(probe.signal[changed]),
      'changed queries', int(changed.sum()*8))

H = 16
rng = np.random.RandomState(8)
shapes = {
    'tok.weight': (5,H), 'off.weight': (64,H),
    'typ.weight': (4,H), 'flg.weight': (2,H),
    'cell.weight_ih': (3*H,H), 'cell.weight_hh': (3*H,H),
    'cell.bias_ih': (3*H,), 'cell.bias_hh': (3*H,),
    'fc1.weight': (H,2*H), 'fc1.bias': (H,),
    'fc2.weight': (5,H), 'fc2.bias': (5,)
}
w = {k: rng.normal(0,.2,shape) for k,shape in shapes.items()}
tokens = rng.randint(0,5,(1,64))
tokens[0,30] = -1
jackpot = np.where(tokens < 0, 4, tokens)
offsets = np.arange(64,0,-1)[None]
args = (offsets, np.array([0]), np.array([0]), H)
print('missing_vs_jackpot maxdelta',
      np.max(abs(forward_np(w,tokens,*args)-forward_np(w,jackpot,*args))))
PY
```

Output:

```text
missing recent tokens [1325, 1165, 989, 821, 660, 501, 331, 165] total 5957
retained nonzero signals 28 changed queries 40
missing_vs_jackpot maxdelta 0.0
```

**Coverage of leakage classes A–F**

| Class | Tests run and results |
|---|---|
| **A — Future ticks** | Generic value/remove/add/count-preserving-move perturbations on saved `seq`, saved `seqloss`, saved `tfm2`, GBDT features, GLM features, `lag_classes`, and `others_win`: **all earlier-output deltas 0.0**. F1–F3 failed observedness tests. Initialized `shortctx.predict` at cutoffs 80/96/112 was invariant to future values, future observedness, and supplied `n_obs`: **0.0**. Pair/GBDT/GLM features also passed the stricter boundary through prediction tick `k`, at `k=32,64,96`. Pooled predictions, M3 with fixed priors, regime features/specialists/change evidence passed pseudo-cutoff corruption tests at those same cutoffs: **0.0**. |
| **B — Cutoff-only quantities** | Audited signal/context occurrences in all requested directories and the fit orchestration. `fit_round(dev_spec(True))` exposed **zero** validation/test label, condition, target, or reference fields. Production tail truncation and every tested pseudo-cutoff had neutral signal, recalculated counts, and no future observed cells. M3’s historical size prior was neutral, with the recorded signal restored only at the actual query. **F5 is the remaining signal-timing defect in the audit helper.** MoE4 condition targets are restricted to `label_splits`; the auxiliary loss does not feed validation conditions into dev fitting. |
| **C — Labels/targets in training** | Traced every arm’s `fit`. `seq`, `seq2`, transformers and MoE arms gate cutoff supervision by the specification; `seqloss` and `shortctx` use histories; GBDT/GLM/regime supervision paths respect the specification. M3 uses train only. Its `truncate` extracts the earlier label then masks that tick and later ticks. CV held-target poisoning changed fitted weights by **0.0**. Halfval masks left **zero** supervised cells in each checked held half. Cached query IDs matched exactly; three available ensembles reproduced cached forecasts on the first two validation episodes within **1.11e-16**. Original training provenance cannot be established from those comparisons alone. |
| **D — Packaging/inference** | Inspected all four ZIPs, clean-ZIP Python sources, `predict.py`, builder, round-two script/blender, and reproduction harness. Clean ZIP has corrected M3 source; all its applicable Python files match current source except the intentional package initializer/path rewrite. Runtime reads test queries/history and metadata, plus released train inputs for M3/fallback. No validation-answer files, cached validation forecasts, `/main_data`, or `data/main` literals were found in ZIP Python code. Reproduction extracts the ZIP and executes its own `predict.py` under isolated Python with ZIP code first on the path. **S1 limits what this proves about filesystem reads.** |
| **E — Evaluation contamination** | Repaired short backtest fits only original training episodes: **24 scored validation episodes, zero overlap**, windows contain 56–63 past observations and neutral signals. Tail evaluation re-derives train labels at earlier cutoffs and masks scored validation outcomes. `halfval` and grouped blend folds passed supervision-isolation checks. Remaining demonstrated issues are MoE/flagged-seq2 historical features and `shortctx`’s missing-token handling. Detailed metric implications follow below. |
| **F — Test contamination** | Loaded released inputs under `audit_dev_files`: **zero forbidden opens**. Train/validation/test episode-ID intersections were all **zero**. Test directory contains only `history.csv`, `queries.csv`, `sample_submission.csv`; the loader reads only the first two. No active path using test signal as a supervised target was found. The unused vendored loader has the latent exception described in S2. |

**P3 — generic differential test for saved neural exports and causal features**

The following reproduces the passing core tests. It preserves the missingness pattern for the value test, changes it for add/remove, and preserves its total count for move.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OPENBLAS_NUM_THREADS=1 ~/.venvs/tdx/bin/python - <<'PY'
import numpy as np
from dataclasses import replace
from tdx.data import Split
from tdx.arms.seq.arm import SeqArm, forward_np as sf
from tdx.arms.tfm2.arm import Tfm2Arm, forward_np as tf
from tdx.arms.gbdt.features import features
from tdx.arms.glm.features import all_features
from tdx.feat import lag_classes, others_win

rng = np.random.RandomState(101)
y = rng.randint(0,5,(2,8,128))
mt = rng.randint(0,3,(2,8))
s = Split('train',['a','b'],y,mt,np.zeros((2,8),int),
          np.array([128,128]),np.full((2,8),128),np.arange(16).reshape(2,8))
seq = SeqArm.load('weights/seqens_ens12_dev/member_00')
sl = SeqArm.load('weights/seqens_size2_ens/member_00')
tfm = Tfm2Arm.load('weights/tfmens_ens8_dev/member_00')

def gru(a,z):
    return sf(a.w,z,mt,96,a.p['h'],a.p['msg'],a.p['factor'])

fns = {
    'seq': lambda z: gru(seq,z),
    'seqloss': lambda z: gru(sl,z),
    'tfm2': lambda z: tf(tfm.w,z,mt,96,tfm.p['d'],tfm.p['L'],
                         tfm.p['heads'],tfm.p['conv'],tfm.p['alibi']),
    'gbdt': lambda z: features(z,mt)[0],
    'glm': lambda z: all_features(replace(s,y=z)),
    'lags': lambda z: lag_classes(z,8),
    'others': lambda z: others_win(z,(1,2,8)),
}
base = y.copy()
base[:,:,64:] = -1
for name,fn in fns.items():
    expected = fn(base)[:,:,:32]
    result = {}
    for kind in ('values','remove','add','move'):
        z = base.copy()
        if kind == 'values': z[:,:,32:64] = (z[:,:,32:64]+1)%5
        if kind == 'remove': z[:,:,63] = -1
        if kind == 'add': z[:,:,100] = 2
        if kind == 'move':
            z[:,:,100] = z[:,:,63]
            z[:,:,63] = -1
        actual = fn(z)[:,:,:32]
        assert np.array_equal(np.isnan(expected),np.isnan(actual))
        result[kind] = float(np.nanmax(abs(expected-actual)))
    print(name,result)
    assert all(d == 0 for d in result.values())
PY
```

Observed output for **each** of the seven entries:

```text
{'values': 0.0, 'remove': 0.0, 'add': 0.0, 'move': 0.0}
```

**Evaluation numbers: what can be inflated**

- **Held-out tail:** no remaining target-in-history or direct cutoff-signal reuse was found in the production constructor. `tail_eval` supplies the train outcome at its new cutoff as supervised training data; validation outcomes at scored ticks remain excluded. `backtest.tail_logloss` instead supplies missing train labels, so these two tools have different training supervision. This difference is not a leak. MoE historical training still has F1; its effect on tail metrics is unmeasured.
- **Short backtest:** the old scored-episode training overlap is removed. No overlap-based inflation remains in the inspected implementation. `shortctx` metrics are nevertheless corrupted by F4; the sign and magnitude cannot be inferred from the number of aliased tokens.
- **Halfval:** the tested split seeds each yielded 48 held episodes, 384 supervised queries in the other half, and **zero held supervised cells**. The sequential model masks NaN targets before using them in its cutoff loss. No mechanical inflation demonstrated.
- **CV blend:** recomputed F’s five OOF scores: **57.108405, 57.112087, 57.076103, 57.106628, 57.124337**, mean **57.105512**. Fitting the same blend on all 96 targets and scoring those same targets gave **57.139058**, approximately **+0.03355 points** over OOF. That measured in-sample difference is not leakage in the OOF implementation.
- **M3 internal selection:** its `train_logloss` is not independent held-out evidence. Priors are estimated from the same training histories/labels before internal historical scoring. Changing one neutral-signal machine’s train cutoff label changed an earlier size probability by **2.9613e-05**, through the estimated prior. This does **not** show validation-answer contamination or direct target inclusion in the per-machine historical filter. It does mean the internal “fresh out-of-sample” description in `common/selection.py:3–5` is too strong. Its aggregate selection optimism was not measured.

**P4 — held-fold poisoning and halfval isolation**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OPENBLAS_NUM_THREADS=1 ~/.venvs/tdx/bin/python - <<'PY'
import numpy as np
from pathlib import Path
from tdx.data import _read_split, Round, with_answers
from tdx.tools.cv_blend import split, folds, fit_weights
from tdx.tools.halfval import halves

root = Path('data/main').resolve()
r = Round('main',root,{'validation':_read_split(root,'validation')},60)
v = with_answers(r).splits['validation']
S = [split(np.load('weights/val_preds/'+n+'.npy'))
     for n in ('m3','gru_ce_11_22','tfm_8')]
fold = folds(v.condition,4,0)
train = np.flatnonzero(fold != 0)
poisoned = v.target.copy()
poisoned[fold == 0] = np.eye(5)[4]
a = fit_weights(S,v.target,train)
b = fit_weights(S,poisoned,train)
print('held-target weight delta',
      max(float(np.max(abs(x-y))) for x,y in zip(a,b)))
for seed in (0,1):
    side = halves(v.condition,seed)
    target = v.target.copy()
    target[side != 0] = np.nan
    print('halfval',seed,'held episodes',sum(side != 0),
          'held supervised cells',np.isfinite(target[side != 0]).sum())
PY
```

Output:

```text
held-target weight delta 0.0
halfval 0 held episodes 48 held supervised cells 0
halfval 1 held episodes 48 held supervised cells 0
```

**Known fixes and shipped-artifact conclusions**

The M3 signal fix is present in both current source and `candidate3b_clean_refit`: earlier pseudo-queries receive signal zero; historical size tuning uses the neutral context; actual cutoff predictions use their recorded signal. Pooled context-prior aggregation matches the organizer baseline’s design. GBDT, GLM, and regime historical features do not consume cutoff signal. Tail-report signal breakdowns use it only to group diagnostics.

The three older ZIPs retain their historical M3 code. They are not evidence that the clean-refit ZIP lacks the fix. The remaining signal-fix gap is specifically **F5**.

The clean ZIP’s README says its blend weights were fitted on **historical tails from 72 validation episodes**, reserving 24 episodes. It does **not** describe those weights as fitted against validation TRUE probabilities. Conversely, `cv_blend.py` explicitly fits TRUE probabilities on training folds. Those are separate procedures. Public validation model selection is allowed by the supplied rules; the historical artifacts needed to independently verify the clean blend’s claimed training procedure are unavailable here.

**What I could not establish**

- **Corrected-model score effects:** no trained MoE, MoE4, flagged-seq2, or shortctx exports were present. Their counterexamples use initialized weights. Assigning contest-point gains would be fabrication.
- **Tiny Torch-fit verification:** the attempted one-epoch CPU fit failed before optimization because Torch’s initialization required writable temporary storage: `FileNotFoundError: No usable temporary directory found ...`. The sandbox denies `/tmp` writes too. No full model training was run.
- **Fresh-folder reproduction:** extracting a ZIP, creating a fresh environment, and writing predictions were blocked by that same filesystem restriction. I inspected the execution path and ZIP contents; I did not claim a fresh reproduction run.
- **Historical dev provenance:** current fit code and sanitation are consistent with dev-only supervision. Checkpoint hyperparameter files do not prove which historical `FitSpec` produced them. The CE seeds 41–52 exports and original per-run metadata are absent; only their cached forecasts are available.
- **All cached rows:** neural-export/cache equality was checked on two episodes for each available ensemble, not all 96. Query-ID alignment was checked for all rows.
- **Historical member selection and clean-blend fold construction:** the frozen candidate specification and documentation exist, but the original training/calibration artifacts needed to independently reconstruct that history do not. No additional mechanical member-selection leak was demonstrated.
- **Organizer test accuracy:** no test answers were accessed or available.

Changed paths: **none**.