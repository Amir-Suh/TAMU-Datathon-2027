# W9 — Fix the leaks found by the adversarial audit, and make the mandatory leak test catch them

## Role and objective
Fix the confirmed defects listed in `claude_output/codex/LEAKAGE_REPORT_2026-11-08.md` (read it first, especially findings F1–F5 and the reproductions P1 and P2) and extend `python -m tdx.tools.leak_test` so the class of bug cannot slip through again. CPU only: the sandbox hides the GPU; use tiny fits (`epochs=1`, `h=16`, small `d`) for tests. Do NOT retrain anything for real, do NOT commit.

## Why
Four research arms and our own mandatory leak test violated the rule "use only observations strictly before the prediction tick, including when building training examples from earlier history":
- F1/F2/F3: `moegate`, `moegate4`, `seq2 (short_flag)` compute a short-history flag from the count of observed ticks over the WHOLE row (`(y>=0).sum(-1)==64`) and feed it to earlier next-tick training steps. Adding or removing a future tick (even one at/after the cutoff in the array) changes outputs at earlier steps (P1: 0.0054, 0.0023, 0.17).
- F4: `shortctx` lets missing ticks (token -1) reach `w['tok.weight'][tokens]`; NumPy indexes the last row, i.e. class 4 (a 20x jackpot).
- F5: `tdx.tools.leak_test.truncate_for_probe` keeps the original cutoff-only `signal` on queries whose cutoff it moved earlier, and the cutoff audit never changes which ticks are observed, so it returned 0.0 for the three leaking arms.

## Owned files
`src/tdx/tools/leak_test.py`, `src/tdx/arms/moegate/**`, `src/tdx/arms/moegate4/**`, `src/tdx/arms/seq2/**`, `src/tdx/arms/shortctx/**`, `src/tdx/arms/seq/arm.py` (only to add `step_outputs`), `src/tdx/arms/tfm2/arm.py` (only to add `step_outputs`), `tests/test_leak.py`, `tests/moegate/**`, `tests/moegate4/**`, `tests/seq2/**`, `tests/shortctx/**`, `claude_output/notes/W9_leakfix.md`. Forbidden: everything else (do not touch `seqloss` internals, `tfmens`, `seqens`, packaging, the vendored m3).

## Work packages
**P1 — `step_outputs(y, mtype, steps)`** (public method, float64, no masking, returns the arm's per-step outputs `[E,8,steps,*]` from the existing numpy forward pass): add to `SeqArm` (so `SeqLossArm` inherits it), `Tfm2Arm`, `MoeGateArm`, `MoeGate4Arm`, `Seq2Arm`. Outputs are logits/log-probs/probabilities as each forward pass returns them (use the mixture probabilities for the MoE arms). No behaviour change otherwise.

**P2 — extend `leak_test`** (keep the existing cutoff audit and the dev-file audit):
1. Fix `truncate_for_probe`: where a query's cutoff is moved earlier set its `signal` to 0 (neutral) and recompute `n_obs`.
2. Add `assert_step_causality(model, y, mtype, *, k)`: for a fitted arm that has `step_outputs`, take `y` already masked at its cutoff, `base = step_outputs(y, mt, steps=k)`, then for each perturbation compare `step_outputs(z, mt, steps=k)[...,:k,:]` (steps `< k`) with `base` and require EXACT equality: (a) values: outcomes at ticks `>= k` replaced by other classes, observedness unchanged; (b) remove: one observed tick at index `>= k` set to -1; (c) add: a tick at index `>= k` that is currently -1 (use an index beyond the cutoff if the row has no gaps) set to an observed class; (d) move: (b)+(c) together so the observed COUNT is preserved. Use a real validation slice of 3 episodes, `k` around 32 and around cutoff-8.
3. Wire it into `check_arm` (after the fit): run it when `hasattr(model, 'step_outputs')`; report `step_causality` in the result dict and in the CLI line; when the arm has no `step_outputs` print `step_causality: not available`. Add a `--set key=json` override option to the CLI so tiny fits are possible (`--set epochs=1 --set h=16`).
4. Tests in `tests/test_leak.py`: (i) a synthetic arm whose `step_outputs` includes the whole-row observed count MUST fail `assert_step_causality`; (ii) a synthetic causal arm passes; (iii) `truncate_for_probe` zeroes signal on moved queries; (iv) `seq`, `tfm2` tiny fits pass the full `check_arm`.

**P3 — causal flag in `moegate`, `moegate4`, `seq2(short_flag=True)`**: replace the whole-row `count == 64` flag by a per-tick causal flag `flag_t = 1 if any tick <= t is missing, else 0`, i.e. `np.cumsum(y < 0, axis=-1) > 0` (shape `[E,8,T]`, then add a trailing axis where the gate/embedding needs it). It uses only ticks `<= t`; at the real cutoff step it equals the old flag on real data (long rows are fully observed, short rows have gaps). Implement identically in the torch training forward and the numpy forward (shapes change from `[B,M,1]` to `[B,M,T,1]` for the MoE gates; for `seq2` the flag embedding is looked up per step). Torch-vs-numpy parity must stay < 1e-9 (the arms check it after every fit). Record `flag='causal_v1'` in each arm's `meta.json`; `load` of an export without it must raise a clear error ("trained with the non-causal flag"). Existing tests must pass or be updated; add to each arm's test file the step-causality test (value/remove/add/move perturbation) using the new `assert_step_causality`.

**P4 — `shortctx` missing tokens** (lowest priority; skip only if P1–P3 are not finished): never let a token of -1 reach an embedding. In `predict` build the 64 items from OBSERVED ticks only (the last 40 observed ticks plus 24 older observed ticks chosen with a fixed seed, offsets = true distance to the cutoff); if a machine has fewer than 64 observed ticks use all of them and mark the rest with a dedicated PAD token (add one embedding row, keep training samples unchanged) so no real class is aliased. In `forward_np` raise `ValueError` if any token is negative or `>= number of embeddings`. Tests: (a) a machine with gaps in the recent block produces no -1 token; (b) PAD never equals a real class output (changing a PAD slot's offset changes nothing); (c) P2 reproduction ("missing_vs_jackpot") now differs from the jackpot case.

## Acceptance (runnable, CPU)
```
~/.venvs/tdx/bin/python -m pytest -q tests/test_leak.py tests/moegate tests/moegate4 tests/seq2 tests/shortctx tests/seq tests/tfm2 tests/seqloss
~/.venvs/tdx/bin/python -m tdx.tools.leak_test --arm seq --variant gru128_factor --set epochs=1 --set h=16
~/.venvs/tdx/bin/python -m tdx.tools.leak_test --arm tfm2 --variant d128l3 --set epochs=1 --set d=16 --set L=1
~/.venvs/tdx/bin/python -m tdx.tools.leak_test --arm moegate --variant pilot --set epochs=1 --set h=16
~/.venvs/tdx/bin/python -m tdx.tools.leak_test --arm moegate4 --variant k3_ref --set epochs=1 --set h=16
~/.venvs/tdx/bin/python -m tdx.tools.leak_test --arm seq2 --variant flag_aug5all --set epochs=1 --set h=16
```
and the P1 script from `claude_output/codex/LEAKAGE_REPORT_2026-11-08.md` adapted to the new flag must print `0.000000000` for `remove`, `add` and `move` for all three arms (put it in the tests). Write `claude_output/notes/W9_leakfix.md` (≤ 40 lines): what changed, the before/after numbers for P1, anything not finished. Report any file you changed outside the owned list.
