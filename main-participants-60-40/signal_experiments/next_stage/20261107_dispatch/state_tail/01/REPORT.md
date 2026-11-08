# Main A · E1 and E2

**Advance P0 hard tail projection to package and assembly checks. Hold finite E2 in favor of P0. Reject E1 for main.** These are development comparisons on 96 previously examined main validation episodes and 768 queries. The archived pair_l31 and factored GRU parents had pre-cutoff history exposure across the main splits. The newly estimated state, compact GRU, and tail parameters used only main training data. The scored main comparisons inherit the archived history exposure.

| Forecast | Points / 60 | R | ΔR vs hybrid | Paired 95% interval |
|---|---:|---:|---:|---:|
| Frozen M3 | 55.268370 | 0.085612 | +0.020668 | [0.016367, 0.025110] |
| Hybrid replay | 56.341022 | 0.064943 | 0 | |
| P0 hard projection | 56.385873 | 0.064096 | −0.000847 | [−0.001265, −0.000449] |
| E1 payout only | 55.369553 | 0.083628 | +0.018685 | [0.014249, 0.023513] |
| E1 offset, decoupled | 55.395352 | 0.083123 | +0.018180 | [0.013284, 0.023456] |
| E1 offset, coupled | 55.392850 | 0.083172 | +0.018229 | [0.013310, 0.023520] |
| E2 finite, α = 512 | 56.386795 | 0.064079 | −0.000864 | [−0.001260, −0.000489] |
| E1 coupled + E2 finite | 55.438178 | 0.082287 | +0.017343 | [0.012444, 0.022582] |

The bootstrap drew 10,000 paired episode samples stratified by condition. The M3 interval above reverses the hybrid-versus-M3 comparison. Full comparisons include [hybrid baseline](scores_vs_hybrid.json), [M3 baseline](scores_vs_m3.json), [matched baselines](scores_vs_matched.json), and [slice data](slices.csv).

| Forecast | Stable R | Short-history R | Drift R | Shared-change R |
|---|---:|---:|---:|---:|
| Hybrid | 0.051074 | 0.092936 | 0.055314 | 0.060449 |
| P0 | 0.050366 | 0.091884 | 0.055127 | 0.059009 |
| E1 payout only | 0.068781 | 0.108390 | 0.079265 | 0.078076 |
| E1 offset, decoupled | 0.067146 | 0.109238 | 0.078664 | 0.077447 |
| E1 offset, coupled | 0.067176 | 0.109229 | 0.078749 | 0.077535 |
| E2 finite, α = 512 | 0.050358 | 0.091883 | 0.055032 | 0.059042 |
| E1 coupled + E2 finite | 0.066551 | 0.108183 | 0.078505 | 0.075908 |

## E1

Coupling added +0.000049 R versus the decoupled offset, with 95% interval [−0.000017, +0.000117]. The decoupled offset changed R by −0.000505 versus payout only, with interval [−0.002105, +0.001071]. Training selected κ = 0 over the full grid. The nonzero arms used κ = −0.45, the best negative choice in the grouped training comparison, while payout-only represents zero.

The offset lowered p0 MSE from 0.004468 to 0.004399. The coupled forecast raised 1× and 2× SSE by 0.001899 and 0.001438 per query versus hybrid. On the 499 balanced or jackpot queries, coupled p0 SSE improved by 0.000150 and total SSE rose by 0.002196. On the 116 balanced or jackpot queries with fitted state posterior between 0.2 and 0.8, coupled total SSE rose by 0.000878 versus hybrid and 0.000076 versus decoupled. The fitted payout levels and strong binomial evidence generated near-endpoint state posteriors for many machines. The compact GRU was trained independently of the archived GRU and may provide different state separation.

| Type | Fitted low 1× | Fitted high 1× | High-state prevalence | Main validation posterior Q1 / median / Q3 |
|---|---:|---:|---:|---:|
| Steady | 0.5435 | 0.7055 | 0.4733 | 0.0003 / 0.0747 / 0.9799 |
| Balanced | 0.2962 | 0.4263 | 0.4532 | 0.0259 / 0.2205 / 0.8524 |
| Jackpot | 0.1874 | 0.3322 | 0.1768 | 0.0068 / 0.0724 / 0.2322 |

The joint mechanism recovered κ = −0.75 in a seeded 60,000-query known-law check with generating state emissions and causal binomial evidence supplied. Its expected-vector SSE was 1.55e−32 and the matched decoupled SSE was 6.86e−05. This checks the coupling and selection under a known static law. The static validation arm failed the main gate, so the sticky arm was deferred. [Train settings, synthetic check and per-split invariants](diagnostics.json) · [Validation state posteriors](validation_state.csv)

## E2

Grouped training tail log loss decreased from 0.770118 at identity to 0.762594 at α = 512 and 0.762338 at α = ∞. The finite arm passed the champion gate, including improvement in all four conditions and lower matched tail error. Its 0.000017 R gain versus P0 had 95% interval [−0.000044, +0.000010], its shared-change ratio rose by 0.000033 versus P0, and its tail-ratio SSE was worse by 0.000027. The 0.000922-point gain over P0 did not earn the extra inference dependency on query history counts.

P0 preserved p0, p1, and total tail mass. On all 768 queries, the per-query changes in 2×, 5× and 20× SSE versus hybrid were −0.000109, −0.000034 and −0.000021. Normalized tail-ratio SSE improved in all three types. Absolute tail SSE increased slightly for steady machines.

| Type | Queries / episodes | P0 absolute tail SSE change | P0 tail-ratio SSE change |
|---|---:|---:|---:|
| Steady | 269 / 90 | +0.000032 | −0.001404 |
| Balanced | 234 / 92 | −0.000260 | −0.001621 |
| Jackpot | 265 / 92 | −0.000279 | −0.005895 |

The finite tail increment improved the rejected coupled E1 forecast by −0.000886 R with interval [−0.001378, −0.000433]. That combination still trailed hybrid. [Type ratios, support slices, and matched errors](slices.csv)

## Fit scope and replay

Four published training episode folds each held out 24 whole episodes. For each outer fit, three inner feature folds held out their own episodes and the outer fold from all upstream parameters. The run fit ordered Beta-Binomial mixtures and the causal 24-unit factored GRU in each partition. Historical steps used neutral signal. Real cutoff labels had weight four in tuning and weight twenty in the GRU's history-plus-label objective. Six pseudo-cutoff shifts were predeclared as 0, 4, 8, 16, 24 and 32 ticks before the actual cutoff. The grouped kappa selector used fold-refitted Split-Match M3 win proxies with neutral signal and fixed prior strength 50. The grouped alpha selector used the held-episode compact GRU tail as a proxy for the archived hybrid tail. The proxy is the main difference between the grouped tuning distribution and the released evaluation distribution. Released forecasts use the exact frozen Split-Match M3 win mass and the exact tree plus archived GRU conditional payout mix. The E1 posterior uses the compact model trained on the full training split. [Episode-level overlap and fit sources](fit_trace.json) · [Source/input hashes and invocation](manifest.json)

The audited fit process froze all CSV hashes before the score process opened validation odds. Both audits report successful script exits. The fit audit logged host introspection through uname and file and contains no validation label, odds, reference or condition reads. At-cutoff and future payout perturbations produced zero forecast changes on validation and test while holding archived parent predictions fixed. The train pseudo-cutoff perturbation also preserved its feature values, and changing the actual signal preserved the historical GRU forecasts. E2 p0 and p1 changes were exactly zero before export and tail-mass change was at most 1.11e−16. Every validation and test CSV passed the kit checker. The 12-episode smoke took 3.69 seconds, the full fit and forecast freeze took 25.05 seconds after imports, and the score process recorded its elapsed time separately. The macOS time wrapper printed 26.49 seconds before its clockrate sysctl was denied by the sandbox. [Fit audit](fit_audit.jsonl) · [Forecast hashes and freeze timestamp](frozen.json) · [Score audit](score_audit.jsonl) · [Format checks](format_checks.log)

Replay from the kit root. Use a new sibling directory to preserve the frozen output.

```sh
d=signal_experiments/next_stage/20261107_dispatch
original="$d/state_tail/01"
replay="$d/state_tail/replay"
mkdir "$replay"
cp "$original/fit_predict.py" "$original/score_frozen.py" "$replay/"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir "$replay/smoke_final"
python3 -B "$d/coordinator/audit_run.py" --stage fit-predict \
  --trace "$replay/smoke_final/fit_audit.jsonl" "$replay/fit_predict.py" --smoke
python3 -B "$d/coordinator/audit_run.py" --stage fit-predict \
  --trace "$replay/fit_audit.jsonl" "$replay/fit_predict.py"
python3 -B "$d/coordinator/audit_run.py" --stage score \
  --trace "$replay/score_audit.jsonl" "$replay/score_frozen.py"
for arm in hybrid p0 e1_payout e1_decoupled e1_coupled e2_finite e1_coupled_e2_finite; do
  python3 -B check_submission.py "$replay/validation_$arm.csv" --split validation
  python3 -B check_submission.py "$replay/test_$arm.csv"
done
```

The replayed hybrid and P0 each differed from their archives by at most 1.11e−16 on validation and 2.22e−16 on test. For packaging, P0 scales the three training type ratios by the hybrid's existing combined tail mass after computing the hybrid. The archive's validation SHA-256 is `d8aca61ddd0a152172a955210d7a655264e3462dce0cb82c79d31de9101f5ffc` and its test SHA-256 is `5ea5f5ff854c380407542db3dd83841519ac412864c0c22d1cc76b49c7e4c23c`. The replayed validation and test hashes are in the freeze file. The coordinator owns final assembly and offline package verification. [Canonical validation forecast](../../../../research_next/P0/raw_counts.csv) · [Canonical test forecast](../../../../research_next/P0/raw_counts_test.csv) · [Replayed test forecast](test_p0.csv)
