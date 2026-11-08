# Session log (append-only; newest at the bottom). Created at the 2026-11-08 closeout because the repo had no log.md.
Score logs per run are in runs/LOG.md (written by the harness). Decisions are in claude_output/notes/EVAL_PROTOCOL_FROZEN.md.

## 2026-11-07 13:30 - 2026-11-08 02:20 CT (one long session)
- Built harness, arms (pooled, gbdt, glm, regime, seq/GRU, tfm2 transformer, seqloss, moegate, moegate4, seq2, shortctx), packaging (offline numpy entry), submissions/ (candidates 1, 2, 3b, andrew's clean refit) - measured validation points 56.17 / 56.58 / 56.90 / 56.75 (dev mode, optimistic).
- Found and fixed: signal-timing contamination and short-backtest label leak (Andrew, cherry-picked), future-observedness flags in moegate/moegate4/seq2 and shortctx missing-token bug (Codex audit, W9 fix verified, 57 tests).
- Corrected claims: size x2 loss gain was seed luck (same-seed 12 vs 12: +0.11, interval [-0.15, +0.35]); transformer size x2 is worse (-0.46 over 8 seeds); moegate K=4+condition loss fails; "own-history" reading of the transductive gap was wrong (data volume: B=53.92 vs A=34.29).
- Frozen protocol + repeated 4-fold grouped CV blend test (tdx.tools.cv_blend); OOF points F 57.11, A 57.20, B 57.28, D 57.28, E 57.16; no candidate meets the replacement rule.
- Round two: round2.sh exists, rehearsed only at toy scale; full-size timing rehearsal NOT done.
- Next: see claude_output/notes/NEXT_SESSION_PROMPT.md.

## 2026-11-08 ~02:00-04:00 CT (second session, candidate 4 and overnight queue)
- Measured: candidate 4 = B (m3 + 12 size x2 GRUs seeds 41-52 + 8 transformers; fold-averaged CV weights, OOF 57.283 vs F 57.106, gain +0.178, interval [+0.095, +0.261], under the 0.2 rule). Built in runs/entry_cand4_B: check_submission VALID FORMAT (1536 queries); repro OK (runs 246.5 s / 233.1 s, 1.22 GB, diff vs CSV 0.0, no torch). NOT uploaded; user decides vs submitted 3b.
- D (24 GRUs) was the default I picked earlier; CV shows no benefit over B (57.276 vs 57.283), so the user chose B. The D package stays in runs/entry_cand4 (repro stopped, unverified).
- CE-80 test already done: 80 epochs worse than 100 at matched seeds (-0.045, [-0.080, -0.013]); 100 stays. MoE retest: all 16 earlier runs failed the numpy/torch parity check (1.07e-9 > 1e-9); threshold set to 1e-6 (as in tfm/tfm2) at src/tdx/arms/moegate4/arm.py:449; no MoE model exists yet.
- round2.sh and round2_blend.py now build the candidate 4 recipe (12 size x2 GRUs seeds 41-52, 8 tfm2 d128l3 seeds 21-28, fixed candB_main blend weights; BLEND_SELECT=1 restores best-preset-on-new-validation). Not yet run to completion.
- Started (user approved) claude_output/sh/overnight_chain.sh at 03:45: full-size round2 rehearsal on main data, leak tests for moegate4 k3_ref/k4_nocond, then the frozen addendum-2 MoE test; queue_epoch150.sh waits for it (size x2 GRU 150-epoch tail curves, seeds 41-43).
- Next: see claude_output/notes/NEXT_SESSION_PROMPT.md.
