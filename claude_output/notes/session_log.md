# Session log (append-only; newest at the bottom). Created at the 2026-11-08 closeout because the repo had no log.md.
Score logs per run are in runs/LOG.md (written by the harness). Decisions are in claude_output/notes/EVAL_PROTOCOL_FROZEN.md.

## 2026-11-07 13:30 - 2026-11-08 02:20 CT (one long session)
- Built harness, arms (pooled, gbdt, glm, regime, seq/GRU, tfm2 transformer, seqloss, moegate, moegate4, seq2, shortctx), packaging (offline numpy entry), submissions/ (candidates 1, 2, 3b, andrew's clean refit) - measured validation points 56.17 / 56.58 / 56.90 / 56.75 (dev mode, optimistic).
- Found and fixed: signal-timing contamination and short-backtest label leak (Andrew, cherry-picked), future-observedness flags in moegate/moegate4/seq2 and shortctx missing-token bug (Codex audit, W9 fix verified, 57 tests).
- Corrected claims: size x2 loss gain was seed luck (same-seed 12 vs 12: +0.11, interval [-0.15, +0.35]); transformer size x2 is worse (-0.46 over 8 seeds); moegate K=4+condition loss fails; "own-history" reading of the transductive gap was wrong (data volume: B=53.92 vs A=34.29).
- Frozen protocol + repeated 4-fold grouped CV blend test (tdx.tools.cv_blend); OOF points F 57.11, A 57.20, B 57.28, D 57.28, E 57.16; no candidate meets the replacement rule.
- Round two: round2.sh exists, rehearsed only at toy scale; full-size timing rehearsal NOT done.
- Next: see claude_output/notes/NEXT_SESSION_PROMPT.md.
