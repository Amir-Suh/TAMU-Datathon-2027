# Antigravity job: evaluation tools for payout size and short_history (new files only, 40-minute budget)

You have no conversation context. Repo: /home/ganfwe/Downloads/TAMU-Datathon-2027. PY=/home/ganfwe/.venvs/tdx/bin/python (always use it; run from the repo root; run python with `PYTHONPATH=src`). GPU is available but shared: at most 3 trainings at a time.

## Time budget
Run `date`, write it to /tmp/agy_evaltools/start.txt (mkdir -p). Hard limit 40 minutes: code + tests by minute 25, the proof runs by minute 40, then report whatever finished.

## Rules
- Create ONLY: `src/tdx/tools/tail_eval.py`, `src/tdx/tools/short_backtest.py`, `tests/test_eval_tools.py`, and files under `/tmp/agy_evaltools/`. Do NOT edit any existing file. No git, no network, no pip, no rm.
- Starting code (working, read it first): `claude_output/agy/inputs/tail_preds.py` (refit a model on histories truncated by HOLD ticks, save forecasts for the held-out tail), `claude_output/agy/inputs/tail_analysis.py` (win part / size part / total log-loss, paired bootstrap), `claude_output/agy/inputs/tail_weights.py` (stacking weights from Brier). Also read `src/tdx/backtest.py`, `src/tdx/data.py`, `src/tdx/run.py`, `src/tdx/arms/m3flip/arm.py`, `src/tdx/arms/seq/arm.py`.

## Definitions (use exactly)
- Outcome class o in {0,1,2,3,4} for payouts 0,1,2,5,20. For a forecast vector p: win part = -log(p0) if o==0 else -log(1-p0); size part (only for wins, o>0) = -log(p_o/(1-p0)); total = -log(p_o) = win part + size part. Clip probabilities at 1e-6. Brier = sum_c (onehot(o)_c - p_c)^2.

## Tool 1: `python -m tdx.tools.tail_eval` (size-part metric with real power)
1. Subcommand `predict --model NAME --seed S --hold H --out DIR`: NAME in `pooled_recent`, `m3`, `gbdt` (variant pair_l31), `gru_fac` (seq, h=128, factor, 100 epochs, transductive), `gru_plain`, `tfm` (arm tfm2, variant conv9_alibi with d=128, L=3, epochs=100). Same logic as tail_preds.py (honest train labels for the earlier cutoff; the scored window is the 30 ticks [cutoff-H, cutoff-H+30) of the VALIDATION split). Save `DIR/tail_<name>_<seed>_h<H>.npz` (preds [30,E,8,5], outs [30,E,8]).
2. Subcommand `report --dir DIR --models a,b,c --holds 30,60,90 [--baseline NAME]`: loads the npz files (averaging probabilities over all seeds found per model), and prints markdown tables:
   a. per model: n outcomes, n wins, win part, size part, total log-loss, Brier;
   b. paired episode bootstrap (2000 resamples, seed 0) of the size-part and total differences versus the baseline, with 95% intervals;
   c. breakdowns of the size part and win part by machine type (steady/balanced/jackpot, from `split.mtype`), by validation condition (use `with_answers(rnd).splits['validation'].condition`), and by signal (-1/0/1);
   d. reliability table of the predicted P(win): 10 equal-count bins, mean predicted vs realized win rate, count, for each model; and a reliability table of the size mix: bins of the predicted 1x share among wins, mean predicted vs realized 1x share.
3. Test (tests/test_eval_tools.py): synthetic arrays where the answer is known (e.g. a perfect-oracle forecast has size part 0 on a one-hot case; the decomposition win+size equals total to 1e-12); the bootstrap returns an interval containing the point estimate.

## Tool 2: `python -m tdx.tools.short_backtest` (real short_history backtest)
Pool every episode whose machines all have exactly 64 observed ticks (`(y>=0).sum(-1)==64`) from train, validation AND test (about 96 episodes). For each such machine the history is the last 40 consecutive ticks before the cutoff plus 24 older sampled ticks. Experiment:
1. Hide the last 8 observed ticks of every machine in those short episodes (ticks cutoff-8 .. cutoff-1) by setting them to -1 and reducing the episode cutoff by 8; other episodes keep their full histories (they remain transductive training data).
2. Refit the arm exactly as `tdx.run.execute`/tail_preds does (honest train labels for train short episodes: the realized outcome at the new cutoff; unobserved -> -1).
3. For j in 0..7 predict with the pseudo cutoff (new cutoff + j) using the original history up to tick < new cutoff + j (the ticks hidden earlier that are now in the past become visible again), and score the realized outcome at that tick.
4. Models: same NAME list as Tool 1. Output: per model n outcomes, win/size/total log-loss and Brier, plus a paired bootstrap by episode versus `pooled_recent`.
CLI: `python -m tdx.tools.short_backtest --model NAME --seed S --out DIR` then `report --dir DIR --models ...`.

## Proof runs (after tests pass)
- `tail_eval predict` for: pooled_recent, m3, gbdt, gru_fac seeds 1-3, tfm seeds 1-2 at holds 30, 60, 90 (CPU models first; at most 3 GPU trainings at a time); then `tail_eval report --models pooled_recent,m3,gbdt,gru_fac,tfm --holds 30,60,90 --baseline pooled_recent`. Existing results from earlier runs for comparison (hold 30 only): size part pooled_recent 1.0379, m3 1.0234, gbdt 1.0306, GRU mean of 3 seeds 1.0099; total m3 0.7235, GRU 0.7428.
- `short_backtest` for pooled_recent, m3, gbdt, gru_fac seeds 1-2, then its report.

## Final report (print only this)
Files created; test result lines; the three most important tables (a: model comparison; b: bootstrap vs baseline; the short_backtest comparison); the reliability table for the best model; any failure with its last 5 log lines; elapsed minutes.
