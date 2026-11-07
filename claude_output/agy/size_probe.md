# Antigravity job: what does the seq GRU know about payout size? (probing, scripts + table, no repo edits)

You have no conversation context. Repo: /home/ganfwe/Downloads/TAMU-Datathon-2027. PY=/home/ganfwe/.venvs/tdx/bin/python (always use it; run from the repo root with `PYTHONPATH=src`).

## Rules
- Write only under `/tmp/agy_probe/`. Do NOT edit any repo file. No git, no network, no pip, no rm. Do not use the GPU (numpy inference only; the GPU is busy).
- Read once: `src/tdx/arms/seq/arm.py` (`forward_np(w, y, mt, steps, H, msg, factor)` returns logits/log-probs [E,8,steps,5] for the whole history; `SeqArm.load(dir).w` holds the weights; `SeqArm.predict(split)`), `src/tdx/data.py`, `src/tdx/arms/gbdt/features.py` (`features(y, mtype)` returns causal per-tick feature columns [E,8,T+1,F] with names).

## Background
Our GRU predicts payout size given a win far better than any hand-built estimator: size-mix error 0.0165 (GRU) vs 0.0316 (machine-type prior) vs 0.0304 (best recency-weighted counts of past win sizes I could fit). A GBDT on win rows with hand features gains only ~1% log-loss over machine type. We want to find out WHAT in the history the GRU uses for size.

## Models and data
Models: the 3 saved factored GRUs `runs/seq/gru128_factor_seed{1,2,3}/model` (average their outputs). Data: `rnd = load_round('data/main', answers=False)`; use splits train, validation and test histories (all observed ticks < cutoff; y has -1 for unobserved). Everything is in-sample for the GRU (it trained on these histories), so probe on VALIDATION and TEST histories only and report that caveat.

## Experiment A (probe the hidden state)
1. Write a numpy function that runs the GRU weights exactly like `forward_np` but ALSO returns the hidden state `h` [E,8,T,H] (copy the loop; verify your logits equal `forward_np`'s to 1e-9 on 2 episodes).
2. Rows: every (episode, machine, tick t) with a WIN at tick t+1 (class 1-4) and t+1 < cutoff, in validation+test. Input = the hidden state after consuming tick t. Label = size class of the win at t+1 (0..3).
3. Fit sklearn `LogisticRegression(max_iter=300, C=1.0)` on h (standardised), 5-fold GroupKFold by episode; report out-of-fold log-loss. Compare with (i) machine-type-only frequency baseline and (ii) the GRU's own implied size distribution at that tick (from its factored head: softmax over columns 1..4 conditional on win, i.e. p_k / (1-p_0)). 
4. Probe quantities that might drive size, each with a ridge/linear or logistic regression from h (5-fold GroupKFold by episode, report R^2 or accuracy): (a) ticks since the machine's last win (log1p), (b) size class of the latest win, (c) number of wins in the last 20 ticks, (d) whether the lag-1 outcome was a win, (e) running share of 1x among all past wins of this machine, (f) running share of 20x among all past wins of this machine, (g) tick position t/256. Print a table quantity | out-of-fold score | baseline (mean/majority).

## Experiment B (which hidden units matter for size)
Compute the GRU's size-mix prediction `s(h)` at each row from Experiment A. For each of the 128 hidden units, set it to its mean over rows, recompute the head output only (head applied to the modified h; this is a one-step ablation at the last layer, no recurrence), and record the increase in out-of-fold-free log-loss on the win rows. Print the top 10 units by increase and how many units account for 50% of the total increase.

## Experiment C (sensitivity to history features, controlled perturbations)
For 300 random validation machines, for each lag k in {1,2,3,5,10,20}: take the k-th most recent LOSS before the cutoff and turn it into a 1x win; separately the k-th most recent win and turn it into a loss. Measure mean absolute change in P(win) and in the size mix (L1 over the 4 conditional size probs) of the cutoff prediction. Print a table lag | action | mean |dP(win)| | mean |dMix|.

## Final report (print only this)
Experiment A tables (log-loss comparison and the quantity probes), Experiment B top-10 and the 50% count, Experiment C table, and any failed step with its last 5 log lines. One sentence of interpretation at most.
