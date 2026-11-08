# Antigravity job: history-noise stress test of seq, gbdt and Amir's m3 (scripts + table, no repo edits)

You have no conversation context. Repo: /home/ganfwe/Downloads/TAMU-Datathon-2027. PY=/home/ganfwe/.venvs/tdx/bin/python (always use it; run from the repo root; run python with `PYTHONPATH=src`).

## Rules
- Write only under `/tmp/agy_stress/`. Do NOT edit any file in the repo and do not touch `/tmp/amir_wt` except through the commands below. No git, no network, no pip, no rm.
- Read once: `src/tdx/arms/seq/arm.py` (`SeqArm.load(dir).predict(split)` returns [E,8,5]), `src/tdx/arms/gbdt/arm.py` (`GBDTArm.load(dir).predict(split)`), `src/tdx/data.py` (`load_round`, Split), `src/tdx/score.py` (`score(truth_round, pred)['points']` and `['by_condition']`), and `/tmp/amir_wt/models/run.py` (how Amir's models are fit and scored from a kit-format data folder; run it with `cd /tmp/amir_wt && PYTHONPATH=/tmp/amir_wt $PY -m models.run --models m3 --data <folder> --out <out>`).

## Question
How fast does each model degrade when the observed history gets noisier? (A rough stand-in for round two, where the lag structure changes.) Models: `seq` = mean of the 3 saved GRUs `runs/seq/gru128_nomsg_seed{1,2,3}/model`; `gbdt` = `runs/gbdt/pair_l31/model`; `m3` = Amir's `m3_lag_pair_flip` (it fits on the data folder you pass it, train only).

## Steps
1. For noise level p in {0.0, 0.02, 0.05, 0.10}: take the VALIDATION split only and corrupt the observed history ticks (ticks < cutoff, value >= 0) of every machine with probability p each, using numpy `RandomState(100 + int(p*1000))`: a win (class 1-4) becomes a loss (class 0); a loss becomes class 1. Keep cutoffs, types, signals, ids unchanged.
2. seq and gbdt: build the corrupted `Split` with `dataclasses.replace(split, y=y_noisy)` from `load_round('data/main', answers=False).splits['validation']`, predict, and score against `with_answers(rnd)` truth (see how `tdx.run.execute` scores validation: `score(with_answers(rnd), pred)`). Record points and the four per-condition loss ratios.
3. m3: copy `/home/ganfwe/Downloads/main-participants-60-40/public` to `/tmp/agy_stress/data_p<p>` (cp -r), then rewrite ONLY `validation/history.csv` there with the same corruption. First read the first 3 lines of that csv to see its layout (it has one row per machine and tick; only rows with ticks < cutoff exist; payout column holds the payout values 0,1,2,5,20; map payout 0 -> loss, any positive payout -> win; a corrupted win becomes payout 0, a corrupted loss becomes payout 1). Use the SAME random draw logic (same seed, same order of machines/ticks) so seq/gbdt and m3 see the same corrupted histories if feasible; if the layouts make that impractical, say so in the report and use an independent draw with the same seed. Run Amir's `models.run --models m3 --data /tmp/agy_stress/data_p<p> --out /tmp/agy_stress/m3_p<p>` and read its validation points from `/tmp/agy_stress/m3_p<p>/scores.json` (or the console line `m3_lag_pair_flip  <points> / 60`).
4. Also compute, for each noise level, the equal-weight blend `(seq + m3 + gbdt)/3` and `(seq + m3)/2` if m3's validation predictions can be loaded into [E,8,5] arrays in the order of `split.query_ids` (they are written as `validation_predictions.csv` with columns query_id,p_0,p_1,p_2,p_5,p_20).
5. Also record each model's mean confidence: the mean over machines of max(p_k) of its predictions, at every noise level.

## Final report (print only this)
Table: noise p | seq points | gbdt points | m3 points | blend3 points | seq+m3 points | seq mean confidence | m3 mean confidence. Then one sentence on which model degrades fastest. List any step that failed (last 5 log lines) and any deviation from the plan.
