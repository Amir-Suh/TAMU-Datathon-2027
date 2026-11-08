# Antigravity job: seq seed farm (mechanical, no code changes)

You have no conversation context. Do exactly this and nothing else.

## Rules
- Work only inside /home/ganfwe/Downloads/TAMU-Datathon-2027. Do NOT edit any file under src/, tests/, kit/, config/. Do NOT use git, network, pip, or rm.
- Write only under runs/seq/ and the log folder /tmp/agy_seedfarm/.
- Use this interpreter for everything: PY=/home/ganfwe/.venvs/tdx/bin/python

## Steps
1. `mkdir -p /tmp/agy_seedfarm`
2. For each seed S in 11 12 13 14 15 16 17 18 19 20 21 22 run (at most 4 at the same time, e.g. with `xargs -P 4`), each about 2.5 minutes on the GPU:
   `cd /home/ganfwe/Downloads/TAMU-Datathon-2027 && $PY -m tdx.run --arm seq --variant gru128_factor --mode dev --set seed=S --out runs/seq/gru128_factor_seed_S > /tmp/agy_seedfarm/factor_S.log 2>&1`
3. Wait for all 12 to finish. Check each log's last line looks like `seq/gru128_factor dev: <number> points (<n>s)`. Rerun any that failed once.
4. Print a table (one row per seed: seed, points) and the mean, min and max of the points, copied from the log lines.
5. Run `$PY -m tdx.tools.compare runs/ --baseline runs/pooled/pooled_recent | head -8` and paste its output.

## Final report (print only this)
- Number of seeds finished (must be 12), table, mean/min/max, the compare output.
- Any command that failed and its last 5 log lines.
