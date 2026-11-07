# W2 read-only review

Reviewer: gpt-6-astra, high effort, selected explicitly through `collaboration.spawn_agent(model=..., reasoning_effort=..., fork_turns="none")` per W2 package table.

Three findings were fixed before the pilot grid:
- Sort final supervision splits to eliminate hash-dependent row order and bagging. Before fix, changing PYTHONHASHSEED changed synthetic p_0 by 0.004232.
- Carry forward each other machine's last observed payout for the neighbour snapshot; 5/10 aggregates remain calendar-tick windows over observed cells.
- Preserve missing divergence as NaN whenever a required rate window has no observations.

Re-review: all findings resolved; no blocking findings. Four synthetic final fits with PYTHONHASHSEED=1,2,3,17 had identical prediction SHA256 c530bb329159c24ae83ec16283a33165edd37c002c764d60d3f64ef1dc92327e, max_abs_difference=0.0, fit_rows=8240, best_iteration=1.
Sparse regression: tick9 other1 win/payout/maxclass=[1,1,1]; tick30 self/episode divergence=[NaN,NaN].

Specification resolution: W2 specifically says cutoff hard/soft supervision enters in final. Dev uses history only, despite the broader common contract permitting train cutoff labels in dev. No validation answers enter dev fitting.
