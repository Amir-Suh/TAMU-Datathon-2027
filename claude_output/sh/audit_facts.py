"""Re-measure the facts quoted in claude_output/01_ADVERSARIAL_REVIEW.md. Stdlib only.
Usage: python3 -I audit_facts.py /path/to/public
"""
import collections
import csv
import statistics
import sys

R = sys.argv[1].rstrip("/") + "/"


def rd(p):
    return list(csv.DictReader(open(R + p)))


print("== sizes, history length, last tick vs cutoff, signal location ==")
for s in ("train", "validation", "test"):
    q, h = rd(f"{s}/queries.csv"), rd(f"{s}/history.csv")
    cut = {x["episode_id"]: int(x["cutoff"]) for x in q}
    cnt = collections.Counter((x["episode_id"], x["machine_id"]) for x in h)
    last = collections.defaultdict(int)
    for r in h:
        last[r["episode_id"]] = max(last[r["episode_id"]], int(r["tick"]))
    print(s, "queries", len(q), "episodes", len(cut), "history rows", len(h),
          "| last tick == cutoff-1 for", sum(last[e] == cut[e] - 1 for e in cut), "of", len(cut),
          "| machines with <=70 rows", sum(v <= 70 for v in cnt.values()),
          "| rows==64:", sum(v == 64 for v in cnt.values()))
print("train labels:", dict(collections.Counter(x["outcome"] for x in rd("train/labels.csv"))))

print("\n== validation targets: true p_0 deciles and spread by condition ==")
t = {x["query_id"]: x for x in rd("validation/targets.csv")}
c = {x["query_id"]: x["condition"] for x in rd("validation/conditions.csv")}
q = {x["query_id"]: x for x in rd("validation/queries.csv")}
print("deciles", sorted(collections.Counter(min(int(float(t[k]["p_0"]) * 10), 9) for k in t).items()))
for cond in ("stable", "short_history", "drift", "shared_change"):
    v = [float(t[k]["p_0"]) for k in t if c[k] == cond]
    print(cond, "mean", round(statistics.mean(v), 3), "sd", round(statistics.pstdev(v), 3))
for key in ("machine_type", "signal"):
    g = collections.defaultdict(list)
    for k, x in q.items():
        g[x[key]].append(float(t[k]["p_0"]))
    print("mean true p_0 by", key, {a: round(statistics.mean(b), 3) for a, b in sorted(g.items())})

print("\n== train history: P(payout>0 | self prev, neighbours prev) and per-machine sign ==")
tq = rd("train/queries.csv")
typ = {(x["episode_id"], x["machine_id"]): x["machine_type"] for x in tq}
H = collections.defaultdict(dict)
for r in rd("train/history.csv"):
    H[(r["episode_id"], r["machine_id"])][int(r["tick"])] = int(r["payout"])
eps = collections.defaultdict(list)
for (e, m) in H:
    eps[e].append(m)
agg = collections.defaultdict(lambda: [0, 0])
perm = collections.defaultdict(lambda: [0, 0])
for (e, m), s in H.items():
    for tick, y in s.items():
        if tick - 1 not in s:
            continue
        nb = [H[(e, o)].get(tick - 1) for o in eps[e] if o != m]
        if None in nb:
            continue
        sw = s[tick - 1] > 0
        nw = sum(v > 0 for v in nb)
        k = (typ[(e, m)], "prev_win" if sw else "prev_loss", "nb>=4won" if nw >= 4 else "nb<4won")
        agg[k][0] += y > 0
        agg[k][1] += 1
        perm[(e, m, sw)][0] += y > 0
        perm[(e, m, sw)][1] += 1
for k in sorted(agg):
    print(k, round(agg[k][0] / agg[k][1], 3), "n", agg[k][1])
sign = collections.defaultdict(lambda: [0, 0])
for (e, m) in H:
    w, l = perm.get((e, m, True)), perm.get((e, m, False))
    if w and l and w[1] >= 15 and l[1] >= 15:
        sign[typ[(e, m)]][0] += w[0] / w[1] > l[0] / l[1]
        sign[typ[(e, m)]][1] += 1
print("machines with P(win|prev win) > P(win|prev loss):", {k: (a, b, round(a / b, 2)) for k, (a, b) in sign.items()})
