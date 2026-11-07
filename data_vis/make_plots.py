"""Exploratory charts for the House Edge (slot-machine forecasting) challenge.

Run from anywhere:  python data_vis/make_plots.py
Reads  ../main-participants-60-40/public  and writes PNGs next to this script.
Only train + validation data are used for anything that needs labels / conditions;
test is used only for the structural (history length / cutoff) chart.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "main-participants-60-40" / "public"

# ---- design tokens (reference palette, light mode) ----------------------------
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3 (validated all-pairs)
TYPE_COLOR = {"steady": C_BLUE, "balanced": C_ORANGE, "jackpot": C_AQUA}
TYPES = ["steady", "balanced", "jackpot"]
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
COND = ["stable", "short_history", "drift", "shared_change"]
COND_LABEL = {"stable": "Stable", "short_history": "Short history", "drift": "Drift", "shared_change": "Shared change"}
PAYOUTS = [0, 1, 2, 5, 20]
PCOLS = ["p_0", "p_1", "p_2", "p_5", "p_20"]

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 160,
})


def hgrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def finish(fig, name, title, subtitle):
    import textwrap
    w, h = fig.get_size_inches()
    lines = textwrap.wrap(subtitle, width=int(w * 9.5))
    head_in = 0.45 + 0.24 * len(lines) + 0.45          # title + subtitle + gap above panel titles
    fig.subplots_adjust(top=1 - head_in / h)
    fig.text(0.01, 1 - 0.10 / h, title, ha="left", va="top", fontsize=15, fontweight="bold")
    fig.text(0.01, 1 - 0.50 / h, chr(10).join(lines), ha="left", va="top", fontsize=10.5, color=INK2, linespacing=1.4)
    fig.savefig(HERE / name, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", name)


# ---- load ---------------------------------------------------------------------
def load(split, labelled=True):
    q = pd.read_csv(DATA / split / "queries.csv")
    h = pd.read_csv(DATA / split / "history.csv")
    if labelled:
        q = q.merge(pd.read_csv(DATA / split / "conditions.csv"))
    return q, h


tq, th = load("train")
vq, vh = load("validation")
sq, sh = load("test", labelled=False)
vt = pd.read_csv(DATA / "validation" / "targets.csv")
vr = pd.read_csv(DATA / "validation" / "reference.csv")

q = pd.concat([tq, vq], ignore_index=True)
h = pd.concat([th, vh], ignore_index=True)
h = h.merge(q[["machine_id", "episode_id", "machine_type", "condition", "cutoff"]], on=["machine_id", "episode_id"])
h["win"] = (h.payout > 0).astype(float)
long_h = h[h.condition != "short_history"]

# =============================================================================
# 01  History structure
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.28})
ax = axes[0]
allq = pd.concat([tq.assign(split="train"), vq.assign(split="validation")])
n_rows = h.groupby("machine_id").size().rename("n")
allq = allq.merge(n_rows, left_on="machine_id", right_index=True)
mean_n = allq.groupby("condition").n.mean().reindex(COND)
bars = ax.bar([COND_LABEL[c] for c in COND], mean_n.values, color=C_BLUE, width=0.55)
for b, v in zip(bars, mean_n.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v:.0f}", ha="center", fontsize=10.5, color=INK)
ax.set_ylabel("Observed ticks per machine (mean)")
ax.set_title("Short-history machines see ~64 ticks, others ~220")
hgrid(ax)
ax.tick_params(axis="x", labelsize=9.5)

ax = axes[1]
sm = vq[vq.condition == "short_history"].machine_id.iloc[0]
lm = vq[vq.condition == "stable"].machine_id.iloc[0]
cut_s = vq.set_index("machine_id").cutoff[sm]
cut_l = vq.set_index("machine_id").cutoff[lm]
for y, mid, cut, label in [(1, lm, cut_l, "Long history"), (0, sm, cut_s, "Short history")]:
    ticks = vh[vh.machine_id == mid].tick.values
    ax.vlines(ticks, y - 0.32, y + 0.32, color=C_BLUE, lw=1.1)
    ax.plot([cut, cut], [y - 0.4, y + 0.4], color=C_ORANGE, lw=2.2)
    ax.text(cut + 2, y + 0.43, f"cutoff {cut}", color=INK2, fontsize=9.5, va="bottom")
    ax.text(-4, y, label, ha="right", va="center", color=INK, fontsize=10.5)
ax.set_xlim(0, 262)
ax.set_ylim(-0.6, 1.85)
ax.set_yticks([])
ax.set_xlabel("Tick")
ax.spines["left"].set_visible(False)
ax.set_title("Observed ticks: a short history is 24 sparse + 40 recent")
ax.tick_params(length=0)
finish(fig, "01_history_structure.png", "How much history does each query get?",
       "Short-history machines have 40 consecutive recent ticks plus 24 scattered older ones; missing ticks are unknown, not zero payouts.")

# =============================================================================
# 02  Payout distributions by machine type (observed vs true)
# =============================================================================
obs = pd.crosstab(h.machine_type, h.payout, normalize="index").reindex(TYPES)
vm = vt.merge(vq[["query_id", "machine_type", "signal", "condition"]])
tru = vm.groupby("machine_type")[PCOLS].mean().reindex(TYPES)
tru.columns = PAYOUTS

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True, gridspec_kw={"wspace": 0.08})
w = 0.26
for ax, df, ttl in [(axes[0], obs, "Observed payouts (train + validation histories)"),
                    (axes[1], tru, "True next-spin probabilities (validation targets)")]:
    for i, t in enumerate(TYPES):
        xs = np.arange(5) + (i - 1) * w
        bars = ax.bar(xs, df.loc[t].values, width=w - 0.03, color=TYPE_COLOR[t], label=t.capitalize())
        for x, v in zip(xs, df.loc[t].values):
            if v >= 0.13 or (t == "jackpot" and v > 0.025 and x > 3):
                ax.text(x, v + 0.01, f"{v*100:.0f}%" if v >= 0.1 else f"{v*100:.1f}%", ha="center", fontsize=8.2, color=INK)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"{p}×" for p in PAYOUTS])
    ax.set_title(ttl, fontsize=11)
    hgrid(ax)
axes[0].set_ylabel("Share of spins")
axes[1].legend(loc="upper right", title=None)
finish(fig, "02_payout_by_machine_type.png", "Machine type sets the baseline payout profile",
       "Jackpot machines lose most often but own the 20× tail; steady machines pay 1× often and almost never 20×. Priors must be per type.")

# =============================================================================
# 03  Signal x type effect on true odds
# =============================================================================
payout_arr = np.array(PAYOUTS)
vm["p_win"] = 1 - vm.p_0
vm["e_pay"] = (vm[PCOLS].values * payout_arr).sum(axis=1)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"wspace": 0.35})
for ax, col, ttl, fmt in [(axes[0], "p_win", "P(any payout)  =  1 − p₀", "{:.0%}"),
                          (axes[1], "e_pay", "Expected payout (chips returned per chip)", "{:.2f}")]:
    piv = vm.pivot_table(index="machine_type", columns="signal", values=col, aggfunc="mean").reindex(TYPES)
    im = ax.imshow(piv.values, cmap=SEQ, aspect="auto", vmin=piv.values.min() * 0.85, vmax=piv.values.max() * 1.02)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["−1", "0", "+1"])
    ax.set_yticks(range(3))
    ax.set_yticklabels([t.capitalize() for t in TYPES])
    ax.set_xlabel("Observed signal")
    ax.set_title(ttl, fontsize=11)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    thr = (piv.values.min() + piv.values.max()) / 2
    for i in range(3):
        for j in range(3):
            v = piv.values[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=11,
                    color="white" if v > thr else INK, fontweight="bold")
finish(fig, "03_signal_effect.png", "The signal shifts the true odds, but not uniformly",
       "Higher signal raises payout odds for jackpot and balanced machines; steady machines are non-monotonic (signal 0 is the weakest). Means over validation queries.")

# =============================================================================
# 04  Lag structure: own-machine win/lag correlation
# =============================================================================
def lag_corrs(g):
    s = g.set_index("tick").win.sort_index()
    full = s.reindex(range(int(s.index.max()) + 1))
    out = []
    for k in range(1, 9):
        a, b = full, full.shift(k)
        ok = a.notna() & b.notna()
        out.append(np.corrcoef(a[ok], b[ok])[0, 1] if a[ok].std() > 0 and b[ok].std() > 0 else np.nan)
    return out


rows = []
for mid, g in long_h.groupby("machine_id"):
    rows.append((mid, g.machine_type.iloc[0], g.condition.iloc[0], *lag_corrs(g)))
LAGS = [f"l{k}" for k in range(1, 9)]
lagdf = pd.DataFrame(rows, columns=["machine_id", "type", "cond"] + LAGS)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.3})
ax = axes[0]
mean_l = lagdf.groupby("type")[LAGS].mean().reindex(TYPES)
norm = TwoSlopeNorm(vcenter=0, vmin=-0.2, vmax=0.2)
ax.imshow(mean_l.values, cmap=DIV, norm=norm, aspect="auto")
ax.set_xticks(range(8))
ax.set_xticklabels(range(1, 9))
ax.set_yticks(range(3))
ax.set_yticklabels([t.capitalize() for t in TYPES])
ax.set_xlabel("Lag k (ticks back)")
ax.set_title("Mean corr( win at t , win at t−k )", fontsize=11)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
for i in range(3):
    for j in range(8):
        v = mean_l.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9.5,
                color="white" if abs(v) > 0.11 else INK, fontweight="bold" if j == 6 else "normal")

ax = axes[1]
bins = np.linspace(-0.4, 0.75, 24)
for t in TYPES:
    ax.hist(lagdf[lagdf.type == t].l7.dropna(), bins=bins, histtype="step", lw=2.2,
            color=TYPE_COLOR[t], label=f"{t.capitalize()} (n={int((lagdf.type == t).sum())})")
ax.axvline(0, color=MUTED, lw=1)
ax.set_xlabel("Per-machine lag-7 correlation")
ax.set_ylabel("Machines")
ax.set_title("Lag 7 is strong but varies a lot machine to machine", fontsize=11)
ax.legend(loc="upper right")
hgrid(ax)
finish(fig, "04_lag_structure.png", "Lag 7 dominates the temporal dependence",
       "Other lags average ~0.03, lag 7 averages ~0.16 with a wide spread (long-history machines, train + validation). Win = payout > 0.")

# =============================================================================
# 05  Machine-specific lag profiles
# =============================================================================
rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.6), sharey=False, gridspec_kw={"wspace": 0.12})
for ax, t in zip(axes, TYPES):
    d = lagdf[lagdf.type == t].dropna(subset=LAGS)
    d = d.iloc[rng.permutation(len(d))[:60]].sort_values("l7", ascending=False)
    ax.imshow(d[LAGS].values, cmap=DIV, norm=TwoSlopeNorm(0, -0.5, 0.5), aspect="auto", interpolation="nearest")
    ax.set_xticks(range(8))
    ax.set_xticklabels(range(1, 9), fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel("Lag k")
    ax.set_title(t.capitalize(), fontsize=11)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
axes[0].set_ylabel("60 random machines, sorted by lag-7 value")
sm_ = plt.cm.ScalarMappable(norm=TwoSlopeNorm(0, -0.5, 0.5), cmap=DIV)
cb = fig.colorbar(sm_, ax=axes, shrink=0.55, pad=0.02, aspect=22)
cb.set_label("corr( win at t , win at t−k )", color=INK2)
cb.outline.set_visible(False)
cb.ax.tick_params(length=0)
finish(fig, "05_machine_lag_profiles.png", "Each machine has its own lag fingerprint",
       "Rows are single machines; red = a win k ticks ago raises today's win chance, blue = lowers it. Lag 7 is mostly positive, but strength (and sign, at other lags) differs machine to machine, so fit per machine and shrink toward the machine type.")

# =============================================================================
# 06  Own vs. other-machine lag information
# =============================================================================
own, oth = {k: [] for k in range(1, 9)}, {k: [] for k in range(1, 9)}
for ep, g in long_h.groupby("episode_id"):
    mat = g.pivot(index="tick", columns="machine_id", values="win").sort_index()
    mat = mat.reindex(range(int(mat.index.max()) + 1))
    for m in mat.columns:
        y = mat[m]
        others = mat.drop(columns=m).mean(axis=1, skipna=True)
        for k in range(1, 9):
            for store, x in ((own, y.shift(k)), (oth, others.shift(k))):
                ok = y.notna() & x.notna()
                if ok.sum() > 30 and y[ok].std() > 0 and x[ok].std() > 0:
                    store[k].append(np.corrcoef(y[ok], x[ok])[0, 1])
fig, ax = plt.subplots(figsize=(8.2, 4.4))
ks = np.arange(1, 9)
for store, col, lab in [(own, C_BLUE, "Same machine's earlier wins"), (oth, C_ORANGE, "Mean of the other 7 machines' earlier wins")]:
    mu = np.array([np.mean(store[k]) for k in ks])
    se = np.array([np.std(store[k]) / np.sqrt(len(store[k])) for k in ks])
    ax.plot(ks, mu, color=col, lw=2.2, marker="o", ms=6, mfc=SURFACE, mew=2, label=lab)
    ax.fill_between(ks, mu - 1.96 * se, mu + 1.96 * se, color=col, alpha=0.15, lw=0)
ax.axhline(0, color=MUTED, lw=1)
ax.set_xticks(ks)
ax.set_xlabel("Lag k (ticks back)")
ax.set_ylabel("Mean correlation with win at t")
ax.legend(loc="upper left")
hgrid(ax)
finish(fig, "06_own_vs_other_machines.png", "A machine's own history matters far more than its neighbours'",
       "Mean per-machine correlation with the win at t (95% CI band), long-history episodes. The other 7 machines' average adds only a weak signal (about 0.02-0.04, peaking at lag 3), so the lag-7 effect is machine-specific.")
print("own lag7 %.3f  other lag7 %.3f" % (np.mean(own[7]), np.mean(oth[7])))

# =============================================================================
# 07  Where the pooled reference loses points
# =============================================================================
ref = vr.set_index("query_id")[PCOLS]
tru_ = vt.set_index("query_id")[PCOLS]
sq_err = (ref - tru_) ** 2
vm = vm.merge(sq_err.sum(axis=1).rename("err"), left_on="query_id", right_index=True)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.22})
ax = axes[0]
piv = vm.pivot_table(index="condition", columns="machine_type", values="err", aggfunc="mean").reindex(COND)[TYPES]
w = 0.26
for i, t in enumerate(TYPES):
    xs = np.arange(4) + (i - 1) * w
    ax.bar(xs, piv[t].values, width=w - 0.03, color=TYPE_COLOR[t], label=t.capitalize())
    for x, v in zip(xs, piv[t].values):
        ax.text(x, v + 0.006, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(range(4))
ax.set_xticklabels([COND_LABEL[c] for c in COND])
ax.set_ylabel("Mean Σ squared error vs. true probs")
ax.set_title("Reference (pooled) error by condition and machine type", fontsize=11)
ax.set_ylim(0, 0.33)
ax.legend(loc="upper center", ncol=3)
hgrid(ax)

ax = axes[1]
share = sq_err.sum() / sq_err.sum().sum()
bars = ax.bar([f"{p}×" for p in PAYOUTS], share.values, color=C_BLUE, width=0.55)
for b, v in zip(bars, share.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.0%}", ha="center", fontsize=10.5)
ax.set_ylabel("Share of total squared error")
ax.set_title("Which payout class carries the error", fontsize=11)
hgrid(ax)
finish(fig, "07_baseline_error.png", "Where the pooled baseline loses accuracy",
       "Validation error of the frozen reference forecast against the simulator's true probabilities. Steady and balanced machines have the most headroom, and 0× alone accounts for 68% of the squared error.")
print(piv.round(3))
print(share.round(3))

# =============================================================================
# 08  Within-history change in win rate (drift check)
# =============================================================================
def change(g):
    c = g.cutoff.iloc[0]
    late = g[g.tick >= c - 64].win
    early = g[g.tick < c - 64].win
    if len(late) < 20 or len(early) < 20:
        return np.nan
    return late.mean() - early.mean()


chg = long_h.groupby("machine_id").apply(change, include_groups=False).rename("d").reset_index()
chg = chg.merge(q[["machine_id", "condition"]]).dropna()
fig, ax = plt.subplots(figsize=(8.5, 4.6))
cl = ["stable", "drift", "shared_change"]
rng = np.random.default_rng(1)
for i, c in enumerate(cl):
    d = chg[chg.condition == c].d.values
    ax.scatter(i + rng.uniform(-0.28, 0.28, len(d)), d, s=14, color=C_BLUE, alpha=0.35, lw=0)
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    ax.plot([i - 0.33, i + 0.33], [med, med], color=INK, lw=2.4)
    ax.plot([i, i], [q1, q3], color=INK, lw=1.4)
    ax.text(i + 0.36, med, f"sd {d.std():.2f}", va="center", fontsize=9.5, color=INK2)
ax.axhline(0, color=MUTED, lw=1)
ax.set_xticks(range(3))
ax.set_xticklabels([COND_LABEL[c] for c in cl])
ax.set_xlim(-0.5, 2.8)
ax.set_ylabel("Win rate: last 64 ticks − earlier ticks")
hgrid(ax)
finish(fig, "08_drift_check.png", "Drift and shared-change machines move about 50% more than stable ones",
       "Per-machine change in win rate, last 64 ticks vs earlier (dots; line = median, bar = IQR). The average change is ~0 in every condition, so the shifts go both ways: it is spread, not direction, that flags a regime change.")
print(chg.groupby("condition").d.agg(["mean", "std"]).round(3))
