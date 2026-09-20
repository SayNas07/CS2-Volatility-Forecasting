"""
Stage 11: generate the paper's figures.

    python make_figures.py

Reads  data/returns.csv, data/daily_prices.csv, data/forecasts.csv,
       data/losses.csv, data/lambda_check.csv, data/panel.csv
Writes figures/fig1_clustering.pdf   ... fig5_shock.pdf

Output is vector PDF at 7x4.5in, sized to sit in a one-column article at
\\includegraphics[width=\\textwidth]. Greyscale-safe: every series is
distinguished by linestyle as well as colour, because printed papers and
photocopied ones lose colour first.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("data")
FIG = Path("figures")
START = "2021-01-01"
SHOCK = pd.Timestamp("2025-10-23")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

INK = "#1a1a1a"
GREY = "#888888"


def short(name, n=28):
    """Item names are long; trim for titles and legends."""
    name = name.replace("(Factory New)", "(FN)").replace("(Field-Tested)", "(FT)")
    name = name.replace("(Minimal Wear)", "(MW)").replace("\u2605 ", "")
    return name if len(name) <= n else name[:n - 1] + "\u2026"


def naive(s):
    """daily_prices.csv carries UTC-aware timestamps, returns.csv does not.
    Strip tz so both compare against plain Timestamps."""
    s = pd.to_datetime(s)
    return s.dt.tz_localize(None) if getattr(s.dt, "tz", None) is not None else s


def load_returns():
    r = pd.read_csv(DATA / "returns.csv", parse_dates=["date"])
    r["date"] = naive(r["date"])
    return r[r["date"] >= START]


def acf(x, lags):
    x = np.asarray(x, float)
    x = x - x.mean()
    denom = (x @ x)
    return np.array([1.0 if L == 0 else (x[L:] @ x[:-L]) / denom for L in range(lags + 1)])


# ---------------------------------------------------------------- figure 1
def fig_clustering(r):
    """Returns over time for one item per tier. The visual case for ARCH."""
    tiers = ["sub5", "5_50", "50_500", "500plus"]
    labels = {"sub5": "Below RM 5", "5_50": "RM 5-50",
              "50_500": "RM 50-500", "500plus": "Above RM 500"}
    picks = []
    for t in tiers:
        sub = r[r["tier"] == t]
        if sub.empty:
            continue
        # longest series in the tier, for a readable panel
        best = sub.groupby("item").size().idxmax()
        picks.append((t, best))

    fig, axes = plt.subplots(len(picks), 1, figsize=(7, 1.5 * len(picks)), sharex=True)
    for ax, (tier, item) in zip(np.atleast_1d(axes), picks):
        g = r[r["item"] == item].sort_values("date")
        ax.plot(g["date"], g["ret"] * 100, lw=0.4, color=INK)
        ax.axhline(0, lw=0.4, color=GREY)
        ax.axvline(SHOCK, lw=0.7, color="#c0392b", ls="--", alpha=0.8)
        ax.set_ylabel("return (%)")
        ax.text(0.01, 0.92, f"{labels[tier]}: {short(item, 34)}",
                transform=ax.transAxes, va="top", fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("")
    fig.align_ylabels()
    fig.savefig(FIG / "fig1_clustering.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_acf(r):
    """ACF of returns vs squared returns, pooled. The two central facts in one panel."""
    LAGS = 20
    ac_r, ac_r2 = [], []
    for _, g in r.groupby("item"):
        x = g.sort_values("date")["ret"].to_numpy()
        if len(x) < 300:
            continue
        ac_r.append(acf(x, LAGS))
        ac_r2.append(acf(x ** 2, LAGS))
    ac_r, ac_r2 = np.array(ac_r), np.array(ac_r2)
    lags = np.arange(LAGS + 1)
    n_med = int(np.median([len(g) for _, g in r.groupby("item")]))
    band = 1.96 / np.sqrt(n_med)

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    for ax, data, title in [
        (axes[0], ac_r, "Returns $r_t$"),
        (axes[1], ac_r2, "Squared returns $r_t^2$"),
    ]:
        ax.axhspan(-band, band, color=GREY, alpha=0.18, lw=0)
        for row in data:
            ax.plot(lags[1:], row[1:], lw=0.5, color=GREY, alpha=0.45)
        ax.plot(lags[1:], data[:, 1:].mean(axis=0), lw=1.6, color=INK, label="panel mean")
        ax.axhline(0, lw=0.5, color=INK)
        ax.set_title(title)
        ax.set_xlabel("lag (days)")
    axes[0].set_ylabel("autocorrelation")
    axes[0].legend(loc="upper right")
    fig.savefig(FIG / "fig2_acf.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_tails(r):
    """QQ plot against the normal, pooled standardised returns."""
    z = []
    for _, g in r.groupby("item"):
        x = g["ret"].to_numpy()
        if len(x) < 300:
            continue
        z.append((x - x.mean()) / x.std())
    z = np.sort(np.concatenate(z))
    n = len(z)
    q_emp = z
    q_norm = np.sqrt(2) * np.vectorize(lambda p: _erfinv(2 * p - 1))(
        (np.arange(1, n + 1) - 0.5) / n)

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    lim = abs(q_norm).max() * 1.05
    ax.plot([-lim, lim], [-lim, lim], lw=0.8, color="#c0392b", ls="--",
            label="normal")
    ax.scatter(q_norm, q_emp, s=1.5, color=INK, alpha=0.35, lw=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(np.percentile(q_emp, 0.02), np.percentile(q_emp, 99.98))
    ax.set_xlabel("theoretical normal quantile")
    ax.set_ylabel("empirical quantile (standardised)")
    ax.legend(loc="upper left")
    fig.savefig(FIG / "fig3_tails.pdf")
    plt.close(fig)


def _erfinv(y):
    # Winitzki approximation - avoids a scipy dependency for one call
    a = 0.147
    ln = np.log(1 - y ** 2)
    t1 = 2 / (np.pi * a) + ln / 2
    return np.sign(y) * np.sqrt(np.sqrt(t1 ** 2 - ln / a) - t1)


# ---------------------------------------------------------------- figure 4
def fig_forecasts():
    """Conditional volatility forecasts vs realised, one representative item."""
    f = pd.read_csv(DATA / "forecasts.csv", parse_dates=["date"])
    f["date"] = naive(f["date"])
    counts = f.groupby("item")["date"].nunique()
    item = counts.idxmax()
    g = f[f["item"] == item]
    wide = g.pivot_table(index="date", columns="model", values="var_hat")
    realised = g.groupby("date")["realised"].first()

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.fill_between(realised.index, 0, np.sqrt(realised), color=GREY, alpha=0.35,
                    lw=0, label="realised $|r_t|$")
    styles = {"EWMA": ("-", 1.2, "#1f4e79"), "GARCH-t": ("--", 1.0, INK),
              "EGARCH-t": (":", 1.0, "#c0392b")}
    for m, (ls, lw, c) in styles.items():
        if m in wide.columns:
            ax.plot(wide.index, np.sqrt(wide[m]), ls=ls, lw=lw, color=c, label=m)
    ax.set_ylabel("conditional volatility (%/day)")
    ax.set_ylim(0, np.sqrt(realised).quantile(0.995) * 1.1)
    ax.set_title(short(item, 40), loc="left")
    ax.legend(ncol=4, loc="upper left")
    fig.savefig(FIG / "fig4_forecasts.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def fig_shock():
    """Normalised price paths around 23 Oct 2025: knives against Covert skins."""
    d = pd.read_csv(DATA / "daily_prices.csv", parse_dates=["date"])
    d["date"] = naive(d["date"])
    panel = pd.read_csv(DATA / "panel.csv")
    treated = ["AWP | Chromatic Aberration (Field-Tested)",
               "AK-47 | Head Shot (Field-Tested)"]
    knives = [i for i in panel["item"] if i.startswith("\u2605")]

    win = d[(d["date"] >= SHOCK - pd.Timedelta(days=45)) &
            (d["date"] <= SHOCK + pd.Timedelta(days=45))]

    fig, ax = plt.subplots(figsize=(7, 3.4))
    for group, colour, ls, lab in [
        (treated, "#1f4e79", "-", "Covert skins (crafting input)"),
        (knives, "#c0392b", "--", "Knives (supply expanded)"),
    ]:
        first = True
        for item in group:
            g = win[win["item"] == item].sort_values("date")
            if g.empty:
                continue
            # Normalise to the day BEFORE the shock: the 23 Oct price already
            # contains the jump, so rebasing on it flattens the entire event.
            base = g[g["date"] < SHOCK]["price"]
            if base.empty:
                continue
            ax.plot(g["date"], 100 * g["price"] / base.iloc[-1],
                    color=colour, ls=ls, lw=1.1,
                    label=lab if first else None)
            first = False
    ax.axvline(SHOCK, lw=0.8, color=INK)
    ax.axhline(100, lw=0.5, color=GREY)
    ax.annotate("Trade-Up update\n23 Oct 2025", xy=(SHOCK, ax.get_ylim()[1]),
                xytext=(6, -4), textcoords="offset points",
                va="top", fontsize=8, color=INK)
    ax.set_yscale("log")
    ax.set_yticks([25, 50, 100, 200, 400])
    ax.set_yticklabels(["25", "50", "100", "200", "400"])
    ax.set_ylabel("price, 22 Oct 2025 = 100 (log scale)")
    ax.legend(loc="upper left")
    fig.savefig(FIG / "fig5_shock.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 6
def fig_shock_losses():
    """QLIKE around the event. Log scale, because the treated spike is 3 orders
    of magnitude and would otherwise flatten everything else to a line."""
    p = DATA / "shock_losses.csv"
    if not p.exists():
        print("  (skipping fig6: run shock_analysis.py first)")
        return
    L = pd.read_csv(p, parse_dates=["date"])
    L = L[(L["rel_day"] >= -10) & (L["rel_day"] <= 30)]

    styles = {"EWMA": ("-", "#1f4e79"), "GARCH-t": ("--", INK),
              "EGARCH-t": (":", "#c0392b"), "GJR-t": ("-.", "#7d6608"),
              "RollSD": ((0, (3, 1, 1, 1, 1, 1)), GREY)}

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0), sharex=True)
    for ax, grp, title in [
        (axes[0], ["treated"], "Treated (Covert skins, $+380\\%$)"),
        (axes[1], ["supply", "control"], "Supply-shocked and control"),
    ]:
        sub = L[L["group"].isin(grp)]
        for m, (ls, c) in styles.items():
            s = sub[sub["model"] == m].groupby("rel_day")["qlike"].mean()
            if s.empty:
                continue
            ax.plot(s.index, s.values, ls=ls, lw=1.1, color=c, label=m)
        ax.axvline(0, lw=0.8, color=INK)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("days from 23 Oct 2025")
    axes[0].set_ylabel("mean QLIKE (log scale)")
    axes[0].legend(ncol=2, loc="upper right", fontsize=7)
    fig.savefig(FIG / "fig6_shock_losses.pdf")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    r = load_returns()
    fig_clustering(r)
    fig_acf(r)
    fig_tails(r)
    fig_forecasts()
    fig_shock()
    fig_shock_losses()
    for p in sorted(FIG.glob("*.pdf")):
        print(f"  {p}  ({p.stat().st_size // 1024} kB)")


if __name__ == "__main__":
    main()