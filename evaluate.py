"""
Stage 9: score the forecasts. QLIKE, MSE, Diebold-Mariano.

    python evaluate.py

Reads  data/forecasts.csv
Writes data/losses.csv     per (item, model) mean losses
       data/dm_tests.csv   pairwise Diebold-Mariano results

WHY QLIKE IS PRIMARY

The realised proxy is the squared daily return - noisy but unbiased, because
no intraday data exists. Under a noisy proxy, MSE rewards models that happen
to spike on the days the proxy spikes, which is partly luck. QLIKE is robust
to proxy noise: the ranking it induces is consistent for the true conditional
variance even when the proxy is noisy. Patton (2011) is the reference.

MSE is reported too, for comparability with the literature. When the two
disagree, QLIKE is believed and the disagreement is discussed.

  QLIKE  = log(sigma_hat^2) + realised / sigma_hat^2
  MSE    = (realised - sigma_hat^2)^2

DIEBOLD-MARIANO

Tests whether the mean loss differential between two models is zero.
One-step-ahead forecasts need no HAC lags in theory, but loss differentials
are autocorrelated in practice, so Newey-West is used with a small bandwidth.
Positive statistic = the first model has HIGHER loss = the second is better.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy import stats

FORECASTS_CSV = Path("data/forecasts.csv")
LOSSES_CSV = Path("data/losses.csv")
DM_CSV = Path("data/dm_tests.csv")

MODELS = ["EGARCH-t", "GARCH-t", "GJR-t", "EWMA", "RollSD"]
BASELINE = "RollSD"
MIN_OBS = 250   # skip items with too few forecasts for a meaningful DM test


def qlike(realised, var_hat):
    return np.log(var_hat) + realised / var_hat


def mse(realised, var_hat):
    return (realised - var_hat) ** 2


def nw_var(d, lags=None):
    """Newey-West long-run variance of a loss differential series."""
    d = np.asarray(d, dtype=float)
    T = len(d)
    if lags is None:
        lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    dm = d - d.mean()
    gamma0 = (dm @ dm) / T
    total = gamma0
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        gL = (dm[L:] @ dm[:-L]) / T
        total += 2 * w * gL
    return max(total, 1e-12)


def dm_test(loss_a, loss_b):
    """Diebold-Mariano. Positive stat => model A has higher loss => B is better."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    T = len(d)
    stat = d.mean() / np.sqrt(nw_var(d) / T)
    # Harvey-Leybourne-Newbold small-sample correction (h=1)
    stat *= np.sqrt((T + 1 - 2) / T) if T > 2 else 1.0
    p = 2 * (1 - stats.t.cdf(abs(stat), df=T - 1))
    return stat, p


def main():
    f = pd.read_csv(FORECASTS_CSV, parse_dates=["date"])
    f["qlike"] = qlike(f["realised"], f["var_hat"])
    f["mse"] = mse(f["realised"], f["var_hat"])

    # ---- mean losses ----
    losses = (f.groupby(["item", "tier", "model"], observed=True)
                .agg(n=("qlike", "size"), qlike=("qlike", "mean"), mse=("mse", "mean"))
                .reset_index())
    LOSSES_CSV.parent.mkdir(parents=True, exist_ok=True)
    losses.to_csv(LOSSES_CSV, index=False)

    pd.set_option("display.width", 200)
    print("=== mean loss by model, averaged over items ===")
    agg = losses.groupby("model", observed=True)[["qlike", "mse"]].mean()
    agg["qlike_rank"] = agg["qlike"].rank()
    agg["mse_rank"] = agg["mse"].rank()
    print(agg.sort_values("qlike").to_string(float_format="%.4f"))

    print("\n=== how often each model wins, per item ===")
    for loss in ["qlike", "mse"]:
        wins = losses.loc[losses.groupby("item")[loss].idxmin(), "model"].value_counts()
        print(f"  {loss.upper():6s}: " + ", ".join(f"{m} {c}" for m, c in wins.items()))

    print("\n=== mean QLIKE by tier ===")
    print(losses.pivot_table(index="tier", columns="model", values="qlike",
                             observed=True).round(3).to_string())

    # ---- Diebold-Mariano, per item, all pairs ----
    rows = []
    for item, g in f.groupby("item"):
        wide_q = g.pivot_table(index="date", columns="model", values="qlike")
        wide_m = g.pivot_table(index="date", columns="model", values="mse")
        wide_q, wide_m = wide_q.dropna(), wide_m.dropna()
        if len(wide_q) < MIN_OBS:
            continue
        present = [m for m in MODELS if m in wide_q.columns]
        for a, b in combinations(present, 2):
            sq, pq = dm_test(wide_q[a], wide_q[b])
            sm, pm = dm_test(wide_m[a], wide_m[b])
            rows.append({"item": item, "n": len(wide_q), "model_a": a, "model_b": b,
                         "dm_qlike": sq, "p_qlike": pq, "dm_mse": sm, "p_mse": pm,
                         "better_qlike": b if sq > 0 else a})

    dm = pd.DataFrame(rows)
    dm.to_csv(DM_CSV, index=False)

    print(f"\n=== Diebold-Mariano, QLIKE, across {dm['item'].nunique()} items ===")
    print("(wins = significantly lower QLIKE at 5%, per item)\n")
    summary = []
    for a, b in combinations(MODELS, 2):
        s = dm[(dm["model_a"] == a) & (dm["model_b"] == b)]
        if s.empty:
            continue
        sig = s[s["p_qlike"] < .05]
        summary.append({
            "pair": f"{a} vs {b}",
            "items": len(s),
            "sig": len(sig),
            f"{a} wins": int((sig["better_qlike"] == a).sum()),
            f"{b} wins": int((sig["better_qlike"] == b).sum()),
        })
    for row in summary:
        a_b = [k for k in row if k.endswith("wins")]
        print(f"  {row['pair']:22s}  {row['sig']:2d}/{row['items']:2d} significant   "
              + "   ".join(f"{k.replace(' wins','')}: {row[k]}" for k in a_b))

    print(f"\n=== each model vs the {BASELINE} baseline (QLIKE) ===")
    for m in MODELS:
        if m == BASELINE:
            continue
        s = dm[((dm["model_a"] == m) & (dm["model_b"] == BASELINE)) |
               ((dm["model_a"] == BASELINE) & (dm["model_b"] == m))]
        sig = s[s["p_qlike"] < .05]
        beats = int((sig["better_qlike"] == m).sum())
        loses = int((sig["better_qlike"] == BASELINE).sum())
        print(f"  {m:10s} beats baseline on {beats}/{len(s)} items, "
              f"loses on {loses}, not significant on {len(s) - len(sig)}")

    print(f"\n{LOSSES_CSV}\n{DM_CSV}")


if __name__ == "__main__":
    main()