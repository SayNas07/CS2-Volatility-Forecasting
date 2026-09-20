"""
Stage 12: does the weekly cycle change anything?

    python ar_check.py

Reads  data/returns.csv
Writes data/ar_check.csv

The ACF figure shows clear spikes at lags 7 and 14 in raw returns. AR(1)
cannot represent that. This script asks two separate questions:

  1. Does AR(7) fit better in sample, and does it clear the lag-7
     autocorrelation out of the residuals?
  2. Does it change the out-of-sample forecast result?

These have different answers and it matters which is which. A better mean
model can improve fit without improving variance forecasts, because the mean
contributes very little to one-step conditional variance.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox

RETURNS = Path("data/returns.csv")
OUT = Path("data/ar_check.csv")
START = "2021-01-01"
SCALE = 100
WINDOW = 500
REFIT_EVERY = 7
EWMA_LAMBDA = 0.94
SANITY = 50

EXCLUDED = ["AWP | Chromatic Aberration (Field-Tested)",
            "AK-47 | Head Shot (Field-Tested)"]

SPECS = {"Constant": dict(mean="Constant"),
         "AR(1)": dict(mean="AR", lags=1),
         "AR(7)": dict(mean="AR", lags=7)}


def acf_at(x, lag):
    x = np.asarray(x, float) - np.mean(x)
    return (x[lag:] @ x[:-lag]) / (x @ x)


def fit(y, kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return arch_model(y, vol="GARCH", p=1, q=1, dist="t", **kwargs).fit(
            disp="off", show_warning=False)


# ---------------------------------------------------------- in-sample
def in_sample(r):
    rows = []
    for name, g in r.groupby("item"):
        y = g.sort_values("date")["ret"] * SCALE
        for label, kw in SPECS.items():
            try:
                res = fit(y, kw)
            except Exception:
                continue
            resid = res.resid.dropna()
            rows.append({
                "item": name, "spec": label, "n": len(y),
                "loglik": res.loglikelihood, "bic": res.bic,
                "resid_acf7": acf_at(resid, 7),
                "resid_lb7_p": acorr_ljungbox(resid, lags=[7],
                                              return_df=True)["lb_pvalue"].iloc[0],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------- out-of-sample
def qlike(realised, var_hat):
    return np.log(var_hat) + realised / var_hat


def rolling_qlike(y, kwargs):
    """Mean QLIKE of rolling one-step forecasts under a given mean spec."""
    losses, res = [], None
    for i in range(WINDOW, len(y)):
        train = y.iloc[i - WINDOW:i]
        tv = float(train.var())
        if res is None or (i - WINDOW) % REFIT_EVERY == 0:
            try:
                cand = fit(train, kwargs)
                cv = float(cand.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
                if np.isfinite(cv) and tv / SANITY < cv < tv * SANITY:
                    res = cand
            except Exception:
                pass
        if res is None:
            continue
        try:
            v = float(res.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
        except Exception:
            continue
        if not np.isfinite(v) or not (tv / SANITY < v < tv * SANITY):
            continue
        losses.append(qlike(y.iloc[i] ** 2, v))
    return float(np.mean(losses)) if losses else np.nan


def ewma_qlike(y):
    var = y.iloc[:WINDOW].var()
    losses = []
    for i in range(WINDOW, len(y)):
        var = EWMA_LAMBDA * var + (1 - EWMA_LAMBDA) * y.iloc[i - 1] ** 2
        losses.append(qlike(y.iloc[i] ** 2, var))
    return float(np.mean(losses))


def main():
    r = pd.read_csv(RETURNS, parse_dates=["date"])
    r = r[(r["date"] >= START) & (~r["item"].isin(EXCLUDED))]

    print("=== in sample ===")
    d = in_sample(r)
    d.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    print(d.groupby("spec")[["bic", "loglik", "resid_acf7"]].mean().round(3).to_string())
    best = d.loc[d.groupby("item")["bic"].idxmin(), "spec"].value_counts()
    print("\nlowest BIC per item:")
    print(best.to_string())
    print("\nitems with residual lag-7 autocorrelation still significant:")
    for s in SPECS:
        sub = d[d["spec"] == s]
        print(f"  {s:9s} {(sub['resid_lb7_p'] < .05).sum()}/{len(sub)}")

    print("\n=== out of sample (3 longest items) ===")
    lengths = r.groupby("item").size().sort_values(ascending=False)
    for name in lengths.index[:3]:
        y = r[r["item"] == name].sort_values("date")["ret"] * SCALE
        if len(y) < WINDOW + 200:
            continue
        q1 = rolling_qlike(y, SPECS["AR(1)"])
        q7 = rolling_qlike(y, SPECS["AR(7)"])
        qe = ewma_qlike(y)
        print(f"  {name[:38]:38s}  AR(1) {q1:.4f}   AR(7) {q7:.4f}   EWMA {qe:.4f}"
              f"   {'EWMA still wins' if qe < min(q1, q7) else 'GARCH wins'}")

    print(f"\n{OUT}")


if __name__ == "__main__":
    main()