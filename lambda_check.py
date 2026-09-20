"""
Stage 10: is EWMA's win about estimation failure, or about one lucky constant?

    python lambda_check.py

Reads  data/returns.csv, data/forecasts.csv
Writes data/lambda_check.csv

The headline result - that no estimated GARCH model beats a parameter-free
benchmark - currently rests on EWMA at lambda = 0.94, the RiskMetrics default.
If 0.94 happens to suit this data and neighbouring values do not, the finding
is a coincidence rather than evidence that estimation fails.

This re-runs EWMA across a grid of lambdas and re-tests each against the
GARCH forecasts already computed. No refitting - EWMA estimates nothing, so
this takes seconds rather than half an hour.

Read the output this way:
  * EWMA wins across the whole grid  -> the result is about estimation failing
  * EWMA wins only near 0.94         -> the result is a lucky constant, and
                                        the write-up must say so
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

RETURNS_CSV = Path("data/returns.csv")
FORECASTS_CSV = Path("data/forecasts.csv")
OUT_CSV = Path("data/lambda_check.csv")

START = "2021-01-01"
SCALE = 100
WINDOW = 500
MIN_OBS = 250

LAMBDAS = [0.85, 0.90, 0.94, 0.97, 0.99]
GARCH_MODELS = ["EGARCH-t", "GARCH-t", "GJR-t"]
BASELINE = "RollSD"


def qlike(realised, var_hat):
    return np.log(var_hat) + realised / var_hat


def nw_var(d, lags=None):
    d = np.asarray(d, dtype=float)
    T = len(d)
    if lags is None:
        lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    dm = d - d.mean()
    total = (dm @ dm) / T
    for L in range(1, lags + 1):
        total += 2 * (1 - L / (lags + 1)) * (dm[L:] @ dm[:-L]) / T
    return max(total, 1e-12)


def dm_test(loss_a, loss_b):
    """Positive stat => A has higher loss => B is better."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    T = len(d)
    stat = d.mean() / np.sqrt(nw_var(d) / T)
    stat *= np.sqrt((T - 1) / T) if T > 2 else 1.0
    return stat, 2 * (1 - stats.t.cdf(abs(stat), df=T - 1))


def ewma_series(y, lam):
    var = y.iloc[:WINDOW].var()
    dates, vals = [], []
    for i in range(WINDOW, len(y)):
        var = lam * var + (1 - lam) * y.iloc[i - 1] ** 2
        dates.append(y.index[i])
        vals.append(var)
    return pd.Series(vals, index=pd.Index(dates, name="date"))


def main():
    r = pd.read_csv(RETURNS_CSV, parse_dates=["date"])
    r = r[r["date"] >= START]
    f = pd.read_csv(FORECASTS_CSV, parse_dates=["date"])

    print(f"half-life implied by each lambda:")
    for lam in LAMBDAS:
        print(f"  lambda {lam:.2f}  ->  {np.log(0.5)/np.log(lam):5.1f} days")

    rows = []
    for item, g in f.groupby("item"):
        y = (r[r["item"] == item].sort_values("date")
             .set_index("date")["ret"] * SCALE)
        realised = y.pow(2)

        wide = g.pivot_table(index="date", columns="model", values="var_hat")
        comparators = [m for m in GARCH_MODELS + [BASELINE] if m in wide.columns]

        for lam in LAMBDAS:
            e = ewma_series(y, lam)
            idx = wide.index.intersection(e.index)
            if len(idx) < MIN_OBS:
                continue
            le = qlike(realised.loc[idx], e.loc[idx])
            row = {"item": item, "lam": lam, "n": len(idx), "qlike_ewma": le.mean()}
            for m in comparators:
                lm = qlike(realised.loc[idx], wide.loc[idx, m])
                stat, p = dm_test(le, lm)
                row[f"qlike_{m}"] = lm.mean()
                row[f"beats_{m}"] = bool(stat < 0 and p < .05)
                row[f"loses_{m}"] = bool(stat > 0 and p < .05)
            rows.append(row)

    d = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 200)
    n_items = d["item"].nunique()
    print(f"\n=== mean QLIKE by lambda, across {n_items} items ===")
    cols = ["qlike_ewma"] + [c for c in d.columns if c.startswith("qlike_") and c != "qlike_ewma"]
    print(d.groupby("lam")[cols].mean().round(4).to_string())

    print(f"\n=== EWMA vs each model, significant DM wins at 5% (of {n_items} items) ===")
    header = "  lambda  " + "  ".join(f"{m:>18s}" for m in GARCH_MODELS + [BASELINE])
    print(header)
    for lam, g in d.groupby("lam"):
        cells = []
        for m in GARCH_MODELS + [BASELINE]:
            if f"beats_{m}" not in g:
                cells.append(f"{'-':>18s}")
                continue
            cells.append(f"{int(g[f'beats_{m}'].sum()):>7d} W {int(g[f'loses_{m}'].sum()):>3d} L")
        print(f"  {lam:.2f}    " + "  ".join(cells))

    print("\n(W = EWMA significantly better, L = EWMA significantly worse)")

    beats_all = []
    for lam, g in d.groupby("lam"):
        losses = sum(int(g[f"loses_{m}"].sum()) for m in GARCH_MODELS if f"loses_{m}" in g)
        beats_all.append((lam, losses))
    clean = [lam for lam, l in beats_all if l == 0]
    print(f"\nLambdas where EWMA never loses to any GARCH model: "
          f"{clean if clean else 'none'}")
    if len(clean) >= 3:
        print("Robust across the grid - the result is about estimation failing,")
        print("not about one fortunate constant. Say so explicitly in the write-up.")
    else:
        print("NOT robust - EWMA's advantage depends on the choice of lambda.")
        print("The headline result must be qualified accordingly.")

    print(f"\n{OUT_CSV}")


if __name__ == "__main__":
    main()