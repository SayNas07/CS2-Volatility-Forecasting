"""
Stage 15: does the headline result survive different cleaning choices?

    python robustness.py --smoke     # 3 items, 2 settings - check it runs
    python robustness.py             # full grid (slow: budget 1-2 hours)

Reads  data/daily_prices.csv, data/panel.csv
Writes data/robustness.csv

The cleaning thresholds in clean_panel.py - the spike-repair band, the
reversal tolerance, and the penny-price floor - were chosen by inspection.
That is defensible but it is also the most obvious line of attack on the
paper: a reader can reasonably ask whether the result is an artefact of
those choices.

This re-runs the whole pipeline (clean -> forecast -> compare) under a grid
of settings and reports whether EWMA still beats every estimated model.

Runtime is dominated by GARCH re-estimation, so it runs on a fixed subset of
items. That is fine for a sensitivity check: the question is whether the
ranking flips, not whether the exact loss values move.
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from arch import arch_model

DAILY = Path("data/daily_prices.csv")
PANEL = Path("data/panel.csv")
OUT = Path("data/robustness.csv")

START = "2021-01-01"
SCALE = 100
WINDOW = 500
REFIT_EVERY = 7
EWMA_LAMBDA = 0.94
SD_WINDOW = 30
SANITY = 50
MAX_ZERO_FRAC = 0.20

EXCLUDED = ["AWP | Chromatic Aberration (Field-Tested)",
            "AK-47 | Head Shot (Field-Tested)"]

# (spike_size, reversal_tol, penny_floor) - the baseline is the middle row
GRID = [
    ("loose",    0.60, 0.25, 0.25),
    ("baseline", 0.80, 0.35, 0.50),
    ("strict",   1.00, 0.45, 1.00),
    ("no repair", np.inf, 0.0, 0.50),   # spike filter disabled entirely
]

SPECS = {"GARCH-t":  dict(vol="GARCH",  p=1, q=1, dist="t"),
         "EGARCH-t": dict(vol="EGARCH", p=1, q=1, dist="t"),
         "GJR-t":    dict(vol="GARCH",  p=1, o=1, q=1, dist="t")}


def repair(prices, size, tol):
    p = prices.to_numpy(float).copy()
    if not np.isfinite(size):
        return pd.Series(p, index=prices.index), 0
    lg = np.log(p)
    n = 0
    for t in range(1, len(p) - 1):
        r1, r2 = lg[t] - lg[t - 1], lg[t + 1] - lg[t]
        if abs(r1) > size and abs(r1 + r2) < tol * abs(r1):
            p[t] = np.exp((lg[t - 1] + lg[t + 1]) / 2)
            lg[t] = np.log(p[t])
            n += 1
    return pd.Series(p, index=prices.index), n


def qlike(realised, var_hat):
    return np.log(var_hat) + realised / var_hat


def forecast_losses(y):
    """Mean QLIKE per model over rolling one-step forecasts."""
    out = {}
    for name, kw in SPECS.items():
        losses, res = [], None
        for i in range(WINDOW, len(y)):
            train = y.iloc[i - WINDOW:i]
            tv = float(train.var())
            if res is None or (i - WINDOW) % REFIT_EVERY == 0:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        cand = arch_model(train, mean="AR", lags=7, **kw).fit(
                            disp="off", show_warning=False)
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
            if np.isfinite(v) and tv / SANITY < v < tv * SANITY:
                losses.append(qlike(y.iloc[i] ** 2, v))
        out[name] = float(np.mean(losses)) if losses else np.nan

    var, ew = y.iloc[:WINDOW].var(), []
    for i in range(WINDOW, len(y)):
        var = EWMA_LAMBDA * var + (1 - EWMA_LAMBDA) * y.iloc[i - 1] ** 2
        ew.append(qlike(y.iloc[i] ** 2, var))
    out["EWMA"] = float(np.mean(ew))

    sd = y.rolling(SD_WINDOW).var().shift(1)
    idx = sd.dropna().index[WINDOW - SD_WINDOW:]
    out["RollSD"] = float(qlike(y.loc[idx] ** 2, sd.loc[idx]).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--items", type=int, default=8)
    a = ap.parse_args()

    grid = GRID[:2] if a.smoke else GRID
    n_items = 3 if a.smoke else a.items

    d = pd.read_csv(DAILY, parse_dates=["date"])
    d["date"] = (d["date"].dt.tz_localize(None)
                 if getattr(d["date"].dt, "tz", None) is not None else d["date"])
    panel = pd.read_csv(PANEL)
    d = d[d["item"].isin(panel["item"]) & ~d["item"].isin(EXCLUDED)]

    # longest series first - they give the most forecasts per unit of runtime
    order = d.groupby("item").size().sort_values(ascending=False)
    items = list(order.index[:n_items])

    rows = []
    for label, size, tol, floor in grid:
        print(f"\n=== {label}  (spike {size}, tol {tol}, floor RM {floor}) ===")
        for item in items:
            g = d[d["item"] == item].sort_values("date").reset_index(drop=True)
            clean, n_fix = repair(g["price"], size, tol)
            if float(clean.median()) < floor:
                print(f"  {item[:38]:38s} dropped (penny floor)")
                continue
            ret = np.log(clean).diff()
            s = pd.Series(ret.to_numpy(), index=g["date"]).dropna()
            s = s[s.index >= START]
            if (s.abs() < 1e-9).mean() > MAX_ZERO_FRAC or len(s) < WINDOW + 150:
                print(f"  {item[:38]:38s} dropped (stale / too short)")
                continue
            res = forecast_losses(s * SCALE)
            best_est = min(res[m] for m in SPECS if np.isfinite(res[m]))
            rows.append({"setting": label, "item": item, "spikes_fixed": n_fix,
                         "n": len(s), **res,
                         "ewma_beats_all_estimated": res["EWMA"] < best_est})
            print(f"  {item[:38]:38s} EWMA {res['EWMA']:.3f}  best est {best_est:.3f}"
                  f"  {'EWMA' if res['EWMA'] < best_est else 'GARCH'}")

    r = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT, index=False)

    pd.set_option("display.width", 200)
    print("\n=== mean QLIKE by setting ===")
    cols = ["EWMA", "RollSD"] + list(SPECS)
    print(r.groupby("setting")[cols].mean().round(3).to_string())

    print("\n=== does EWMA still beat every estimated model? ===")
    chk = r.groupby("setting")["ewma_beats_all_estimated"].agg(["sum", "count"])
    for s, row in chk.iterrows():
        print(f"  {s:10s} {int(row['sum'])}/{int(row['count'])} items")

    if chk["sum"].sum() == chk["count"].sum():
        print("\nResult holds under every cleaning setting tested.")
    else:
        print("\nResult does NOT hold uniformly - report which settings flip it.")

    print(f"\n{OUT}")


if __name__ == "__main__":
    main()