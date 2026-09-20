"""
Stage 8: rolling one-day-ahead volatility forecasts, all models, all items.

    python forecast.py                 # full run (slow - see note below)
    python forecast.py --quick         # 3 items, to check it works first

Reads  data/returns.csv
Writes data/forecasts.csv    one row per (item, model, date): forecast vs realised

RUNTIME. 22 items x 3 GARCH-family models x ~230 weekly refits over a 500-day
window. Expect 20-60 minutes. Run --quick first; do not discover a bug at
minute 45.

DESIGN, per the locked decisions:
  * Rolling 500-day window, not expanding. This market has regime shifts and
    an expanding window would drag stale pre-shock data through the forecast
    forever.
  * Refit weekly, forecast daily. Between refits the last fitted parameters
    are used to filter forward. Daily refitting is more correct but ~7x the
    runtime for a difference that is small in practice.
  * Every forecast is genuinely out of sample: the model at time t has never
    seen return t+1. This is the thing naive backtests get wrong.

MODELS
  EGARCH-t   primary - never hits a parameter boundary
  GARCH-t    comparison
  GJR-t      asymmetry, threshold form
  EWMA       RiskMetrics lambda=0.94, no estimation
  RollSD     rolling standard deviation, the honest baseline
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from arch import arch_model

RETURNS_CSV = Path("data/returns.csv")
OUT_CSV = Path("data/forecasts.csv")

START = "2021-01-01"
SCALE = 100
WINDOW = 500      # rolling estimation window, days
REFIT_EVERY = 7   # refit cadence, days
EWMA_LAMBDA = 0.94
SD_WINDOW = 30

EXCLUDED = [
    "AWP | Chromatic Aberration (Field-Tested)",
    "AK-47 | Head Shot (Field-Tested)",
]

GARCH_SPECS = {
    "EGARCH-t": dict(vol="EGARCH", p=1, q=1, dist="t"),
    "GARCH-t":  dict(vol="GARCH",  p=1, q=1, dist="t"),
    "GJR-t":    dict(vol="GARCH",  p=1, o=1, q=1, dist="t"),
}


# A refit is rejected if its one-step forecast is this far from the training
# sample variance. EGARCH occasionally converges to a degenerate solution that
# forecasts variances of 1e30+; because refits are weekly, one bad fit would
# otherwise be reused for seven days and destroy every loss average.
SANITY_BAND = 50


def garch_forecasts(y, model, kwargs):
    """Rolling one-step-ahead conditional variance forecasts.

    At each date t we fit (or reuse) on the previous WINDOW observations and
    forecast variance for t+1. y[t+1] is never in the training sample.

    Refits are sanity-checked before being accepted. A fit that converges to a
    degenerate solution is discarded and the previous parameters are kept - the
    same thing you would do in production, and it is reported, not hidden.
    """
    out, res = [], None
    rejected = 0
    for i in range(WINDOW, len(y)):
        train = y.iloc[i - WINDOW:i]
        train_var = float(train.var())

        if res is None or (i - WINDOW) % REFIT_EVERY == 0:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cand = arch_model(train, mean="AR", lags=1, **kwargs).fit(
                        disp="off", show_warning=False)
                cand_var = float(cand.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
                ok = (np.isfinite(cand_var)
                      and train_var / SANITY_BAND < cand_var < train_var * SANITY_BAND)
                if ok:
                    res = cand
                else:
                    rejected += 1
            except Exception:
                rejected += 1

        if res is None:
            continue
        try:
            var = float(res.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
        except Exception:
            continue
        if not np.isfinite(var) or var <= 0:
            continue
        # The held-over fit can also drift out of range as the window moves.
        if not (train_var / SANITY_BAND < var < train_var * SANITY_BAND):
            continue
        out.append({"date": y.index[i], "model": model, "var_hat": var})
    return out, rejected


def ewma_forecasts(y):
    """RiskMetrics EWMA. No parameters estimated - lambda is fixed by convention."""
    var = y.iloc[:WINDOW].var()
    out = []
    for i in range(WINDOW, len(y)):
        var = EWMA_LAMBDA * var + (1 - EWMA_LAMBDA) * y.iloc[i - 1] ** 2
        out.append({"date": y.index[i], "model": "EWMA", "var_hat": var})
    return out


def rollsd_forecasts(y):
    """Rolling standard deviation. The baseline a GARCH model must beat to justify itself."""
    out = []
    for i in range(WINDOW, len(y)):
        out.append({"date": y.index[i], "model": "RollSD",
                    "var_hat": y.iloc[i - SD_WINDOW:i].var()})
    return out


def main():
    quick = "--quick" in sys.argv

    r = pd.read_csv(RETURNS_CSV, parse_dates=["date"])
    r = r[(r["date"] >= START) & (~r["item"].isin(EXCLUDED))]
    meta = r.groupby("item")[["tier", "category"]].first()

    items = sorted(r["item"].unique())
    if quick:
        items = items[:3]
        print("QUICK MODE - 3 items only\n")

    frames, rejections = [], []
    for n, name in enumerate(items, 1):
        y = r[r["item"] == name].sort_values("date").set_index("date")["ret"] * SCALE
        if len(y) < WINDOW + 100:
            print(f"[{n}/{len(items)}] {name} - too short, skipped")
            continue
        print(f"[{n}/{len(items)}] {name}  ({len(y) - WINDOW} forecasts)")

        rows = []
        for model, kwargs in GARCH_SPECS.items():
            got, rej = garch_forecasts(y, model, kwargs)
            rows += got
            if rej:
                rejections.append({"item": name, "model": model, "rejected_refits": rej})
        rows += ewma_forecasts(y)
        rows += rollsd_forecasts(y)

        df = pd.DataFrame(rows)
        df["item"] = name
        # Realised proxy: squared return on the forecast date. Noisy but
        # unbiased - the reason QLIKE is the primary loss function.
        df["realised"] = df["date"].map(y.pow(2))
        frames.append(df)

    allf = pd.concat(frames, ignore_index=True).join(meta, on="item")
    allf = allf.dropna(subset=["var_hat", "realised"])
    allf = allf[allf["var_hat"] > 0]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    allf.to_csv(OUT_CSV, index=False)

    print(f"\n{len(allf):,} forecasts across {allf['item'].nunique()} items")
    print(allf.groupby("model").size().to_string())

    if rejections:
        rej = pd.DataFrame(rejections)
        print("\nRefits rejected as degenerate (parameters held over instead):")
        print(rej.groupby("model")["rejected_refits"].sum().to_string())
        print("Report this in the write-up - it is a property of the model, not a bug.")
    print(f"\n{OUT_CSV}")
    print("Next: evaluate.py for QLIKE, MSE and Diebold-Mariano.")


if __name__ == "__main__":
    main()