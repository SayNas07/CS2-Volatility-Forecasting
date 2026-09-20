"""
Stage 4: clean the panel and build the return series.

    python clean_panel.py

Reads  data/daily_prices.csv, data/panel.csv
Writes data/returns.csv        cleaned log returns, model-ready
       data/clean_report.csv   what was dropped or repaired, per item

Two problems this fixes, both found by inspecting the raw panel:

1. Stale pricing. Thinly traded items repeat the same median price for days,
   producing exact-zero returns. GARCH fits these happily and returns a
   meaningless persistence estimate, so items above MAX_ZERO_FRAC are dropped.

2. Penny items. Below about RM0.50 the median price is quantised and can be
   dragged for several days by a few trades. The spike filter below only sees
   one-day round trips, so these are dropped outright rather than repaired.

3. Bad prints. Steam's median occasionally reports a price orders of magnitude
   away from trend for exactly one day, then reverts. Example: a Bayonet
   Doppler printing RM 11.74 between two RM 2,200 days. These are not
   volatility - they are bad data, and they dominate kurtosis and any
   variance estimate if left in.

NOT cleaned: the 23 October 2025 Valve trade-up update. Covert skins roughly
quadrupled and knives fell 40% on enormous volume, and the moves persisted.
That is a genuine exogenous supply shock and the most interesting event in the
sample. Any filter aggressive enough to remove it is too aggressive.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DAILY_CSV = Path("data/daily_prices.csv")
PANEL_CSV = Path("data/panel.csv")
RETURNS_CSV = Path("data/returns.csv")
REPORT_CSV = Path("data/clean_report.csv")

MAX_ZERO_FRAC = 0.20   # drop items whose returns are this often exactly zero
MIN_MEDIAN_PRICE = 0.50  # sub-RM0.50 items are penny stocks - median is noise
SPIKE_SIZE = 0.80      # a move this large (log) is a spike candidate
REVERSAL_TOL = 0.35    # ...if the next day undoes this much of it, it's a bad print
MIN_OBS = 500          # minimum surviving observations to keep an item


def repair_spikes(prices):
    """Replace one-day round-trip prints with the geometric mean of neighbours.

    A genuine price move persists. A bad print reverses almost exactly the next
    day, so r_t and r_t+1 are large, opposite, and nearly cancel. Only prices
    matching that signature are touched; real jumps survive untouched.
    """
    p = prices.to_numpy(dtype=float).copy()
    logp = np.log(p)
    fixed = []
    for t in range(1, len(p) - 1):
        r1 = logp[t] - logp[t - 1]
        r2 = logp[t + 1] - logp[t]
        if abs(r1) > SPIKE_SIZE and abs(r1 + r2) < REVERSAL_TOL * abs(r1):
            p[t] = np.exp((logp[t - 1] + logp[t + 1]) / 2)
            logp[t] = np.log(p[t])
            fixed.append(t)
    return pd.Series(p, index=prices.index), len(fixed)


def main():
    daily = pd.read_csv(DAILY_CSV, parse_dates=["date"])
    panel = pd.read_csv(PANEL_CSV, parse_dates=["start", "end"])
    daily = daily[daily["item"].isin(panel["item"])]

    meta = panel.set_index("item")[["tier", "category", "median_price", "median_volume"]]

    report, kept = [], []
    for name, g in daily.groupby("item"):
        g = g.sort_values("date").reset_index(drop=True)

        raw_ret = np.log(g["price"]).diff().dropna()
        zero_frac = float((raw_ret.abs() < 1e-9).mean())

        clean_price, n_fixed = repair_spikes(g["price"])
        ret = np.log(clean_price).diff()

        row = {
            "item": name,
            "n_obs": len(raw_ret),
            "zero_frac": zero_frac,
            "spikes_repaired": n_fixed,
            "vol_before": raw_ret.std() * np.sqrt(365),
            "vol_after": ret.std() * np.sqrt(365),
            "kurt_before": raw_ret.kurtosis(),
            "kurt_after": ret.dropna().kurtosis(),
        }

        med_price = float(g["price"].median())
        row["median_price_obs"] = med_price

        if zero_frac > MAX_ZERO_FRAC:
            row["verdict"] = "dropped: stale pricing"
        elif med_price < MIN_MEDIAN_PRICE:
            # Below ~RM0.50 the median price is quantised to a few ticks and
            # gets dragged around for days at a time by a handful of trades.
            # The one-day reversal filter cannot repair a multi-day excursion.
            row["verdict"] = "dropped: penny item"
        elif len(raw_ret) < MIN_OBS:
            row["verdict"] = "dropped: too short"
        else:
            row["verdict"] = "kept"
            out = g[["item", "date"]].copy()
            out["price"] = clean_price
            out["volume"] = g["volume"]
            out["ret"] = ret
            kept.append(out.dropna(subset=["ret"]))

        report.append(row)

    rep = pd.DataFrame(report).join(meta, on="item").sort_values(["tier", "zero_frac"])
    returns = pd.concat(kept, ignore_index=True).join(meta, on="item")

    RETURNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    returns.to_csv(RETURNS_CSV, index=False)
    rep.to_csv(REPORT_CSV, index=False)

    n_kept = (rep["verdict"] == "kept").sum()
    print(f"{len(rep)} items -> {n_kept} kept")
    print(rep["verdict"].value_counts().to_string())
    print(f"\nSpikes repaired: {int(rep['spikes_repaired'].sum())} "
          f"across {int((rep['spikes_repaired'] > 0).sum())} items")

    print("\nAnnualised vol, before vs after repair (kept items, by tier):")
    k = rep[rep["verdict"] == "kept"]
    print(k.groupby("tier", observed=True)[["vol_before", "vol_after"]].mean().round(3).to_string())

    print("\nExcess kurtosis, before vs after:")
    print(k.groupby("tier", observed=True)[["kurt_before", "kurt_after"]].mean().round(1).to_string())

    print(f"\nCommon window: {returns['date'].min().date()} to {returns['date'].max().date()}")
    print(f"{len(returns):,} returns across {returns['item'].nunique()} items")
    print(f"\n{RETURNS_CSV}\n{REPORT_CSV}")


if __name__ == "__main__":
    main()