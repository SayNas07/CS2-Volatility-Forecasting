"""
Stage 5: do the return series actually have ARCH effects?

    python stylised_facts.py

Reads  data/returns.csv
Writes data/stylised_facts.csv   per-item test statistics

This runs BEFORE any GARCH fitting, and it is the step people skip. GARCH
assumes volatility clusters. If squared returns show no autocorrelation, a
GARCH model will still converge and still produce parameters - they just
won't mean anything. Check first.

Four things get tested per item:
  * Ljung-Box on returns          - is the mean predictable? (hopefully not)
  * Ljung-Box on squared returns  - does volatility cluster? (needs to be yes)
  * ARCH-LM                       - formal test for conditional heteroskedasticity
  * Jarque-Bera                   - are returns non-normal? (they will be)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

RETURNS_CSV = Path("data/returns.csv")
OUT_CSV = Path("data/stylised_facts.csv")

LAGS = 10
START = "2021-01-01"   # common window; keeps the Oct 2025 event centred


def main():
    r = pd.read_csv(RETURNS_CSV, parse_dates=["date"])
    r = r[r["date"] >= START]

    rows = []
    for name, g in r.groupby("item"):
        x = g.sort_values("date")["ret"].dropna().to_numpy()
        if len(x) < 250:
            continue

        lb_r = acorr_ljungbox(x, lags=[LAGS], return_df=True)
        lb_r2 = acorr_ljungbox(x ** 2, lags=[LAGS], return_df=True)
        arch_lm = het_arch(x, nlags=LAGS)
        jb = stats.jarque_bera(x)

        rows.append({
            "item": name,
            "tier": g["tier"].iloc[0],
            "category": g["category"].iloc[0],
            "n": len(x),
            "ann_vol": x.std() * np.sqrt(365),
            "skew": stats.skew(x),
            "ex_kurt": stats.kurtosis(x),
            "lb_ret_p": lb_r["lb_pvalue"].iloc[0],
            "lb_sq_p": lb_r2["lb_pvalue"].iloc[0],
            "arch_lm_stat": arch_lm[0],
            "arch_lm_p": arch_lm[1],
            "jb_p": jb.pvalue,
            # persistence proxy: first-order autocorrelation of |returns|
            "abs_acf1": pd.Series(np.abs(x)).autocorr(1),
        })

    d = pd.DataFrame(rows).sort_values(["tier", "ann_vol"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 200)
    print(f"Window: {START} onwards, {len(d)} items\n")
    print(d[["item", "tier", "n", "ann_vol", "ex_kurt", "lb_sq_p",
             "arch_lm_p", "abs_acf1"]].to_string(index=False, float_format="%.4g"))

    print("\n--- how many items pass each test at 5% ---")
    print(f"  volatility clusters (LB on squared):  {(d['lb_sq_p'] < .05).sum()}/{len(d)}")
    print(f"  ARCH effects present (ARCH-LM):       {(d['arch_lm_p'] < .05).sum()}/{len(d)}")
    print(f"  returns non-normal (Jarque-Bera):     {(d['jb_p'] < .05).sum()}/{len(d)}")
    print(f"  mean UNpredictable (LB on returns):   {(d['lb_ret_p'] >= .05).sum()}/{len(d)}")

    print("\n--- by tier ---")
    print(d.groupby("tier", observed=True).agg(
        n_items=("item", "size"),
        ann_vol=("ann_vol", "mean"),
        ex_kurt=("ex_kurt", "mean"),
        abs_acf1=("abs_acf1", "mean"),
        arch_pass=("arch_lm_p", lambda s: (s < .05).sum()),
    ).round(3).to_string())

    print(f"\n{OUT_CSV}")


if __name__ == "__main__":
    main()