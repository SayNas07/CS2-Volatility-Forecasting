"""
Stage 14: Model Confidence Set.

    python mcs_test.py

Reads  data/forecasts.csv
Writes data/mcs_results.csv

Diebold-Mariano is pairwise. With five models that is ten tests per item, and
running ten tests at 5% without correction means roughly a 40% chance of at
least one spurious rejection per item. The Model Confidence Set of Hansen,
Lunde and Nason (2011) solves this: it starts from the full set of models and
sequentially eliminates those that are significantly worse, returning the
subset that cannot be statistically separated from the best at a given
confidence level.

The claim "no estimated model beats the benchmark" is much stronger if the
estimated models are EXCLUDED from the confidence set, rather than merely
losing pairwise comparisons.

Note on interpretation: a model surviving in the MCS does not mean it is good.
It means the data cannot distinguish it from the best. With short samples the
set is often large, which is itself informative.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from arch.bootstrap import MCS

FORECASTS = Path("data/forecasts.csv")
OUT = Path("data/mcs_results.csv")

MODELS = ["EGARCH-t", "GARCH-t", "GJR-t", "EWMA", "RollSD"]
ESTIMATED = ["EGARCH-t", "GARCH-t", "GJR-t"]
ALPHA = 0.10        # 90% confidence set
BOOT_REPS = 1000
BLOCK = 10          # stationary bootstrap block length, for autocorrelated losses
MIN_OBS = 250


def qlike(realised, var_hat):
    return np.log(var_hat) + realised / var_hat


def main():
    f = pd.read_csv(FORECASTS, parse_dates=["date"])
    f["qlike"] = qlike(f["realised"], f["var_hat"])

    rows = []
    items = sorted(f["item"].unique())
    for n, item in enumerate(items, 1):
        g = f[f["item"] == item]
        wide = g.pivot_table(index="date", columns="model", values="qlike").dropna()
        present = [m for m in MODELS if m in wide.columns]
        if len(wide) < MIN_OBS or len(present) < 2:
            continue
        print(f"[{n}/{len(items)}] {item[:42]}", end="  ")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mcs = MCS(wide[present], size=ALPHA, reps=BOOT_REPS,
                          block_size=BLOCK, method="R", seed=42)
                mcs.compute()
            included = list(mcs.included)
            pvals = mcs.pvalues["Pvalue"].to_dict()
        except Exception as e:
            print(f"failed: {type(e).__name__}")
            continue

        print(f"-> {', '.join(included)}")
        for m in present:
            rows.append({"item": item, "model": m, "n": len(wide),
                         "in_mcs": m in included,
                         "mcs_pvalue": pvals.get(m, np.nan),
                         "mean_qlike": wide[m].mean(),
                         "tier": g["tier"].iloc[0]})

    d = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)

    n_items = d["item"].nunique()
    pd.set_option("display.width", 200)

    print(f"\n=== inclusion in the {int((1-ALPHA)*100)}% MCS, {n_items} items ===")
    summ = (d.groupby("model")["in_mcs"]
              .agg(items_in="sum", rate="mean")
              .assign(rate=lambda x: (x["rate"] * 100).round(1))
              .sort_values("items_in", ascending=False))
    summ["items_in"] = summ["items_in"].astype(int)
    print(summ.to_string())

    print("\n=== size of the confidence set per item ===")
    sizes = d.groupby("item")["in_mcs"].sum()
    print(sizes.value_counts().sort_index().rename("items").to_string())

    print("\n=== the key question ===")
    per_item = d.pivot_table(index="item", columns="model", values="in_mcs")
    est = [m for m in ESTIMATED if m in per_item.columns]
    all_est_out = (~per_item[est].any(axis=1)).sum()
    ewma_in = int(per_item["EWMA"].sum()) if "EWMA" in per_item else 0
    print(f"  items where EVERY estimated model is excluded: {all_est_out}/{n_items}")
    print(f"  items where EWMA survives:                     {ewma_in}/{n_items}")
    singleton = sizes[sizes == 1].index
    if len(singleton):
        win = d[(d["item"].isin(singleton)) & d["in_mcs"]]["model"].value_counts()
        print(f"\n  items with a singleton MCS ({len(singleton)}), sole survivor:")
        print("   " + win.to_string().replace("\n", "\n   "))

    print(f"\n{OUT}")


if __name__ == "__main__":
    main()