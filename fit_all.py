"""
Stage 7: fit every item, every GARCH-family specification.

    python fit_all.py

Reads  data/returns.csv, data/stylised_facts.csv
Writes data/fit_results.csv     one row per (item, model): params + diagnostics

Design choices baked in, per the locked decisions:
  * AR(1) mean for every item, fixed. Per-item mean selection would make the
    cross-sectional beta comparison incoherent - different mean models mean
    the variance equations aren't comparable. AR(2) occasionally wins on BIC;
    that gets reported, not acted on.
  * Student-t errors throughout, Gaussian GARCH fitted alongside as a check.
  * The two event-broken items are excluded here and handled separately.

Convergence is recorded, never silently dropped. An item that fails to
converge is a result - it goes in the table with its reason.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox

RETURNS_CSV = Path("data/returns.csv")
OUT_CSV = Path("data/fit_results.csv")

START = "2021-01-01"
SCALE = 100
LAGS = 10

# Excluded from the main panel per decision 4 - the 23 Oct 2025 Valve shock
# dominates their squared-return series (ARCH-LM p ~ 1.0). Section 6 material.
EXCLUDED = [
    "AWP | Chromatic Aberration (Field-Tested)",
    "AK-47 | Head Shot (Field-Tested)",
]

SPECS = {
    "GARCH-t":  dict(vol="GARCH",  p=1, q=1, dist="t"),
    "GARCH-n":  dict(vol="GARCH",  p=1, q=1, dist="normal"),
    "EGARCH-t": dict(vol="EGARCH", p=1, q=1, dist="t"),
    "GJR-t":    dict(vol="GARCH",  p=1, o=1, q=1, dist="t"),
}


def persistence_of(res, model):
    """alpha+beta for GARCH/GJR; beta for EGARCH (log-variance AR coefficient).

    GJR adds gamma/2 because the asymmetry term is active half the time under
    a symmetric error distribution.
    """
    p = res.params
    if model.startswith("EGARCH"):
        return p.get("beta[1]", np.nan)
    tot = p.get("alpha[1]", np.nan) + p.get("beta[1]", np.nan)
    if "gamma[1]" in p:
        tot += p["gamma[1]"] / 2
    return tot


def fit_one(y, name, model, kwargs):
    row = {"item": name, "model": model, "n": len(y)}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(y, mean="AR", lags=1, **kwargs).fit(disp="off", show_warning=False)
    except Exception as e:
        row["converged"] = False
        row["note"] = f"{type(e).__name__}: {e}"[:120]
        return row

    z = res.std_resid.dropna()
    pers = persistence_of(res, model)

    row.update({
        "converged": bool(res.convergence_flag == 0),
        "note": "" if res.convergence_flag == 0 else f"flag {res.convergence_flag}",
        "loglik": res.loglikelihood,
        "aic": res.aic,
        "bic": res.bic,
        "omega": res.params.get("omega", np.nan),
        "alpha": res.params.get("alpha[1]", np.nan),
        "beta": res.params.get("beta[1]", np.nan),
        "gamma": res.params.get("gamma[1]", np.nan),
        "nu": res.params.get("nu", np.nan),
        "persistence": pers,
        "half_life": np.log(0.5) / np.log(pers) if 0 < pers < 1 else np.nan,
        "alpha_p": res.pvalues.get("alpha[1]", np.nan),
        "beta_p": res.pvalues.get("beta[1]", np.nan),
        "gamma_p": res.pvalues.get("gamma[1]", np.nan),
        "lb_z_p": acorr_ljungbox(z, lags=[LAGS], return_df=True)["lb_pvalue"].iloc[0],
        "lb_z2_p": acorr_ljungbox(z ** 2, lags=[LAGS], return_df=True)["lb_pvalue"].iloc[0],
        "z_kurt": z.kurtosis(),
    })
    return row


def main():
    r = pd.read_csv(RETURNS_CSV, parse_dates=["date"])
    r = r[(r["date"] >= START) & (~r["item"].isin(EXCLUDED))]
    meta = r.groupby("item")[["tier", "category"]].first()

    rows = []
    items = sorted(r["item"].unique())
    for i, name in enumerate(items, 1):
        y = r[r["item"] == name].sort_values("date")["ret"] * SCALE
        if len(y) < 250:
            continue
        print(f"[{i}/{len(items)}] {name}")
        for model, kwargs in SPECS.items():
            rows.append(fit_one(y, name, model, kwargs))

    d = pd.DataFrame(rows).join(meta, on="item")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 220)
    ok = d[d["converged"]]
    print(f"\n{len(items)} items x {len(SPECS)} specs = {len(d)} fits, "
          f"{len(ok)} converged, {len(d) - len(ok)} failed")
    if len(d) > len(ok):
        print("\nFailures:")
        print(d[~d["converged"]][["item", "model", "note"]].to_string(index=False))

    print("\n--- residual diagnostics: how many fits leave ARCH effects behind? ---")
    for m in SPECS:
        s = ok[ok["model"] == m]
        bad = (s["lb_z2_p"] < .05).sum()
        print(f"  {m:9s}  {bad}/{len(s)} fail LB on squared std residuals")

    print("\n--- persistence by tier (GARCH-t) ---")
    g = ok[ok["model"] == "GARCH-t"]
    print(g.groupby("tier", observed=True).agg(
        n=("item", "size"),
        persistence=("persistence", "mean"),
        half_life=("half_life", "median"),
        nu=("nu", "mean"),
    ).round(3).to_string())

    print("\n--- which spec wins on BIC, per item ---")
    best = ok.loc[ok.groupby("item")["bic"].idxmin()]
    print(best["model"].value_counts().to_string())

    print(f"\n{OUT_CSV}")


if __name__ == "__main__":
    main()