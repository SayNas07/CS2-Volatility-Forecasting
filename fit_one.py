"""
Stage 6: fit ONE item end to end, and check the fit is actually valid.

    python fit_one.py                                  # default item
    python fit_one.py "AK-47 | Ice Coaled (Field-Tested)"

Reads data/returns.csv

The point of this stage is not to get numbers - it is to understand what the
numbers mean before scaling to 22 items. Read every section of the output.

What gets checked, in order:
  1. The fitted model and its parameters
  2. Persistence and half-life - does the model imply sensible dynamics?
  3. Standardised residuals - did the model actually absorb the ARCH effects?
  4. Specification comparison - does AR(1) beat constant mean? does t beat normal?
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox

RETURNS_CSV = Path("data/returns.csv")
START = "2021-01-01"
DEFAULT_ITEM = "AK-47 | Elite Build (Field-Tested)"

# arch works better on percentage returns - log returns are ~0.01, and the
# optimiser struggles with parameters that small. Scale up, remember to scale
# variance forecasts back down by SCALE**2 later.
SCALE = 100


def load(item):
    r = pd.read_csv(RETURNS_CSV, parse_dates=["date"])
    g = r[(r["item"] == item) & (r["date"] >= START)].sort_values("date")
    if g.empty:
        raise SystemExit(f"No data for {item!r}. Check the name against returns.csv.")
    return g.set_index("date")["ret"] * SCALE


def diagnostics(res, y, label):
    """Did the model absorb what it was supposed to absorb?"""
    z = res.std_resid.dropna()

    lb_z = acorr_ljungbox(z, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    lb_z2 = acorr_ljungbox(z ** 2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]

    print(f"\n--- residual diagnostics: {label} ---")
    print(f"  LB(10) on standardised residuals    p = {lb_z:.4f}"
          f"   {'OK - mean absorbed' if lb_z > .05 else 'FAIL - mean structure left over'}")
    print(f"  LB(10) on squared std residuals     p = {lb_z2:.4f}"
          f"   {'OK - ARCH absorbed' if lb_z2 > .05 else 'FAIL - ARCH effects left over'}")
    print(f"  std resid kurtosis                  {z.kurtosis():.2f}"
          f"   (near 0 means the error distribution fits)")
    return lb_z, lb_z2


def main():
    item = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ITEM
    y = load(item)

    print(f"Item:   {item}")
    print(f"Sample: {y.index.min().date()} to {y.index.max().date()}  ({len(y)} obs)")
    print(f"Mean {y.mean():.4f}%  SD {y.std():.4f}%  annualised {y.std()*np.sqrt(365)/SCALE:.2f}")

    # ---- the model we committed to: AR(1) mean, GARCH(1,1), Student-t ----
    print("\n" + "=" * 70)
    print("AR(1)-GARCH(1,1), Student-t errors")
    print("=" * 70)
    m = arch_model(y, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="t")
    res = m.fit(disp="off")
    print(res.summary())

    omega = res.params["omega"]
    alpha = res.params["alpha[1]"]
    beta = res.params["beta[1]"]
    persistence = alpha + beta

    print(f"\n--- persistence ---")
    print(f"  alpha (shock impact)     {alpha:.4f}")
    print(f"  beta  (memory)           {beta:.4f}")
    print(f"  alpha + beta             {persistence:.4f}")
    if persistence >= 1:
        print("  >= 1: variance is NOT stationary. The unconditional variance does")
        print("  not exist and long-horizon forecasts diverge. Note this, consider IGARCH.")
    else:
        half_life = np.log(0.5) / np.log(persistence)
        uncond = np.sqrt(omega / (1 - persistence)) * np.sqrt(365) / SCALE
        print(f"  half-life of a shock     {half_life:.1f} days")
        print(f"  implied long-run vol     {uncond:.2f} annualised")
        print("  (compare that to the sample annualised vol above - they should be close)")

    print(f"\n  nu (t degrees of freedom) {res.params['nu']:.2f}")
    print("  low nu = fat tails. Below ~4 the kurtosis of the t is undefined.")

    diagnostics(res, y, "AR(1)-GARCH(1,1)-t")

    # ---- does the mean model matter? ----
    print("\n" + "=" * 70)
    print("Specification comparison")
    print("=" * 70)
    alts = {
        "AR(1) mean, t errors":     arch_model(y, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="t"),
        "Constant mean, t errors":  arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="t"),
        "AR(1) mean, normal errors": arch_model(y, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="normal"),
        "AR(2) mean, t errors":     arch_model(y, mean="AR", lags=2, vol="GARCH", p=1, q=1, dist="t"),
    }
    rows = []
    for label, mod in alts.items():
        r = mod.fit(disp="off")
        rows.append({"spec": label, "loglik": r.loglikelihood, "aic": r.aic, "bic": r.bic})
    comp = pd.DataFrame(rows).sort_values("bic")
    print(comp.to_string(index=False, float_format="%.2f"))
    print("\nLower BIC is better. Confirms decisions 2 and 3 for this item -")
    print("check it holds on two or three others before trusting it generally.")


if __name__ == "__main__":
    main()