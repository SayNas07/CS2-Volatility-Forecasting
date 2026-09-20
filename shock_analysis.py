"""
Stage 13: Section 7 - do volatility forecasts survive a structural break?

    python shock_analysis.py

Reads  data/returns.csv
Writes data/shock_losses.csv    daily QLIKE per (item, model) around the event
       data/shock_summary.csv   per (item, model) deterioration and recovery

DESIGN

The test is deliberately one of parameter staleness, not of re-estimation.
Every model is estimated ONCE on pre-shock data only, then filtered forward
through the event with those parameters frozen. This is fair across model
classes because each model still updates its conditional variance recursively
as new returns arrive - GARCH through its recursion, EWMA through its decay.
What differs is only how much each model had to learn in the first place.

Three groups:
  treated   - Covert skins that became a crafting input (+378%, +394%)
  supply    - knives whose supply constraint was removed (-29% to -43%)
  control   - everything else in the panel

Three questions:
  1. How much does forecast loss deteriorate, by specification?
  2. How long until it recovers?
  3. Does asymmetry (EGARCH, GJR) help when the shock is opposite-signed
     across groups on the same day?
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from arch import arch_model

RETURNS = Path("data/returns.csv")
LOSSES_OUT = Path("data/shock_losses.csv")
SUMMARY_OUT = Path("data/shock_summary.csv")

SHOCK = pd.Timestamp("2025-10-23")
SCALE = 100
TRAIN_DAYS = 500     # estimation window, ending the day before the shock
PRE_EVAL = 250       # pre-shock days used to set the baseline loss level
POST = 60            # days after the shock to track
ROLL = 20            # smoothing window for the recovery criterion
TOL = 1.10           # recovered = rolling loss within 10% of baseline

TREATED = ["AWP | Chromatic Aberration (Field-Tested)",
           "AK-47 | Head Shot (Field-Tested)"]

SPECS = {
    "GARCH-t":  dict(vol="GARCH",  p=1, q=1, dist="t"),
    "EGARCH-t": dict(vol="EGARCH", p=1, q=1, dist="t"),
    "GJR-t":    dict(vol="GARCH",  p=1, o=1, q=1, dist="t"),
}
EWMA_LAMBDA = 0.94
SD_WINDOW = 30


def qlike(realised, var_hat):
    return np.log(var_hat) + realised / var_hat


def group_of(item):
    if item in TREATED:
        return "treated"
    return "supply" if item.startswith("\u2605") else "control"


def frozen_garch(y, train, kwargs):
    """Estimate on `train`, then filter the full series with parameters fixed.

    arch's .fix() runs the variance recursion without re-estimating, so the
    conditional variance at t uses information through t-1 only. That is
    exactly a one-step-ahead forecast with stale parameters.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = arch_model(train, mean="AR", lags=7, **kwargs).fit(
            disp="off", show_warning=False)
        held = arch_model(y, mean="AR", lags=7, **kwargs).fix(
            fitted.params, first_obs=y.index[0])
    var = pd.Series(held.conditional_volatility ** 2, index=y.index)
    return var.replace([np.inf, -np.inf], np.nan)


def frozen_ewma(y, seed_var):
    var, out = seed_var, {}
    for i in range(1, len(y)):
        var = EWMA_LAMBDA * var + (1 - EWMA_LAMBDA) * y.iloc[i - 1] ** 2
        out[y.index[i]] = var
    return pd.Series(out)


def frozen_rollsd(y):
    return y.rolling(SD_WINDOW).var().shift(1)


def main():
    r = pd.read_csv(RETURNS, parse_dates=["date"])
    r["date"] = pd.to_datetime(r["date"]).dt.tz_localize(None) \
        if getattr(r["date"].dt, "tz", None) is not None else r["date"]

    rows = []
    for item, g in r.groupby("item"):
        y = g.sort_values("date").set_index("date")["ret"] * SCALE
        pre = y[y.index < SHOCK]
        post = y[(y.index >= SHOCK) & (y.index <= SHOCK + pd.Timedelta(days=POST))]
        if len(pre) < TRAIN_DAYS + PRE_EVAL or len(post) < 20:
            continue

        train = pre.iloc[-TRAIN_DAYS:]
        window = y[(y.index >= pre.index[-(TRAIN_DAYS + PRE_EVAL)]) &
                   (y.index <= SHOCK + pd.Timedelta(days=POST))]

        forecasts = {}
        for name, kw in SPECS.items():
            try:
                forecasts[name] = frozen_garch(window, train, kw)
            except Exception as e:
                print(f"  {item[:30]} {name}: {type(e).__name__}")
        forecasts["EWMA"] = frozen_ewma(window, float(train.var()))
        forecasts["RollSD"] = frozen_rollsd(window)

        realised = window.pow(2)
        for name, var in forecasts.items():
            idx = var.dropna().index.intersection(realised.index)
            var = var.loc[idx]
            idx = var[var > 0].index
            L = qlike(realised.loc[idx], var.loc[idx])
            for d, v in L.items():
                rows.append({"item": item, "group": group_of(item), "model": name,
                             "date": d, "qlike": v,
                             "rel_day": (d - SHOCK).days})

    L = pd.DataFrame(rows)
    LOSSES_OUT.parent.mkdir(parents=True, exist_ok=True)
    L.to_csv(LOSSES_OUT, index=False)

    # ---- per item x model: baseline, event loss, recovery ----
    summ = []
    for (item, model), g in L.groupby(["item", "model"]):
        g = g.sort_values("date")
        base = g[(g["rel_day"] < 0)]["qlike"]
        if len(base) < 100:
            continue
        b = base.mean()
        ev = g[(g["rel_day"] >= 0) & (g["rel_day"] <= 20)]["qlike"]
        after = g[g["rel_day"] >= 0].set_index("rel_day")["qlike"]
        roll = after.rolling(ROLL).mean()
        rec = roll[roll <= b * TOL]
        summ.append({
            "item": item, "group": group_of(item), "model": model,
            "baseline": b,
            "event_20d": ev.mean(),
            "deterioration": ev.mean() - b,
            "peak_day": int(after.idxmax()) if len(after) else np.nan,
            "recovery_days": int(rec.index[0]) if len(rec) else np.nan,
        })
    S = pd.DataFrame(summ)
    S.to_csv(SUMMARY_OUT, index=False)

    pd.set_option("display.width", 220)
    print(f"Items analysed: {S['item'].nunique()}  "
          f"({S.groupby('group')['item'].nunique().to_dict()})\n")

    print("=== QLIKE deterioration in the 20 days after the shock ===")
    piv = S.pivot_table(index="group", columns="model", values="deterioration",
                        observed=True)
    print(piv.round(3).to_string())

    print("\n=== baseline vs event QLIKE, by group and model ===")
    print(S.pivot_table(index=["group", "model"], values=["baseline", "event_20d"],
                        observed=True).round(3).to_string())

    print("\n=== recovery: days until 20-day rolling QLIKE returns to baseline ===")
    print(S.pivot_table(index="group", columns="model", values="recovery_days",
                        observed=True).round(1).to_string())
    never = S[S["recovery_days"].isna()]
    if len(never):
        print(f"\nNever recovered within {POST} days: {len(never)} of {len(S)} "
              f"item-model pairs")
        print(never.groupby("model").size().to_string())

    print("\n=== does asymmetry help? (deterioration, treated vs supply) ===")
    a = S[S["group"].isin(["treated", "supply"])]
    print(a.pivot_table(index="model", columns="group", values="deterioration",
                        observed=True).round(3).to_string())
    print("\ntreated = positive shock, supply = negative shock.")
    print("Asymmetric models weight negative shocks more heavily, so if asymmetry")
    print("helps we expect them to do relatively better on 'supply' than on 'treated'.")

    print(f"\n{LOSSES_OUT}\n{SUMMARY_OUT}")


if __name__ == "__main__":
    main()