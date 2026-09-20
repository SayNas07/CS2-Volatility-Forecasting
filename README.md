# Volatility Forecasting in an Unregulated Retail Asset Market

Do standard volatility models offer any forecast advantage in a market with no
market makers, no arbitrageurs, no short selling and no circuit breakers?

This project builds a daily price panel for Counter-Strike 2 weapon skins from
the Steam Community Market and runs a full volatility-forecasting study on it:
GARCH, EGARCH and GJR-GARCH against parameter-free benchmarks, evaluated out of
sample with QLIKE, Diebold–Mariano tests and the Model Confidence Set.

**Headline result: no estimated GARCH-family specification beats a
parameter-free benchmark on any item.** An exponentially weighted moving
average lies inside the 90% Model Confidence Set on all 19 testable items and
is the sole survivor on five. EGARCH wins BIC in-sample on half the panel and
ranks last out of sample.

## Findings

| | |
|---|---|
| **ARCH effects are real** | 22 of 24 items show volatility clustering and significant ARCH-LM statistics |
| **The conditional mean is predictable** | Ljung–Box rejects on raw returns for all 24 items — equity returns essentially never do this |
| **There is a weekly cycle** | Mean returns run from −0.64% on Wednesdays to +0.40% on Saturdays; AR(7) beats AR(1) on BIC for 18 of 22 items |
| **Two microstructure regimes coexist** | Knives show bid–ask bounce (ρ₁ = −0.41); cheap cases show stale pricing (ρ₁ = +0.19) |
| **Persistence does not scale with value** | Contradicts the gradient visible in raw absolute-return autocorrelation — a caution about proxy choice |
| **Estimated models lose out of sample** | 0 of 19 items where any GARCH variant beats a 30-day rolling standard deviation |
| **The result is robust** | Holds across λ ∈ [0.85, 0.99] and four cleaning specifications, including no outlier repair at all |

## Pipeline

Run in order. Each stage writes to `data/` and the next stage reads from it.

| Stage | Script | What it does | Runtime |
|---|---|---|---|
| 1 | `steam_pull.py --discover` | Scan the market, fill four price tiers with candidates | ~4 min |
| 2 | `steam_pull.py --pull` | Fetch price history, cached and resumable | ~11 min |
| 3 | `steam_pull.py --screen` | Screen on trading activity, stratify into a panel | instant |
| 4 | `clean_panel.py` | Drop stale and penny items, repair isolated bad prints | instant |
| 5 | `stylised_facts.py` | Ljung–Box, ARCH-LM, Jarque–Bera — **before** fitting anything | ~10 s |
| 6 | `fit_one.py [item]` | Fit one item end to end with full diagnostics | ~20 s |
| 7 | `fit_all.py` | All items, all specifications, convergence tracked | ~3 min |
| 8 | `ar_check.py` | Does the weekly cycle change the mean model, or the forecasts? | ~15 min |
| 9 | `forecast.py` | Rolling one-step-ahead forecasts, weekly refit | ~30 min |
| 10 | `evaluate.py` | QLIKE, MSE, Diebold–Mariano | ~30 s |
| 11 | `lambda_check.py` | Is EWMA's win an artefact of λ = 0.94? | ~10 s |
| 12 | `mcs_test.py` | Model Confidence Set — handles the multiplicity problem | ~3 min |
| 13 | `robustness.py` | Re-run the pipeline under four cleaning specifications | 1–2 h |
| 14 | `shock_analysis.py` | Forecast behaviour across the 23 Oct 2025 structural break | ~5 min |
| 15 | `make_figures.py` | All six paper figures as vector PDFs | ~20 s |

## Setup

```bash
pip install -r requirements.txt
```

The Steam price-history endpoint requires an authenticated session. Log in at
steamcommunity.com, open devtools → Application → Cookies, and copy the
`steamLoginSecure` value:

```bash
export STEAM_LOGIN_SECRET='...'        # macOS / Linux
$env:STEAM_LOGIN_SECRET = '...'        # Windows PowerShell
```

The cookie expires every few days. Prices return in the account's own currency
(MYR here); log returns are scale-free, so this affects only the tier labels.

## Notes on the data

Three problems, all found by inspection rather than assumed:

- **Stale pricing.** Two sticker items printed an identical median on ~75% of
  days. Dropped — GARCH fits a series of exact zeros happily and reports
  meaningless persistence.
- **Penny items.** Below ~RM 0.50 the median is quantised and gets dragged for
  days by a handful of trades. Dropped; cheap-tier annualised volatility fell
  from 5.06 to 0.75 as a result.
- **Isolated bad prints.** A Bayonet priced around RM 2,200 printed RM 11.74 for
  one day. 75 repaired across 7 items, with the filter restricted to moves that
  reverse the next day.

**What is deliberately *not* cleaned:** on 23 October 2025 Valve extended the
Trade-Up Contract, and two Covert skins rose ~380% while knives fell 29–43%.
Those moves persisted. Any filter aggressive enough to remove them is too
aggressive, and `clean_panel.py` documents this so the thresholds don't get
tuned until the finding disappears.

## Method choices, and why

- **QLIKE primary, MSE secondary.** No intraday data means the realised proxy is
  the squared daily return — unbiased but noisy. QLIKE's ranking is robust to
  proxy noise; MSE's is not (Patton 2011). EGARCH's mean MSE is double every
  other model's at comparable QLIKE, which shows the problem directly.
- **Rolling window, not expanding.** The market has regime shifts; an expanding
  window would carry stale pre-shock data indefinitely.
- **AR(7) mean, fixed across items.** The weekly cycle needs it. Fixed rather
  than per-item selected, because different mean models make variance
  parameters non-comparable across the cross-section.
- **Degenerate refits rejected, not hidden.** EGARCH occasionally converges to
  solutions forecasting variances above 1e30. These are rejected against a
  sanity band and the previous parameters held over — what you would do in
  production. 51 refits were rejected; the count is reported.

## Repo layout

```
├── steam_pull.py        stages 1–3   data collection
├── clean_panel.py       stage 4      cleaning and return construction
├── stylised_facts.py    stage 5      pre-model diagnostics
├── fit_one.py           stage 6      single-item fit with diagnostics
├── fit_all.py           stage 7      full panel estimation
├── ar_check.py          stage 8      mean specification check
├── forecast.py          stage 9      rolling out-of-sample forecasts
├── evaluate.py          stage 10     loss functions and DM tests
├── lambda_check.py      stage 11     EWMA sensitivity
├── mcs_test.py          stage 12     Model Confidence Set
├── robustness.py        stage 13     cleaning sensitivity
├── shock_analysis.py    stage 14     structural break analysis
├── make_figures.py      stage 15     figures
├── paper/               LaTeX source and figures
└── data/                generated, gitignored
```

## Limitations

Twenty-two items in the main panel, as few as one per tier after exclusions —
cross-sectional claims are patterns, not tests. Squared daily returns are a
noisy volatility proxy. Steam Community Market only, so third-party
marketplace volume is invisible. Selection on liquidity over full history
favours items that stayed liquid. Full list in the paper.
