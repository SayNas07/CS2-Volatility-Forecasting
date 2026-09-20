"""
Steam Community Market price-history puller for CS2 (appid 730).

Three stages. Run them in order.

    export STEAM_LOGIN_SECRET='<steamLoginSecure cookie value>'

    python steam_pull.py --discover   # scan the market, fill each price tier
    python steam_pull.py --pull       # fetch history for every candidate
    python steam_pull.py --screen     # keep what's actually modellable, stratify

Outputs:
    data/candidates.json     raw candidate pool
    data/raw/*.json          one raw response per item
    data/daily_prices.csv    every candidate, daily
    data/item_stats.csv      per-item liquidity diagnostics
    data/panel.csv           the screened, stratified sample you model on
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

APPID = 730
RAW = Path("data/raw")
CANDIDATES = Path("data/candidates.json")
DAILY_CSV = Path("data/daily_prices.csv")
STATS_CSV = Path("data/item_stats.csv")
PANEL_CSV = Path("data/panel.csv")

# Screening thresholds. Loosen if too little survives, but know what you're
# giving up: each one exists to protect the conditional-variance estimate.
MIN_DAYS = 500          # at least ~2 years of observations
MIN_MEDIAN_VOLUME = 5   # typical day must have real trades, not one print
MAX_MISSING_FRAC = 0.05 # no long dead stretches inside the sample window
PER_CELL = 3            # items sampled per (tier x category) cell

# Price tiers in MYR. Defined once, used at both discovery and screening, so
# the pool is built to fill the same buckets the panel is stratified over.
TIER_BINS = [0, 5, 50, 500, np.inf]
TIER_LABELS = ["sub5", "5_50", "50_500", "500plus"]

# Discovery settings. Scanning is cheap (one request per 100 items); it's the
# history pull that costs 4s per item. So scan wide, then keep selectively.
SCAN_DEPTH = 6000       # how deep the price-descending scan goes
PER_TIER = 40           # candidates kept per price tier
MIN_LISTINGS = 10       # free liquidity proxy - skip obvious dead items

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
}


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------

def session():
    token = os.environ.get("STEAM_LOGIN_SECRET")
    if not token:
        raise SystemExit("Set STEAM_LOGIN_SECRET to your steamLoginSecure cookie value.")
    s = requests.Session()
    s.headers.update(HEADERS)
    s.cookies.set("steamLoginSecure", token, domain="steamcommunity.com")
    return s


def get(s, url, params=None, tries=5):
    """GET with backoff. Steam throttles hard past ~20 requests per minute."""
    for attempt in range(tries):
        r = s.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r
        if r.status_code in (429, 502, 503):
            wait = (2 ** attempt) * 10 + random.uniform(0, 5)
            print(f"  {r.status_code} - sleeping {wait:.0f}s")
            time.sleep(wait)
            continue
        if r.status_code == 400:
            raise SystemExit("400 - cookie is expired or wrong. Grab a fresh one.")
        r.raise_for_status()
    raise RuntimeError(f"Gave up on {url}")


def parse_price(text):
    """'£12.34' -> 12.34. Currency symbol depends on your Steam account."""
    if not text:
        return np.nan
    cleaned = re.sub(r"[^\d.,]", "", text).replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def categorise(name):
    """Crude bucketing by item type. Good enough for stratification."""
    n = name.lower()
    if "case" in n and "hardened" not in n:
        return "case"
    if "sticker" in n or "patch" in n or "graffiti" in n:
        return "sticker"
    if name.startswith("\u2605"):  # knives and gloves carry the star prefix
        return "knife_glove"
    if "key" in n or "capsule" in n or "package" in n or "pass" in n:
        return "container"
    return "skin"


# --------------------------------------------------------------------------
# stage 1 - discovery
# --------------------------------------------------------------------------

def sweep(s, n, sort_column, sort_dir, source):
    """One pass through the market search, in a given sort order."""
    items, start = [], 0
    while len(items) < n:
        r = get(s, "https://steamcommunity.com/market/search/render/", params={
            "appid": APPID, "norender": 1, "count": 100, "start": start,
            "sort_column": sort_column, "sort_dir": sort_dir,
        })
        results = r.json().get("results", [])
        if not results:
            break
        for it in results:
            items.append({
                "name": it["hash_name"],
                "listings": it.get("sell_listings") or 0,
                "price": parse_price(it.get("sell_price_text")),
                "source": source,
            })
        start += 100
        time.sleep(3)
    return items[:n]


def discover(s, scan_depth=SCAN_DEPTH, per_tier=PER_TIER):
    """Build a candidate pool that spans the price distribution.

    Taking the top of any single sort fails: listing count returns nothing but
    cases, price returns nothing but knives that trade twice a month. Neither
    reaches the mid-market, which is where liquid expensive items live.

    So scan deep instead of taking the top, then within each price tier keep
    the most-listed items. Listing count is a free liquidity proxy from the
    search response - it isn't the real screen (that's --screen, on actual
    trading data), it just stops us spending 4s per item on obvious corpses.
    """
    print(f"Scanning {scan_depth} items by price, high to low...")
    pool = sweep(s, scan_depth, "price", "desc", "price_scan")

    # Sorting by price never reaches the cheap end in reasonable depth - there
    # are tens of thousands of sub-RM5 items. Grab that tier separately.
    print("Scanning 1000 items by listing count (cheap end)...")
    pool += sweep(s, 1000, "quantity", "desc", "volume_scan")

    df = pd.DataFrame(pool).drop_duplicates(subset="name")
    df = df[df["price"].notna() & (df["listings"] >= MIN_LISTINGS)]
    df["tier"] = pd.cut(df["price"], bins=TIER_BINS, labels=TIER_LABELS)

    print(f"\nScanned {len(pool)} listings, {len(df)} usable. By tier:")
    print(df.groupby("tier", observed=True).size().to_string())

    kept = (
        df.sort_values("listings", ascending=False)
        .groupby("tier", observed=True)
        .head(per_tier)
    )

    thin = [t for t in TIER_LABELS if (kept["tier"] == t).sum() < per_tier // 2]
    if thin:
        print(f"\nThin tiers (raise SCAN_DEPTH to reach deeper): {', '.join(thin)}")

    records = kept.drop(columns="tier").to_dict("records")
    CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATES.write_text(json.dumps(records, indent=2))

    print(f"\nKeeping {len(records)} candidates -> {CANDIDATES}")
    print(kept.groupby("tier", observed=True)["price"].agg(["count", "min", "max"]).to_string())
    print(f"\nPull time: roughly {len(records) * 4 / 60:.0f} min "
          f"(cached items are skipped, so reruns are faster)")


# --------------------------------------------------------------------------
# stage 2 - pull
# --------------------------------------------------------------------------

def pull_one(s, name):
    r = get(s, "https://steamcommunity.com/market/pricehistory/", params={
        "appid": APPID, "market_hash_name": name,
    })
    payload = r.json()
    if not payload.get("success"):
        return None
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"{quote(name, safe='')}.json").write_text(json.dumps(payload))
    return payload


def to_frame(name, payload):
    df = pd.DataFrame(payload["prices"], columns=["ts", "price", "volume"])
    # Steam's format: "Mar 03 2026 14: +0"
    df["ts"] = pd.to_datetime(df["ts"], format="%b %d %Y %H: +0", utc=True)
    df["volume"] = df["volume"].astype(int)
    df["item"] = name
    return df


def pull_all():
    s = session()
    items = json.loads(CANDIDATES.read_text())

    # Resume support - a 300-item pull will get interrupted at some point.
    frames = []
    for i, it in enumerate(items, 1):
        name = it["name"]
        cached = RAW / f"{quote(name, safe='')}.json"
        if cached.exists():
            frames.append(to_frame(name, json.loads(cached.read_text())))
            continue
        print(f"[{i}/{len(items)}] {name}")
        try:
            payload = pull_one(s, name)
        except Exception as e:
            print(f"  failed: {e}")
            continue
        if payload:
            frames.append(to_frame(name, payload))
        time.sleep(4)

    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit("No history pulled. Check the cookie and rerun.")
    allrows = pd.concat(frames, ignore_index=True)
    # Recent ~30 days arrive hourly, older data daily. Collapse to daily so the
    # return series has one observation per day throughout - otherwise the tail
    # of the sample is hourly and the volatility estimate is garbage.
    daily = (
        allrows.set_index("ts")
        .groupby("item")[["price", "volume"]]
        .resample("1D")
        .agg(price=("price", "last"), volume=("volume", "sum"))
        .reset_index()
        .rename(columns={"ts": "date"})
        .dropna(subset=["price"])
    )
    DAILY_CSV.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(DAILY_CSV, index=False)
    print(f"\n{len(daily)} daily rows across {daily['item'].nunique()} items -> {DAILY_CSV}")


# --------------------------------------------------------------------------
# stage 3 - screen and stratify
# --------------------------------------------------------------------------

def item_stats(daily):
    out = []
    for name, g in daily.groupby("item"):
        g = g.sort_values("date")
        span = (g["date"].max() - g["date"].min()).days + 1
        out.append({
            "item": name,
            "n_days": len(g),
            "span_days": span,
            "missing_frac": 1 - len(g) / span if span else 1.0,
            "median_volume": g["volume"].median(),
            "median_price": g["price"].median(),
            "start": g["date"].min().date(),
            "end": g["date"].max().date(),
        })
    return pd.DataFrame(out)


def screen():
    daily = pd.read_csv(DAILY_CSV, parse_dates=["date"])
    stats = item_stats(daily)

    keep = stats[
        (stats["n_days"] >= MIN_DAYS)
        & (stats["median_volume"] >= MIN_MEDIAN_VOLUME)
        & (stats["missing_frac"] <= MAX_MISSING_FRAC)
    ].copy()

    print(f"{len(stats)} candidates -> {len(keep)} pass the liquidity screen")
    for label, mask in [
        ("too short", stats["n_days"] < MIN_DAYS),
        ("too thin", stats["median_volume"] < MIN_MEDIAN_VOLUME),
        ("too gappy", stats["missing_frac"] > MAX_MISSING_FRAC),
    ]:
        print(f"  {label}: {int(mask.sum())}")

    keep["tier"] = pd.cut(keep["median_price"], bins=TIER_BINS, labels=TIER_LABELS)
    keep["category"] = keep["item"].map(categorise)
    stats["tier"] = pd.cut(stats["median_price"], bins=TIER_BINS, labels=TIER_LABELS)

    # Survival by price tier. A tier wiped out here is a finding worth writing
    # down, not a bug: it means items at that price trade too thinly to model
    # at daily frequency.
    print("\n  survivors by tier:")
    for t in TIER_LABELS:
        tot = int((stats["tier"] == t).sum())
        surv = int((keep["tier"] == t).sum())
        if tot:
            print(f"    {t}: {surv}/{tot}")

    # Stratified sample: take the most liquid items within each cell, so the
    # panel spans price levels and item types rather than being 40 cases.
    panel = (
        keep.sort_values("median_volume", ascending=False)
        .groupby(["tier", "category"], observed=True)
        .head(PER_CELL)
        .sort_values(["tier", "category", "median_volume"], ascending=[True, True, False])
    )

    stats.to_csv(STATS_CSV, index=False)
    panel.to_csv(PANEL_CSV, index=False)

    print(f"\nPanel: {len(panel)} items")
    print(panel.groupby(["tier", "category"], observed=True).size().to_string())
    print(f"\n{STATS_CSV} (all diagnostics)\n{PANEL_CSV} (model on this)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--discover", action="store_true")
    p.add_argument("--scan-depth", type=int, default=SCAN_DEPTH,
                   help=f"how deep the price scan goes (default {SCAN_DEPTH})")
    p.add_argument("--pull", action="store_true")
    p.add_argument("--screen", action="store_true")
    a = p.parse_args()
    if a.discover:
        discover(session(), scan_depth=a.scan_depth)
    if a.pull:
        pull_all()
    if a.screen:
        screen()