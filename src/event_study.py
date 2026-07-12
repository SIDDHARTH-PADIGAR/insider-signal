"""
event_study.py — the actual quantitative core of this project.

1. Detects "cluster buys": >=2 distinct insiders at the same company making
   open-market purchases (transaction_code == 'P') within a short window.
2. For each cluster event, computes the stock's cumulative abnormal return
   (CAR) over several horizons after the event — return relative to the
   S&P 500 (SPY) over the same period.
3. Runs a one-sample t-test: is the average CAR across all cluster events
   significantly different from zero?

Run: python src/event_study.py
Requires: data/insider_transactions.csv (from build_dataset.py) already exists.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
import joblib
import os

CLUSTER_WINDOW_DAYS = 10
MIN_DISTINCT_INSIDERS = 2
HORIZONS = [5, 10, 20, 30]


def detect_cluster_buys(df: pd.DataFrame) -> pd.DataFrame:
    buys = df[df["transaction_code"] == "P"].copy()
    if buys.empty:
        return pd.DataFrame()

    clusters = []
    for ticker, grp in buys.groupby("ticker"):
        grp = grp.sort_values("transaction_date")
        dates = grp["transaction_date"].tolist()

        i = 0
        while i < len(dates):
            window_start = dates[i]
            window_end = window_start + pd.Timedelta(days=CLUSTER_WINDOW_DAYS)
            in_window = grp[(grp["transaction_date"] >= window_start) &
                             (grp["transaction_date"] <= window_end)]
            distinct_owners = in_window["owner_name"].nunique()

            if distinct_owners >= MIN_DISTINCT_INSIDERS:
                clusters.append({
                    "ticker": ticker,
                    "cluster_start": window_start,
                    "cluster_end": in_window["transaction_date"].max(),
                    "n_insiders": distinct_owners,
                    "total_shares": in_window["shares"].sum(),
                    "total_value": (in_window["shares"] * in_window["price_per_share"]).sum(),
                })
                i = grp.index.get_loc(in_window.index[-1]) + 1
            else:
                i += 1

    return pd.DataFrame(clusters)


def compute_car(ticker: str, event_date: pd.Timestamp, benchmark_prices: pd.Series) -> dict:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(
            start=event_date - pd.Timedelta(days=5),
            end=pd.Timestamp.now(),
        )
    except Exception:
        return {}

    if hist.empty:
        return {}

    hist.index = hist.index.tz_localize(None)
    future_dates = hist.index[hist.index > event_date]
    if len(future_dates) == 0:
        return {}   # only bail if we have literally zero future data

    entry_price = hist.loc[hist.index[hist.index <= event_date][-1], "Close"] \
        if any(hist.index <= event_date) else hist["Close"].iloc[0]

    result = {}
    for h in HORIZONS:
        if h > len(future_dates):
            continue   # skip just this horizon, keep the others
        exit_date = future_dates[h - 1]
        exit_price = hist.loc[exit_date, "Close"]
        stock_return = (exit_price / entry_price) - 1

        bench_window = benchmark_prices[(benchmark_prices.index > event_date) &
                                         (benchmark_prices.index <= exit_date)]
        if bench_window.empty:
            continue
        bench_entry = benchmark_prices[benchmark_prices.index <= event_date].iloc[-1]
        bench_exit = bench_window.iloc[-1]
        bench_return = (bench_exit / bench_entry) - 1
        result[f"car_{h}d"] = round(float(stock_return - bench_return), 4)

    return result


def main():
    df = pd.read_csv("data/insider_transactions.csv")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["transaction_date"])
    print(f"Loaded {len(df)} transactions")

    clusters = detect_cluster_buys(df)
    if clusters.empty:
        print("No cluster buys detected. Transaction code breakdown:")
        print(df["transaction_code"].value_counts())
        return

    print(f"Detected {len(clusters)} cluster buy events")
    print(clusters[["ticker", "cluster_start", "cluster_end", "n_insiders"]].to_string())

    print("Fetching S&P 500 (SPY) benchmark prices...")
    spy = yf.Ticker("SPY")
    spy_hist = spy.history(
        start=clusters["cluster_start"].min() - pd.Timedelta(days=10),
        end=pd.Timestamp.now(),
    )
    if spy_hist.empty:
        print("ERROR: could not fetch SPY data — check yfinance version "
              "(try: pip install --upgrade yfinance) or your connection.")
        return
    spy_hist.index = spy_hist.index.tz_localize(None)
    benchmark_prices = spy_hist["Close"]

    car_rows = []
    for _, row in clusters.iterrows():
        car = compute_car(row["ticker"], row["cluster_end"], benchmark_prices)
        if car:
            car_rows.append({**row.to_dict(), **car})

    results = pd.DataFrame(car_rows)
    if results.empty:
        print("No events had enough subsequent trading data to compute CAR.")
        return

    results.to_csv("data/cluster_buy_events.csv", index=False)
    print(f"\nSaved {len(results)} scored events to data/cluster_buy_events.csv")

    print("\n--- Significance test: is mean CAR different from zero? ---")
    summary = {}
    for h in HORIZONS:
        col = f"car_{h}d"
        if col not in results.columns:
            continue
        values = results[col].dropna()
        if len(values) < 3:
            print(f"  {h}d: only {len(values)} events with data — too few for a meaningful test")
            continue
        t_stat, p_value = stats.ttest_1samp(values, 0)
        summary[col] = {
            "mean_car": round(float(values.mean()), 4),
            "n_events": int(len(values)),
            "t_stat": round(float(t_stat), 3),
            "p_value": round(float(p_value), 4),
            "significant_at_5pct": bool(p_value < 0.05),
        }
        print(f"  {h}d: mean CAR={summary[col]['mean_car']:+.2%}  n={summary[col]['n_events']}  "
              f"p={summary[col]['p_value']:.3f}  {'SIGNIFICANT' if summary[col]['significant_at_5pct'] else 'not significant'}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(summary, "models/event_study_summary.joblib")
    print("\nSaved summary to models/event_study_summary.joblib")


if __name__ == "__main__":
    main()