"""
build_dataset.py — pulls Form 4 insider transactions for a fixed watchlist,
saves raw data to data/insider_transactions.csv.

Run: python src/build_dataset.py

This will take a few minutes — it's making real, rate-limited calls to
SEC's servers (one submissions call + one XML fetch per Form 4 filing,
per ticker). That's expected, not a bug. Don't run it repeatedly in a
tight loop.
"""

import pandas as pd
from sec_client import get_cik_map, get_recent_form4_filings, parse_form4

def get_sp500_tickers() -> list[str]:
    """
    Pulls current S&P 500 constituents from a standard public dataset —
    an unbiased universe, not hand-picked by expected insider activity
    (picking companies because they're known for insider trading would
    bias the event study toward finding a signal).
    """
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    df = pd.read_csv(url)
    return df["Symbol"].str.replace(".", "-", regex=False).tolist()  # BRK.B -> BRK-B style fix


WATCHLIST = get_sp500_tickers()

def main():
    print("Fetching CIK map (ticker -> SEC company ID)...")
    cik_map = get_cik_map()

    all_transactions = []
    for ticker in WATCHLIST:
        cik = cik_map.get(ticker)
        if not cik:
            print(f"  {ticker}: no CIK found, skipping")
            continue

        print(f"  {ticker}: fetching recent Form 4 filings...")
        filings = get_recent_form4_filings(cik, max_filings=20)

        for filing in filings:
            txns = parse_form4(filing["doc_url"])
            for t in txns:
                t["filing_date"] = filing["filing_date"]
                t["ticker"] = t["ticker"] or ticker
                all_transactions.append(t)

        print(f"    -> {len([t for t in all_transactions if t.get('ticker') == ticker])} transactions so far")

    df = pd.DataFrame(all_transactions)
    if df.empty:
        print("WARNING: no transactions collected at all. Check USER_AGENT in "
              "sec_client.py and your internet connection before debugging further.")
        return

    df.to_csv("data/insider_transactions_raw.csv", index=False)
    print(f"Raw save complete: {len(df)} rows -> data/insider_transactions_raw.csv")

    df["transaction_date"] = pd.to_datetime(df["transaction_date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["transaction_date"])
    df = df.sort_values(["ticker", "transaction_date"])
    df.to_csv("data/insider_transactions.csv", index=False)
    print(f"\nSaved {len(df)} transactions to data/insider_transactions.csv")
    print(df["acquired_disposed"].value_counts())


if __name__ == "__main__":
    main()