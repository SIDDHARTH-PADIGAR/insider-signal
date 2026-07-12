"""
FastAPI backend for the Insider Trading Signal Tracker.

Run: uvicorn src.api:app --reload --port 8001
Docs: http://localhost:8001/docs
"""

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Insider Trading Signal Tracker",
    description="Detects insider cluster buys from live SEC Form 4 filings "
                "and tests whether they predict abnormal stock returns.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "V", "WMT", "PG", "KO", "DIS", "NFLX", "AMD",
    "CRM", "ADBE", "COST", "PEP", "XOM",
]


def _load_events():
    try:
        return pd.read_csv("data/cluster_buy_events.csv", parse_dates=["cluster_start", "cluster_end"])
    except FileNotFoundError:
        return pd.DataFrame()


def _load_summary():
    try:
        return joblib.load("models/event_study_summary.joblib")
    except FileNotFoundError:
        return {}


@app.get("/watchlist")
def get_watchlist():
    return {"tickers": WATCHLIST}


@app.get("/summary")
def get_summary():
    summary = _load_summary()
    if not summary:
        raise HTTPException(404, "No event study results yet — run src/event_study.py first.")
    return summary


@app.get("/clusters")
def get_clusters():
    events = _load_events()
    if events.empty:
        raise HTTPException(404, "No cluster buy events found — run build_dataset.py then event_study.py.")
    events = events.sort_values("cluster_end", ascending=False)
    events = events.replace({np.nan: None})
    return events.to_dict(orient="records")


@app.get("/clusters/{ticker}")
def get_clusters_for_ticker(ticker: str):
    events = _load_events()
    if events.empty:
        raise HTTPException(404, "No cluster buy events found.")
    subset = events[events["ticker"] == ticker.upper()]
    if subset.empty:
        raise HTTPException(404, f"No cluster buy events found for {ticker.upper()}.")
    subset = subset.replace({np.nan: None})
    return subset.sort_values("cluster_end", ascending=False).to_dict(orient="records")


@app.get("/")
def root():
    return {"status": "ok", "message": "Insider Trading Signal Tracker API. See /docs."}