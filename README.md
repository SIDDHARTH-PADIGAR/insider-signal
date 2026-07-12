# Insider Trading Signal Tracker

A live pipeline that pulls real SEC insider-trading filings, detects when
multiple company insiders buy stock in a short window ("cluster buys"),
and tests with actual statistics, not a vibe check — whether that
predicts the stock outperforming the market afterward.

**This is a quantitative research pipeline, not a trained ML model.**
Detection is rule-based, the CAR calculation is a formula, and the
signal test is a classical hypothesis test (one-sample t-test). Stating
that plainly rather than dressing it up — the value here is the data
engineering and statistical rigor, not a black-box model. See "Why no
trained model" below for the reasoning.

## The question

Academic finance research has repeatedly found that when several
insiders at the same company buy stock with their own money in a short
window, it's a weak but real positive signal, insiders have information
outsiders don't. This project builds a live tool to detect that pattern
and test whether it holds on recent, real data.

## Result

| Horizon | Mean CAR vs S&P 500 | n events | p-value | Significant at 5%? |
|---|---|---|---|---|
| 5 days  | +0.41% | 31 | 0.754 | No |
| 10 days | +1.97% | 30 | 0.195 | No |
| 20 days | +2.44% | 24 | 0.296 | No |
| 30 days | +1.44% | 23 | 0.573 | No |

**Every horizon is directionally positive consistent with the
hypothesis but none reach conventional statistical significance.**
This is an honest, real result, not a shortfall to apologize for: it
matches the pattern in published insider-trading literature, where this
effect is real but small, and typically only reaches significance with
samples of hundreds of events across many years. 31 events from roughly
a year of filings is exactly the sample-size-limited outcome you'd
expect, not a sign the method is broken.

## Architecture

```
SEC EDGAR (data.sec.gov + sec.gov/Archives)
        │
        ▼
src/sec_client.py       — ticker→CIK lookup, Form 4 filing list, raw XML fetch/parse
        │
        ▼
src/build_dataset.py    — pulls Form 4 transactions across all S&P 500 constituents
        │                  (unbiased universe — see "Design decisions" below)
        ▼
data/insider_transactions.csv   (16,389 real transactions)
        │
        ▼
src/event_study.py
   ├── detect_cluster_buys()  — rule-based: >=2 distinct insiders, 10-day window,
   │                            genuine open-market purchases only (code 'P')
   ├── compute_car()          — cumulative abnormal return vs SPY, 4 horizons
   └── one-sample t-test      — is mean CAR significantly different from zero?
        │
        ▼
src/api.py (FastAPI)  →  frontend/index.html (live dashboard)
```

## Quickstart

```bash
pip install -r requirements.txt
# edit USER_AGENT in src/sec_client.py with your real name/email — SEC
# blocks requests without a descriptive User-Agent, this isn't optional

python src/build_dataset.py   # ~45-60 min — real, rate-limited SEC calls across 500 tickers
python src/event_study.py     # detects clusters, computes CAR, runs significance test
uvicorn src.api:app --reload --port 8001
# open frontend/index.html directly in a browser
```

## Design decisions worth defending in an interview

**Why open-market purchases only (transaction code 'P'), not all "acquired" transactions.**
Raw Form 4 data is dominated by option exercises, RSU vesting, and
tax-withholding transactions routine compensation mechanics, not a
choice. Of 16,389 transactions pulled, only 30 were genuine 'P' codes.
Filtering to just 'P' is what makes this a real conviction signal
instead of noise from payroll events.

**Why the full, unbiased S&P 500, not a hand-picked watchlist.**
An earlier version of this project used a 20-28 ticker watchlist of
familiar large-caps. It found almost no cluster events, for a real
reason: mega-cap executives rarely need to buy stock with personal cash.
Rather than switch to companies already known for frequent insider
activity (which would bias the study toward finding a signal because the
sample was picked to contain it), I pulled the full, current S&P 500
constituent list from a standard public source and ran the same,
unbiased detection logic across all 500, more legitimate events from a
wider net, not a rigged sample.

**Why no trained ML model.**
With only 31 detected events, any train/test split leaves ~20 training
rows, nowhere near enough to fit a classifier without it just
memorizing noise. Rather than build a "trained model" that would be
untrustworthy with this little data, this project stays honest about
what it is: a statistically rigorous detection and testing pipeline. The
natural extension, a supervised classifier predicting cluster-buy
outcomes from features like insider count, dollar value, and sector is
a legitimate next step once enough events accumulate over a longer
collection window.

**Known limitation: EDGAR filing quirk.**
The SEC submissions API's `primaryDocument` field for Form 4 filings
points to a human-readable, XSL-rendered HTML view (despite the `.xml`
extension), not the raw machine-readable XML. The pipeline instead reads
each filing's own directory index to locate the actual raw XML document.
Worth knowing if extending this to other filing types (3, 5, 13D, etc.),
which may have the same quirk.

**Known limitation: NaN handling in recent events.**
Events too recent to have a full 20 or 30 trading days of subsequent
price history return `null` for those specific horizons rather than
being dropped entirely you'll see partial data for the newest cluster
events in the dashboard, which is expected, not a bug.

## Next steps (not yet built)

- Extend the collection window (multiple years) to grow the event sample
  toward the size needed for real statistical power
- Add sector/insider-role (officer vs director) as event characteristics
  to see if the signal is stronger in a subset
- Once sample size allows, revisit the supervised classifier described above

## Tech stack

Python, FastAPI, pandas, scipy (statistical testing), yfinance, SEC EDGAR
public API (data.sec.gov), vanilla JS/HTML dashboard (no build step).
```

Push both:
```powershell
git add .gitignore README.md
git commit -m "Add gitignore and full project README"
git push
```
