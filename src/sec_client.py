"""
sec_client.py — talks to the real, free, public SEC EDGAR API.

IMPORTANT: SEC requires a descriptive User-Agent identifying you (name +
email/contact). Requests without one, or with a generic/browser UA, get
blocked. Edit USER_AGENT below with your real info before running.

Rate limit: SEC asks for max 10 requests/second. We sleep between calls
to stay well under that — don't remove the sleep, you WILL get temporarily
blocked if you hammer it.
"""

import time
import requests
import xml.etree.ElementTree as ET

USER_AGENT = "Siddharth Padigar research-project siddharthpadigar22@gmail.com"  # EDIT if needed
HEADERS = {"User-Agent": USER_AGENT}
SLEEP_BETWEEN_CALLS = 0.15  # ~6-7 req/sec, safely under SEC's 10/sec limit


def get_cik_map() -> dict:
    """
    Ticker -> zero-padded 10-digit CIK, from SEC's official ticker list.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    mapping = {}
    for entry in data.values():
        ticker = entry["ticker"].upper()
        cik = str(entry["cik_str"]).zfill(10)
        mapping[ticker] = cik
    return mapping


def get_recent_form4_filings(cik: str, max_filings: int = 40) -> list[dict]:
    """
    Returns recent Form 4 filing metadata. For each filing, looks up the
    filing's own index.json to find the RAW xml document (not the rendered
    xslF345X0X/ HTML view, which the submissions API's primaryDocument
    field actually points to).
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    time.sleep(SLEEP_BETWEEN_CALLS)
    if resp.status_code != 200:
        return []

    data = resp.json()
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])

    results = []
    for i, form in enumerate(forms):
        if form != "4":
            continue
        accession_nodash = accessions[i].replace("-", "")
        raw_doc_url = _find_raw_form4_xml(cik, accession_nodash)
        if raw_doc_url:
            results.append({
                "cik": cik,
                "accession": accessions[i],
                "filing_date": filing_dates[i],
                "doc_url": raw_doc_url,
            })
        if len(results) >= max_filings:
            break
    return results


def _find_raw_form4_xml(cik: str, accession_nodash: str) -> str | None:
    """
    Looks at the filing's own directory listing and picks the raw XML doc —
    identified as an .xml file NOT inside an xsl.../ subfolder (that
    subfolder holds the rendered HTML view, not machine-readable data).
    """
    index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/index.json"
    resp = requests.get(index_url, headers=HEADERS, timeout=30)
    time.sleep(SLEEP_BETWEEN_CALLS)
    if resp.status_code != 200:
        return None

    try:
        items = resp.json().get("directory", {}).get("item", [])
    except Exception:
        return None

    for item in items:
        name = item.get("name", "")
        if name.endswith(".xml") and "xsl" not in name.lower():
            return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/{name}"
    return None


def parse_form4(doc_url: str) -> list[dict]:
    """
    Fetches and parses a single Form 4 XML document. Returns one dict per
    non-derivative transaction (i.e. actual open-market stock buys/sells,
    not option grants — those live in a separate derivativeTable we skip
    for this project since we care about open-market conviction buys).

    Transaction code 'P' = open market purchase (buy), 'S' = sale.
    Acquired/Disposed code 'A' = acquired, 'D' = disposed — we use this as
    the primary signal since it's more reliable than the transaction code
    field across filing variants.
    """
    resp = requests.get(doc_url, headers=HEADERS, timeout=30)
    time.sleep(SLEEP_BETWEEN_CALLS)
    if resp.status_code != 200:
        return []

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        return []  # some primaryDocument links are HTML wrappers, not raw XML — skip

    def text_or_none(elem, path):
        node = elem.find(path)
        return node.text.strip() if node is not None and node.text else None

    issuer_ticker = text_or_none(root, ".//issuer/issuerTradingSymbol")
    owner_name = text_or_none(root, ".//reportingOwner/reportingOwnerId/rptOwnerName")
    owner_title = text_or_none(root, ".//reportingOwnerRelationship/officerTitle")
    is_director = text_or_none(root, ".//reportingOwnerRelationship/isDirector") == "1"
    is_officer = text_or_none(root, ".//reportingOwnerRelationship/isOfficer") == "1"

    transactions = []
    for txn in root.findall(".//nonDerivativeTable/nonDerivativeTransaction"):
        txn_date = text_or_none(txn, ".//transactionDate/value")
        shares = text_or_none(txn, ".//transactionAmounts/transactionShares/value")
        price = text_or_none(txn, ".//transactionAmounts/transactionPricePerShare/value")
        ad_code = text_or_none(txn, ".//transactionAmounts/transactionAcquiredDisposedCode/value")
        txn_code = text_or_none(txn, ".//transactionCoding/transactionCode")

        if not txn_date or not shares:
            continue

        transactions.append({
            "ticker": issuer_ticker,
            "owner_name": owner_name,
            "owner_title": owner_title,
            "is_director": is_director,
            "is_officer": is_officer,
            "transaction_date": txn_date,
            "shares": float(shares),
            "price_per_share": float(price) if price else None,
            "acquired_disposed": ad_code,  # 'A' = buy, 'D' = sell
            "transaction_code": txn_code,
        })
    return transactions