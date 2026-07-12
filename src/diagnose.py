# diagnose2.py
import requests
from sec_client import HEADERS, get_recent_form4_filings, parse_form4

cik = "0000320193"
filings = get_recent_form4_filings(cik, max_filings=3)

print(f"Got {len(filings)} filing records")
for f in filings:
    print("\ndoc_url:", f["doc_url"])
    resp = requests.get(f["doc_url"], headers=HEADERS, timeout=30)
    print("  status:", resp.status_code)
    print("  content-type:", resp.headers.get("content-type"))
    print("  first 300 chars:", resp.text[:300])