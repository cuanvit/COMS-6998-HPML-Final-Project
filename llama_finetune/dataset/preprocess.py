import requests
import pandas as pd

# 1) fetch the page with a real-browser User-Agent
url = "https://www.slickcharts.com/sp500"
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    )
}
resp = requests.get(url, headers=headers)
resp.raise_for_status()

# 2) parse out the first table (the S&P 500 constituents)
sp500 = pd.read_html(resp.text)[0]

# 3) clean off any suffixes (e.g. “BRK.B” → “BRK”)
sp500["Symbol"] = sp500["Symbol"].str.replace(r"\..*$", "", regex=True)

# # 4) pull it into a Python list
# symbol_list = sp500["Symbol"].tolist()

# # 5) (optionally) print or return it
# print(symbol_list)

symbol_list = [f"{sym}.US" for sym in sp500["Symbol"].tolist()]

print(symbol_list)

import os
import json
import time
import requests


from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("API_KEY")

LIMIT      = 100
OFFSET     = 0
OUTPUT_FN  = "news_data.json"

# 2) Fetch & accumulate
all_articles = []
for symbol in symbol_list:
    try:
        url = (
            f'https://eodhd.com/api/news?s={symbol}&offset={OFFSET}&limit={LIMIT}&api_token={API_KEY}&fmt=json'
        )
        resp = requests.get(url)
        resp.raise_for_status()
        news = resp.json() or []
        # annotate each item with its ticker
        for item in news:
            item["ticker"] = symbol
        all_articles.extend(news)
        
    except requests.exceptions.HTTPError as http_err:
        print(f"[HTTP {resp.status_code}] {symbol}: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"[NETWORK ERROR] {symbol}: {req_err}")
    except Exception as err:
        print(f"[UNEXPECTED] {symbol}: {err}")

    finally:
        time.sleep(0.5)  # be kind to the API

# 3) Write out as one big JSON array
with open(OUTPUT_FN, "w") as f:
    json.dump(all_articles, f, indent=2)

print(f"Wrote {len(all_articles)} articles to {OUTPUT_FN}")