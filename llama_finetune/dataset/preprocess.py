import requests
import pandas as pd

# fetch the page using  User-Agent
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

# parse out the first table (the S&P 500 constituents)
sp500 = pd.read_html(resp.text)[0]

# clean suffixes (e.g. “BRK.B” → “BRK”)
sp500["Symbol"] = sp500["Symbol"].str.replace(r"\..*$", "", regex=True)


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

#  fetch 
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
        time.sleep(0.5)  # be kind to the API haha

# Write out as in one big JSON 
with open(OUTPUT_FN, "w") as f:
    json.dump(all_articles, f, indent=2)

print(f"Wrote {len(all_articles)} articles to {OUTPUT_FN}")