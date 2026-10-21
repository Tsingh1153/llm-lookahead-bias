import yfinance as yf
import pandas as pd

TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "XOM", "UNH", "HD", "MA", "KO", "PEP", "DIS", "NFLX",
    "INTC", "CSCO", "BAC", "PFE", "NKE", "MCD", "ORCL", "CRM", "ADBE", "AMD",
]

#closing prices
prices = yf.download(TICKERS, start= "2020-12-01", end="2026-01-01", interval="1mo", auto_adjust=True)["Close"]

#Percent change
returns = prices.pct_change().dropna(how="all")

#reshape
df = returns.stack().reset_index()
df.columns = ["month", "ticker", "return"]
df = df.dropna()
df["month"] = df["month"].dt.strftime("%Y-%m")
df["actual"] = df["return"].apply(lambda r: "UP" if r > 0 else "DOWN")

df.to_csv("data/ground_truth.csv", index=False)
print(len(df), "rows")
print("Share of UP months:", round((df["actual"] == "UP").mean(), 3))
