import yfinance as yf
import pandas as pd


# Default tickers for single-stock backtesting
TICKERS = ["AAPL", "MSFT", "TSLA"]
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"

# Candidate universe for the portfolio builder.
# Spans 6 sectors so the Markowitz optimizer has genuine diversification
# to work with, if all candidates were from the same sector, the
# covariance matrix would be high and the optimizer wouldn't gain much.
PORTFOLIO_UNIVERSE = [
    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    # Consumer / E-commerce
    "AMZN", "TSLA", "COST",
    # Financials
    "JPM", "GS",
    # Healthcare
    "JNJ", "UNH",
    # Energy
    "XOM", "CVX",
    # Industrials / Aerospace
    "BA", "CAT",
]

# Broad validation universe — intentionally includes less prominent names
# across many sectors to stress-test the model on stocks the DQN hasn't
# been tuned against. Avoids survivorship bias by mixing large, mid, and
# cyclical names rather than just the most famous mega-cap performers.
BROAD_UNIVERSE = [
    # Large-cap tech (well-known but different dynamics)
    "INTC", "IBM", "QCOM", "TXN",
    # Mid-cap tech
    "FTNT", "CDNS", "ANSS",
    # Financials (banks + insurance)
    "BAC", "WFC", "AXP", "TRV", "AFL",
    # Healthcare (pharma + devices)
    "ABBV", "BMY", "MDT", "BDX", "ZBH",
    # Consumer staples
    "WMT", "TGT", "KO", "PG", "CL",
    # Consumer discretionary
    "NKE", "MCD", "YUM", "SBUX",
    # Industrials
    "HON", "MMM", "EMR", "ETN",
    # Energy
    "SLB", "EOG", "MPC", "VLO",
    # Materials
    "NEM", "FCX", "LIN",
    # Real estate
    "AMT", "PLD", "O",
    # Utilities
    "NEE", "DUK", "SO",
]


def download_stock_data(ticker: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

    # yfinance >=0.2 returns MultiIndex columns when downloading a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Timezone-aware indices cause alignment issues when joining with SPY data
    df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index

    df.dropna(inplace=True)
    df.sort_index(inplace=True)

    print(f"[{ticker}] Downloaded {len(df)} trading days ({df.index[0].date()} – {df.index[-1].date()})")
    return df


def download_spy_context(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    # SPY is used as a market-wide benchmark feature, not as a trading target.
    # Only Close is needed — returns and momentum are derived in features.py.
    return download_stock_data("SPY", start=start, end=end)[["Close"]]


def download_all(tickers: list = TICKERS, start: str = START_DATE, end: str = END_DATE) -> dict:
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = download_stock_data(ticker, start, end)
        except Exception as e:
            print(f"[WARNING] Failed to download {ticker}: {e}")
    return data
