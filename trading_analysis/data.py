"""Load a single adjusted daily close series from a CSV or Yahoo Finance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_close_csv(path: str | Path) -> pd.Series:
    data = pd.read_csv(path)
    if not {"Date", "Close"}.issubset(data.columns):
        raise ValueError("CSV must contain Date and adjusted Close columns")
    dates = pd.to_datetime(data["Date"], errors="raise")
    close = pd.to_numeric(data["Close"], errors="raise").astype(float)
    return pd.Series(close.to_numpy(), index=pd.DatetimeIndex(dates), name="Close")


def download_close(ticker: str, *, start: str = "2023-01-01", end: str | None = None) -> pd.Series:
    """Download split/dividend-adjusted daily closes for one ticker."""
    import yfinance as yf

    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"no price data returned for {ticker}")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise ValueError("expected exactly one ticker")
        close = close.iloc[:, 0]
    return close.rename("Close")
