import yfinance as yf
import pandas as pd

def get_info(name: str) -> dict:
    """
    Fetches general metadata and descriptive statistics for a given ticker.

    Args:
        name (str): The ticker symbol of the asset (e.g., 'AAPL', 'MSFT').

    Returns:
        dict: A dictionary containing company information, such as sector,
              market cap, and fundamental ratios.
    """
    tick = yf.Ticker(name)
    return tick.info



def get_close_price(name: str, start=None, end=None) -> pd.Series:
    """
    Retrieves historical closing prices for a specific asset.

    If no dates are provided, the function returns the maximum available
    historical data.

    Args:
        name (str): The ticker symbol of the asset.
        start (str, optional): Start date in 'YYYY-MM-DD' format.
        end (str, optional): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.Series: A pandas Series of closing prices indexed by date.
    """

    tick = yf.Ticker(name)

    if start is None and end is None:
        data = tick.history(period="max")
    else:
        data = tick.history(start=start, end=end)

    return data["Close"]


def get_option_chain(name: str) -> dict:
    """
    Retrieves the complete Call option chain for all available expiries.

    The function filters the raw data to retain only relevant columns for
    volatility analysis and pricing.

    Args:
        name (str): The ticker symbol of the asset.

    Returns:
        dict: A dictionary where keys are expiration dates (str) and values
              are pandas DataFrames containing columns such as 'strike',
              'lastPrice', 'bid', 'ask', and 'impliedVolatility'.

    Note:
        Only Call options are processed in this implementation.
    """

    tick = yf.Ticker(name)
    expiries = tick.options
    chains = {}

    wanted = ["strike", "lastPrice", "bid", "ask", "impliedVol", "impliedVolatility"]

    for exp in expiries:
        chain = tick.option_chain(exp)
        calls = chain.calls

        available = [col for col in wanted if col in calls.columns]
        chains[exp] = calls[available]

    return chains
