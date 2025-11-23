"""
Volatility Calculation Functions

This module provides functions to calculate various volatility metrics
for use in trading strategies.

All volatility functions take a prices Series/DataFrame and return a Series
of volatility values. Functions can be partialed to create specific volatility
calculators (e.g., simple_vol(20) for 20-day volatility).
"""

import pandas as pd
import numpy as np
from functools import partial


# Constants
TRADING_DAYS_PER_YEAR = 252


def simple_vol(prices, N=20, annualize=True):
    """
    Calculate trailing N-day standard deviation of daily returns.

    This is the simplest volatility measure: the standard deviation of
    log returns over a rolling N-day window.

    Args:
        prices (pd.Series or pd.DataFrame): Price series or DataFrame with 'Close' column
        N (int): Number of days for rolling window (default: 20)
        annualize (bool): If True, annualize the volatility (default: True)

    Returns:
        pd.Series: Rolling N-day volatility as a percentage

    Example:
        # Create a 20-day volatility calculator
        vol_20 = partial(simple_vol, N=20)

        # Use it
        prices = fetch_stock_prices('AAPL')
        volatility = vol_20(prices)
    """
    # Extract Close prices if DataFrame
    if isinstance(prices, pd.DataFrame):
        if 'Close' not in prices.columns:
            raise ValueError("DataFrame must have 'Close' column")
        price_series = prices['Close']
    else:
        price_series = prices

    # Calculate daily returns (today's close / yesterday's close - 1)
    returns = price_series.pct_change()

    # Calculate rolling standard deviation
    rolling_std = returns.rolling(window=N, min_periods=N).std()

    # Annualize if requested
    if annualize:
        rolling_std = rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Convert to percentage
    volatility = rolling_std * 100

    return volatility


def parkinson_vol(prices, N=20, annualize=True):
    """
    Calculate Parkinson volatility using High-Low range.

    Parkinson volatility uses the high-low range of prices, which can be
    more efficient than close-to-close volatility.

    Formula: sqrt(1/(4*N*ln(2)) * sum((ln(High/Low))^2))

    Args:
        prices (pd.DataFrame): DataFrame with 'High' and 'Low' columns
        N (int): Number of days for rolling window (default: 20)
        annualize (bool): If True, annualize the volatility (default: True)

    Returns:
        pd.Series: Rolling N-day Parkinson volatility as a percentage

    Note:
        Requires High and Low price data. More accurate than simple volatility
        when intraday range is significant.
    """
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("Parkinson volatility requires DataFrame with High and Low columns")

    required_cols = ['High', 'Low']
    missing = [col for col in required_cols if col not in prices.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Calculate ln(High/Low)^2
    hl_ratio = np.log(prices['High'] / prices['Low'])
    hl_ratio_sq = hl_ratio ** 2

    # Rolling sum
    sum_hl = hl_ratio_sq.rolling(window=N, min_periods=N).sum()

    # Parkinson formula
    parkinson = np.sqrt(sum_hl / (4 * N * np.log(2)))

    # Annualize if requested
    if annualize:
        parkinson = parkinson * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Convert to percentage
    volatility = parkinson * 100

    return volatility


def garman_klass_vol(prices, N=20, annualize=True):
    """
    Calculate Garman-Klass volatility using OHLC data.

    Garman-Klass extends Parkinson by also using Open and Close prices,
    providing a more accurate volatility estimate.

    Args:
        prices (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close' columns
        N (int): Number of days for rolling window (default: 20)
        annualize (bool): If True, annualize the volatility (default: True)

    Returns:
        pd.Series: Rolling N-day Garman-Klass volatility as a percentage

    Note:
        Most accurate of the range-based volatility estimators.
        Requires full OHLC data.
    """
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("Garman-Klass volatility requires DataFrame with OHLC columns")

    required_cols = ['Open', 'High', 'Low', 'Close']
    missing = [col for col in required_cols if col not in prices.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Calculate components
    hl = np.log(prices['High'] / prices['Low'])
    co = np.log(prices['Close'] / prices['Open'])

    # Garman-Klass formula
    gk_sq = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2

    # Rolling average
    gk_var = gk_sq.rolling(window=N, min_periods=N).mean()

    # Take square root to get volatility
    gk_vol = np.sqrt(gk_var)

    # Annualize if requested
    if annualize:
        gk_vol = gk_vol * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Convert to percentage
    volatility = gk_vol * 100

    return volatility


def ewma_vol(prices, span=20, annualize=True):
    """
    Calculate exponentially-weighted moving average volatility.

    EWMA gives more weight to recent observations, making it more
    responsive to recent changes in volatility.

    Args:
        prices (pd.Series or pd.DataFrame): Price series or DataFrame with 'Close' column
        span (int): Span for EWMA (roughly equivalent to N-day window, default: 20)
        annualize (bool): If True, annualize the volatility (default: True)

    Returns:
        pd.Series: EWMA volatility as a percentage
    """
    # Extract Close prices if DataFrame
    if isinstance(prices, pd.DataFrame):
        if 'Close' not in prices.columns:
            raise ValueError("DataFrame must have 'Close' column")
        price_series = prices['Close']
    else:
        price_series = prices

    # Calculate daily returns
    returns = price_series.pct_change()

    # Calculate EWMA of squared returns
    ewma_var = returns.pow(2).ewm(span=span, adjust=False).mean()

    # Take square root
    ewma_std = np.sqrt(ewma_var)

    # Annualize if requested
    if annualize:
        ewma_std = ewma_std * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Convert to percentage
    volatility = ewma_std * 100

    return volatility


def realized_vol(prices, N=20, annualize=True):
    """
    Calculate realized volatility (sum of squared returns).

    Also known as historical volatility. This is the same as simple_vol
    but uses a different calculation method (sum of squared returns).

    Args:
        prices (pd.Series or pd.DataFrame): Price series or DataFrame with 'Close' column
        N (int): Number of days for rolling window (default: 20)
        annualize (bool): If True, annualize the volatility (default: True)

    Returns:
        pd.Series: Rolling N-day realized volatility as a percentage
    """
    # Extract Close prices if DataFrame
    if isinstance(prices, pd.DataFrame):
        if 'Close' not in prices.columns:
            raise ValueError("DataFrame must have 'Close' column")
        price_series = prices['Close']
    else:
        price_series = prices

    # Calculate daily returns
    returns = price_series.pct_change()

    # Calculate rolling sum of squared returns
    sum_sq_returns = (returns ** 2).rolling(window=N, min_periods=N).sum()

    # Realized volatility
    realized = np.sqrt(sum_sq_returns / N)

    # Annualize if requested
    if annualize:
        realized = realized * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Convert to percentage
    volatility = realized * 100

    return volatility


# Convenience function to create partialed volatility functions
def create_vol_function(vol_func, **kwargs):
    """
    Create a partialed volatility function with fixed parameters.

    Args:
        vol_func: Volatility function (e.g., simple_vol, parkinson_vol)
        **kwargs: Parameters to partial out (e.g., N=20)

    Returns:
        Partialed function that takes only prices

    Example:
        # Create a 30-day simple volatility calculator
        vol_30 = create_vol_function(simple_vol, N=30)

        # Use it
        volatility = vol_30(prices)
    """
    return partial(vol_func, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("Volatility Functions Examples")
    print("=" * 80)

    # Create sample price data
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    np.random.seed(42)

    # Simulate random walk prices
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices_close = 100 * np.exp(np.cumsum(returns))

    # Create OHLC data
    prices = pd.DataFrame({
        'Close': prices_close,
        'Open': prices_close * (1 + np.random.normal(0, 0.005, len(dates))),
        'High': prices_close * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        'Low': prices_close * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
    }, index=dates)

    print("\nSample price data:")
    print(prices.head())

    # Example 1: Simple volatility with different windows
    print("\n1. Simple Volatility (20-day)")
    print("-" * 80)
    vol_20 = simple_vol(prices, N=20)
    print(f"Latest 20-day vol: {vol_20.iloc[-1]:.2f}%")

    # Example 2: Using partial
    print("\n2. Using partial to create 30-day volatility calculator")
    print("-" * 80)
    vol_30_func = partial(simple_vol, N=30)
    vol_30 = vol_30_func(prices)
    print(f"Latest 30-day vol: {vol_30.iloc[-1]:.2f}%")

    # Example 3: Parkinson volatility
    print("\n3. Parkinson Volatility (using High-Low)")
    print("-" * 80)
    parkinson = parkinson_vol(prices, N=20)
    print(f"Latest Parkinson vol: {parkinson.iloc[-1]:.2f}%")

    # Example 4: Garman-Klass volatility
    print("\n4. Garman-Klass Volatility (using OHLC)")
    print("-" * 80)
    gk = garman_klass_vol(prices, N=20)
    print(f"Latest Garman-Klass vol: {gk.iloc[-1]:.2f}%")

    # Example 5: EWMA volatility
    print("\n5. EWMA Volatility")
    print("-" * 80)
    ewma = ewma_vol(prices, span=20)
    print(f"Latest EWMA vol: {ewma.iloc[-1]:.2f}%")

    # Example 6: Compare all methods
    print("\n6. Comparison of Different Volatility Measures")
    print("-" * 80)

    comparison = pd.DataFrame({
        'Simple_Vol': simple_vol(prices, N=20),
        'Parkinson': parkinson_vol(prices, N=20),
        'Garman_Klass': garman_klass_vol(prices, N=20),
        'EWMA': ewma_vol(prices, span=20),
        'Realized': realized_vol(prices, N=20)
    })

    print("\nLast 10 days:")
    print(comparison.tail(10).to_string())

    print("\nSummary statistics:")
    print(comparison.describe().to_string())

    # Example 7: Creating reusable volatility functions
    print("\n7. Creating Reusable Volatility Functions")
    print("-" * 80)

    # Create different volatility calculators
    vol_10 = create_vol_function(simple_vol, N=10)
    vol_50 = create_vol_function(simple_vol, N=50)
    vol_gk_30 = create_vol_function(garman_klass_vol, N=30)

    print(f"10-day simple vol: {vol_10(prices).iloc[-1]:.2f}%")
    print(f"50-day simple vol: {vol_50(prices).iloc[-1]:.2f}%")
    print(f"30-day Garman-Klass vol: {vol_gk_30(prices).iloc[-1]:.2f}%")

    print("\n" + "=" * 80)
    print("Examples completed successfully!")
