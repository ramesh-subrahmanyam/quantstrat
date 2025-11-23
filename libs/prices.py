"""
Price Data Fetching Module

This module provides functions to fetch stock price data from Yahoo Finance
with intelligent caching using the cache module.

Compatible with the fetch_stock_prices interface from optionspnl/libs/dma_analysis.py
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

from .cache import PriceCache, CacheConfig, get_default_cache, set_default_cache


logger = logging.getLogger(__name__)


def fetch_stock_prices(symbol, num_days=201, use_cache=True, force_refresh=False):
    """
    Fetch historical adjusted closing prices for a stock.

    This function is compatible with the original from dma_analysis.py
    but adds caching functionality.

    Uses adjusted close prices to account for stock splits and dividends,
    ensuring accurate calculations across corporate actions.

    Args:
        symbol (str): Stock ticker symbol
        num_days (int): Number of days of historical data to fetch (default 201)
        use_cache (bool): Whether to use caching (default True)
        force_refresh (bool): Force refresh even if cached data is fresh (default False)

    Returns:
        pd.Series: Series of adjusted closing prices indexed by date, or None if error
    """
    try:
        # Calculate date range (fetch extra to account for weekends/holidays)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=num_days * 2)

        if use_cache:
            cache = get_default_cache()

            # Try to load from cache first
            if not force_refresh:
                cached_data = cache.get(symbol, auto_update=False)
                if cached_data is not None and 'Close' in cached_data.columns:
                    # Check if we have enough data
                    prices = cached_data['Close'].tail(num_days)
                    if len(prices) >= num_days:
                        logger.info(f"Using cached data for {symbol}")
                        return prices

            # Fetch fresh data
            logger.info(f"Fetching fresh data for {symbol}")
            hist = _fetch_from_yfinance(symbol, start_date, end_date)

            if hist is not None:
                # Cache the full OHLCV data
                cache.put(symbol, hist)

                # Return just Close prices
                prices = hist['Close'].tail(num_days)

                if len(prices) < num_days:
                    logger.warning(f"Only {len(prices)} days available for {symbol}, need {num_days}")
                    return None

                return prices
            else:
                return None

        else:
            # No caching - fetch directly
            hist = _fetch_from_yfinance(symbol, start_date, end_date)

            if hist is None:
                return None

            prices = hist['Close'].tail(num_days)

            if len(prices) < num_days:
                logger.warning(f"Only {len(prices)} days available for {symbol}, need {num_days}")
                return None

            return prices

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None


def fetch_price_range(symbol, start_date, end_date, use_cache=True, force_refresh=False):
    """
    Fetch OHLCV data for a specific date range.

    Args:
        symbol (str): Stock ticker symbol
        start_date (str|datetime): Start date (YYYY-MM-DD format or datetime)
        end_date (str|datetime): End date (YYYY-MM-DD format or datetime)
        use_cache (bool): Whether to use caching (default True)
        force_refresh (bool): Force refresh even if cached (default False)

    Returns:
        pd.DataFrame: DataFrame with Open, High, Low, Close, Volume columns
    """
    # Convert dates to strings if needed
    if isinstance(start_date, datetime):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, datetime):
        end_date = end_date.strftime('%Y-%m-%d')

    try:
        if use_cache:
            cache = get_default_cache()

            # Try cache first
            if not force_refresh:
                cached_data = cache.get(symbol, auto_update=False)

                if cached_data is not None:
                    # Filter to requested date range
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)

                    # Make sure timezone matches cached data
                    if cached_data.index.tz is not None:
                        if start_dt.tz is None:
                            start_dt = start_dt.tz_localize(cached_data.index.tz)
                        else:
                            start_dt = start_dt.tz_convert(cached_data.index.tz)
                        if end_dt.tz is None:
                            end_dt = end_dt.tz_localize(cached_data.index.tz)
                        else:
                            end_dt = end_dt.tz_convert(cached_data.index.tz)

                    filtered = cached_data[
                        (cached_data.index >= start_dt) &
                        (cached_data.index <= end_dt)
                    ]

                    # Check if cache covers the requested range
                    if not filtered.empty and \
                       filtered.index.min() <= start_dt and \
                       filtered.index.max() >= end_dt:
                        logger.info(f"Using cached data for {symbol}")
                        return filtered

            # Fetch fresh data
            logger.info(f"Fetching fresh data for {symbol} from {start_date} to {end_date}")
            data = _fetch_from_yfinance(symbol, start_date, end_date)

            if data is not None:
                cache.put(symbol, data)

            return data

        else:
            # No caching
            return _fetch_from_yfinance(symbol, start_date, end_date)

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None


def _fetch_from_yfinance(symbol, start_date, end_date):
    """
    Internal function to fetch data from Yahoo Finance.

    Args:
        symbol (str): Stock ticker
        start_date (str|datetime): Start date
        end_date (str|datetime): End date

    Returns:
        pd.DataFrame: OHLCV data with DatetimeIndex, or None if error
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)

        if hist.empty:
            logger.warning(f"No data returned for {symbol}")
            return None

        # Ensure we have standard columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [col for col in required_cols if col in hist.columns]

        if not available_cols:
            logger.warning(f"No standard OHLCV columns found for {symbol}")
            return None

        return hist[available_cols]

    except Exception as e:
        logger.error(f"Error fetching from yfinance for {symbol}: {e}")
        return None


def refresh_cache(symbol=None):
    """
    Force refresh of cached data.

    Args:
        symbol (str): If provided, refresh only this symbol.
                     If None, clear entire cache (next fetch will refresh).
    """
    cache = get_default_cache()

    if symbol is None:
        logger.info("Clearing entire cache")
        cache.clear()
    else:
        logger.info(f"Clearing cache for {symbol}")
        cache.clear(symbol)


def get_cache_info(symbol=None):
    """
    Get information about cached data.

    Args:
        symbol (str): If provided, get info for this symbol only.
                     If None, get summary of all cached symbols.

    Returns:
        dict or pd.DataFrame: Cache information
    """
    cache = get_default_cache()

    if symbol is not None:
        return cache.get_info(symbol)
    else:
        return cache.get_cache_summary()


def configure_cache(cache_dir=None, staleness_days=1, history_days=252, columns=None):
    """
    Configure the default cache settings.

    This should typically be called once at the start of your script/application.

    Args:
        cache_dir (str): Directory for cache files
        staleness_days (float): Days before cache is stale (-1 = never)
        history_days (int): Number of days to store
        columns (list): Columns to cache (default: OHLCV)

    Example:
        # At the start of your script:
        configure_cache(staleness_days=0.5, history_days=500)
    """
    config = CacheConfig(
        cache_dir=cache_dir,
        staleness_days=staleness_days,
        history_days=history_days,
        columns=columns
    )

    cache = PriceCache(config)
    set_default_cache(cache)

    logger.info(f"Cache configured: dir={config.cache_dir}, staleness={staleness_days}d, history={history_days}d")


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Testing price fetching with caching")
    print("=" * 60)

    # Example 1: Basic usage (compatible with original function)
    print("\n1. fetch_stock_prices (original interface)")
    print("-" * 60)
    prices = fetch_stock_prices('AAPL', num_days=201)
    if prices is not None:
        print(f"Fetched {len(prices)} days of AAPL closing prices")
        print(f"Latest price: ${prices.iloc[-1]:.2f}")
        print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    # Example 2: Fetch specific date range
    print("\n2. fetch_price_range")
    print("-" * 60)
    data = fetch_price_range('MSFT', '2024-01-01', '2024-12-31')
    if data is not None:
        print(f"Fetched {len(data)} days of MSFT OHLCV data")
        print(f"Columns: {list(data.columns)}")
        print(f"\nFirst few rows:")
        print(data.head())

    # Example 3: Check cache
    print("\n3. Cache status")
    print("-" * 60)
    cache_summary = get_cache_info()
    print(cache_summary.to_string(index=False))

    # Example 4: Fetch again (should use cache)
    print("\n4. Fetching AAPL again (should use cache)")
    print("-" * 60)
    prices2 = fetch_stock_prices('AAPL', num_days=201)
    print(f"Successfully retrieved: {prices2 is not None}")

    # Example 5: Force refresh
    print("\n5. Force refresh MSFT")
    print("-" * 60)
    data_refreshed = fetch_price_range('MSFT', '2024-01-01', '2024-12-31', force_refresh=True)
    print(f"Refreshed: {data_refreshed is not None}")

    # Example 6: Configure custom cache settings
    print("\n6. Custom cache configuration")
    print("-" * 60)
    configure_cache(staleness_days=0.5, history_days=500)
    print("Cache reconfigured with 12-hour staleness and 500-day history")

    # Example 7: Clear cache
    print("\n7. Clearing cache")
    print("-" * 60)
    refresh_cache()
    print("Cache cleared")

    final_summary = get_cache_info()
    print(f"Cached symbols: {len(final_summary)}")
