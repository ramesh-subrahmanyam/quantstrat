"""
Quantstrat Libraries

Core modules for the backtesting engine.
"""

from .cache import CacheConfig, PriceCache, get_default_cache, set_default_cache
from .prices import (
    fetch_stock_prices,
    fetch_price_range,
    refresh_cache,
    get_cache_info,
    configure_cache
)
from .volatility import (
    simple_vol,
    parkinson_vol,
    garman_klass_vol,
    ewma_vol,
    realized_vol,
    create_vol_function
)
from .performance import (
    stats,
    calculate_pnl,
    extended_stats,
    summary_table
)
from .backtester import Backtester

__all__ = [
    # Cache
    'CacheConfig',
    'PriceCache',
    'get_default_cache',
    'set_default_cache',

    # Prices
    'fetch_stock_prices',
    'fetch_price_range',
    'refresh_cache',
    'get_cache_info',
    'configure_cache',

    # Volatility
    'simple_vol',
    'parkinson_vol',
    'garman_klass_vol',
    'ewma_vol',
    'realized_vol',
    'create_vol_function',

    # Performance
    'stats',
    'calculate_pnl',
    'extended_stats',
    'summary_table',

    # Backtester
    'Backtester',
]
