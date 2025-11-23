"""
Stock Price Caching Module

This module provides caching functionality for stock price data with:
- Parquet file storage (one file per stock for efficient storage/retrieval)
- Configurable staleness checking
- Force refresh capability
- Stock-specific staleness tracking

Based on the HistoricalCache from optionspnl but enhanced with:
- Parquet format instead of JSON for better performance with time series data
- More flexible configuration options
- Better separation of concerns
"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from pathlib import Path
import logging


class CacheConfig:
    """
    Configuration for price caching behavior.

    This allows configuration to be created once and reused across multiple
    cache instances, or passed down from top-level scripts.
    """

    def __init__(self,
                 cache_dir=None,
                 staleness_days=1,
                 history_days=252,
                 columns=None):
        """
        Initialize cache configuration.

        Args:
            cache_dir (str|Path): Directory to store cache files.
                                 Defaults to .cache/prices in project root
            staleness_days (float): Days before cache is considered stale.
                                   -1 means never update automatically.
                                   Can be fractional (e.g., 0.5 = 12 hours)
            history_days (int): Number of trading days to store
            columns (list): List of OHLCV columns to cache.
                          Defaults to ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        if cache_dir is None:
            # Default: project_root/.cache/prices
            project_root = Path(__file__).parent.parent
            cache_dir = project_root / ".cache" / "prices"

        self.cache_dir = Path(cache_dir)
        self.staleness_days = staleness_days
        self.history_days = history_days
        self.columns = columns if columns is not None else ['Open', 'High', 'Low', 'Close', 'Volume']

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict):
        """Create CacheConfig from dictionary (e.g., loaded from JSON)."""
        return cls(**config_dict)

    def to_dict(self):
        """Convert configuration to dictionary for serialization."""
        return {
            'cache_dir': str(self.cache_dir),
            'staleness_days': self.staleness_days,
            'history_days': self.history_days,
            'columns': self.columns
        }


class PriceCache:
    """
    Manages cached stock price data using Parquet files.

    Features:
    - One parquet file per stock for efficient storage and retrieval
    - Stock-specific staleness tracking (metadata stored with each stock)
    - Configurable cache behavior via CacheConfig
    - Force refresh capability
    """

    def __init__(self, config=None):
        """
        Initialize the price cache.

        Args:
            config (CacheConfig): Cache configuration. If None, uses defaults.
        """
        self.config = config if config is not None else CacheConfig()

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _get_cache_path(self, symbol):
        """Get the parquet file path for a symbol."""
        return self.config.cache_dir / f"{symbol}.parquet"

    def _get_metadata_path(self, symbol):
        """Get the metadata file path for a symbol."""
        return self.config.cache_dir / f"{symbol}_meta.json"

    def _load_metadata(self, symbol):
        """
        Load metadata for a cached symbol.

        Metadata includes: last_updated, columns, days_stored
        """
        meta_path = self._get_metadata_path(symbol)

        if not meta_path.exists():
            return None

        try:
            import json
            with open(meta_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Error loading metadata for {symbol}: {e}")
            return None

    def _save_metadata(self, symbol, metadata):
        """Save metadata for a symbol."""
        meta_path = self._get_metadata_path(symbol)

        try:
            import json
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving metadata for {symbol}: {e}")

    def needs_update(self, symbol):
        """
        Check if a symbol's cache needs updating.

        Staleness is checked on a per-stock basis using metadata.

        Args:
            symbol (str): Stock symbol

        Returns:
            bool: True if cache needs update, False otherwise
        """
        cache_path = self._get_cache_path(symbol)

        # No cache file exists
        if not cache_path.exists():
            return True

        # Load metadata
        metadata = self._load_metadata(symbol)
        if metadata is None:
            return True

        # If staleness_days is -1, never auto-update
        if self.config.staleness_days == -1:
            return False

        # Check if columns have changed
        cached_columns = set(metadata.get('columns', []))
        requested_columns = set(self.config.columns)
        if cached_columns != requested_columns:
            return True

        # Check staleness
        try:
            last_updated = datetime.fromisoformat(metadata.get('last_updated'))
            age_days = (datetime.now() - last_updated).total_seconds() / (24 * 3600)

            return age_days > self.config.staleness_days

        except (ValueError, KeyError):
            return True

    def _load_from_cache(self, symbol):
        """Load price data from parquet file."""
        cache_path = self._get_cache_path(symbol)

        if not cache_path.exists():
            return None

        try:
            df = pd.read_parquet(cache_path)
            self.logger.info(f"Loaded {symbol} from cache ({len(df)} rows)")
            return df
        except Exception as e:
            self.logger.error(f"Error loading cache for {symbol}: {e}")
            return None

    def _save_to_cache(self, symbol, data):
        """
        Save price data to parquet file with metadata.

        Args:
            symbol (str): Stock symbol
            data (pd.DataFrame): Price data with DatetimeIndex
        """
        cache_path = self._get_cache_path(symbol)

        try:
            # Save data as parquet
            data.to_parquet(cache_path, compression='snappy')

            # Save metadata
            metadata = {
                'symbol': symbol,
                'last_updated': datetime.now().isoformat(),
                'days_stored': len(data),
                'columns': list(data.columns),
                'date_range': {
                    'start': data.index.min().isoformat(),
                    'end': data.index.max().isoformat()
                }
            }
            self._save_metadata(symbol, metadata)

            self.logger.info(f"Saved {symbol} to cache ({len(data)} rows)")

        except Exception as e:
            self.logger.error(f"Error saving cache for {symbol}: {e}")

    def get(self, symbol, auto_update=True, force_refresh=False):
        """
        Get cached price data for a symbol.

        Args:
            symbol (str): Stock symbol
            auto_update (bool): Automatically update if stale (default True)
            force_refresh (bool): Force refresh even if fresh (default False)

        Returns:
            pd.DataFrame: Price data with DatetimeIndex, or None if not available
        """
        # Force refresh overrides everything
        if force_refresh:
            return None  # Will trigger fetch in caller

        # Check if update needed
        if auto_update and self.needs_update(symbol):
            return None  # Will trigger fetch in caller

        # Load from cache
        return self._load_from_cache(symbol)

    def put(self, symbol, data):
        """
        Store price data in cache.

        Args:
            symbol (str): Stock symbol
            data (pd.DataFrame): Price data to cache
        """
        if data is None or data.empty:
            self.logger.warning(f"Cannot cache empty data for {symbol}")
            return

        # Ensure data has datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.error(f"Data for {symbol} must have DatetimeIndex")
            return

        # Keep only requested columns if they exist
        available_cols = [col for col in self.config.columns if col in data.columns]
        if not available_cols:
            self.logger.warning(f"None of requested columns found in data for {symbol}")
            return

        data_to_save = data[available_cols].copy()

        # Keep only most recent history_days if specified
        if self.config.history_days > 0:
            data_to_save = data_to_save.tail(self.config.history_days)

        self._save_to_cache(symbol, data_to_save)

    def get_info(self, symbol):
        """
        Get cache information for a symbol.

        Args:
            symbol (str): Stock symbol

        Returns:
            dict: Cache metadata including staleness info
        """
        metadata = self._load_metadata(symbol)

        if metadata is None:
            return None

        # Add staleness info
        metadata['is_stale'] = self.needs_update(symbol)

        if metadata.get('last_updated'):
            try:
                last_updated = datetime.fromisoformat(metadata['last_updated'])
                age = datetime.now() - last_updated
                metadata['age_hours'] = round(age.total_seconds() / 3600, 1)
            except:
                pass

        return metadata

    def list_cached_symbols(self):
        """
        List all symbols with cached data.

        Returns:
            list: List of symbol strings
        """
        symbols = []

        for cache_file in self.config.cache_dir.glob("*.parquet"):
            symbol = cache_file.stem
            symbols.append(symbol)

        return sorted(symbols)

    def clear(self, symbol=None):
        """
        Clear cached data.

        Args:
            symbol (str): If provided, clear only this symbol.
                         If None, clear all cached data.
        """
        if symbol is None:
            # Clear all cache files
            for cache_file in self.config.cache_dir.glob("*.parquet"):
                cache_file.unlink()
            for meta_file in self.config.cache_dir.glob("*_meta.json"):
                meta_file.unlink()
            self.logger.info("Cleared entire cache")
        else:
            # Clear specific symbol
            cache_path = self._get_cache_path(symbol)
            meta_path = self._get_metadata_path(symbol)

            if cache_path.exists():
                cache_path.unlink()
            if meta_path.exists():
                meta_path.unlink()

            self.logger.info(f"Cleared cache for {symbol}")

    def get_cache_summary(self):
        """
        Get summary of all cached data.

        Returns:
            pd.DataFrame: Summary with columns: Symbol, Last Updated, Days, Is Stale
        """
        symbols = self.list_cached_symbols()

        if not symbols:
            return pd.DataFrame(columns=['Symbol', 'Last Updated', 'Days Stored',
                                        'Age (hours)', 'Is Stale'])

        summary = []
        for symbol in symbols:
            info = self.get_info(symbol)
            if info:
                summary.append({
                    'Symbol': symbol,
                    'Last Updated': info.get('last_updated', 'Unknown'),
                    'Days Stored': info.get('days_stored', 0),
                    'Age (hours)': info.get('age_hours', 0),
                    'Is Stale': info.get('is_stale', True)
                })

        return pd.DataFrame(summary)


# Global default cache instance
_default_cache = None


def get_default_cache():
    """Get or create the default cache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = PriceCache()
    return _default_cache


def set_default_cache(cache):
    """Set the default cache instance (useful for custom configurations)."""
    global _default_cache
    _default_cache = cache


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)

    # Example 1: Create cache with default config
    print("Example 1: Default cache configuration")
    cache = PriceCache()
    print(f"Cache directory: {cache.config.cache_dir}")
    print(f"Staleness threshold: {cache.config.staleness_days} days")
    print(f"History days: {cache.config.history_days}")
    print()

    # Example 2: Create cache with custom config
    print("Example 2: Custom cache configuration")
    custom_config = CacheConfig(
        cache_dir="/tmp/custom_cache",
        staleness_days=0.5,  # 12 hours
        history_days=500,
        columns=['Close', 'Volume']  # Only store Close and Volume
    )
    custom_cache = PriceCache(custom_config)
    print(f"Custom config: {custom_config.to_dict()}")
    print()

    # Example 3: Cache operations (simulated with sample data)
    print("Example 3: Simulating cache operations")

    # Create sample data
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    sample_data = pd.DataFrame({
        'Open': range(len(dates)),
        'High': range(len(dates)),
        'Low': range(len(dates)),
        'Close': range(len(dates)),
        'Volume': range(len(dates))
    }, index=dates)

    # Store in cache
    cache.put('DEMO', sample_data)

    # Check if needs update
    print(f"DEMO needs update: {cache.needs_update('DEMO')}")

    # Get cache info
    info = cache.get_info('DEMO')
    print(f"DEMO cache info: {info}")

    # Load from cache
    loaded = cache.get('DEMO')
    print(f"Loaded {len(loaded)} rows from cache")

    # List cached symbols
    print(f"Cached symbols: {cache.list_cached_symbols()}")

    # Get cache summary
    print("\nCache summary:")
    print(cache.get_cache_summary())

    # Clean up
    cache.clear()
    print("\nCache cleared")
