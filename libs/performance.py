"""
Performance Analytics Module

This module provides functions to calculate performance statistics
for trading strategies.
"""

import pandas as pd
import numpy as np


# Constants
TRADING_DAYS_PER_YEAR = 252


def stats(pnl_series):
    """
    Calculate performance statistics from a P&L series.

    Args:
        pnl_series (pd.Series): Series of daily profit and loss values
                               Index should be dates, values should be daily PnL

    Returns:
        dict: Dictionary containing:
            - sharpe: Annualized Sharpe ratio
            - total_pnl: Total profit/loss
            - num_trades: Number of trades (entries/exits)
            - mean_pnl_per_trade: Average PnL per trade

    Example:
        # Assuming you have a strategy result with positions
        pnl = calculate_pnl(df['Close'], df['Position'])
        performance = stats(pnl)
        print(f"Sharpe: {performance['sharpe']:.2f}")
        print(f"Total PnL: ${performance['total_pnl']:.2f}")
    """
    if pnl_series is None or len(pnl_series) == 0:
        return {
            'sharpe': 0.0,
            'total_pnl': 0.0,
            'num_trades': 0,
            'mean_pnl_per_trade': 0.0
        }

    # Total PnL
    total_pnl = pnl_series.sum()

    # Sharpe ratio (annualized)
    mean_pnl = pnl_series.mean()
    std_pnl = pnl_series.std()

    if std_pnl == 0 or np.isnan(std_pnl):
        sharpe = 0.0
    else:
        sharpe = (mean_pnl / std_pnl) * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Number of trades (count non-zero PnL days as proxy)
    # A more accurate count would need position changes
    num_trades = (pnl_series != 0).sum()

    # Mean PnL per trade
    if num_trades > 0:
        mean_pnl_per_trade = total_pnl / num_trades
    else:
        mean_pnl_per_trade = 0.0

    return {
        'sharpe': sharpe,
        'total_pnl': total_pnl,
        'num_trades': num_trades,
        'mean_pnl_per_trade': mean_pnl_per_trade
    }


def calculate_pnl(prices, positions, initial_capital=100000):
    """
    Calculate daily P&L from prices and positions.

    Args:
        prices (pd.Series): Series of prices (Close prices)
        positions (pd.Series): Series of positions (1=long, 0=flat, -1=short)
        initial_capital (float): Initial capital (default: 100,000)

    Returns:
        pd.Series: Daily P&L series

    Note:
        This is a simplified P&L calculation. For more accurate results,
        you'd want to account for:
        - Transaction costs
        - Slippage
        - Position sizing
        - Margin requirements (for shorts)
    """
    # Calculate returns
    returns = prices.pct_change()

    # Lag positions by 1 day (trade on today's close, realize PnL tomorrow)
    lagged_positions = positions.shift(1).fillna(0)

    # Strategy returns = position * market returns
    strategy_returns = lagged_positions * returns

    # Dollar PnL
    pnl = strategy_returns * initial_capital

    return pnl


def extended_stats(pnl_series, prices=None, positions=None):
    """
    Calculate extended performance statistics.

    Args:
        pnl_series (pd.Series): Daily P&L series
        prices (pd.Series, optional): Price series for additional metrics
        positions (pd.Series, optional): Position series for additional metrics

    Returns:
        dict: Extended statistics including:
            - Basic stats (sharpe, total_pnl, num_trades, mean_pnl_per_trade)
            - max_drawdown: Maximum drawdown percentage
            - max_drawdown_duration: Maximum drawdown duration in days
            - win_rate: Percentage of profitable trades
            - profit_factor: Ratio of gross profits to gross losses
            - avg_win: Average winning trade
            - avg_loss: Average losing trade
            - largest_win: Largest winning trade
            - largest_loss: Largest losing trade
    """
    # Get basic stats
    basic = stats(pnl_series)

    # Extended stats
    extended = {}

    # Cumulative PnL for drawdown calculation
    cum_pnl = pnl_series.cumsum()

    # Maximum drawdown
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_drawdown = drawdown.min()

    # Maximum drawdown as percentage of running max
    if running_max.max() > 0:
        max_drawdown_pct = (max_drawdown / running_max.max()) * 100
    else:
        max_drawdown_pct = 0.0

    extended['max_drawdown'] = max_drawdown
    extended['max_drawdown_pct'] = max_drawdown_pct

    # Maximum drawdown duration
    # Find periods where we're in drawdown
    in_drawdown = drawdown < 0
    if in_drawdown.any():
        # Find the longest consecutive True sequence
        groups = (in_drawdown != in_drawdown.shift()).cumsum()
        drawdown_lengths = in_drawdown.groupby(groups).sum()
        max_dd_duration = drawdown_lengths.max() if len(drawdown_lengths) > 0 else 0
    else:
        max_dd_duration = 0

    extended['max_drawdown_duration'] = int(max_dd_duration)

    # Win/Loss statistics (only for non-zero PnL days)
    non_zero_pnl = pnl_series[pnl_series != 0]

    if len(non_zero_pnl) > 0:
        wins = non_zero_pnl[non_zero_pnl > 0]
        losses = non_zero_pnl[non_zero_pnl < 0]

        # Win rate
        win_rate = (len(wins) / len(non_zero_pnl)) * 100 if len(non_zero_pnl) > 0 else 0.0
        extended['win_rate'] = win_rate

        # Profit factor
        gross_profit = wins.sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float('inf') if gross_profit > 0 else 0.0

        extended['profit_factor'] = profit_factor

        # Average win/loss
        extended['avg_win'] = wins.mean() if len(wins) > 0 else 0.0
        extended['avg_loss'] = losses.mean() if len(losses) > 0 else 0.0

        # Largest win/loss
        extended['largest_win'] = wins.max() if len(wins) > 0 else 0.0
        extended['largest_loss'] = losses.min() if len(losses) > 0 else 0.0

        # Number of wins/losses
        extended['num_wins'] = len(wins)
        extended['num_losses'] = len(losses)
    else:
        extended['win_rate'] = 0.0
        extended['profit_factor'] = 0.0
        extended['avg_win'] = 0.0
        extended['avg_loss'] = 0.0
        extended['largest_win'] = 0.0
        extended['largest_loss'] = 0.0
        extended['num_wins'] = 0
        extended['num_losses'] = 0

    # Combine basic and extended stats
    return {**basic, **extended}


def summary_table(stats_dict):
    """
    Create a formatted summary table of statistics.

    Args:
        stats_dict (dict): Dictionary of statistics (from stats() or extended_stats())

    Returns:
        str: Formatted string table
    """
    lines = []
    lines.append("=" * 60)
    lines.append("PERFORMANCE SUMMARY")
    lines.append("=" * 60)

    # Basic stats
    lines.append(f"Total P&L:              ${stats_dict.get('total_pnl', 0):,.2f}")
    lines.append(f"Sharpe Ratio:           {stats_dict.get('sharpe', 0):.3f}")
    lines.append(f"Number of Trades:       {stats_dict.get('num_trades', 0):,}")
    lines.append(f"Mean P&L per Trade:     ${stats_dict.get('mean_pnl_per_trade', 0):,.2f}")

    # Extended stats if available
    if 'max_drawdown' in stats_dict:
        lines.append("")
        lines.append("Drawdown Metrics:")
        lines.append(f"  Max Drawdown:         ${stats_dict.get('max_drawdown', 0):,.2f}")
        lines.append(f"  Max Drawdown %:       {stats_dict.get('max_drawdown_pct', 0):.2f}%")
        lines.append(f"  Max DD Duration:      {stats_dict.get('max_drawdown_duration', 0)} days")

    if 'win_rate' in stats_dict:
        lines.append("")
        lines.append("Win/Loss Metrics:")
        lines.append(f"  Win Rate:             {stats_dict.get('win_rate', 0):.2f}%")
        lines.append(f"  Profit Factor:        {stats_dict.get('profit_factor', 0):.3f}")
        lines.append(f"  Number of Wins:       {stats_dict.get('num_wins', 0)}")
        lines.append(f"  Number of Losses:     {stats_dict.get('num_losses', 0)}")
        lines.append(f"  Average Win:          ${stats_dict.get('avg_win', 0):,.2f}")
        lines.append(f"  Average Loss:         ${stats_dict.get('avg_loss', 0):,.2f}")
        lines.append(f"  Largest Win:          ${stats_dict.get('largest_win', 0):,.2f}")
        lines.append(f"  Largest Loss:         ${stats_dict.get('largest_loss', 0):,.2f}")

    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("Performance Analytics Examples")
    print("=" * 80)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')

    # Simulate a strategy with positive drift and volatility
    daily_returns = np.random.normal(0.001, 0.02, len(dates))
    prices = pd.Series(100 * np.exp(np.cumsum(daily_returns)), index=dates)

    # Simple strategy: long when price > 20-day MA, flat otherwise
    ma_20 = prices.rolling(20).mean()
    positions = pd.Series(0, index=dates)
    positions[prices > ma_20] = 1

    # Calculate P&L
    pnl = calculate_pnl(prices, positions, initial_capital=100000)

    print("\nExample 1: Basic Statistics")
    print("-" * 80)
    basic_stats = stats(pnl)
    print(f"Sharpe Ratio: {basic_stats['sharpe']:.3f}")
    print(f"Total P&L: ${basic_stats['total_pnl']:,.2f}")
    print(f"Number of Trades: {basic_stats['num_trades']}")
    print(f"Mean P&L per Trade: ${basic_stats['mean_pnl_per_trade']:.2f}")

    print("\nExample 2: Extended Statistics")
    print("-" * 80)
    ext_stats = extended_stats(pnl, prices, positions)
    print(summary_table(ext_stats))

    print("\nExample 3: Comparing Strategies")
    print("-" * 80)

    # Strategy 2: Buy and hold
    positions_bh = pd.Series(1, index=dates)
    pnl_bh = calculate_pnl(prices, positions_bh, initial_capital=100000)
    stats_bh = stats(pnl_bh)

    print("MA Strategy:")
    print(f"  Sharpe: {basic_stats['sharpe']:.3f}, Total P&L: ${basic_stats['total_pnl']:,.2f}")

    print("\nBuy & Hold:")
    print(f"  Sharpe: {stats_bh['sharpe']:.3f}, Total P&L: ${stats_bh['total_pnl']:,.2f}")

    print("\n" + "=" * 80)
    print("Examples completed successfully!")
