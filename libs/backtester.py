"""
Backtester Module

This module provides the Backtester class to run trading strategies
with volatility-adjusted position sizing and slippage modeling.
"""

import pandas as pd
import numpy as np
import logging
from .performance import stats


logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtester class to run trading strategies with vol-adjusted position sizing.

    The Backtester:
    1. Runs a strategy on a symbol/date range
    2. Calculates dollar volatility (return vol × price)
    3. Adjusts positions by dollar_size / dollar_volatility
    4. Calculates unslipped and slipped PnL
    5. Applies slippage based on position changes and slippage_bps
    6. Computes performance statistics for both

    All calculations are fully vectorized for efficiency.
    """

    def __init__(self, strategy, dollar_size=100000):
        """
        Initialize the Backtester.

        Args:
            strategy: Strategy object (instance of BaseStrategy or subclass)
            dollar_size (float): Target dollar size for position sizing (default: 100,000)

        Example:
            from strategies import MovingAverageCrossoverStrategy
            from libs.backtester import Backtester

            strategy = MovingAverageCrossoverStrategy(short_period=20, long_period=50)
            backtester = Backtester(strategy, dollar_size=100000)

            backtester('AAPL', '2024-01-01', '2024-12-31', slippage_bps=5)
        """
        self.strategy = strategy
        self.dollar_size = dollar_size

        # Will be set when __call__ is invoked
        self.symbol = None
        self.start_date = None
        self.end_date = None
        self.slippage_bps = None
        self.slipped_performance = None
        self.unslipped_performance = None

        logger.info(f"Initialized Backtester with {strategy.__class__.__name__}")
        logger.info(f"Dollar size: ${dollar_size:,.2f}")

    def __call__(self, symbol, start_date, end_date, slippage_bps=0):
        """
        Run the strategy and calculate vol-adjusted PnL with slippage.

        Args:
            symbol (str): Stock ticker symbol
            start_date (str|datetime): Start date (YYYY-MM-DD)
            end_date (str|datetime): End date (YYYY-MM-DD)
            slippage_bps (float): Slippage in basis points (default: 0)
                                 Example: 5 bps = 0.05% per trade

        Returns:
            None (results stored in self attributes)

        Attributes set:
            - slipped_performance: Performance stats with slippage
            - unslipped_performance: Performance stats without slippage

        Example:
            backtester('AAPL', '2024-01-01', '2024-12-31', slippage_bps=5)
            print(backtester.slipped_performance)
            print(backtester.unslipped_performance)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.slippage_bps = slippage_bps

        logger.info(f"Running backtest on {symbol} from {start_date} to {end_date}")
        logger.info(f"Slippage: {slippage_bps} bps")

        # Step 1: Run strategy
        logger.info("Running strategy...")
        self.strategy(symbol, start_date, end_date)

        if self.strategy.df is None or self.strategy.df.empty:
            logger.error(f"Strategy execution failed for {symbol}")
            self.slipped_performance = None
            self.unslipped_performance = None
            return

        df = self.strategy.df.copy()

        # Step 2: Calculate dollar volatility (vectorized)
        # Dollar volatility = (return volatility / 100) × price
        # Volatility is already in percentage form from volatility functions
        logger.info("Calculating dollar volatility...")
        df['Dollar_Volatility'] = (df['Volatility'] / 100) * df['Close']

        # Handle division by zero or NaN
        df['Dollar_Volatility'] = df['Dollar_Volatility'].replace(0, np.nan)

        # Step 3: Calculate vol-adjusted positions (vectorized)
        # Vol-adjusted position = position × (dollar_size / dollar_volatility)
        logger.info("Calculating vol-adjusted positions...")
        df['Vol_Adjusted_Position'] = df['Position'] * (self.dollar_size / df['Dollar_Volatility'])

        # Fill NaN positions with 0
        df['Vol_Adjusted_Position'] = df['Vol_Adjusted_Position'].fillna(0)

        # Step 4: Calculate unslipped PnL (vectorized)
        # Daily return
        df['Return'] = df['Close'].pct_change()

        # Lag positions by 1 day (today's position applies to tomorrow's return)
        df['Lagged_Position'] = df['Vol_Adjusted_Position'].shift(1).fillna(0)

        # Unslipped PnL = lagged_position × return × price
        # = lagged_position × (price_change)
        df['Price_Change'] = df['Close'].diff()
        df['Unslipped_PnL'] = df['Lagged_Position'] * df['Price_Change']

        # Fill NaN with 0 (first day)
        df['Unslipped_PnL'] = df['Unslipped_PnL'].fillna(0)

        logger.info("Calculated unslipped PnL")

        # Step 5: Calculate slippage (vectorized)
        # Slippage = |position_change| × price × slippage_bps / 10000
        # Note: slippage_bps / 10000 converts basis points to decimal
        # Example: 5 bps = 5/10000 = 0.0005 = 0.05%

        logger.info(f"Calculating slippage ({slippage_bps} bps)...")

        # Position change from previous day
        df['Position_Change'] = df['Vol_Adjusted_Position'].diff().fillna(0)

        # Slippage = |position_change| × price × (slippage_bps / 10000)
        df['Slippage'] = np.abs(df['Position_Change']) * df['Close'] * (slippage_bps / 10000)

        # Step 6: Calculate slipped PnL (vectorized)
        df['Slipped_PnL'] = df['Unslipped_PnL'] - df['Slippage']

        logger.info("Calculated slipped PnL")

        # Step 7: Calculate cumulative PnL
        df['Cumulative_Unslipped_PnL'] = df['Unslipped_PnL'].cumsum()
        df['Cumulative_Slipped_PnL'] = df['Slipped_PnL'].cumsum()

        # Store the enhanced dataframe back
        self.strategy.df = df

        # Step 8: Calculate performance statistics
        logger.info("Calculating performance statistics...")

        # Unslipped performance
        self.unslipped_performance = stats(df['Unslipped_PnL'])
        logger.info(f"Unslipped - Sharpe: {self.unslipped_performance['sharpe']:.3f}, "
                   f"Total PnL: ${self.unslipped_performance['total_pnl']:,.2f}")

        # Slipped performance
        self.slipped_performance = stats(df['Slipped_PnL'])
        logger.info(f"Slipped - Sharpe: {self.slipped_performance['sharpe']:.3f}, "
                   f"Total PnL: ${self.slipped_performance['total_pnl']:,.2f}")

        # Calculate slippage cost
        total_slippage = df['Slippage'].sum()
        logger.info(f"Total slippage cost: ${total_slippage:,.2f}")

        logger.info(f"Backtest complete for {symbol}")

    def get_summary(self):
        """
        Get a formatted summary of backtest results.

        Returns:
            str: Formatted summary string
        """
        if self.slipped_performance is None or self.unslipped_performance is None:
            return "No backtest results available. Run backtest first."

        lines = []
        lines.append("=" * 80)
        lines.append(f"BACKTEST RESULTS: {self.symbol}")
        lines.append(f"Period: {self.start_date} to {self.end_date}")
        lines.append(f"Dollar Size: ${self.dollar_size:,.2f}")
        lines.append(f"Slippage: {self.slippage_bps} bps")
        lines.append("=" * 80)

        lines.append("\nUNSLIPPED PERFORMANCE:")
        lines.append("-" * 80)
        unslipped = self.unslipped_performance
        lines.append(f"Total P&L:              ${unslipped['total_pnl']:,.2f}")
        lines.append(f"Sharpe Ratio:           {unslipped['sharpe']:.3f}")
        lines.append(f"Number of Trades:       {unslipped['num_trades']:,}")
        lines.append(f"Mean P&L per Trade:     ${unslipped['mean_pnl_per_trade']:,.2f}")

        lines.append("\nSLIPPED PERFORMANCE:")
        lines.append("-" * 80)
        slipped = self.slipped_performance
        lines.append(f"Total P&L:              ${slipped['total_pnl']:,.2f}")
        lines.append(f"Sharpe Ratio:           {slipped['sharpe']:.3f}")
        lines.append(f"Number of Trades:       {slipped['num_trades']:,}")
        lines.append(f"Mean P&L per Trade:     ${slipped['mean_pnl_per_trade']:,.2f}")

        lines.append("\nSLIPPAGE IMPACT:")
        lines.append("-" * 80)
        slippage_cost = unslipped['total_pnl'] - slipped['total_pnl']
        slippage_pct = (slippage_cost / abs(unslipped['total_pnl']) * 100) if unslipped['total_pnl'] != 0 else 0
        lines.append(f"Total Slippage Cost:    ${slippage_cost:,.2f} ({slippage_pct:.2f}%)")
        sharpe_impact = unslipped['sharpe'] - slipped['sharpe']
        lines.append(f"Sharpe Impact:          {sharpe_impact:.3f}")

        lines.append("=" * 80)

        return "\n".join(lines)

    def get_dataframe(self):
        """
        Get the strategy dataframe with all calculated fields.

        Returns:
            pd.DataFrame: DataFrame with prices, positions, PnL, etc.
        """
        if self.strategy.df is None:
            logger.warning("No dataframe available. Run backtest first.")
            return None

        return self.strategy.df


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path

    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from strategies.base import MovingAverageCrossoverStrategy, BuyAndHoldStrategy
    from functools import partial
    from libs.volatility import simple_vol
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("Backtester Examples (Vol-Adjusted with Slippage)")
    print("=" * 80)

    # Example 1: Single backtest with slippage
    print("\nExample 1: MA Strategy with 5 bps slippage")
    print("-" * 80)

    strategy = MovingAverageCrossoverStrategy(
        volatility_function=partial(simple_vol, N=20),
        short_period=20,
        long_period=50
    )
    backtester = Backtester(strategy, dollar_size=100000)

    backtester('AAPL', '2024-01-01', '2024-12-31', slippage_bps=5)

    print(backtester.get_summary())

    # Example 2: Compare different slippage levels
    print("\nExample 2: Slippage Sensitivity Analysis")
    print("-" * 80)

    slippage_levels = [0, 2, 5, 10, 20]
    results = []

    for slippage in slippage_levels:
        # Create fresh strategy instance
        strat = MovingAverageCrossoverStrategy(short_period=20, long_period=50)
        bt = Backtester(strat, dollar_size=100000)

        bt('AAPL', '2024-01-01', '2024-12-31', slippage_bps=slippage)

        results.append({
            'Slippage (bps)': slippage,
            'Unslipped PnL': bt.unslipped_performance['total_pnl'],
            'Slipped PnL': bt.slipped_performance['total_pnl'],
            'Cost': bt.unslipped_performance['total_pnl'] - bt.slipped_performance['total_pnl'],
            'Unslipped Sharpe': bt.unslipped_performance['sharpe'],
            'Slipped Sharpe': bt.slipped_performance['sharpe']
        })

    import pandas as pd
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Example 3: View detailed data
    print("\nExample 3: Sample Detailed Data (Last 10 Days)")
    print("-" * 80)

    df = backtester.get_dataframe()
    if df is not None:
        cols = ['Close', 'Volatility', 'Dollar_Volatility', 'Position',
                'Vol_Adjusted_Position', 'Unslipped_PnL', 'Slippage', 'Slipped_PnL']
        print(df[cols].tail(10).to_string())

    # Example 4: Different dollar sizes
    print("\nExample 4: Different Dollar Sizes")
    print("-" * 80)

    dollar_sizes = [50000, 100000, 250000, 500000]
    size_results = []

    for size in dollar_sizes:
        strat = MovingAverageCrossoverStrategy(short_period=20, long_period=50)
        bt = Backtester(strat, dollar_size=size)

        bt('AAPL', '2024-01-01', '2024-12-31', slippage_bps=5)

        size_results.append({
            'Dollar Size': f'${size:,}',
            'Total PnL': bt.slipped_performance['total_pnl'],
            'Sharpe': bt.slipped_performance['sharpe'],
            'Num Trades': bt.slipped_performance['num_trades']
        })

    size_df = pd.DataFrame(size_results)
    print(size_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("Examples completed successfully!")
    print("\nKey Features Demonstrated:")
    print("- Volatility-adjusted position sizing")
    print("- Slippage modeling (basis points)")
    print("- Unslipped vs Slipped performance comparison")
    print("- Fully vectorized calculations")
    print("=" * 80)
