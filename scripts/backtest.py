#!/usr/bin/env python3
"""
Main Backtest Script

This script runs a backtest using the 200-day moving average (DMA) strategy
and displays formatted performance results.

Usage:
    python backtest.py SYMBOL [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--slippage BPS] [--dollar-size SIZE]

Examples:
    python backtest.py AAPL
    python backtest.py AAPL --start-date 2020-01-01
    python backtest.py AAPL --start-date 2020-01-01 --end-date 2024-12-31
    python backtest.py AAPL --slippage 5 --dollar-size 250000
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime
import logging

# Add parent directory to path to import libs
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.dma import DMA
from libs.backtester import Backtester


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Run backtest using 200-day moving average strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s AAPL
  %(prog)s AAPL --start-date 2020-01-01
  %(prog)s AAPL --start-date 2020-01-01 --end-date 2024-12-31
  %(prog)s AAPL --slippage 5 --dollar-size 250000
        """
    )

    parser.add_argument(
        'symbol',
        type=str,
        help='Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default='2010-01-01',
        help='Start date for backtest (YYYY-MM-DD) [default: 2010-01-01]'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default=datetime.today().strftime('%Y-%m-%d'),
        help='End date for backtest (YYYY-MM-DD) [default: today]'
    )

    parser.add_argument(
        '--slippage',
        type=float,
        default=5.0,
        help='Slippage in basis points [default: 5]'
    )

    parser.add_argument(
        '--dollar-size',
        type=float,
        default=100000,
        help='Dollar size for position sizing [default: 100000]'
    )

    parser.add_argument(
        '--lookback',
        type=int,
        default=200,
        help='Moving average lookback period [default: 200]'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def validate_date(date_string):
    """
    Validate date string format.

    Args:
        date_string (str): Date string to validate

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def main():
    """
    Main function to run the backtest.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Validate dates
    if not validate_date(args.start_date):
        print(f"Error: Invalid start date format: {args.start_date}")
        print("Expected format: YYYY-MM-DD")
        sys.exit(1)

    if not validate_date(args.end_date):
        print(f"Error: Invalid end date format: {args.end_date}")
        print("Expected format: YYYY-MM-DD")
        sys.exit(1)

    # Print header
    print()
    print("=" * 80)
    print("QUANTSTRAT BACKTEST")
    print("=" * 80)
    print(f"Symbol:         {args.symbol}")
    print(f"Strategy:       {args.lookback}-Day Moving Average (DMA)")
    print(f"Period:         {args.start_date} to {args.end_date}")
    print(f"Dollar Size:    ${args.dollar_size:,.2f}")
    print(f"Slippage:       {args.slippage} bps")
    print("=" * 80)
    print()

    try:
        # Create strategy instance
        strategy = DMA(lookback=args.lookback)

        # Create backtester instance
        backtester = Backtester(strategy, dollar_size=args.dollar_size)

        # Run backtest
        print("Running backtest...")
        backtester(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            slippage_bps=args.slippage
        )

        # Check if backtest was successful
        if backtester.slipped_performance is None:
            print()
            print("=" * 80)
            print("ERROR: Backtest failed")
            print("=" * 80)
            print()
            print("Possible reasons:")
            print("  - Invalid ticker symbol")
            print("  - No data available for the specified date range")
            print("  - Network connectivity issues")
            print()
            sys.exit(1)

        # Display results
        print()
        print(backtester.get_summary())
        print()

        # Display additional statistics
        df = backtester.get_dataframe()
        if df is not None:
            print("ADDITIONAL STATISTICS:")
            print("-" * 80)

            # Position statistics
            long_days = (df['Position'] == 1).sum()
            flat_days = (df['Position'] == 0).sum()
            total_days = len(df)

            print(f"Total Trading Days:     {total_days:,}")
            print(f"Long Days:              {long_days:,} ({100*long_days/total_days:.1f}%)")
            print(f"Flat Days:              {flat_days:,} ({100*flat_days/total_days:.1f}%)")

            # PnL statistics
            winning_days = (df['Slipped_PnL'] > 0).sum()
            losing_days = (df['Slipped_PnL'] < 0).sum()
            win_rate = (winning_days / (winning_days + losing_days) * 100) if (winning_days + losing_days) > 0 else 0

            print(f"\nWinning Days:           {winning_days:,}")
            print(f"Losing Days:            {losing_days:,}")
            print(f"Win Rate:               {win_rate:.1f}%")

            # Return statistics
            avg_win = df[df['Slipped_PnL'] > 0]['Slipped_PnL'].mean() if winning_days > 0 else 0
            avg_loss = df[df['Slipped_PnL'] < 0]['Slipped_PnL'].mean() if losing_days > 0 else 0

            print(f"\nAverage Win:            ${avg_win:,.2f}")
            print(f"Average Loss:           ${avg_loss:,.2f}")
            if avg_loss != 0:
                print(f"Win/Loss Ratio:         {abs(avg_win/avg_loss):.2f}")

            # Drawdown
            cumulative_pnl = df['Cumulative_Slipped_PnL']
            running_max = cumulative_pnl.cummax()
            drawdown = cumulative_pnl - running_max
            max_drawdown = drawdown.min()

            print(f"\nMax Drawdown:           ${max_drawdown:,.2f}")

            print("=" * 80)
            print()

    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print("Backtest interrupted by user")
        print("=" * 80)
        print()
        sys.exit(1)

    except Exception as e:
        print()
        print("=" * 80)
        print(f"ERROR: {str(e)}")
        print("=" * 80)
        print()
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
