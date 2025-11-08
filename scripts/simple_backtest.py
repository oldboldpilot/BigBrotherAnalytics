#!/usr/bin/env python3
"""
Simple Iron Condor Backtest - Tier 1 POC

Quick validation of iron condor strategy on historical data.
Uses Python for rapid prototyping before C++ implementation.

Usage:
    uv run python scripts/simple_backtest.py
    uv run python scripts/simple_backtest.py --symbol SPY --start 2023-01-01
"""

import argparse
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_iv_rank(symbol, current_date, lookback_days=252):
    """
    Calculate IV rank (simplified - using price volatility as proxy).

    IV Rank = (Current IV - IV Low) / (IV High - IV Low) * 100

    For POC: Using realized volatility as proxy for IV.
    """
    # Get historical prices
    conn = duckdb.connect('data/bigbrother.duckdb')

    query = f"""
        SELECT close, date
        FROM stock_prices
        WHERE symbol = '{symbol}'
          AND date <= '{current_date}'
        ORDER BY date DESC
        LIMIT {lookback_days}
    """

    df = conn.execute(query).df()

    if len(df) < 30:
        return 0.0  # Not enough data

    # Calculate realized volatility (annualized)
    returns = np.log(df['close'] / df['close'].shift(1)).dropna()
    realized_vol = returns.std() * np.sqrt(252)

    # Calculate IV rank (using 252-day range)
    vol_series = returns.rolling(30).std() * np.sqrt(252)
    vol_min = vol_series.min()
    vol_max = vol_series.max()

    if vol_max == vol_min:
        return 50.0

    iv_rank = (realized_vol - vol_min) / (vol_max - vol_min) * 100

    return iv_rank

def simulate_iron_condor(
    symbol,
    entry_date,
    stock_price,
    iv_rank,
    dte=45
):
    """
    Simulate iron condor entry and track to expiration.

    Simplified for Tier 1:
    - No real options data (estimating prices)
    - Using historical stock movements
    - Simple P/L calculation
    """
    # Entry criteria
    if iv_rank < 50:
        return None  # Skip if IV rank too low

    # Calculate expected move (1 SD)
    iv = 0.10 + (iv_rank / 100.0) * 0.40  # Map IV rank to ~10-50% IV
    expected_move = stock_price * iv * np.sqrt(dte / 365)

    # Iron condor structure
    short_put = stock_price - expected_move
    long_put = short_put - 5.0  # $5 wing
    short_call = stock_price + expected_move
    long_call = short_call + 5.0  # $5 wing

    # Estimate credit (simplified)
    put_credit = 0.30  # ~$30 per spread
    call_credit = 0.30  # ~$30 per spread
    total_credit = put_credit + call_credit  # $60 total

    max_loss = 5.0 - total_credit  # $440

    # Calculate exit date
    exit_date = entry_date + timedelta(days=dte)

    # Get stock price at exit
    conn = duckdb.connect('data/bigbrother.duckdb')
    exit_query = f"""
        SELECT close
        FROM stock_prices
        WHERE symbol = '{symbol}'
          AND date >= '{exit_date}'
        ORDER BY date
        LIMIT 1
    """

    exit_df = conn.execute(exit_query).df()

    if exit_df.empty:
        return None  # No data for exit

    exit_price = exit_df['close'].iloc[0]

    # Calculate P/L
    # If price stays within strikes: keep full credit
    # If price breaches: lose money

    if exit_price < short_put:
        # Breached downside
        breach_amount = short_put - exit_price
        loss = min(breach_amount, max_loss)
        pnl = total_credit - loss
    elif exit_price > short_call:
        # Breached upside
        breach_amount = exit_price - short_call
        loss = min(breach_amount, max_loss)
        pnl = total_credit - loss
    else:
        # Price stayed in range - full profit
        pnl = total_credit

    return {
        'symbol': symbol,
        'entry_date': entry_date,
        'exit_date': exit_date,
        'entry_price': stock_price,
        'exit_price': exit_price,
        'iv_rank': iv_rank,
        'short_put': short_put,
        'short_call': short_call,
        'credit': total_credit,
        'max_loss': max_loss,
        'pnl': pnl,
        'pnl_pct': (pnl / max_loss) * 100 if max_loss > 0 else 0
    }

def run_backtest(symbol='SPY', start_date='2023-01-01', end_date='2025-11-01'):
    """
    Run simple iron condor backtest.

    Entry rules:
    - IV rank > 50
    - 45 DTE
    - Enter on first trading day of each month

    Exit rules:
    - Hold to expiration (simplified)
    """
    print(f"\n🔬 Running Iron Condor Backtest")
    print(f"   Symbol: {symbol}")
    print(f"   Period: {start_date} to {end_date}")
    print(f"   Strategy: Monthly iron condors, 45 DTE, IV rank > 50")
    print()

    conn = duckdb.connect('data/bigbrother.duckdb')

    # Get all trading days
    query = f"""
        SELECT date, close
        FROM stock_prices
        WHERE symbol = '{symbol}'
          AND date >= '{start_date}'
          AND date <= '{end_date}'
        ORDER BY date
    """

    df = conn.execute(query).df()

    print(f"   Data: {len(df)} trading days")
    print()

    # Simulate trades (monthly entries)
    trades = []
    current_date = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    while current_date < end:
        # Find first trading day of month
        month_start = current_date.replace(day=1)
        month_data = df[df['date'] >= month_start]

        if month_data.empty:
            break

        entry_date = month_data.iloc[0]['date']
        stock_price = month_data.iloc[0]['close']

        # Calculate IV rank
        iv_rank = calculate_iv_rank(symbol, entry_date)

        # Simulate trade
        trade = simulate_iron_condor(
            symbol,
            entry_date,
            stock_price,
            iv_rank,
            dte=45
        )

        if trade:
            trades.append(trade)
            print(f"   Trade {len(trades)}: {entry_date.date()} @ ${stock_price:.2f} (IV Rank: {iv_rank:.0f}) → P/L: ${trade['pnl']:.2f}")

        # Move to next month
        current_date = (month_start + timedelta(days=32)).replace(day=1)

    if not trades:
        print("   No trades executed (IV rank criteria not met)")
        return

    # Calculate statistics
    trades_df = pd.DataFrame(trades)

    total_pnl = trades_df['pnl'].sum()
    win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) * 100
    avg_pnl = trades_df['pnl'].mean()
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean()
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean()
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    print()
    print("="*60)
    print("📊 Backtest Results:")
    print(f"   Total Trades: {len(trades_df)}")
    print(f"   Win Rate: {win_rate:.1f}%")
    print(f"   Total P/L: ${total_pnl:.2f}")
    print(f"   Average P/L: ${avg_pnl:.2f}")
    print(f"   Average Win: ${avg_win:.2f}")
    print(f"   Average Loss: ${avg_loss:.2f}")
    print(f"   Profit Factor: {profit_factor:.2f}")
    print(f"   ROC per trade: {trades_df['pnl_pct'].mean():.1f}%")
    print("="*60)
    print()

    if win_rate > 65 and total_pnl > 0:
        print("✅ Strategy shows promise! Win rate > 65% and profitable.")
    elif total_pnl > 0:
        print("⚠️  Profitable but win rate could be better.")
    else:
        print("❌ Strategy needs refinement.")

    return trades_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='SPY', help='Symbol to backtest')
    parser.add_argument('--start', default='2023-01-01', help='Start date')
    parser.add_argument('--end', default='2025-11-01', help='End date')

    args = parser.parse_args()

    try:
        results = run_backtest(args.symbol, args.start, args.end)

        if results is not None:
            # Save results
            results.to_csv('data/backtest_results.csv', index=False)
            print(f"📁 Results saved to: data/backtest_results.csv")

    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
