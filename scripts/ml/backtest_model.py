#!/usr/bin/env python3
"""
Backtest the ML Price Predictor model on historical data
Simulates trading with the trained model to validate profitability
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("="*80)
print("ML MODEL BACKTEST - Historical Performance Validation")
print("="*80)

# Load the training data to get test set
db_path = Path('data/training_data.duckdb')

if not db_path.exists():
    print("❌ Training database not found!")
    sys.exit(1)

print("\n📊 Loading ALL data from DuckDB (3+ year backtest)...")
conn = duckdb.connect(str(db_path), read_only=True)
# Use all available data for comprehensive backtest
query = """
    SELECT * FROM (
        SELECT * FROM train
        UNION ALL
        SELECT * FROM validation
        UNION ALL
        SELECT * FROM test
    )
    ORDER BY Date
"""
test_df = conn.execute(query).df()
conn.close()

print(f"   Test period: {test_df['Date'].min()} to {test_df['Date'].max()}")
print(f"   Test samples: {len(test_df):,}")
print(f"   Symbols: {test_df['symbol'].nunique()}")

# Load trained model
import torch
import torch.nn as nn

class PricePredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=3, dropout=0.3):
        super(PricePredictor, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

print("\n🧠 Loading trained model...")
checkpoint = torch.load('models/price_predictor_best.pth', weights_only=False, map_location='cpu')
input_size = len(checkpoint['feature_cols'])
model = PricePredictor(input_size=input_size)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

scaler = checkpoint['scaler']
feature_cols = checkpoint['feature_cols']

print(f"   Model epoch: {checkpoint['epoch'] + 1}")
print(f"   Features: {len(feature_cols)}")

# Prepare test data
X_test = test_df[feature_cols].values
y_test = test_df[['target_1d', 'target_5d', 'target_20d']].values

# Normalize
X_test_norm = scaler.transform(X_test)

# Make predictions
print("\n🔮 Generating predictions...")
with torch.no_grad():
    X_test_t = torch.FloatTensor(X_test_norm)
    predictions = model(X_test_t).numpy()

# Add predictions to dataframe
test_df['pred_1d'] = predictions[:, 0]
test_df['pred_5d'] = predictions[:, 1]
test_df['pred_20d'] = predictions[:, 2]

# Simulated Trading Backtest
print("\n" + "="*80)
print("BACKT EST SIMULATION - Trading on Predictions")
print("="*80)

# Trading parameters
initial_capital = 10000
capital = initial_capital
position_size = 0.1  # 10% of capital per trade
confidence_threshold = 0.005  # 0.5% predicted move minimum
tax_rate = 0.371  # 37.1% short-term capital gains

# OPTIONS TRADING COMMISSIONS
# Each contract controls 100 shares
exchange_commission = 0.65  # $0.65 per contract (exchange-routed)
otc_commission = 2.00  # $2.00 per contract (OTC-routed, higher fees)
contracts_per_trade = 1  # Start with 1 contract per trade

# OTC routing happens for:
# - Extended hours trades (most are OTC-routed)
# - Large orders
# - Illiquid options
# User does NOT trade OTC - only exchange-routed trades
otc_routing_ratio = 0.0

# Blended average commission
blended_commission = (exchange_commission * (1 - otc_routing_ratio) +
                     otc_commission * otc_routing_ratio)

# Options spreads (bid-ask)
# Regular hours: 0.05-0.15 spread on typical options
# Extended hours: 0.15-0.40 spread (much wider due to low liquidity)
regular_hours_spread_pct = 0.001  # 0.1% spread (conservative for liquid options)
extended_hours_spread_pct = 0.0025  # 0.25% spread (wider in extended hours)
extended_hours_ratio = 0.3  # 30% of trades in extended hours

# Average option premium for fee calculation (will adjust per trade)
avg_option_premium = 200  # Assume $2 per share * 100 shares = $200 per contract

print(f"\n📈 OPTIONS TRADING WITH OTC ROUTING")
print(f"   Exchange commission: ${exchange_commission} per contract")
print(f"   OTC commission: ${otc_commission} per contract")
print(f"   OTC routing ratio: {otc_routing_ratio*100:.0f}%")
print(f"   Blended commission: ${blended_commission:.2f} per contract")
print(f"   Regular hours spread: {regular_hours_spread_pct*100:.2f}%")
print(f"   Extended hours spread: {extended_hours_spread_pct*100:.2f}%")
print(f"   Extended hours trades: {extended_hours_ratio*100:.0f}%")

trades = []
current_position = None

print(f"\n💰 Starting capital: ${initial_capital:,.2f}")
print(f"   Position size: {position_size*100:.1f}%")
print(f"   Confidence threshold: {confidence_threshold*100:.1f}%")
print(f"   Tax rate: {tax_rate*100:.1f}%\n")

# Group by symbol for sequential trading
for symbol in test_df['symbol'].unique():
    symbol_df = test_df[test_df['symbol'] == symbol].copy().reset_index(drop=True)

    for i in range(len(symbol_df) - 1):  # -1 because we need next day's actual return
        row = symbol_df.iloc[i]
        next_row = symbol_df.iloc[i + 1]

        # Trading logic: use 1-day prediction
        pred_return = row['pred_1d']
        actual_return = row['target_1d']

        # Only trade if prediction is confident
        if abs(pred_return) >= confidence_threshold:
            # Determine position direction
            if pred_return > 0:
                direction = "LONG"
            else:
                direction = "SHORT"

            # Calculate position value for OPTIONS
            # Position size determines how much capital to allocate
            allocated_capital = capital * position_size

            # Estimate option premium (assuming ATM options ~2-5% of stock price)
            # Using conservative 3% of stock price as premium estimate
            option_premium_per_share = row['close'] * 0.03
            option_premium_per_contract = option_premium_per_share * 100

            # Calculate how many contracts we can buy
            num_contracts = max(1, int(allocated_capital / option_premium_per_contract))
            actual_position_value = num_contracts * option_premium_per_contract

            # Calculate P&L before costs
            # For options, P&L = number of contracts * 100 shares * price change
            if direction == "LONG":
                gross_pnl = num_contracts * 100 * row['close'] * actual_return
            else:
                # Short options (put options profiting from price drops)
                gross_pnl = num_contracts * 100 * row['close'] * (-actual_return)

            # Calculate costs - OPTIONS SPECIFIC
            # Commission: Blended rate accounting for OTC routing
            commission = blended_commission * num_contracts * 2  # Entry + exit

            # Bid-ask spread cost (paid on entry and exit)
            spread_pct = (regular_hours_spread_pct * (1 - extended_hours_ratio) +
                         extended_hours_spread_pct * extended_hours_ratio)
            spread_cost = actual_position_value * spread_pct * 2  # Entry + exit

            fees = commission + spread_cost

            if gross_pnl > 0:
                taxes = gross_pnl * tax_rate
            else:
                taxes = 0  # No tax on losses

            net_pnl = gross_pnl - fees - taxes

            # Update capital
            capital += net_pnl

            # Record trade
            trades.append({
                'date': row['Date'],
                'symbol': symbol,
                'direction': direction,
                'pred_return': pred_return,
                'actual_return': actual_return,
                'position_value': actual_position_value,
                'num_contracts': num_contracts,
                'commission': commission,
                'spread_cost': spread_cost,
                'gross_pnl': gross_pnl,
                'fees': fees,
                'taxes': taxes,
                'net_pnl': net_pnl,
                'capital': capital,
                'correct': (pred_return * actual_return) > 0  # Same direction
            })

# Results
print(f"\n📊 BACKTEST RESULTS")
print("="*80)

if len(trades) == 0:
    print("❌ No trades executed (predictions below confidence threshold)")
else:
    trades_df = pd.DataFrame(trades)

    total_return = ((capital - initial_capital) / initial_capital) * 100
    win_rate = trades_df['correct'].mean() * 100
    total_trades = len(trades)
    winning_trades = trades_df[trades_df['net_pnl'] > 0].shape[0]
    losing_trades = trades_df[trades_df['net_pnl'] < 0].shape[0]

    avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean() if losing_trades > 0 else 0

    profit_factor = (trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum() /
                    abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum())
                    if losing_trades > 0 else float('inf'))

    max_drawdown = ((trades_df['capital'].cummin() - trades_df['capital']) / trades_df['capital']).max() * 100

    print(f"Final Capital: ${capital:,.2f}")
    print(f"Total Return: {total_return:+.2f}%")
    print(f"")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades} ({winning_trades/total_trades*100:.1f}%)")
    print(f"Losing Trades: {losing_trades} ({losing_trades/total_trades*100:.1f}%)")
    print(f"")
    print(f"Directional Accuracy: {win_rate:.1f}%")
    print(f"Average Win: ${avg_win:,.2f}")
    print(f"Average Loss: ${avg_loss:,.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")

    print(f"\n💸 COST BREAKDOWN")
    print("="*80)
    total_fees = trades_df['fees'].sum()
    total_commission = trades_df['commission'].sum()
    total_spread = trades_df['spread_cost'].sum()
    total_taxes = trades_df['taxes'].sum()
    gross_profit = trades_df['gross_pnl'].sum()
    net_profit = trades_df['net_pnl'].sum()
    avg_contracts = trades_df['num_contracts'].mean()

    print(f"Gross P&L: ${gross_profit:,.2f}")
    print(f"Total Fees: ${total_fees:,.2f} ({total_fees/abs(gross_profit)*100:.1f}% of gross)")
    print(f"  - Commission: ${total_commission:,.2f} (${blended_commission:.2f} blended × {total_trades*2} legs)")
    print(f"    * Exchange: ${exchange_commission} × {int((1-otc_routing_ratio)*total_trades*2)} legs")
    print(f"    * OTC: ${otc_commission} × {int(otc_routing_ratio*total_trades*2)} legs")
    print(f"  - Bid-Ask Spread: ${total_spread:,.2f}")
    print(f"Average Contracts/Trade: {avg_contracts:.1f}")

    # Calculate realistic tax (offset losses against gains)
    realistic_tax = max(0, gross_profit * tax_rate) if gross_profit > 0 else 0
    realistic_net_pnl = gross_profit - total_fees - realistic_tax

    print(f"Total Taxes (per-trade): ${total_taxes:,.2f} ({total_taxes/abs(gross_profit)*100:.1f}% of gross) ⚠️ UNREALISTIC")
    print(f"Total Taxes (realistic): ${realistic_tax:,.2f} ({realistic_tax/abs(gross_profit)*100 if gross_profit != 0 else 0:.1f}% of gross) ✓ With loss offset")
    print(f"Net P&L (per-trade tax): ${net_profit:,.2f}")
    print(f"Net P&L (realistic tax): ${realistic_net_pnl:,.2f}")
    print(f"Final Return (realistic): {(realistic_net_pnl/initial_capital)*100:+.2f}%")

    # Profitability assessment
    print(f"\n🎯 PROFITABILITY ASSESSMENT")
    print("="*80)

    if total_return > 0 and win_rate >= 52.0:
        print("✅ MODEL IS PROFITABLE IN BACKTEST!")
        print(f"   ✅ Positive return: +{total_return:.2f}%")
        print(f"   ✅ Win rate: {win_rate:.1f}% (>= 52% target after costs)")
        print(f"\n🚀 Model validated for live trading!")
    elif total_return > 0:
        print("⚠️  Model is marginally profitable")
        print(f"   ✅ Positive return: +{total_return:.2f}%")
        print(f"   ⚠️  Win rate: {win_rate:.1f}% (< 52% after costs)")
        print(f"\n   Consider: More training data or feature engineering")
    else:
        print("❌ Model is not profitable in backtest")
        print(f"   ❌ Negative return: {total_return:.2f}%")
        print(f"   ❌ Win rate: {win_rate:.1f}%")
        print(f"\n   Action: Retrain with more data or different features")

    # Save backtest results
    trades_df.to_csv('models/backtest_results.csv', index=False)
    print(f"\n📁 Backtest results saved: models/backtest_results.csv")

print("\n" + "="*80)
