#!/usr/bin/env python3
"""
Sample Data Generator for BigBrother Trading Dashboard
Populates the database with realistic sample data for testing
"""

import duckdb
import random
from datetime import datetime, timedelta
from pathlib import Path

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "bigbrother.duckdb"

# Sample symbols for different sectors
SECTOR_SYMBOLS = {
    10: ['XLE', 'CVX', 'XOM', 'COP'],  # Energy
    15: ['XLB', 'DD', 'DOW', 'LYB'],   # Materials
    20: ['XLI', 'CAT', 'BA', 'GE'],    # Industrials
    25: ['XLY', 'AMZN', 'TSLA', 'HD'], # Consumer Discretionary
    30: ['XLP', 'PG', 'KO', 'WMT'],    # Consumer Staples
    35: ['XLV', 'JNJ', 'UNH', 'PFE'],  # Health Care
    40: ['XLF', 'JPM', 'BAC', 'WFC'],  # Financials
    45: ['XLK', 'AAPL', 'MSFT', 'NVDA'], # Information Technology
    50: ['XLC', 'GOOGL', 'META', 'DIS'], # Communication Services
    55: ['XLU', 'NEE', 'DUK', 'SO'],   # Utilities
    60: ['XLRE', 'AMT', 'PLD', 'SPG']  # Real Estate
}

STRATEGIES = [
    'SECTOR_ROTATION',
    'MOMENTUM',
    'MEAN_REVERSION',
    'EMPLOYMENT_SIGNAL',
    'MANUAL'
]

def generate_positions(conn, num_positions=20):
    """Generate sample positions"""
    print(f"Generating {num_positions} sample positions...")

    positions_data = []

    # Get current max ID
    max_id_result = conn.execute("SELECT COALESCE(MAX(id), 0) FROM positions").fetchone()
    current_id = max_id_result[0] + 1

    for i in range(num_positions):
        # Random sector and symbol
        sector_code = random.choice(list(SECTOR_SYMBOLS.keys()))
        symbol = random.choice(SECTOR_SYMBOLS[sector_code])

        # Random position details
        quantity = random.uniform(10, 500)
        avg_cost = random.uniform(50, 500)
        current_price = avg_cost * random.uniform(0.85, 1.15)  # -15% to +15% from entry
        market_value = quantity * current_price
        unrealized_pnl = (current_price - avg_cost) * quantity

        # Bot management
        is_bot_managed = random.choice([True, False])
        managed_by = 'BOT' if is_bot_managed else 'MANUAL'
        bot_strategy = random.choice(STRATEGIES) if is_bot_managed else None

        # Timestamps
        opened_at = datetime.now() - timedelta(days=random.randint(1, 90))

        positions_data.append({
            'id': current_id,
            'account_id': f'ACC{random.randint(1000, 9999)}',
            'symbol': symbol,
            'quantity': quantity,
            'avg_cost': avg_cost,
            'current_price': current_price,
            'market_value': market_value,
            'unrealized_pnl': unrealized_pnl,
            'is_bot_managed': is_bot_managed,
            'managed_by': managed_by,
            'bot_strategy': bot_strategy,
            'opened_at': opened_at,
            'opened_by': managed_by
        })
        current_id += 1

    # Insert positions
    conn.executemany("""
        INSERT INTO positions (
            id, account_id, symbol, quantity, avg_cost, current_price,
            market_value, unrealized_pnl, is_bot_managed, managed_by,
            bot_strategy, opened_at, opened_by
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [(
        p['id'], p['account_id'], p['symbol'], p['quantity'], p['avg_cost'], p['current_price'],
        p['market_value'], p['unrealized_pnl'], p['is_bot_managed'], p['managed_by'],
        p['bot_strategy'], p['opened_at'], p['opened_by']
    ) for p in positions_data])

    print(f"✓ Created {num_positions} positions")
    return positions_data

def generate_positions_history(conn, positions_data, days=30):
    """Generate historical position snapshots"""
    print(f"Generating position history for last {days} days...")

    history_data = []
    num_snapshots = 0

    for position in positions_data:
        # Create snapshots at different time points
        num_snaps = random.randint(5, 15)  # 5-15 snapshots per position

        for snap in range(num_snaps):
            days_ago = random.randint(0, days)
            timestamp = datetime.now() - timedelta(days=days_ago, hours=random.randint(0, 23))

            # Price variation over time
            price_variation = random.uniform(0.90, 1.10)
            current_price = position['avg_cost'] * price_variation
            unrealized_pnl = (current_price - position['avg_cost']) * position['quantity']

            history_data.append({
                'timestamp': timestamp,
                'symbol': position['symbol'],
                'quantity': position['quantity'],
                'average_price': position['avg_cost'],
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl,
                'is_bot_managed': position['is_bot_managed'],
                'strategy': position['bot_strategy']
            })
            num_snapshots += 1

    # Insert history
    conn.executemany("""
        INSERT INTO positions_history (
            timestamp, symbol, quantity, average_price, current_price,
            unrealized_pnl, is_bot_managed, strategy
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [(
        h['timestamp'], h['symbol'], h['quantity'], h['average_price'],
        h['current_price'], h['unrealized_pnl'], h['is_bot_managed'], h['strategy']
    ) for h in history_data])

    print(f"✓ Created {num_snapshots} history snapshots")

def check_data(conn):
    """Check the generated data"""
    print("\n=== Data Summary ===")

    # Positions
    result = conn.execute("SELECT COUNT(*) as count FROM positions").fetchone()
    print(f"Positions: {result[0]}")

    # Positions History
    result = conn.execute("SELECT COUNT(*) as count FROM positions_history").fetchone()
    print(f"Position History: {result[0]}")

    # Sectors
    result = conn.execute("SELECT COUNT(*) as count FROM sectors").fetchone()
    print(f"Sectors: {result[0]}")

    # Sector Employment
    result = conn.execute("SELECT COUNT(*) as count FROM sector_employment").fetchone()
    print(f"Sector Employment Records: {result[0]}")

    # Sample position
    print("\n=== Sample Position ===")
    result = conn.execute("""
        SELECT symbol, quantity, avg_cost, current_price, unrealized_pnl, bot_strategy
        FROM positions
        LIMIT 1
    """).fetchone()
    if result:
        print(f"Symbol: {result[0]}")
        print(f"Quantity: {result[1]:.2f}")
        print(f"Avg Cost: ${result[2]:.2f}")
        print(f"Current Price: ${result[3]:.2f}")
        print(f"Unrealized P&L: ${result[4]:.2f}")
        print(f"Strategy: {result[5]}")

    # P&L Summary
    print("\n=== P&L Summary ===")
    result = conn.execute("""
        SELECT
            COUNT(*) as total_positions,
            SUM(unrealized_pnl) as total_pnl,
            AVG(unrealized_pnl) as avg_pnl,
            SUM(CASE WHEN unrealized_pnl > 0 THEN 1 ELSE 0 END) as winners,
            SUM(CASE WHEN unrealized_pnl < 0 THEN 1 ELSE 0 END) as losers
        FROM positions
    """).fetchone()
    if result:
        print(f"Total Positions: {result[0]}")
        print(f"Total P&L: ${result[1]:.2f}")
        print(f"Average P&L: ${result[2]:.2f}")
        print(f"Winners: {result[3]}")
        print(f"Losers: {result[4]}")

def clear_existing_data(conn):
    """Clear existing sample data"""
    print("Clearing existing data...")

    try:
        conn.execute("DELETE FROM positions_history")
        print("✓ Cleared positions_history")
    except Exception as e:
        print(f"  Note: {e}")

    try:
        conn.execute("DELETE FROM positions")
        print("✓ Cleared positions")
    except Exception as e:
        print(f"  Note: {e}")

def main():
    """Main function to generate sample data"""
    print("BigBrother Trading Dashboard - Sample Data Generator")
    print("=" * 60)

    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        print("Please ensure the database exists before running this script.")
        return

    print(f"Connecting to database: {DB_PATH}")
    conn = duckdb.connect(str(DB_PATH))

    # Ask user if they want to clear existing data
    print("\nThis will add sample data to the database.")
    response = input("Do you want to clear existing positions first? (y/N): ").strip().lower()

    if response == 'y':
        clear_existing_data(conn)

    print("\nGenerating sample data...")

    # Generate positions
    positions_data = generate_positions(conn, num_positions=25)

    # Generate history
    generate_positions_history(conn, positions_data, days=30)

    # Commit changes
    conn.commit()

    # Check the data
    check_data(conn)

    conn.close()

    print("\n✓ Sample data generation complete!")
    print("\nYou can now run the dashboard:")
    print("  cd dashboard")
    print("  uv run streamlit run app.py")

if __name__ == "__main__":
    main()
