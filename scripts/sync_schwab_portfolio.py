#!/usr/bin/env python3
"""
Sync Schwab Portfolio to Database
Fetches real positions from Schwab API and updates the database
"""

import schwab
import duckdb
from pathlib import Path
from datetime import datetime

def main():
    print("=" * 80)
    print("Syncing Schwab Portfolio to Database")
    print("=" * 80)

    # Load Schwab client
    token_path = Path('configs/schwab_tokens.json')
    app_key = '8fOTwEsi51wbdmEsn5dSYxk6Y38ZJKH09etLXf3uJgyUXcIa'
    app_secret = 'PKy7ILBYmEnxEkm0BQNt28AHtHoYs3c4y09cG51LqMsVkdZkmwuOqBBUDJWzMamT'

    print(f"Loading token from: {token_path}")
    client = schwab.auth.client_from_token_file(str(token_path), app_key, app_secret)

    # Get account numbers
    print("Fetching account numbers...")
    resp = client.get_account_numbers()
    if resp.status_code != 200:
        print(f"❌ Failed to get account numbers: {resp.status_code}")
        return 1

    accounts = resp.json()
    print(f"✅ Found {len(accounts)} account(s)")

    # Get account details with positions
    for account_info in accounts:
        account_num = account_info['accountNumber']
        account_hash = account_info['hashValue']

        print(f"\nAccount: {account_num}")
        print(f"Hash: {account_hash}")

        # Get account details with positions
        resp = client.get_account(account_hash, fields=client.Account.Fields.POSITIONS)
        if resp.status_code != 200:
            print(f"❌ Failed to get account details: {resp.status_code}")
            continue

        account_data = resp.json()['securitiesAccount']

        # Extract positions
        positions = account_data.get('positions', [])
        print(f"📊 Found {len(positions)} position(s)")

        if not positions:
            print("   No positions to sync")
            continue

        # Connect to database
        db_path = Path('data/bigbrother.duckdb')
        print(f"\nConnecting to database: {db_path}")
        conn = duckdb.connect(str(db_path))

        # Create account_balances table if it doesn't exist
        conn.execute("""
            CREATE TABLE IF NOT EXISTS account_balances (
                account_id VARCHAR PRIMARY KEY,
                liquidation_value DOUBLE,
                equity DOUBLE,
                long_market_value DOUBLE,
                margin_balance DOUBLE,
                available_funds DOUBLE,
                buying_power DOUBLE,
                updated_at TIMESTAMP
            )
        """)

        # Clear existing positions (for fresh sync)
        print("Clearing old positions...")
        conn.execute("DELETE FROM positions")

        # Insert positions
        print("Inserting positions...")
        position_id = 1
        for pos in positions:
            instrument = pos['instrument']
            symbol = instrument['symbol']
            asset_type = instrument['assetType']

            quantity = pos.get('longQuantity', 0) - pos.get('shortQuantity', 0)
            avg_price = pos.get('averagePrice', 0)
            market_value = pos.get('marketValue', 0)
            # Calculate current price from market value (more accurate than currentDayProfitLoss)
            current_price = market_value / quantity if quantity != 0 else 0

            # All real positions are manual (not bot-managed)
            is_bot_managed = False
            bot_strategy = None

            opened_at = datetime.now().isoformat()

            print(f"   {symbol}: {quantity} @ ${avg_price:.2f} = ${market_value:.2f}")

            conn.execute("""
                INSERT INTO positions (
                    id, account_id, symbol, quantity, avg_cost, current_price, market_value,
                    unrealized_pnl, is_bot_managed, bot_strategy, opened_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                position_id, account_num, symbol, quantity, avg_price, current_price, market_value,
                market_value - (avg_price * quantity),
                is_bot_managed, bot_strategy, opened_at, opened_at
            ])
            position_id += 1

        # Store account-level balances
        balances = account_data.get('currentBalances', {})
        liquidation_value = balances.get('liquidationValue', 0)
        equity = balances.get('equity', 0)
        long_market_value = balances.get('longMarketValue', 0)
        margin_balance = balances.get('marginBalance', 0)
        available_funds = balances.get('availableFunds', 0)
        buying_power = balances.get('buyingPower', 0)

        print(f"\nStoring account balances...")
        print(f"   Liquidation Value: ${liquidation_value:,.2f}")
        print(f"   Margin Balance: ${margin_balance:,.2f}")

        # Delete existing balance record for this account
        conn.execute("DELETE FROM account_balances WHERE account_id = ?", [account_num])

        # Insert updated balance
        conn.execute("""
            INSERT INTO account_balances (
                account_id, liquidation_value, equity, long_market_value,
                margin_balance, available_funds, buying_power, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            account_num, liquidation_value, equity, long_market_value,
            margin_balance, available_funds, buying_power, datetime.now().isoformat()
        ])

        conn.commit()
        conn.close()

        print(f"\n✅ Synced {len(positions)} positions to database")
        print(f"✅ Stored account balance: ${liquidation_value:,.2f}")

    print("\n" + "=" * 80)
    print("✅ Portfolio sync complete!")
    print("=" * 80)
    print("\nRefresh the dashboard to see your positions: http://localhost:8501")

    return 0

if __name__ == "__main__":
    exit(main())
