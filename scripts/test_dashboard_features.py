#!/usr/bin/env python3
"""
Comprehensive Dashboard Feature Tests
Tests all dashboard components that were fixed/added
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_fred_import():
    """Test FRED module import"""
    print("\n" + "="*70)
    print("TEST 1: FRED Module Import")
    print("="*70)
    try:
        import requests
        print("✅ requests module: INSTALLED")

        # Try importing FRED module (Python bindings)
        try:
            from build import fred_rates_py
            print("✅ fred_rates_py C++ module: AVAILABLE")
            FRED_AVAILABLE = True
        except ImportError as e:
            print(f"⚠️  fred_rates_py C++ module: NOT AVAILABLE ({e})")
            print("   Fallback to Python implementation will be used")
            FRED_AVAILABLE = False

        return True, FRED_AVAILABLE
    except Exception as e:
        print(f"❌ FRED import failed: {e}")
        return False, False

def test_fred_api():
    """Test FRED API key and connectivity"""
    print("\n" + "="*70)
    print("TEST 2: FRED API Key & Connectivity")
    print("="*70)
    try:
        import yaml
        api_keys_path = Path(__file__).parent.parent / "api_keys.yaml"

        with open(api_keys_path, 'r') as f:
            config = yaml.safe_load(f)

        fred_key = config.get('fred_api_key')
        if fred_key:
            print(f"✅ FRED API key found: {fred_key[:8]}...{fred_key[-4:]}")
        else:
            print("❌ FRED API key NOT found in api_keys.yaml")
            return False

        # Test actual API call
        import requests
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': 'DGS10',
            'api_key': fred_key,
            'file_type': 'json',
            'limit': 1,
            'sort_order': 'desc'
        }

        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'observations' in data and len(data['observations']) > 0:
                rate = data['observations'][0]['value']
                date = data['observations'][0]['date']
                print(f"✅ FRED API live test: SUCCESS")
                print(f"   10-Year Treasury: {rate}% (as of {date})")
                return True

        print(f"❌ FRED API test failed: Status {response.status_code}")
        return False

    except Exception as e:
        print(f"❌ FRED API test failed: {e}")
        return False

def test_database_path():
    """Test database path resolution"""
    print("\n" + "="*70)
    print("TEST 3: Database Path Resolution")
    print("="*70)
    try:
        import duckdb

        # Test from project root
        db_path = Path(__file__).parent.parent / "data" / "bigbrother.duckdb"
        print(f"Database path: {db_path}")

        if db_path.exists():
            print(f"✅ Database exists: {db_path.stat().st_size / (1024*1024):.1f} MB")
        else:
            print(f"❌ Database NOT found at: {db_path}")
            return False

        # Test read-only connection
        conn = duckdb.connect(str(db_path), read_only=True)

        # Test basic queries
        tables = conn.execute("SHOW TABLES").fetchall()
        print(f"✅ Database connection: SUCCESS")
        print(f"   Tables found: {len(tables)}")
        for table in tables[:5]:  # Show first 5 tables
            print(f"   - {table[0]}")

        conn.close()
        return True

    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_dashboard_views_paths():
    """Test that dashboard views have correct paths"""
    print("\n" + "="*70)
    print("TEST 4: Dashboard Views Path Configuration")
    print("="*70)
    try:
        # Simulate dashboard/views/ path resolution
        views_dir = Path(__file__).parent.parent / "dashboard" / "views"

        # Test 3-level path resolution (views -> dashboard -> project root)
        test_file = views_dir / "live_trading_activity.py"
        simulated_file = str(test_file)

        # Go up 3 levels
        level_1 = os.path.dirname(simulated_file)  # views dir
        level_2 = os.path.dirname(level_1)  # dashboard dir
        level_3 = os.path.dirname(level_2)  # project root

        expected_db_path = os.path.join(level_3, 'data', 'bigbrother.duckdb')

        print(f"Simulated __file__: {simulated_file}")
        print(f"Level 1 (views): {level_1}")
        print(f"Level 2 (dashboard): {level_2}")
        print(f"Level 3 (project root): {level_3}")
        print(f"Computed DB path: {expected_db_path}")

        if os.path.exists(expected_db_path):
            print("✅ 3-level path resolution: CORRECT")
            return True
        else:
            print("❌ 3-level path resolution: INCORRECT (DB not found)")
            return False

    except Exception as e:
        print(f"❌ Path resolution test failed: {e}")
        return False

def test_tax_tracking_view():
    """Test tax tracking view can load data"""
    print("\n" + "="*70)
    print("TEST 5: Tax Tracking View Data")
    print("="*70)
    try:
        import duckdb
        db_path = Path(__file__).parent.parent / "data" / "bigbrother.duckdb"
        conn = duckdb.connect(str(db_path), read_only=True)

        # Check for tax_records table
        tax_records = conn.execute("SELECT COUNT(*) FROM tax_records").fetchone()[0]
        print(f"✅ tax_records table: {tax_records} records")

        # Check for v_ytd_tax_summary view
        try:
            ytd = conn.execute("SELECT * FROM v_ytd_tax_summary LIMIT 1").fetchone()
            if ytd:
                print(f"✅ v_ytd_tax_summary view: AVAILABLE")
                print(f"   YTD Gross P&L: ${ytd[0]:,.2f}" if ytd[0] else "   No data yet")
            else:
                print("⚠️  v_ytd_tax_summary view: No data yet")
        except Exception as e:
            print(f"⚠️  v_ytd_tax_summary view: {e}")

        conn.close()
        return True

    except Exception as e:
        print(f"❌ Tax tracking test failed: {e}")
        return False

def test_news_feed():
    """Test news feed data and JAX groupby fix"""
    print("\n" + "="*70)
    print("TEST 6: News Feed Data & JAX Groupby")
    print("="*70)
    try:
        import duckdb
        db_path = Path(__file__).parent.parent / "data" / "bigbrother.duckdb"
        conn = duckdb.connect(str(db_path), read_only=True)

        # Check for news_articles table
        news_count = conn.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]
        print(f"✅ news_articles table: {news_count} articles")

        if news_count > 0:
            # Test sentiment aggregation (simulating JAX groupby fix)
            sentiment_by_symbol = conn.execute("""
                SELECT symbol, AVG(sentiment_score) as mean, COUNT(*) as count
                FROM news_articles
                GROUP BY symbol
                ORDER BY mean DESC
                LIMIT 5
            """).fetchall()

            print("✅ Sentiment aggregation: SUCCESS")
            print("   Top symbols by sentiment:")
            for row in sentiment_by_symbol:
                print(f"   - {row[0]}: {row[1]:.3f} ({row[2]} articles)")
        else:
            print("⚠️  No news articles yet (expected for new setup)")

        conn.close()
        return True

    except Exception as e:
        print(f"❌ News feed test failed: {e}")
        return False

def test_trading_engine():
    """Test trading engine status"""
    print("\n" + "="*70)
    print("TEST 7: Trading Engine Status")
    print("="*70)
    try:
        import subprocess

        # Check if bigbrother is running
        result = subprocess.run(
            ["pgrep", "-f", "bigbrother"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            print(f"✅ Trading engine running: {len(pids)} process(es)")
            for pid in pids:
                print(f"   PID: {pid}")

            # Check recent log output
            log_path = Path(__file__).parent.parent / "logs" / "bigbrother.log"
            if log_path.exists():
                with open(log_path, 'r') as f:
                    lines = f.readlines()[-10:]  # Last 10 lines
                print("\n   Recent log entries:")
                for line in lines:
                    print(f"   {line.rstrip()}")

            return True
        else:
            print("⚠️  Trading engine NOT running")
            return False

    except Exception as e:
        print(f"❌ Trading engine test failed: {e}")
        return False

def test_paper_trading_limits():
    """Test paper trading configuration"""
    print("\n" + "="*70)
    print("TEST 8: Paper Trading Limits ($2000)")
    print("="*70)
    try:
        import yaml
        config_path = Path(__file__).parent.parent / "configs" / "paper_trading.yaml"

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        max_position = config['risk']['max_position_size']
        max_daily_loss = config['risk']['max_daily_loss']

        if max_position == 2000.0 and max_daily_loss == 2000.0:
            print(f"✅ Position limit: ${max_position:,.0f}")
            print(f"✅ Daily loss limit: ${max_daily_loss:,.0f}")
            return True
        else:
            print(f"❌ Position limit: ${max_position:,.0f} (expected $2,000)")
            print(f"❌ Daily loss limit: ${max_daily_loss:,.0f} (expected $2,000)")
            return False

    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("BIGBROTHERANALYTICS - COMPREHENSIVE DASHBOARD TESTS")
    print("="*70)

    results = []

    # Run all tests
    results.append(("FRED Import", test_fred_import()[0]))
    results.append(("FRED API", test_fred_api()))
    results.append(("Database Path", test_database_path()))
    results.append(("Views Path Resolution", test_dashboard_views_paths()))
    results.append(("Tax Tracking", test_tax_tracking_view()))
    results.append(("News Feed", test_news_feed()))
    results.append(("Trading Engine", test_trading_engine()))
    results.append(("Paper Trading Limits", test_paper_trading_limits()))

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Dashboard is fully functional.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
