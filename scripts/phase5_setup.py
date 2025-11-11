#!/usr/bin/env python3
"""
Phase 5 Setup - Complete Paper Trading Initialization

One script to handle all Phase 5 setup tasks:
- OAuth token management (with automatic refresh)
- Tax configuration verification
- Database initialization
- System health checks
- Schwab API connectivity testing
- Dashboard readiness verification
- Optional auto-start of dashboard and trading engine

Features:
- Automatic token refresh when expired (uses refresh_token)
- Complete system verification in one command
- Portable across Unix systems (auto-detects paths)
- Color-coded status output
- Comprehensive final report

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10

Usage:
    uv run python scripts/phase5_setup.py                    # Full setup
    uv run python scripts/phase5_setup.py --skip-oauth       # Skip OAuth check
    uv run python scripts/phase5_setup.py --quick            # Quick verification
    uv run python scripts/phase5_setup.py --start-all        # Setup + auto-start services
    uv run python scripts/phase5_setup.py --quick --start-all # Quick check + start services
"""

import json
import sys
import os
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
import subprocess
import shutil
import yaml
import requests
import base64

# Color output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")

def run_command(cmd, description):
    """Run a command and return success status"""
    try:
        print(f"   Running: {description}...", end=" ", flush=True)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print_success("OK")
            return True, result.stdout
        else:
            print_error(f"FAILED")
            if result.stderr:
                print(f"      Error: {result.stderr[:200]}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print_error("TIMEOUT")
        return False, "Command timed out"
    except Exception as e:
        print_error(f"ERROR: {e}")
        return False, str(e)

class Phase5Setup:
    def __init__(self, skip_oauth=False, quick=False, start_dashboard=False, start_trading=False):
        self.base_dir = Path(__file__).parent.parent
        self.skip_oauth = skip_oauth
        self.start_dashboard = start_dashboard
        self.start_trading = start_trading
        self.quick = quick
        self.checks_passed = []
        self.checks_failed = []
        self.warnings = []

    def refresh_oauth_token(self):
        """Refresh expired OAuth token using refresh_token"""
        try:
            # Load app configuration
            app_config_file = self.base_dir / "configs" / "schwab_app_config.yaml"
            token_file = self.base_dir / "configs" / "schwab_tokens.json"

            if not app_config_file.exists():
                print_error(f"App config not found: {app_config_file}")
                return False

            with open(app_config_file, 'r') as f:
                app_config = yaml.safe_load(f)

            app_key = app_config['app_key']
            app_secret = app_config['app_secret']

            # Load current tokens
            with open(token_file, 'r') as f:
                token_data = json.load(f)

            refresh_token = token_data['token']['refresh_token']

            print_info("Refreshing OAuth token...")

            # Prepare refresh request
            token_url = "https://api.schwabapi.com/v1/oauth/token"

            # Create Basic Auth header
            credentials = f"{app_key}:{app_secret}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()

            headers = {
                "Authorization": f"Basic {encoded_credentials}",
                "Content-Type": "application/x-www-form-urlencoded"
            }

            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token
            }

            # Make refresh request
            response = requests.post(token_url, headers=headers, data=data, timeout=10)

            if response.status_code == 200:
                new_tokens = response.json()

                # Update token file
                token_data['creation_timestamp'] = int(datetime.now().timestamp())
                token_data['token']['access_token'] = new_tokens['access_token']
                token_data['token']['refresh_token'] = new_tokens['refresh_token']
                token_data['token']['expires_in'] = new_tokens['expires_in']
                token_data['token']['expires_at'] = int(datetime.now().timestamp()) + new_tokens['expires_in']

                # Save updated tokens
                with open(token_file, 'w') as f:
                    json.dump(token_data, f, indent=4)

                expires_dt = datetime.fromtimestamp(token_data['token']['expires_at'])
                minutes_valid = new_tokens['expires_in'] // 60

                print_success(f"Token refreshed! Valid for {minutes_valid} minutes")
                print(f"      New expiration: {expires_dt}")
                return True
            else:
                print_error(f"Token refresh failed (status {response.status_code})")
                if response.text:
                    print(f"      Response: {response.text[:200]}")
                return False

        except Exception as e:
            print_error(f"Token refresh error: {e}")
            return False

    def check_oauth_tokens(self):
        """Check OAuth token status and offer refresh"""
        print_header("Step 1: OAuth Token Management")

        token_file = self.base_dir / "configs" / "schwab_tokens.json"

        if not token_file.exists():
            print_error(f"Token file not found: {token_file}")
            print_info("Run OAuth authentication first:")
            print(f"   uv run python scripts/run_schwab_oauth_interactive.py")
            self.checks_failed.append("OAuth tokens not found")
            return False

        try:
            with open(token_file) as f:
                tokens = json.load(f)

            access_exp = datetime.fromtimestamp(tokens['token']['expires_at'], tz=timezone.utc)
            now = datetime.now(timezone.utc)
            hours_left = (access_exp - now).total_seconds() / 3600

            print(f"   Access Token Status:")
            print(f"      Expires: {access_exp}")
            print(f"      Valid for: {hours_left:.1f} hours")

            if hours_left > 0:
                print_success(f"Access token valid for {hours_left:.1f} hours")
                self.checks_passed.append("OAuth access token valid")
            else:
                print_warning("Access token expired - needs refresh")
                self.warnings.append("OAuth token expired")

                if not self.skip_oauth:
                    # Attempt automatic token refresh
                    if self.refresh_oauth_token():
                        self.checks_passed.append("OAuth token refreshed automatically")
                        # Re-check token after refresh
                        with open(token_file) as f:
                            tokens = json.load(f)
                        access_exp = datetime.fromtimestamp(tokens['token']['expires_at'], tz=timezone.utc)
                        now = datetime.now(timezone.utc)
                        hours_left = (access_exp - now).total_seconds() / 3600
                        print_success(f"Token now valid for {hours_left:.1f} hours")
                        return True
                    else:
                        print_warning("Automatic refresh failed - manual refresh needed:")
                        print_info("Run: uv run python scripts/run_schwab_oauth_interactive.py")
                        self.checks_failed.append("Token refresh failed")
                        return False

            # Check refresh token
            if 'refresh_token_expires_at' in tokens['token']:
                refresh_exp = datetime.fromtimestamp(tokens['token']['refresh_token_expires_at'], tz=timezone.utc)
                days_left = (refresh_exp - now).total_seconds() / 86400

                print(f"\n   Refresh Token Status:")
                print(f"      Expires: {refresh_exp}")
                print(f"      Valid for: {days_left:.1f} days")

                if days_left > 0:
                    print_success(f"Refresh token valid for {days_left:.1f} days")
                else:
                    print_error("Refresh token expired - full re-authentication needed")
                    self.checks_failed.append("Refresh token expired")
                    return False

            return True

        except Exception as e:
            print_error(f"Error checking tokens: {e}")
            self.checks_failed.append("OAuth token check failed")
            return False

    def verify_tax_configuration(self):
        """Verify tax configuration for married filing jointly"""
        print_header("Step 2: Tax Configuration Verification")

        try:
            import duckdb
            db_path = self.base_dir / "data" / "bigbrother.duckdb"

            if not db_path.exists():
                print_warning(f"Database not found: {db_path}")
                print_info("Database will be created when system starts")
                self.warnings.append("Database not initialized yet")
                return True

            conn = duckdb.connect(str(db_path))

            # Check tax_config table
            try:
                config = conn.execute("""
                    SELECT
                        base_annual_income,
                        tax_year,
                        short_term_rate,
                        long_term_rate,
                        state_tax_rate,
                        medicare_surtax,
                        filing_status
                    FROM tax_config WHERE id = 1
                """).fetchone()

                if config:
                    print(f"   Tax Configuration:")
                    print(f"      Filing Status: {config[6] or 'Not set'}")
                    print(f"      Tax Year: {config[1]}")
                    print(f"      Base Income: ${config[0]:,.0f}")
                    print(f"      Short-term rate: {config[2]*100:.1f}% federal")
                    print(f"      Long-term rate: {config[3]*100:.1f}% federal")
                    print(f"      State tax: {config[4]*100:.1f}%")
                    print(f"      Medicare surtax: {config[5]*100:.1f}%")

                    effective_st = (config[2] + config[4] + config[5]) * 100
                    effective_lt = (config[3] + config[4] + config[5]) * 100

                    print(f"\n   Effective Rates:")
                    print(f"      Short-term: {effective_st:.1f}%")
                    print(f"      Long-term: {effective_lt:.1f}%")

                    # Verify correct married filing jointly rates
                    if config[2] == 0.24 and config[3] == 0.15 and config[6] == 'married_joint':
                        print_success("Tax configuration correct (Married Filing Jointly)")
                        self.checks_passed.append("Tax rates configured correctly")
                    else:
                        print_warning("Tax configuration may need update")
                        print_info("Expected: 24% ST, 15% LT for married filing jointly")
                        self.warnings.append("Tax rates may need adjustment")
                else:
                    print_warning("Tax configuration not initialized")
                    print_info("Will initialize with defaults")
                    self.warnings.append("Tax config needs initialization")

            except Exception as e:
                print_warning(f"Tax config table not found: {e}")
                print_info("Will be created during setup")
                self.warnings.append("Tax tables need creation")

            conn.close()
            return True

        except Exception as e:
            print_error(f"Error checking tax configuration: {e}")
            self.checks_failed.append("Tax config check failed")
            return False

    def initialize_tax_database(self):
        """Initialize or verify tax database"""
        print_header("Step 3: Tax Database Initialization")

        if self.quick:
            print_info("Skipping in quick mode")
            return True

        # Run setup script
        success, output = run_command(
            f"cd {self.base_dir} && uv run python scripts/monitoring/setup_tax_database.py",
            "Tax database setup"
        )

        if success:
            self.checks_passed.append("Tax database initialized")
            return True
        else:
            self.checks_failed.append("Tax database initialization failed")
            return False

    def initialize_news_database(self):
        """Initialize or verify news database"""
        print_header("Step 3.5: News Database Initialization")

        if self.quick:
            print_info("Skipping in quick mode")
            return True

        # Run news setup script
        success, output = run_command(
            f"cd {self.base_dir} && uv run python scripts/monitoring/setup_news_database.py",
            "News database setup"
        )

        if success:
            self.checks_passed.append("News database initialized")
            return True
        else:
            self.warnings.append("News database initialization failed (non-critical)")
            print_warning("News features may not be available")
            return True  # Non-critical, don't fail setup

    def verify_paper_trading_config(self):
        """Verify paper trading configuration"""
        print_header("Step 4: Paper Trading Configuration")

        config_file = self.base_dir / "configs" / "config.yaml"
        paper_config = self.base_dir / "configs" / "paper_trading.yaml"

        # Check main config
        if config_file.exists():
            with open(config_file) as f:
                content = f.read()
                if 'paper_trading: true' in content or 'paper_trading:true' in content:
                    print_success("Paper trading enabled in config.yaml")
                    self.checks_passed.append("Paper trading enabled")
                else:
                    print_warning("Paper trading may not be enabled in config.yaml")
                    self.warnings.append("Verify paper_trading: true in config")
        else:
            print_warning(f"Config file not found: {config_file}")
            self.warnings.append("Main config not found")

        # Check paper trading config
        if paper_config.exists():
            print_success(f"Paper trading config found: {paper_config.name}")

            with open(paper_config) as f:
                content = f.read()
                print_info("Paper trading limits:")
                # Parse key values
                for line in content.split('\n'):
                    if 'max_position_size:' in line or 'max_daily_loss:' in line or 'max_concurrent_positions:' in line:
                        print(f"      {line.strip()}")

            self.checks_passed.append("Paper trading config verified")
        else:
            print_warning("Paper trading config not found")
            self.warnings.append("Paper trading config missing")

        return True

    def test_schwab_api(self):
        """Test Schwab API connectivity"""
        print_header("Step 5: Schwab API Connectivity Test")

        if self.skip_oauth:
            print_info("Skipping API test (--skip-oauth flag)")
            return True

        # Try to get a simple quote
        try:
            import requests

            token_file = self.base_dir / "configs" / "schwab_tokens.json"
            if not token_file.exists():
                print_warning("Token file not found - skipping API test")
                return True

            with open(token_file) as f:
                tokens = json.load(f)

            access_token = tokens['token']['access_token']

            # Test quote endpoint
            print_info("Testing market data API...")
            url = "https://api.schwabapi.com/marketdata/v1/SPY/quotes"
            headers = {'Authorization': f'Bearer {access_token}'}

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                quote = response.json()['SPY']['quote']
                print_success("API connectivity verified")
                print(f"      SPY Quote: ${quote['lastPrice']:.2f}")
                print(f"      Volume: {quote['totalVolume']:,}")
                self.checks_passed.append("Schwab API connectivity verified")
                return True
            elif response.status_code == 401:
                print_warning("API authentication failed - token may be expired")
                self.warnings.append("API token needs refresh")
                return False
            else:
                print_warning(f"API returned status {response.status_code}")
                self.warnings.append("API connectivity issues")
                return False

        except Exception as e:
            print_warning(f"API test failed: {e}")
            print_info("API connectivity will be tested again when trading starts")
            self.warnings.append("API test inconclusive")
            return True

    def verify_system_components(self):
        """Verify key system components"""
        print_header("Step 6: System Components Check")

        components = [
            ("Dashboard", self.base_dir / "dashboard" / "app.py"),
            ("Orders Manager", self.base_dir / "src" / "schwab_api" / "orders_manager.cppm"),
            ("Risk Manager", self.base_dir / "src" / "risk_management" / "risk_management.cppm"),
            ("Tax Calculator", self.base_dir / "scripts" / "monitoring" / "calculate_taxes.py"),
            ("Config", self.base_dir / "configs" / "config.yaml"),
        ]

        all_found = True
        for name, path in components:
            if path.exists():
                print_success(f"{name} found")
            else:
                print_error(f"{name} not found: {path}")
                all_found = False

        if all_found:
            self.checks_passed.append("All system components present")
        else:
            self.checks_failed.append("Some components missing")

        return all_found

    def start_dashboard_process(self):
        """Start the Streamlit dashboard"""
        print_header("Starting Dashboard")

        dashboard_path = self.base_dir / "dashboard" / "app.py"
        if not dashboard_path.exists():
            print_error(f"Dashboard not found: {dashboard_path}")
            return False

        try:
            # Kill existing streamlit processes to avoid port conflicts
            print("   Stopping existing dashboard processes...", end=" ", flush=True)
            subprocess.run(["pkill", "-f", "streamlit"],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
            time.sleep(1)  # Give processes time to terminate
            print("done")

            print("   Starting Streamlit dashboard...", end=" ", flush=True)
            # Start dashboard in background
            subprocess.Popen(
                ["uv", "run", "streamlit", "run", "app.py",
                 "--server.headless=true", "--server.port=8501"],
                cwd=self.base_dir / "dashboard",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            time.sleep(3)  # Give it time to start
            print_success("Started")
            print_info("Dashboard available at: http://localhost:8501")
            return True
        except Exception as e:
            print_error(f"Failed: {e}")
            return False

    def start_trading_engine(self):
        """Start the trading engine"""
        print_header("Starting Trading Engine")

        bigbrother_path = self.base_dir / "build" / "bin" / "bigbrother"
        if not bigbrother_path.exists():
            print_error(f"Trading engine not found: {bigbrother_path}")
            print_info("Build the project first: cd build && ninja bigbrother")
            return False

        try:
            # Kill existing bigbrother processes to avoid duplicates
            print("   Stopping existing trading engine processes...", end=" ", flush=True)
            subprocess.run(["pkill", "-f", "bigbrother"],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
            time.sleep(1)  # Give processes time to terminate
            print("done")

            print("   Starting trading engine...", end=" ", flush=True)

            # Auto-detect library paths (portable across systems)
            # Priority: ENV LIBCXX_PATH > /usr/local/lib > /usr/lib
            lib_paths = []

            # Add architecture-specific paths
            for base in ["/usr/local/lib", "/usr/lib"]:
                arch_path = os.path.join(base, "x86_64-unknown-linux-gnu")
                if os.path.exists(arch_path):
                    lib_paths.append(arch_path)
                if os.path.exists(base):
                    lib_paths.append(base)

            # Add existing LD_LIBRARY_PATH
            if os.environ.get("LD_LIBRARY_PATH"):
                lib_paths.append(os.environ.get("LD_LIBRARY_PATH"))

            ld_library_path = ":".join(filter(None, lib_paths))

            # Auto-detect compiler locations (portable across systems)
            # Priority: ENV CC/CXX > /usr/local/bin > /usr/bin > which
            cc = os.environ.get("CC") or shutil.which("clang") or "/usr/local/bin/clang"
            cxx = os.environ.get("CXX") or shutil.which("clang++") or "/usr/local/bin/clang++"

            env = {
                **os.environ,
                "CC": cc,
                "CXX": cxx,
                "LD_LIBRARY_PATH": ld_library_path,
                "PATH": os.environ.get("PATH", "")
            }

            subprocess.Popen(
                [str(bigbrother_path)],
                cwd=self.base_dir,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            time.sleep(2)  # Give it time to start
            print_success("Started")
            print_info("Trading engine running in background")
            return True
        except Exception as e:
            print_error(f"Failed: {e}")
            return False

    def generate_report(self):
        """Generate final setup report"""
        print_header("Phase 5 Setup Report")

        total_checks = len(self.checks_passed) + len(self.checks_failed)
        success_rate = (len(self.checks_passed) / total_checks * 100) if total_checks > 0 else 0

        print(f"\n{Colors.BOLD}Summary:{Colors.END}")
        print(f"   Total Checks: {total_checks}")
        print(f"   {Colors.GREEN}Passed: {len(self.checks_passed)}{Colors.END}")
        print(f"   {Colors.RED}Failed: {len(self.checks_failed)}{Colors.END}")
        print(f"   {Colors.YELLOW}Warnings: {len(self.warnings)}{Colors.END}")
        print(f"   Success Rate: {success_rate:.0f}%")

        if self.checks_passed:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✅ Checks Passed:{Colors.END}")
            for check in self.checks_passed:
                print(f"   • {check}")

        if self.warnings:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  Warnings:{Colors.END}")
            for warning in self.warnings:
                print(f"   • {warning}")

        if self.checks_failed:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ Checks Failed:{Colors.END}")
            for check in self.checks_failed:
                print(f"   • {check}")

        # Overall status
        print(f"\n{Colors.BOLD}Overall Status:{Colors.END}")

        if len(self.checks_failed) == 0:
            if len(self.warnings) == 0:
                print_success("READY FOR PHASE 5 - All systems go! 🚀")

                # Start services if requested
                services_started = []
                if self.start_dashboard:
                    if self.start_dashboard_process():
                        services_started.append("Dashboard")

                if self.start_trading:
                    if self.start_trading_engine():
                        services_started.append("Trading Engine")

                # Show next steps
                print_info("Next steps:")
                if services_started:
                    print(f"\n{Colors.BOLD}✅ Started Services:{Colors.END}")
                    for service in services_started:
                        print(f"   • {service}")
                    if not self.start_dashboard:
                        print("\n   1. Start dashboard: uv run streamlit run dashboard/app.py")
                    if not self.start_trading:
                        print("   2. Start trading: ./build/bigbrother")
                    print("   3. Review execution plan: cat /tmp/phase5_execution_plan.md")
                else:
                    print("   1. Start dashboard: uv run streamlit run dashboard/app.py")
                    print("   2. Start trading: ./build/bigbrother")
                    print("   3. Review execution plan: cat /tmp/phase5_execution_plan.md")
                    print("\n   Tip: Use --start-all flag to auto-start dashboard and trading engine")

                return 0
            else:
                print_warning("MOSTLY READY - Review warnings before proceeding")
                print_info("Address warnings if possible, then proceed to Phase 5")
                return 0
        else:
            print_error("NOT READY - Critical issues must be resolved")
            print_info("Fix failed checks before starting Phase 5")
            return 1

    def run(self):
        """Run all setup steps"""
        print_header("BigBrotherAnalytics - Phase 5 Setup")
        print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Directory: {self.base_dir}")
        print(f"   Mode: {'Quick Check' if self.quick else 'Full Setup'}")

        # Run all checks
        self.check_oauth_tokens()
        self.verify_tax_configuration()

        if not self.quick:
            self.initialize_tax_database()
            self.initialize_news_database()

        self.verify_paper_trading_config()
        self.test_schwab_api()
        self.verify_system_components()

        # Generate report
        return self.generate_report()

def main():
    parser = argparse.ArgumentParser(
        description='Phase 5 Paper Trading Setup - Complete Initialization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/phase5_setup.py                    # Full setup
  uv run python scripts/phase5_setup.py --skip-oauth       # Skip OAuth check
  uv run python scripts/phase5_setup.py --quick            # Quick verification only
  uv run python scripts/phase5_setup.py --start-all        # Setup + start dashboard & trading
  uv run python scripts/phase5_setup.py --start-dashboard  # Setup + start dashboard only

This script performs:
  1. OAuth token management and verification (with automatic refresh)
  2. Tax configuration check (married filing jointly, $300K base, CA)
  3. Database initialization (tax tables)
  4. Paper trading configuration verification
  5. Schwab API connectivity test
  6. System component verification
  7. Comprehensive status report
  8. (Optional) Start dashboard and trading engine

Token Management:
  - Automatically refreshes expired access tokens using refresh_token
  - No manual intervention needed for expired tokens (< 30 min validity)
  - Warns if refresh token is expired (requires full re-authentication)
        """
    )

    parser.add_argument('--skip-oauth', action='store_true',
                       help='Skip OAuth token checks and API tests')
    parser.add_argument('--quick', action='store_true',
                       help='Quick verification only (skip initialization)')
    parser.add_argument('--start-dashboard', action='store_true',
                       help='Start Streamlit dashboard after successful setup')
    parser.add_argument('--start-trading', action='store_true',
                       help='Start trading engine after successful setup')
    parser.add_argument('--start-all', action='store_true',
                       help='Start both dashboard and trading engine after successful setup')

    args = parser.parse_args()

    # If --start-all is specified, enable both
    start_dashboard = args.start_dashboard or args.start_all
    start_trading = args.start_trading or args.start_all

    setup = Phase5Setup(
        skip_oauth=args.skip_oauth,
        quick=args.quick,
        start_dashboard=start_dashboard,
        start_trading=start_trading
    )
    return setup.run()

if __name__ == "__main__":
    sys.exit(main())
