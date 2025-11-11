#!/usr/bin/env python3
"""
Phase 5 End-of-Day Shutdown Script

Gracefully shuts down all Phase 5 trading processes:
- Stops trading engine (bigbrother)
- Stops dashboard (streamlit)
- Generates end-of-day reports
- Calculates taxes for closed trades
- Backs up database
- Clean shutdown confirmation

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10

Usage:
    uv run python scripts/phase5_shutdown.py
    uv run python scripts/phase5_shutdown.py --force      # Skip confirmations
    uv run python scripts/phase5_shutdown.py --no-backup  # Skip database backup
"""

import subprocess
import signal
import psutil
import time
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import sys

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

class Phase5Shutdown:
    def __init__(self, force=False, no_backup=False):
        self.base_dir = Path(__file__).parent.parent
        self.force = force
        self.no_backup = no_backup
        self.shutdown_time = datetime.now()
        self.processes_stopped = []
        self.processes_failed = []

    def confirm_shutdown(self):
        """Ask user to confirm shutdown"""
        if self.force:
            return True

        print_header("Phase 5 End-of-Day Shutdown")
        print_warning("This will stop all trading processes:")
        print("   • Trading engine (bigbrother)")
        print("   • Dashboard (streamlit)")
        print("   • News ingestion (if running)")
        print("   • Any background processes")
        print()

        response = input(f"{Colors.YELLOW}Continue with shutdown? (yes/no): {Colors.END}")
        return response.lower() in ['yes', 'y']

    def find_processes(self):
        """Find all Phase 5 related processes"""
        print_header("Step 1: Identifying Running Processes")

        process_patterns = [
            ("bigbrother", "Trading Engine"),
            ("streamlit", "Dashboard"),
            ("news_ingestion", "News Ingestion"),
            ("python.*phase5", "Phase 5 Scripts"),
        ]

        found_processes = []

        for pattern, description in process_patterns:
            matching_procs = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if pattern in cmdline or pattern in proc.info['name']:
                        matching_procs.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': cmdline[:80],
                            'description': description
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            if matching_procs:
                print(f"\n{Colors.BOLD}{description}:{Colors.END}")
                for proc in matching_procs:
                    print(f"   PID {proc['pid']:6d} | {proc['name']:15s} | {proc['cmdline']}")
                    found_processes.append(proc)
            else:
                print_info(f"No {description.lower()} processes found")

        return found_processes

    def stop_process(self, proc_info, timeout=10):
        """Stop a single process gracefully"""
        pid = proc_info['pid']
        name = proc_info['description']

        try:
            process = psutil.Process(pid)

            # Try SIGTERM first (graceful)
            print(f"   Stopping {name} (PID {pid})...", end=" ", flush=True)
            process.send_signal(signal.SIGTERM)

            # Wait for process to terminate
            try:
                process.wait(timeout=timeout)
                print_success("Stopped gracefully")
                self.processes_stopped.append(proc_info)
                return True
            except psutil.TimeoutExpired:
                # Force kill if timeout
                print_warning("Timeout, force killing...")
                process.kill()
                process.wait(timeout=2)
                print_success("Killed")
                self.processes_stopped.append(proc_info)
                return True

        except psutil.NoSuchProcess:
            print_info("Already stopped")
            return True
        except Exception as e:
            print_error(f"Failed: {e}")
            self.processes_failed.append(proc_info)
            return False

    def stop_all_processes(self, processes):
        """Stop all identified processes"""
        if not processes:
            print_info("No processes to stop")
            return True

        print_header("Step 2: Stopping Processes")

        all_stopped = True
        for proc in processes:
            if not self.stop_process(proc):
                all_stopped = False
            time.sleep(0.5)  # Brief pause between stops

        return all_stopped

    def generate_eod_report(self):
        """Generate end-of-day report"""
        print_header("Step 3: Generating End-of-Day Report")

        try:
            import duckdb
            db_path = self.base_dir / "data" / "bigbrother.duckdb"

            if not db_path.exists():
                print_warning("Database not found - skipping report")
                return

            conn = duckdb.connect(str(db_path))

            # Get trading activity for today
            today = self.shutdown_time.strftime('%Y-%m-%d')

            print(f"\n{Colors.BOLD}Trading Activity for {today}:{Colors.END}")

            # Closed trades today
            trades_today = conn.execute(f"""
                SELECT COUNT(*), SUM(net_pnl_after_tax)
                FROM tax_records
                WHERE DATE(exit_time) = '{today}'
            """).fetchone()

            if trades_today and trades_today[0] > 0:
                print(f"   Trades closed: {trades_today[0]}")
                print(f"   Net P&L (after tax): ${trades_today[1]:,.2f}")
            else:
                print_info("No trades closed today")

            # Open positions
            open_positions = conn.execute("""
                SELECT COUNT(*), SUM(current_price * quantity), SUM(unrealized_pnl)
                FROM positions
                WHERE quantity > 0 AND is_bot_managed = true
            """).fetchone()

            if open_positions and open_positions[0] > 0:
                print(f"\n{Colors.BOLD}Open Bot Positions:{Colors.END}")
                print(f"   Count: {open_positions[0]}")
                print(f"   Value: ${open_positions[1]:,.2f}")
                print(f"   Unrealized P&L: ${open_positions[2]:,.2f}")
            else:
                print_info("No open bot positions")

            # YTD summary
            ytd = conn.execute("SELECT * FROM v_ytd_tax_summary").fetchone()
            if ytd and ytd[0]:
                print(f"\n{Colors.BOLD}2025 YTD Summary:{Colors.END}")
                print(f"   Total trades: {ytd[6]}")
                print(f"   Gross P&L: ${ytd[0]:,.2f}")
                print(f"   Net after tax: ${ytd[4]:,.2f}")
                print(f"   Tax owed: ${ytd[3]:,.2f}")

                # Win rate
                wins = conn.execute("SELECT COUNT(*) FROM tax_records WHERE net_pnl_after_tax > 0").fetchone()[0]
                if ytd[6] > 0:
                    win_rate = (wins / ytd[6]) * 100
                    print(f"   Win rate: {win_rate:.1f}% ({wins}/{ytd[6]})")

            conn.close()
            print_success("Report generated")

        except Exception as e:
            print_error(f"Report generation failed: {e}")

    def calculate_taxes(self):
        """Calculate taxes for today's closed trades"""
        print_header("Step 4: Tax Calculation")

        try:
            result = subprocess.run(
                ["uv", "run", "python", "scripts/monitoring/calculate_taxes.py"],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                print_success("Taxes calculated for closed trades")
                # Show relevant output
                for line in result.stdout.split('\n'):
                    if '✅' in line or 'YTD' in line or 'Tax' in line:
                        print(f"   {line}")
            else:
                print_warning("Tax calculation had issues")
                if result.stderr:
                    print(f"   {result.stderr[:200]}")

        except subprocess.TimeoutExpired:
            print_warning("Tax calculation timed out")
        except Exception as e:
            print_error(f"Tax calculation failed: {e}")

    def backup_database(self):
        """Backup database"""
        if self.no_backup:
            print_info("Database backup skipped (--no-backup flag)")
            return

        print_header("Step 5: Database Backup")

        db_path = self.base_dir / "data" / "bigbrother.duckdb"
        if not db_path.exists():
            print_warning("Database not found - skipping backup")
            return

        # Create backup filename with timestamp
        backup_dir = self.base_dir / "data" / "backups"
        backup_dir.mkdir(exist_ok=True)

        backup_name = f"bigbrother_{self.shutdown_time.strftime('%Y%m%d_%H%M%S')}.duckdb"
        backup_path = backup_dir / backup_name

        try:
            print(f"   Backing up to: {backup_name}", end=" ", flush=True)
            shutil.copy2(db_path, backup_path)

            file_size = backup_path.stat().st_size / (1024 * 1024)  # MB
            print_success(f"OK ({file_size:.1f} MB)")

            # Keep only last 7 days of backups
            cutoff = self.shutdown_time.timestamp() - (7 * 24 * 60 * 60)
            for old_backup in backup_dir.glob("bigbrother_*.duckdb"):
                if old_backup.stat().st_mtime < cutoff:
                    old_backup.unlink()
                    print_info(f"Removed old backup: {old_backup.name}")

        except Exception as e:
            print_error(f"Backup failed: {e}")

    def cleanup_temp_files(self):
        """Clean up temporary files"""
        print_header("Step 6: Cleanup")

        # Clean up temp files
        temp_patterns = [
            "*.log.1",
            "*.tmp",
            "__pycache__",
        ]

        cleaned = 0
        for pattern in temp_patterns:
            for temp_file in self.base_dir.rglob(pattern):
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                        cleaned += 1
                    elif temp_file.is_dir():
                        shutil.rmtree(temp_file)
                        cleaned += 1
                except Exception:
                    pass

        if cleaned > 0:
            print_success(f"Cleaned {cleaned} temporary files")
        else:
            print_info("No cleanup needed")

    def generate_shutdown_summary(self):
        """Generate final shutdown summary"""
        print_header("Shutdown Summary")

        print(f"\n{Colors.BOLD}Processes Stopped:{Colors.END} {len(self.processes_stopped)}")
        for proc in self.processes_stopped:
            print(f"   • {proc['description']} (PID {proc['pid']})")

        if self.processes_failed:
            print(f"\n{Colors.BOLD}Processes Failed:{Colors.END} {len(self.processes_failed)}")
            for proc in self.processes_failed:
                print(f"   • {proc['description']} (PID {proc['pid']})")

        print(f"\n{Colors.BOLD}Shutdown Time:{Colors.END}")
        print(f"   {self.shutdown_time.strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"\n{Colors.BOLD}Next Session:{Colors.END}")
        print("   1. Run: uv run python scripts/phase5_setup.py --quick")
        print("   2. Start dashboard: uv run streamlit run dashboard/app.py")
        print("   3. Start trading: ./build/bigbrother")

        if len(self.processes_stopped) > 0 and len(self.processes_failed) == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✅ Clean shutdown complete!{Colors.END}")
            return 0
        elif len(self.processes_failed) > 0:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  Shutdown complete with warnings{Colors.END}")
            return 1
        else:
            print(f"\n{Colors.BLUE}{Colors.BOLD}ℹ️  No processes were running{Colors.END}")
            return 0

    def run(self):
        """Run complete shutdown sequence"""
        if not self.confirm_shutdown():
            print_info("Shutdown cancelled")
            return 0

        # Execute shutdown sequence
        processes = self.find_processes()
        self.stop_all_processes(processes)
        self.generate_eod_report()
        self.calculate_taxes()
        self.backup_database()
        self.cleanup_temp_files()

        return self.generate_shutdown_summary()

def main():
    parser = argparse.ArgumentParser(
        description='Phase 5 End-of-Day Shutdown - Stop all trading processes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/phase5_shutdown.py              # Interactive shutdown
  uv run python scripts/phase5_shutdown.py --force      # Skip confirmations
  uv run python scripts/phase5_shutdown.py --no-backup  # Skip database backup

Shutdown sequence:
  1. Identify running processes (bigbrother, streamlit)
  2. Stop all processes gracefully (SIGTERM)
  3. Generate end-of-day trading report
  4. Calculate taxes for closed trades
  5. Backup database (last 7 days kept)
  6. Clean up temporary files
  7. Show shutdown summary

Use this script every day at market close to ensure clean shutdown.
        """
    )

    parser.add_argument('--force', action='store_true',
                       help='Skip confirmation prompts')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip database backup')

    args = parser.parse_args()

    shutdown = Phase5Shutdown(force=args.force, no_backup=args.no_backup)
    sys.exit(shutdown.run())

if __name__ == "__main__":
    main()
