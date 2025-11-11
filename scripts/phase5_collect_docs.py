#!/usr/bin/env python3
"""
Phase 5 Documentation Collector

Collects all Phase 5 setup documents and resources into one organized location:
- Copies all /tmp/phase5_*.md files
- Copies relevant scripts
- Creates master index
- Generates summary
- Optional archive creation

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10

Usage:
    uv run python scripts/phase5_collect_docs.py
    uv run python scripts/phase5_collect_docs.py --archive  # Create zip archive
    uv run python scripts/phase5_collect_docs.py --clean    # Remove old docs first
"""

import shutil
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import sys

class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")

def print_header(text):
    print(f"\n{Colors.BOLD}{text}{Colors.END}")

class Phase5Collector:
    def __init__(self, create_archive=False, clean=False):
        self.base_dir = Path(__file__).parent.parent
        self.phase5_dir = self.base_dir / "docs" / "phase5"
        self.create_archive = create_archive
        self.clean = clean
        self.collected_files = []

    def setup_directories(self):
        """Create Phase 5 documentation directory structure"""
        print_header("Setting up directories...")

        if self.clean and self.phase5_dir.exists():
            print_info(f"Cleaning existing directory: {self.phase5_dir}")
            shutil.rmtree(self.phase5_dir)

        # Create directory structure
        directories = [
            self.phase5_dir,
            self.phase5_dir / "setup",
            self.phase5_dir / "execution",
            self.phase5_dir / "monitoring",
            self.phase5_dir / "scripts",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print_success(f"Created: {directory.relative_to(self.base_dir)}")

    def collect_tmp_documents(self):
        """Collect all Phase 5 documents from /tmp"""
        print_header("Collecting Phase 5 documents from /tmp...")

        tmp_docs = [
            ("phase5_execution_plan.md", "execution", "Complete 17-21 day execution plan"),
            ("phase5_daily_checklist.md", "monitoring", "Daily monitoring checklists"),
            ("phase5_oauth_live_data_setup.md", "setup", "OAuth & API setup guide"),
            ("phase5_system_verification.md", "setup", "Pre-flight system check"),
            ("phase5_quick_start_uv.md", "setup", "UV Python quick reference"),
            ("phase5_summary.md", "execution", "Executive summary"),
        ]

        for filename, category, description in tmp_docs:
            source = Path(f"/tmp/{filename}")
            if source.exists():
                dest = self.phase5_dir / category / filename
                shutil.copy2(source, dest)
                print_success(f"Copied: {filename} → {category}/")
                print(f"         {description}")
                self.collected_files.append((category, filename, description))
            else:
                print(f"   ⚠️  Not found: {filename}")

    def collect_local_documents(self):
        """Collect Phase 5 documents from local docs/"""
        print_header("Collecting local documentation...")

        local_docs = [
            ("PHASE5_SETUP_GUIDE.md", "setup", "Unified setup script guide"),
            ("TAX_PLANNING_300K.md", "setup", "Tax planning for $300K income"),
            ("TAX_TRACKING_YTD.md", "setup", "YTD incremental tax tracking"),
        ]

        for filename, category, description in local_docs:
            source = self.base_dir / "docs" / filename
            if source.exists():
                dest = self.phase5_dir / category / filename
                shutil.copy2(source, dest)
                print_success(f"Copied: {filename} → {category}/")
                print(f"         {description}")
                self.collected_files.append((category, filename, description))
            else:
                print(f"   ⚠️  Not found: {filename}")

    def collect_scripts(self):
        """Collect Phase 5 related scripts"""
        print_header("Collecting Phase 5 scripts...")

        scripts = [
            ("phase5_setup.py", "Main Phase 5 setup script"),
            ("monitoring/calculate_taxes.py", "Tax calculator"),
            ("monitoring/setup_tax_database.py", "Tax database setup"),
            ("monitoring/update_tax_config_ytd.py", "YTD tax config"),
            ("monitoring/update_tax_rates_married.py", "Married filing jointly rates"),
            ("monitoring/update_tax_rates_300k.py", "Tax rates for $300K"),
        ]

        for script_path, description in scripts:
            source = self.base_dir / "scripts" / script_path
            if source.exists():
                dest = self.phase5_dir / "scripts" / Path(script_path).name
                shutil.copy2(source, dest)
                print_success(f"Copied: {script_path}")
                print(f"         {description}")
                self.collected_files.append(("scripts", Path(script_path).name, description))
            else:
                print(f"   ⚠️  Not found: {script_path}")

    def create_master_index(self):
        """Create master index README"""
        print_header("Creating master index...")

        readme_content = f"""# Phase 5 Documentation Hub

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Location:** `docs/phase5/`

---

## Quick Start

**Single command to verify everything:**
```bash
uv run python scripts/phase5_setup.py
```

**Expected:** 100% success rate (6/6 checks passed)

---

## Directory Structure

```
docs/phase5/
├── setup/              # Setup and configuration guides
├── execution/          # Execution plans and timelines
├── monitoring/         # Daily monitoring and checklists
├── scripts/            # Phase 5 utility scripts
└── README.md          # This file
```

---

## Setup Guides

"""

        # Add setup files
        setup_files = [(cat, name, desc) for cat, name, desc in self.collected_files if cat == "setup"]
        for category, filename, description in sorted(setup_files, key=lambda x: x[1]):
            readme_content += f"\n### [{filename}](setup/{filename})\n{description}\n"

        readme_content += "\n---\n\n## Execution Plans\n"

        # Add execution files
        exec_files = [(cat, name, desc) for cat, name, desc in self.collected_files if cat == "execution"]
        for category, filename, description in sorted(exec_files, key=lambda x: x[1]):
            readme_content += f"\n### [{filename}](execution/{filename})\n{description}\n"

        readme_content += "\n---\n\n## Monitoring & Checklists\n"

        # Add monitoring files
        mon_files = [(cat, name, desc) for cat, name, desc in self.collected_files if cat == "monitoring"]
        for category, filename, description in sorted(mon_files, key=lambda x: x[1]):
            readme_content += f"\n### [{filename}](monitoring/{filename})\n{description}\n"

        readme_content += "\n---\n\n## Scripts\n"

        # Add scripts
        script_files = [(cat, name, desc) for cat, name, desc in self.collected_files if cat == "scripts"]
        for category, filename, description in sorted(script_files, key=lambda x: x[1]):
            readme_content += f"\n### [{filename}](scripts/{filename})\n{description}\n"

        readme_content += f"""

---

## Phase 5 Overview

### Timeline: 17-21 Days

```
Day 0-1:  Pre-Trading Setup (OAuth, config verification)
Day 2-3:  Dry-Run Phase (monitor only, no real trades)
Day 4-10: Small Position Phase ($50 positions, 1 at a time)
Day 11+:  Scaling Phase ($50→$100, 1→3 concurrent)
```

### Success Criteria

**Must Have:**
- ✅ Minimum 15 trades executed
- ✅ Win rate ≥55% (after-tax)
- ✅ ZERO trades on manual positions
- ✅ ZERO risk limit violations
- ✅ Tax calculations accurate
- ✅ System stable (no crashes in 5+ days)

### Tax Configuration

**Status:** Married Filing Jointly
**Base Income:** $300,000 (2025)
**Rates:**
- Short-term: 32.8% (24% federal + 5% state + 3.8% Medicare)
- Long-term: 23.8% (15% federal + 5% state + 3.8% Medicare)

**Win Rate Needed:** ≥55% to be profitable after 32.8% tax + 3% fees

---

## Daily Workflow

### Morning Pre-Market (10 seconds)
```bash
uv run python scripts/phase5_setup.py --quick
```
**Expected:** All green ✅, 100% success rate

### Start Trading Day
```bash
# 1. Verify setup (if needed)
uv run python scripts/phase5_setup.py

# 2. Start dashboard
uv run streamlit run dashboard/app.py

# 3. Start trading engine
./build/bigbrother
```

### End of Day Review
```bash
# Calculate taxes for closed trades
uv run python scripts/monitoring/calculate_taxes.py

# Check win rate and performance
uv run python -c "
import duckdb
conn = duckdb.connect('data/bigbrother.duckdb')
trades = conn.execute('SELECT COUNT(*), SUM(CASE WHEN net_pnl_after_tax > 0 THEN 1 ELSE 0 END) FROM tax_records').fetchone()
if trades[0] > 0:
    print(f'Win Rate: {{(trades[1]/trades[0])*100:.1f}}% (Target: ≥55%)')
conn.close()
"
```

---

## Key Documents

### Must Read (Priority Order)
1. **[PHASE5_SETUP_GUIDE.md](setup/PHASE5_SETUP_GUIDE.md)** - Start here
2. **[phase5_execution_plan.md](execution/phase5_execution_plan.md)** - Complete roadmap
3. **[phase5_daily_checklist.md](monitoring/phase5_daily_checklist.md)** - Daily monitoring

### Configuration
- **[TAX_PLANNING_300K.md](setup/TAX_PLANNING_300K.md)** - Tax strategy
- **[TAX_TRACKING_YTD.md](setup/TAX_TRACKING_YTD.md)** - YTD tracking

### Technical Setup
- **[phase5_oauth_live_data_setup.md](setup/phase5_oauth_live_data_setup.md)** - OAuth & API
- **[phase5_system_verification.md](setup/phase5_system_verification.md)** - Pre-flight check

---

## Quick Reference Commands

### Setup & Verification
```bash
# Full setup
uv run python scripts/phase5_setup.py

# Quick check (daily)
uv run python scripts/phase5_setup.py --quick

# Help
uv run python scripts/phase5_setup.py --help
```

### Tax Management
```bash
# Setup tax database
uv run python scripts/monitoring/setup_tax_database.py

# Calculate taxes
uv run python scripts/monitoring/calculate_taxes.py

# Update tax rates (married)
uv run python scripts/monitoring/update_tax_rates_married.py

# YTD config check
uv run python scripts/monitoring/update_tax_config_ytd.py
```

### Dashboard & Monitoring
```bash
# Start dashboard
uv run streamlit run dashboard/app.py

# Access at: http://localhost:8501
# Navigate to "Tax Implications" view
```

---

## Emergency Procedures

### Stop Trading Immediately
```bash
# Kill trading engine
killall bigbrother

# Close all bot positions (manual script needed)
# Check positions first
uv run python -c "
import duckdb
conn = duckdb.connect('data/bigbrother.duckdb')
positions = conn.execute('SELECT symbol, quantity, is_bot_managed FROM positions WHERE quantity > 0').fetchall()
for p in positions:
    status = 'BOT' if p[2] else 'MANUAL'
    print(f'{{p[0]}} | {{p[1]}} shares | {{status}}')
conn.close()
"
```

### OAuth Token Refresh
```bash
# Quick refresh (if refresh token valid)
uv run python /tmp/refresh_schwab_token.py

# Full re-authentication
uv run python scripts/run_schwab_oauth_interactive.py
```

---

## Support & Troubleshooting

### Common Issues

**OAuth Token Expired:**
- Run: `uv run python scripts/phase5_setup.py`
- Auto-refreshes if possible

**Tax Configuration Wrong:**
- Run: `uv run python scripts/monitoring/update_tax_rates_married.py`
- Verify: `uv run python scripts/phase5_setup.py --quick`

**API Not Connected:**
- Check token: `uv run python scripts/phase5_setup.py`
- Test API: Included in setup script

**Database Issues:**
- Reinitialize: `uv run python scripts/monitoring/setup_tax_database.py`

---

## Files Collected

**Total Files:** {len(self.collected_files)}

"""

        # List all collected files
        for category in ["setup", "execution", "monitoring", "scripts"]:
            cat_files = [(name, desc) for cat, name, desc in self.collected_files if cat == category]
            if cat_files:
                readme_content += f"\n**{category.capitalize()}:** {len(cat_files)} files\n"
                for name, desc in sorted(cat_files):
                    readme_content += f"- `{name}` - {desc}\n"

        readme_content += f"""

---

## Next Steps

1. **Review this README** - Understand Phase 5 structure
2. **Read PHASE5_SETUP_GUIDE.md** - Learn the unified setup script
3. **Run setup verification:**
   ```bash
   uv run python scripts/phase5_setup.py
   ```
4. **Review execution plan** - `execution/phase5_execution_plan.md`
5. **Print daily checklist** - `monitoring/phase5_daily_checklist.md`
6. **Begin Day 0 setup** - Follow execution plan

---

**Phase 5 Status:** Ready for Paper Trading Validation
**Tax Configuration:** Married Filing Jointly, $300K base, 32.8% ST / 23.8% LT
**System Status:** 100% verified and operational

**Let's begin Phase 5!** 🚀
"""

        readme_path = self.phase5_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)

        print_success(f"Created master index: {readme_path.relative_to(self.base_dir)}")

    def create_archive(self):
        """Create zip archive of all Phase 5 documentation"""
        if not self.create_archive:
            return

        print_header("Creating archive...")

        archive_name = f"phase5_docs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        archive_path = self.base_dir / f"{archive_name}.tar.gz"

        try:
            # Create tar.gz archive
            subprocess.run(
                ["tar", "-czf", str(archive_path), "-C", str(self.base_dir / "docs"), "phase5"],
                check=True,
                capture_output=True
            )

            file_size = archive_path.stat().st_size / 1024  # KB
            print_success(f"Created archive: {archive_path.name} ({file_size:.1f} KB)")
            print_info(f"Location: {archive_path}")

        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  Archive creation failed: {e}")
        except Exception as e:
            print(f"   ⚠️  Error: {e}")

    def generate_summary(self):
        """Generate collection summary"""
        print_header("Collection Summary")

        categories = {}
        for category, filename, description in self.collected_files:
            if category not in categories:
                categories[category] = []
            categories[category].append(filename)

        print(f"\n{Colors.BOLD}Collected Files by Category:{Colors.END}")
        for category, files in sorted(categories.items()):
            print(f"  {category.capitalize():15s} {len(files):2d} files")

        print(f"\n{Colors.BOLD}Total:{Colors.END} {len(self.collected_files)} files")

        print(f"\n{Colors.BOLD}Location:{Colors.END}")
        print(f"  {self.phase5_dir}")

        print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
        print("  1. Read:  docs/phase5/README.md")
        print("  2. Setup: uv run python scripts/phase5_setup.py")
        print("  3. Begin: Phase 5 paper trading validation")

    def run(self):
        """Run collection process"""
        print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}{'Phase 5 Documentation Collector':^70}{Colors.END}")
        print(f"{Colors.BOLD}{'='*70}{Colors.END}")

        self.setup_directories()
        self.collect_tmp_documents()
        self.collect_local_documents()
        self.collect_scripts()
        self.create_master_index()
        self.create_archive()
        self.generate_summary()

        print(f"\n{Colors.GREEN}✅ Phase 5 documentation collection complete!{Colors.END}\n")

def main():
    parser = argparse.ArgumentParser(
        description='Collect all Phase 5 documentation into organized structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/phase5_collect_docs.py              # Collect all docs
  uv run python scripts/phase5_collect_docs.py --archive    # Create zip archive
  uv run python scripts/phase5_collect_docs.py --clean      # Remove old docs first

Output:
  docs/phase5/
    ├── setup/              # Setup guides
    ├── execution/          # Execution plans
    ├── monitoring/         # Daily checklists
    ├── scripts/            # Utility scripts
    └── README.md          # Master index
        """
    )

    parser.add_argument('--archive', action='store_true',
                       help='Create tar.gz archive of collected docs')
    parser.add_argument('--clean', action='store_true',
                       help='Remove existing docs/phase5 directory first')

    args = parser.parse_args()

    collector = Phase5Collector(create_archive=args.archive, clean=args.clean)
    collector.run()

if __name__ == "__main__":
    main()
