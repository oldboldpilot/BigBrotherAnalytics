#!/usr/bin/env python3
"""
BigBrotherAnalytics: Signal Recalculation Script
Recalculate sector rotation signals after data updates

This script:
1. Recalculates employment signals based on latest BLS data
2. Recalculates sector rotation signals
3. Compares with previous signals to detect significant changes
4. Logs significant changes (>5% confidence change or >10% allocation change)
5. Generates signal change report for notifications

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
"""

import os
import sys
import logging
import json
import duckdb
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import employment signals module
try:
    from employment_signals import (
        generate_employment_signals,
        generate_rotation_signals,
        check_jobless_claims_spike
    )
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "employment_signals",
        Path(__file__).parent.parent / 'employment_signals.py'
    )
    employment_signals = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(employment_signals)
    generate_employment_signals = employment_signals.generate_employment_signals
    generate_rotation_signals = employment_signals.generate_rotation_signals
    check_jobless_claims_spike = employment_signals.check_jobless_claims_spike

# Configuration
BASE_DIR = Path(__file__).parent.parent.parent
LOG_DIR = BASE_DIR / 'logs' / 'automated_updates'
DB_PATH = BASE_DIR / 'data' / 'bigbrother.duckdb'
SIGNALS_DIR = LOG_DIR / 'signals'

# Ensure directories exist
LOG_DIR.mkdir(parents=True, exist_ok=True)
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOG_DIR / f'signal_recalc_{datetime.now().strftime("%Y%m%d")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SignalRecalculator:
    """Recalculates and tracks employment and sector rotation signals."""

    def __init__(self, db_path: Path = DB_PATH, test_mode: bool = False):
        """
        Initialize signal recalculator.

        Args:
            db_path: Path to DuckDB database
            test_mode: If True, run in test mode (calculate but don't save)
        """
        self.db_path = db_path
        self.test_mode = test_mode
        self.recalc_summary = {
            'timestamp': datetime.now().isoformat(),
            'signals_recalculated': {},
            'significant_changes': [],
            'errors': []
        }

        logger.info("=" * 80)
        logger.info("BigBrotherAnalytics Signal Recalculation")
        logger.info(f"Mode: {'TEST' if test_mode else 'LIVE'}")
        logger.info(f"Database: {db_path}")
        logger.info("=" * 80)

    def load_previous_signals(self, signal_type: str) -> Optional[Dict]:
        """
        Load the most recent previous signals from file.

        Args:
            signal_type: 'employment' or 'rotation'

        Returns:
            Dictionary of previous signals or None
        """
        signal_files = sorted(SIGNALS_DIR.glob(f'{signal_type}_signals_*.json'))

        if not signal_files:
            logger.info(f"No previous {signal_type} signals found")
            return None

        latest_file = signal_files[-1]
        logger.info(f"Loading previous {signal_type} signals from: {latest_file}")

        try:
            with open(latest_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading previous signals: {e}")
            return None

    def save_signals(self, signals: List[Dict], signal_type: str) -> None:
        """
        Save signals to JSON file.

        Args:
            signals: List of signal dictionaries
            signal_type: 'employment' or 'rotation'
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = SIGNALS_DIR / f'{signal_type}_signals_{timestamp}.json'

        try:
            with open(filename, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'signal_type': signal_type,
                    'count': len(signals),
                    'signals': signals
                }, f, indent=2)

            logger.info(f"Saved {len(signals)} {signal_type} signals to: {filename}")

        except Exception as e:
            logger.error(f"Error saving signals: {e}")

    def compare_employment_signals(
        self,
        previous: Optional[Dict],
        current: List[Dict]
    ) -> List[Dict]:
        """
        Compare previous and current employment signals.

        Args:
            previous: Previous signals dictionary
            current: Current signals list

        Returns:
            List of significant changes
        """
        if not previous or 'signals' not in previous:
            logger.info("No previous employment signals to compare")
            return []

        significant_changes = []
        prev_signals = {
            (s['sector_code'], s['type']): s
            for s in previous['signals']
        }

        for curr_signal in current:
            key = (curr_signal['sector_code'], curr_signal['type'])

            if key in prev_signals:
                prev_signal = prev_signals[key]

                # Check for significant confidence change (>5%)
                conf_change = abs(curr_signal['confidence'] - prev_signal['confidence'])

                # Check for signal strength change (>0.15 on -1.0 to +1.0 scale)
                strength_change = abs(
                    curr_signal['signal_strength'] - prev_signal['signal_strength']
                )

                if conf_change > 0.05 or strength_change > 0.15:
                    significant_changes.append({
                        'sector_code': curr_signal['sector_code'],
                        'sector_name': curr_signal['sector_name'],
                        'signal_type': curr_signal['type'],
                        'previous_confidence': prev_signal['confidence'],
                        'current_confidence': curr_signal['confidence'],
                        'confidence_change': conf_change,
                        'previous_strength': prev_signal['signal_strength'],
                        'current_strength': curr_signal['signal_strength'],
                        'strength_change': strength_change,
                        'description': (
                            f"{curr_signal['sector_name']}: "
                            f"Confidence {prev_signal['confidence']:.2f} -> "
                            f"{curr_signal['confidence']:.2f} "
                            f"(Δ {conf_change:+.2f}), "
                            f"Strength {prev_signal['signal_strength']:.2f} -> "
                            f"{curr_signal['signal_strength']:.2f} "
                            f"(Δ {strength_change:+.2f})"
                        )
                    })

            else:
                # New signal detected
                significant_changes.append({
                    'sector_code': curr_signal['sector_code'],
                    'sector_name': curr_signal['sector_name'],
                    'signal_type': curr_signal['type'],
                    'previous_confidence': 0.0,
                    'current_confidence': curr_signal['confidence'],
                    'confidence_change': curr_signal['confidence'],
                    'previous_strength': 0.0,
                    'current_strength': curr_signal['signal_strength'],
                    'strength_change': curr_signal['signal_strength'],
                    'description': (
                        f"NEW SIGNAL: {curr_signal['sector_name']} - "
                        f"{curr_signal['type']} "
                        f"(Confidence: {curr_signal['confidence']:.2f}, "
                        f"Strength: {curr_signal['signal_strength']:.2f})"
                    )
                })

        return significant_changes

    def compare_rotation_signals(
        self,
        previous: Optional[Dict],
        current: List[Dict]
    ) -> List[Dict]:
        """
        Compare previous and current rotation signals.

        Args:
            previous: Previous signals dictionary
            current: Current signals list

        Returns:
            List of significant changes
        """
        if not previous or 'signals' not in previous:
            logger.info("No previous rotation signals to compare")
            return []

        significant_changes = []
        prev_signals = {s['sector_code']: s for s in previous['signals']}

        for curr_signal in current:
            sector_code = curr_signal['sector_code']

            if sector_code in prev_signals:
                prev_signal = prev_signals[sector_code]

                # Check for allocation change (>10% relative change)
                alloc_change = abs(
                    curr_signal['target_allocation'] - prev_signal['target_allocation']
                )
                alloc_pct_change = (
                    alloc_change / prev_signal['target_allocation'] * 100.0
                    if prev_signal['target_allocation'] > 0 else 100.0
                )

                # Check for action change (e.g., Overweight -> Neutral)
                action_changed = curr_signal['action'] != prev_signal['action']

                # Check for composite score change (>0.15 on -1.0 to +1.0 scale)
                score_change = abs(
                    curr_signal['composite_score'] - prev_signal['composite_score']
                )

                if alloc_pct_change > 10.0 or action_changed or score_change > 0.15:
                    significant_changes.append({
                        'sector_code': curr_signal['sector_code'],
                        'sector_name': curr_signal['sector_name'],
                        'sector_etf': curr_signal['sector_etf'],
                        'previous_action': prev_signal['action'],
                        'current_action': curr_signal['action'],
                        'action_changed': action_changed,
                        'previous_allocation': prev_signal['target_allocation'],
                        'current_allocation': curr_signal['target_allocation'],
                        'allocation_change': alloc_change,
                        'allocation_pct_change': alloc_pct_change,
                        'previous_score': prev_signal['composite_score'],
                        'current_score': curr_signal['composite_score'],
                        'score_change': score_change,
                        'description': (
                            f"{curr_signal['sector_name']} ({curr_signal['sector_etf']}): "
                            f"{prev_signal['action']} -> {curr_signal['action']}, "
                            f"Allocation {prev_signal['target_allocation']:.1f}% -> "
                            f"{curr_signal['target_allocation']:.1f}% "
                            f"(Δ {alloc_change:+.1f}%)"
                        )
                    })

        return significant_changes

    def recalculate_employment_signals(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Recalculate employment signals.

        Returns:
            (signals, significant_changes)
        """
        logger.info("\nRecalculating employment signals...")

        try:
            # Load previous signals
            previous = self.load_previous_signals('employment')

            # Generate new signals
            current_signals = generate_employment_signals(str(self.db_path))
            logger.info(f"Generated {len(current_signals)} employment signals")

            # Compare with previous
            significant_changes = self.compare_employment_signals(previous, current_signals)

            if significant_changes:
                logger.info(f"Found {len(significant_changes)} significant employment signal changes:")
                for change in significant_changes:
                    logger.info(f"  - {change['description']}")
            else:
                logger.info("No significant employment signal changes detected")

            # Save new signals
            if not self.test_mode:
                self.save_signals(current_signals, 'employment')

            return current_signals, significant_changes

        except Exception as e:
            logger.error(f"Error recalculating employment signals: {e}", exc_info=True)
            self.recalc_summary['errors'].append({
                'type': 'employment_signals',
                'error': str(e)
            })
            return [], []

    def recalculate_rotation_signals(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Recalculate sector rotation signals.

        Returns:
            (signals, significant_changes)
        """
        logger.info("\nRecalculating sector rotation signals...")

        try:
            # Load previous signals
            previous = self.load_previous_signals('rotation')

            # Generate new signals
            current_signals = generate_rotation_signals(str(self.db_path))
            logger.info(f"Generated {len(current_signals)} rotation signals")

            # Compare with previous
            significant_changes = self.compare_rotation_signals(previous, current_signals)

            if significant_changes:
                logger.info(f"Found {len(significant_changes)} significant rotation signal changes:")
                for change in significant_changes:
                    logger.info(f"  - {change['description']}")
            else:
                logger.info("No significant rotation signal changes detected")

            # Save new signals
            if not self.test_mode:
                self.save_signals(current_signals, 'rotation')

            return current_signals, significant_changes

        except Exception as e:
            logger.error(f"Error recalculating rotation signals: {e}", exc_info=True)
            self.recalc_summary['errors'].append({
                'type': 'rotation_signals',
                'error': str(e)
            })
            return [], []

    def check_recession_signals(self) -> Optional[Dict]:
        """
        Check for recession warning signals (jobless claims spike).

        Returns:
            Recession signal dictionary or None
        """
        logger.info("\nChecking for recession warning signals...")

        try:
            signal = check_jobless_claims_spike(str(self.db_path))

            if signal:
                logger.warning(f"RECESSION WARNING: {signal}")
                self.recalc_summary['significant_changes'].append({
                    'type': 'recession_warning',
                    'signal': signal,
                    'description': 'Jobless claims spike detected - recession warning'
                })

            return signal

        except Exception as e:
            logger.error(f"Error checking recession signals: {e}", exc_info=True)
            return None

    def run_recalculation(self) -> Dict:
        """
        Run the full signal recalculation process.

        Returns:
            Summary dictionary of recalculation results
        """
        logger.info("\n" + "=" * 80)
        logger.info("Starting Signal Recalculation Process")
        logger.info("=" * 80 + "\n")

        # Recalculate employment signals
        employment_signals, employment_changes = self.recalculate_employment_signals()
        self.recalc_summary['signals_recalculated']['employment'] = {
            'count': len(employment_signals),
            'significant_changes': len(employment_changes)
        }
        self.recalc_summary['significant_changes'].extend([
            {'type': 'employment', **change} for change in employment_changes
        ])

        # Recalculate rotation signals
        rotation_signals, rotation_changes = self.recalculate_rotation_signals()
        self.recalc_summary['signals_recalculated']['rotation'] = {
            'count': len(rotation_signals),
            'significant_changes': len(rotation_changes)
        }
        self.recalc_summary['significant_changes'].extend([
            {'type': 'rotation', **change} for change in rotation_changes
        ])

        # Check recession signals
        recession_signal = self.check_recession_signals()

        # Save summary
        self._save_recalc_summary()

        logger.info("\n" + "=" * 80)
        logger.info("Signal Recalculation Complete")
        logger.info(f"Employment signals: {len(employment_signals)}")
        logger.info(f"Rotation signals: {len(rotation_signals)}")
        logger.info(f"Significant changes: {len(self.recalc_summary['significant_changes'])}")
        logger.info(f"Errors: {len(self.recalc_summary['errors'])}")
        logger.info("=" * 80 + "\n")

        return self.recalc_summary

    def _save_recalc_summary(self) -> None:
        """Save recalculation summary to JSON file."""
        summary_file = LOG_DIR / f'recalc_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

        try:
            with open(summary_file, 'w') as f:
                json.dump(self.recalc_summary, f, indent=2)
            logger.info(f"Recalculation summary saved to: {summary_file}")
        except Exception as e:
            logger.error(f"Error saving recalculation summary: {e}")


def main():
    """Main entry point for signal recalculation script."""
    import argparse

    parser = argparse.ArgumentParser(
        description='BigBrotherAnalytics Signal Recalculation Script'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (calculate but don\'t save)'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=str(DB_PATH),
        help='Path to DuckDB database'
    )

    args = parser.parse_args()

    recalculator = SignalRecalculator(
        db_path=Path(args.db_path),
        test_mode=args.test
    )

    summary = recalculator.run_recalculation()

    # Exit with error code if there were errors
    if summary['errors']:
        logger.error("Signal recalculation completed with errors")
        sys.exit(1)
    else:
        logger.info("Signal recalculation completed successfully")
        sys.exit(0)


if __name__ == '__main__':
    main()
