#!/usr/bin/env python3
"""
Daily Trading Report Generator

Generates comprehensive daily reports including:
- Executive summary (account value, trades, win rate)
- Trade execution details
- Signal analysis (generated, executed, rejected)
- Risk compliance status
- Market conditions summary
"""

import duckdb
import pandas as pd
from datetime import datetime, date
from pathlib import Path
import json
import sys
from typing import Dict, List, Tuple, Optional
import traceback


class DailyReportGenerator:
    """Generate comprehensive daily trading reports"""

    def __init__(self, db_path: str = "data/bigbrother.duckdb"):
        """Initialize report generator"""
        self.db_path = Path(db_path)
        self.report_date = date.today()
        self.timestamp = datetime.now()

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Create database connection"""
        return duckdb.connect(str(self.db_path), read_only=True)

    def get_executive_summary(self) -> Dict:
        """Get executive summary metrics"""
        conn = self.connect()

        try:
            # Account value and performance
            account = conn.execute("""
                SELECT
                    account_id,
                    liquidation_value,
                    equity,
                    buying_power,
                    updated_at
                FROM account_balances
                ORDER BY updated_at DESC
                LIMIT 1
            """).fetchall()

            account_data = {}
            if account:
                account_data = {
                    'account_id': account[0][0],
                    'liquidation_value': float(account[0][1]) if account[0][1] else 0,
                    'equity': float(account[0][2]) if account[0][2] else 0,
                    'buying_power': float(account[0][3]) if account[0][3] else 0,
                    'updated_at': str(account[0][4]) if account[0][4] else 'N/A'
                }

            # Today's signal execution
            signals = conn.execute(f"""
                SELECT
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN status = 'EXECUTED' THEN 1 END) as executed,
                    COUNT(CASE WHEN status != 'EXECUTED' THEN 1 END) as rejected,
                    AVG(confidence) as avg_confidence,
                    SUM(expected_return) as total_expected_return,
                    SUM(CASE WHEN status = 'EXECUTED' THEN expected_return ELSE 0 END) as executed_return
                FROM trading_signals
                WHERE DATE(timestamp) = '{self.report_date}'
            """).fetchall()

            signals_data = {}
            if signals and signals[0][0]:
                signals_data = {
                    'total_signals': int(signals[0][0]),
                    'executed': int(signals[0][1]),
                    'rejected': int(signals[0][2]),
                    'execution_rate': (signals[0][1] / signals[0][0] * 100) if signals[0][0] > 0 else 0,
                    'avg_confidence': float(signals[0][3]) if signals[0][3] else 0,
                    'total_expected_return': float(signals[0][4]) if signals[0][4] else 0,
                    'executed_return': float(signals[0][5]) if signals[0][5] else 0
                }

            return {
                'report_date': str(self.report_date),
                'generated_at': self.timestamp.isoformat(),
                'account': account_data,
                'signals': signals_data
            }

        finally:
            conn.close()

    def get_trade_execution_details(self) -> Dict:
        """Get detailed trade execution information"""
        conn = self.connect()

        try:
            # Executed trades
            trades = conn.execute(f"""
                SELECT
                    timestamp,
                    strategy,
                    symbol,
                    signal_type,
                    confidence,
                    expected_return,
                    estimated_cost,
                    max_risk,
                    days_to_expiration,
                    greeks_delta,
                    greeks_theta,
                    greeks_vega,
                    iv_percentile,
                    market_conditions
                FROM trading_signals
                WHERE DATE(timestamp) = '{self.report_date}'
                AND status = 'EXECUTED'
                ORDER BY timestamp
            """).fetchall()

            # Get column names
            description = conn.description
            if hasattr(conn, 'execute'):
                # Re-execute to get description
                result = conn.execute(f"""
                    SELECT *
                    FROM trading_signals
                    WHERE DATE(timestamp) = '{self.report_date}'
                    AND status = 'EXECUTED'
                    ORDER BY timestamp
                    LIMIT 0
                """)

            trade_list = []
            for trade in trades:
                trade_dict = {
                    'timestamp': str(trade[0]),
                    'strategy': trade[1],
                    'symbol': trade[2],
                    'signal_type': trade[3],
                    'confidence': float(trade[4]) if trade[4] else 0,
                    'expected_return': float(trade[5]) if trade[5] else 0,
                    'estimated_cost': float(trade[6]) if trade[6] else 0,
                    'max_risk': float(trade[7]) if trade[7] else 0,
                    'days_to_expiration': int(trade[8]) if trade[8] else 0,
                    'greeks': {
                        'delta': float(trade[9]) if trade[9] else 0,
                        'theta': float(trade[10]) if trade[10] else 0,
                        'vega': float(trade[11]) if trade[11] else 0,
                    },
                    'iv_percentile': float(trade[12]) if trade[12] else 0,
                }
                trade_list.append(trade_dict)

            # Summary stats
            summary = {
                'total_executed': len(trade_list),
                'strategies': list(set([t['strategy'] for t in trade_list])),
                'symbols': list(set([t['symbol'] for t in trade_list])),
                'avg_confidence': sum([t['confidence'] for t in trade_list]) / len(trade_list) if trade_list else 0,
                'total_expected_return': sum([t['expected_return'] for t in trade_list]),
                'total_estimated_cost': sum([t['estimated_cost'] for t in trade_list]),
                'total_max_risk': sum([t['max_risk'] for t in trade_list]),
            }

            return {
                'summary': summary,
                'trades': trade_list
            }

        finally:
            conn.close()

    def get_signal_analysis(self) -> Dict:
        """Get comprehensive signal analysis"""
        conn = self.connect()

        try:
            # Overall signal flow
            signals = conn.execute(f"""
                SELECT
                    status,
                    COUNT(*) as count,
                    AVG(confidence) as avg_confidence,
                    AVG(expected_return) as avg_return,
                    SUM(expected_return) as total_return
                FROM trading_signals
                WHERE DATE(timestamp) = '{self.report_date}'
                GROUP BY status
                ORDER BY count DESC
            """).fetchall()

            signal_breakdown = {}
            for row in signals:
                signal_breakdown[row[0]] = {
                    'count': int(row[1]),
                    'avg_confidence': float(row[2]) if row[2] else 0,
                    'avg_return': float(row[3]) if row[3] else 0,
                    'total_return': float(row[4]) if row[4] else 0,
                }

            # Rejection reasons
            rejections = conn.execute(f"""
                SELECT
                    status,
                    COUNT(*) as count
                FROM trading_signals
                WHERE DATE(timestamp) = '{self.report_date}'
                AND status != 'EXECUTED'
                GROUP BY status
                ORDER BY count DESC
            """).fetchall()

            rejection_reasons = {}
            for row in rejections:
                rejection_reasons[row[0]] = {
                    'count': int(row[1]),
                    'percentage': 0  # Will be calculated
                }

            # Calculate percentages
            total_rejected = sum([r['count'] for r in rejection_reasons.values()])
            for reason in rejection_reasons:
                if total_rejected > 0:
                    rejection_reasons[reason]['percentage'] = (
                        rejection_reasons[reason]['count'] / total_rejected * 100
                    )

            # By strategy
            by_strategy = conn.execute(f"""
                SELECT
                    strategy,
                    COUNT(*) as total,
                    COUNT(CASE WHEN status = 'EXECUTED' THEN 1 END) as executed,
                    COUNT(CASE WHEN status != 'EXECUTED' THEN 1 END) as rejected
                FROM trading_signals
                WHERE DATE(timestamp) = '{self.report_date}'
                GROUP BY strategy
                ORDER BY total DESC
            """).fetchall()

            strategy_stats = {}
            for row in by_strategy:
                strategy_stats[row[0]] = {
                    'total': int(row[1]),
                    'executed': int(row[2]),
                    'rejected': int(row[3]),
                    'acceptance_rate': (row[2] / row[1] * 100) if row[1] > 0 else 0,
                }

            return {
                'signal_breakdown': signal_breakdown,
                'rejection_analysis': rejection_reasons,
                'by_strategy': strategy_stats
            }

        finally:
            conn.close()

    def get_risk_compliance(self) -> Dict:
        """Get risk compliance status"""
        conn = self.connect()

        try:
            # Risk-related metrics
            risk_data = conn.execute(f"""
                SELECT
                    COUNT(CASE WHEN status = 'REJECTED_RISK' THEN 1 END) as risk_rejections,
                    MAX(max_risk) as max_single_risk,
                    SUM(max_risk) as total_max_risk,
                    COUNT(CASE WHEN max_risk > 500 THEN 1 END) as high_risk_signals
                FROM trading_signals
                WHERE DATE(timestamp) = '{self.report_date}'
            """).fetchall()

            risk_rejections = 0
            max_single_risk = 0
            total_max_risk = 0
            high_risk_count = 0

            if risk_data and risk_data[0]:
                risk_rejections = int(risk_data[0][0]) if risk_data[0][0] else 0
                max_single_risk = float(risk_data[0][1]) if risk_data[0][1] else 0
                total_max_risk = float(risk_data[0][2]) if risk_data[0][2] else 0
                high_risk_count = int(risk_data[0][3]) if risk_data[0][3] else 0

            # Budget compliance
            budget_data = conn.execute(f"""
                SELECT
                    COUNT(CASE WHEN status = 'FILTERED_BUDGET' THEN 1 END) as budget_rejections,
                    MAX(estimated_cost) as max_position_cost,
                    AVG(estimated_cost) as avg_position_cost
                FROM trading_signals
                WHERE DATE(timestamp) = '{self.report_date}'
            """).fetchall()

            budget_rejections = 0
            max_position_cost = 0
            avg_position_cost = 0

            if budget_data and budget_data[0]:
                budget_rejections = int(budget_data[0][0]) if budget_data[0][0] else 0
                max_position_cost = float(budget_data[0][1]) if budget_data[0][1] else 0
                avg_position_cost = float(budget_data[0][2]) if budget_data[0][2] else 0

            return {
                'risk_management': {
                    'risk_rejections': risk_rejections,
                    'max_single_risk': max_single_risk,
                    'total_max_risk': total_max_risk,
                    'high_risk_signals': high_risk_count,
                    'status': 'COMPLIANT' if risk_rejections == 0 else 'ALERTS'
                },
                'budget_compliance': {
                    'budget_rejections': budget_rejections,
                    'max_position_cost': max_position_cost,
                    'avg_position_cost': avg_position_cost,
                    'limit': 500,
                    'status': 'COMPLIANT' if budget_rejections == 0 else 'EXCEEDED'
                }
            }

        finally:
            conn.close()

    def get_market_conditions(self) -> Dict:
        """Get market conditions summary"""
        conn = self.connect()

        try:
            # Get market conditions from signals
            conditions = conn.execute(f"""
                SELECT DISTINCT
                    market_conditions
                FROM trading_signals
                WHERE DATE(timestamp) = '{self.report_date}'
                AND market_conditions IS NOT NULL
                LIMIT 1
            """).fetchall()

            market_conditions = {}
            if conditions and conditions[0][0]:
                try:
                    market_conditions = json.loads(conditions[0][0])
                except (json.JSONDecodeError, TypeError):
                    market_conditions = {'raw': str(conditions[0][0])}

            # Get IV metrics
            iv_data = conn.execute(f"""
                SELECT
                    AVG(iv_percentile) as avg_iv_percentile,
                    MIN(iv_percentile) as min_iv,
                    MAX(iv_percentile) as max_iv
                FROM trading_signals
                WHERE DATE(timestamp) = '{self.report_date}'
                AND iv_percentile IS NOT NULL
            """).fetchall()

            iv_metrics = {}
            if iv_data and iv_data[0][0]:
                iv_metrics = {
                    'avg_iv_percentile': float(iv_data[0][0]) if iv_data[0][0] else 0,
                    'min_iv': float(iv_data[0][1]) if iv_data[0][1] else 0,
                    'max_iv': float(iv_data[0][2]) if iv_data[0][2] else 0,
                }

            # Get DTE distribution
            dte_data = conn.execute(f"""
                SELECT
                    AVG(days_to_expiration) as avg_dte,
                    MIN(days_to_expiration) as min_dte,
                    MAX(days_to_expiration) as max_dte
                FROM trading_signals
                WHERE DATE(timestamp) = '{self.report_date}'
                AND days_to_expiration IS NOT NULL
            """).fetchall()

            dte_metrics = {}
            if dte_data and dte_data[0][0]:
                dte_metrics = {
                    'avg_dte': float(dte_data[0][0]) if dte_data[0][0] else 0,
                    'min_dte': int(dte_data[0][1]) if dte_data[0][1] else 0,
                    'max_dte': int(dte_data[0][2]) if dte_data[0][2] else 0,
                }

            return {
                'market_conditions': market_conditions,
                'iv_metrics': iv_metrics,
                'dte_metrics': dte_metrics,
                'overall_assessment': 'Normal' if not market_conditions else 'See market_conditions'
            }

        finally:
            conn.close()

    def generate_report(self) -> Dict:
        """Generate complete daily report"""
        print(f"Generating daily report for {self.report_date}...")

        report = {
            'metadata': {
                'report_type': 'Daily Trading Report',
                'date': str(self.report_date),
                'generated_at': self.timestamp.isoformat(),
                'version': '1.0'
            },
            'executive_summary': self.get_executive_summary(),
            'trade_execution': self.get_trade_execution_details(),
            'signal_analysis': self.get_signal_analysis(),
            'risk_compliance': self.get_risk_compliance(),
            'market_conditions': self.get_market_conditions()
        }

        return report

    def save_report(self, report: Dict, format: str = 'json') -> Path:
        """Save report to file"""
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)

        filename = f"daily_report_{self.report_date.strftime('%Y%m%d')}.{format}"
        filepath = reports_dir / filename

        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif format == 'html':
            html_content = self._generate_html(report)
            with open(filepath, 'w') as f:
                f.write(html_content)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return filepath

    def _generate_html(self, report: Dict) -> str:
        """Generate HTML version of report"""
        exec_summary = report['executive_summary']
        signals = exec_summary['signals']
        account = exec_summary['account']

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Daily Trading Report - {self.report_date}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .metric {{ background: #f9f9f9; padding: 15px; border-left: 4px solid #007bff; }}
        .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #007bff; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        .success {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .danger {{ color: #dc3545; }}
        .timestamp {{ font-size: 12px; color: #999; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Daily Trading Report</h1>
        <p class="timestamp">Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Executive Summary</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Liquidation Value</div>
                <div class="metric-value">${account.get('liquidation_value', 0):,.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Signals</div>
                <div class="metric-value">{signals.get('total_signals', 0)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Executed</div>
                <div class="metric-value success">{signals.get('executed', 0)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Execution Rate</div>
                <div class="metric-value">{signals.get('execution_rate', 0):.1f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Expected Return</div>
                <div class="metric-value">${signals.get('executed_return', 0):,.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Avg Confidence</div>
                <div class="metric-value">{signals.get('avg_confidence', 0):.1%}</div>
            </div>
        </div>

        <h2>Signal Analysis</h2>
        <p>Total signals generated: {signals.get('total_signals', 0)}</p>
        <p>Execution rate: {signals.get('execution_rate', 0):.1f}%</p>
        <p>Expected return from executed trades: ${signals.get('executed_return', 0):,.2f}</p>

        <h2>Risk Compliance</h2>
        <p>Status: {report['risk_compliance']['risk_management']['status']}</p>
        <p>Risk rejections: {report['risk_compliance']['risk_management']['risk_rejections']}</p>
        <p>Budget status: {report['risk_compliance']['budget_compliance']['status']}</p>

        <footer style="text-align: center; color: #999; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd;">
            BigBrother Analytics - Automated Trading Report System
        </footer>
    </div>
</body>
</html>
"""
        return html

    def print_summary(self, report: Dict):
        """Print report summary to console"""
        print("\n" + "=" * 80)
        print(f"DAILY TRADING REPORT - {self.report_date}")
        print("=" * 80)

        exec_summary = report['executive_summary']
        signals = exec_summary['signals']
        account = exec_summary['account']

        print("\nEXECUTIVE SUMMARY")
        print("-" * 80)
        if account:
            print(f"  Account Liquidation Value: ${account.get('liquidation_value', 0):,.2f}")
            print(f"  Equity: ${account.get('equity', 0):,.2f}")
            print(f"  Buying Power: ${account.get('buying_power', 0):,.2f}")

        print(f"\n  Total Signals Generated: {signals.get('total_signals', 0)}")
        print(f"  Executed: {signals.get('executed', 0)}")
        print(f"  Rejected: {signals.get('rejected', 0)}")
        print(f"  Execution Rate: {signals.get('execution_rate', 0):.1f}%")
        print(f"  Avg Confidence: {signals.get('avg_confidence', 0):.1%}")
        print(f"  Expected Return (Executed): ${signals.get('executed_return', 0):,.2f}")

        print("\nTRADE EXECUTION")
        print("-" * 80)
        trade_exec = report['trade_execution']
        summary = trade_exec['summary']
        print(f"  Total Executed Trades: {summary['total_executed']}")
        print(f"  Strategies: {', '.join(summary['strategies']) if summary['strategies'] else 'None'}")
        print(f"  Symbols: {', '.join(summary['symbols']) if summary['symbols'] else 'None'}")
        print(f"  Total Expected Return: ${summary['total_expected_return']:,.2f}")
        print(f"  Total Estimated Cost: ${summary['total_estimated_cost']:,.2f}")
        print(f"  Total Max Risk: ${summary['total_max_risk']:,.2f}")

        print("\nSIGNAL ANALYSIS")
        print("-" * 80)
        signal_analysis = report['signal_analysis']
        for status, data in signal_analysis['signal_breakdown'].items():
            print(f"  {status}: {data['count']} (Avg Return: ${data['total_return']:,.2f})")

        print("\nRISK COMPLIANCE")
        print("-" * 80)
        risk = report['risk_compliance']
        print(f"  Risk Management Status: {risk['risk_management']['status']}")
        print(f"  Risk Rejections: {risk['risk_management']['risk_rejections']}")
        print(f"  Budget Compliance Status: {risk['budget_compliance']['status']}")
        print(f"  Budget Rejections: {risk['budget_compliance']['budget_rejections']}")

        print("\n" + "=" * 80)


def main():
    """Main entry point"""
    try:
        generator = DailyReportGenerator()

        # Generate report
        report = generator.generate_report()

        # Save as JSON
        json_path = generator.save_report(report, format='json')
        print(f"✅ JSON report saved: {json_path}")

        # Save as HTML
        html_path = generator.save_report(report, format='html')
        print(f"✅ HTML report saved: {html_path}")

        # Print summary
        generator.print_summary(report)

        return 0

    except Exception as e:
        print(f"❌ Error generating report: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
