#!/usr/bin/env python3
"""
Weekly Trading Report Generator

Generates comprehensive weekly reports including:
- Performance summary (trades, win rate, P&L, Sharpe)
- Strategy comparison table
- Signal acceptance rate by strategy
- Risk analysis
- Recommendations based on data
"""

import duckdb
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
import json
import sys
from typing import Dict, List, Tuple, Optional
import traceback
import statistics


class WeeklyReportGenerator:
    """Generate comprehensive weekly trading reports"""

    def __init__(self, db_path: str = "data/bigbrother.duckdb", week_offset: int = 0):
        """
        Initialize report generator

        Args:
            db_path: Path to DuckDB database
            week_offset: Number of weeks back (0 = current week, 1 = last week, etc.)
        """
        self.db_path = Path(db_path)
        self.week_offset = week_offset

        # Calculate week dates
        today = date.today()
        start_of_week = today - timedelta(days=today.weekday())
        self.week_start = start_of_week - timedelta(weeks=week_offset)
        self.week_end = self.week_start + timedelta(days=6)
        self.timestamp = datetime.now()

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Create database connection"""
        return duckdb.connect(str(self.db_path), read_only=True)

    def get_performance_summary(self) -> Dict:
        """Get weekly performance summary"""
        conn = self.connect()

        try:
            # Overall weekly stats
            stats = conn.execute(f"""
                SELECT
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN status = 'EXECUTED' THEN 1 END) as executed,
                    COUNT(CASE WHEN status != 'EXECUTED' THEN 1 END) as rejected,
                    AVG(confidence) as avg_confidence,
                    AVG(expected_return) as avg_expected_return,
                    SUM(expected_return) as total_expected_return,
                    SUM(CASE WHEN status = 'EXECUTED' THEN expected_return ELSE 0 END) as executed_return,
                    SUM(CASE WHEN status = 'EXECUTED' THEN max_risk ELSE 0 END) as total_risk,
                    COUNT(DISTINCT DATE(timestamp)) as trading_days,
                    COUNT(DISTINCT strategy) as num_strategies
                FROM trading_signals
                WHERE DATE(timestamp) >= '{self.week_start}'
                AND DATE(timestamp) <= '{self.week_end}'
            """).fetchall()

            if not stats or not stats[0][0]:
                return {
                    'week_start': str(self.week_start),
                    'week_end': str(self.week_end),
                    'total_signals': 0,
                    'executed': 0,
                    'rejected': 0,
                    'execution_rate': 0,
                    'avg_confidence': 0,
                    'total_expected_return': 0,
                    'executed_return': 0,
                    'trading_days': 0,
                    'num_strategies': 0,
                }

            row = stats[0]
            total = int(row[0])
            executed = int(row[1])
            total_expected = float(row[5]) if row[5] else 0
            executed_return = float(row[6]) if row[6] else 0
            total_risk = float(row[7]) if row[7] else 0

            # Calculate Sharpe ratio (simplified - using expected returns)
            sharpe_ratio = 0
            if total_risk > 0:
                sharpe_ratio = executed_return / total_risk if total_risk > 0 else 0

            return {
                'week_start': str(self.week_start),
                'week_end': str(self.week_end),
                'total_signals': total,
                'executed': executed,
                'rejected': int(row[2]),
                'execution_rate': (executed / total * 100) if total > 0 else 0,
                'avg_confidence': float(row[3]) if row[3] else 0,
                'avg_expected_return': float(row[4]) if row[4] else 0,
                'total_expected_return': total_expected,
                'executed_return': executed_return,
                'total_risk': total_risk,
                'risk_reward_ratio': (executed_return / total_risk) if total_risk > 0 else 0,
                'sharpe_ratio': sharpe_ratio,
                'trading_days': int(row[8]) if row[8] else 0,
                'num_strategies': int(row[9]) if row[9] else 0,
            }

        finally:
            conn.close()

    def get_strategy_comparison(self) -> Dict:
        """Get strategy performance comparison table"""
        conn = self.connect()

        try:
            # Get strategy stats
            strategies = conn.execute(f"""
                SELECT
                    strategy,
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN status = 'EXECUTED' THEN 1 END) as executed,
                    AVG(confidence) as avg_confidence,
                    AVG(expected_return) as avg_return,
                    SUM(expected_return) as total_return,
                    SUM(CASE WHEN status = 'EXECUTED' THEN expected_return ELSE 0 END) as executed_return,
                    SUM(CASE WHEN status = 'EXECUTED' THEN max_risk ELSE 0 END) as total_risk,
                    COUNT(DISTINCT symbol) as num_symbols,
                    COUNT(DISTINCT DATE(timestamp)) as trading_days
                FROM trading_signals
                WHERE DATE(timestamp) >= '{self.week_start}'
                AND DATE(timestamp) <= '{self.week_end}'
                GROUP BY strategy
                ORDER BY executed DESC
            """).fetchall()

            comparison = {}
            for row in strategies:
                strategy_name = row[0]
                total = int(row[1])
                executed = int(row[2])
                total_return = float(row[5]) if row[5] else 0
                executed_return = float(row[6]) if row[6] else 0
                total_risk = float(row[7]) if row[7] else 0

                comparison[strategy_name] = {
                    'total_signals': total,
                    'executed': executed,
                    'rejected': total - executed,
                    'execution_rate': (executed / total * 100) if total > 0 else 0,
                    'avg_confidence': float(row[3]) if row[3] else 0,
                    'avg_return': float(row[4]) if row[4] else 0,
                    'total_return': total_return,
                    'executed_return': executed_return,
                    'total_risk': total_risk,
                    'risk_reward_ratio': (executed_return / total_risk) if total_risk > 0 else 0,
                    'num_symbols': int(row[8]) if row[8] else 0,
                    'trading_days': int(row[9]) if row[9] else 0,
                    'signals_per_day': (total / int(row[9])) if row[9] > 0 else 0,
                }

            return comparison

        finally:
            conn.close()

    def get_signal_acceptance_rates(self) -> Dict:
        """Get signal acceptance rates by strategy"""
        conn = self.connect()

        try:
            acceptance = conn.execute(f"""
                SELECT
                    strategy,
                    DATE(timestamp) as date,
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN status = 'EXECUTED' THEN 1 END) as executed,
                    COUNT(CASE WHEN status = 'FILTERED_CONFIDENCE' THEN 1 END) as confidence_rejected,
                    COUNT(CASE WHEN status = 'FILTERED_RETURN' THEN 1 END) as return_rejected,
                    COUNT(CASE WHEN status = 'FILTERED_WIN_PROB' THEN 1 END) as win_prob_rejected,
                    COUNT(CASE WHEN status = 'FILTERED_BUDGET' THEN 1 END) as budget_rejected,
                    COUNT(CASE WHEN status = 'REJECTED_RISK' THEN 1 END) as risk_rejected
                FROM trading_signals
                WHERE DATE(timestamp) >= '{self.week_start}'
                AND DATE(timestamp) <= '{self.week_end}'
                GROUP BY strategy, DATE(timestamp)
                ORDER BY date DESC, strategy
            """).fetchall()

            acceptance_data = {}
            for row in acceptance:
                strategy = row[0]
                date_str = str(row[1])
                total = int(row[2])

                if strategy not in acceptance_data:
                    acceptance_data[strategy] = []

                acceptance_data[strategy].append({
                    'date': date_str,
                    'total': total,
                    'executed': int(row[3]),
                    'acceptance_rate': (int(row[3]) / total * 100) if total > 0 else 0,
                    'rejections': {
                        'confidence': int(row[4]) if row[4] else 0,
                        'return': int(row[5]) if row[5] else 0,
                        'win_prob': int(row[6]) if row[6] else 0,
                        'budget': int(row[7]) if row[7] else 0,
                        'risk': int(row[8]) if row[8] else 0,
                    }
                })

            return acceptance_data

        finally:
            conn.close()

    def get_risk_analysis(self) -> Dict:
        """Get comprehensive risk analysis"""
        conn = self.connect()

        try:
            # Overall risk metrics
            risk_metrics = conn.execute(f"""
                SELECT
                    COUNT(CASE WHEN status = 'REJECTED_RISK' THEN 1 END) as risk_rejections,
                    MAX(max_risk) as max_single_risk,
                    AVG(max_risk) as avg_risk,
                    SUM(CASE WHEN status = 'EXECUTED' THEN max_risk ELSE 0 END) as executed_risk,
                    COUNT(CASE WHEN max_risk > 500 THEN 1 END) as high_risk_signals,
                    COUNT(CASE WHEN status = 'FILTERED_BUDGET' THEN 1 END) as budget_rejections,
                    AVG(CASE WHEN status = 'FILTERED_BUDGET' THEN estimated_cost END) as avg_rejected_cost
                FROM trading_signals
                WHERE DATE(timestamp) >= '{self.week_start}'
                AND DATE(timestamp) <= '{self.week_end}'
            """).fetchall()

            row = risk_metrics[0] if risk_metrics else None
            if not row:
                return {
                    'risk_rejections': 0,
                    'max_single_risk': 0,
                    'avg_risk': 0,
                    'executed_risk': 0,
                    'high_risk_signals': 0,
                    'budget_rejections': 0,
                    'avg_rejected_cost': 0,
                }

            return {
                'risk_rejections': int(row[0]) if row[0] else 0,
                'max_single_risk': float(row[1]) if row[1] else 0,
                'avg_risk': float(row[2]) if row[2] else 0,
                'executed_risk': float(row[3]) if row[3] else 0,
                'high_risk_signals': int(row[4]) if row[4] else 0,
                'budget_rejections': int(row[5]) if row[5] else 0,
                'avg_rejected_cost': float(row[6]) if row[6] else 0,
                'overall_risk_status': 'COMPLIANT' if (not row[0] or row[0] == 0) else 'ALERTS'
            }

        finally:
            conn.close()

    def get_recommendations(self, report: Dict) -> List[Dict]:
        """Generate recommendations based on data"""
        recommendations = []

        perf = report['performance_summary']
        strat = report['strategy_comparison']
        risk = report['risk_analysis']

        # Execution rate recommendation
        if perf['execution_rate'] < 30:
            recommendations.append({
                'severity': 'HIGH',
                'category': 'Execution Rate',
                'message': f"Execution rate is low ({perf['execution_rate']:.1f}%). Review signal filter thresholds.",
                'action': 'Consider adjusting confidence or return thresholds in config.yaml'
            })
        elif perf['execution_rate'] > 80:
            recommendations.append({
                'severity': 'LOW',
                'category': 'Execution Rate',
                'message': f"Execution rate is very high ({perf['execution_rate']:.1f}%). Review risk controls.",
                'action': 'Ensure risk parameters are appropriately conservative'
            })

        # Best performing strategy
        if strat:
            best_strategy = max(strat.items(), key=lambda x: x[1]['executed_return'])
            recommendations.append({
                'severity': 'INFO',
                'category': 'Strategy Performance',
                'message': f"Best performing strategy: {best_strategy[0]} (${best_strategy[1]['executed_return']:.2f} return)",
                'action': 'Consider increasing allocation to this strategy if risk profile allows'
            })

        # Budget rejections
        if risk['budget_rejections'] > perf['total_signals'] * 0.2:
            recommendations.append({
                'severity': 'MEDIUM',
                'category': 'Budget Constraint',
                'message': f"Budget rejections high ({risk['budget_rejections']} signals). Position limit may be too restrictive.",
                'action': f"Review max_position_size in config (currently $500, rejected avg: ${risk['avg_rejected_cost']:.2f})"
            })

        # Risk rejections
        if risk['risk_rejections'] > 0:
            recommendations.append({
                'severity': 'MEDIUM',
                'category': 'Risk Management',
                'message': f"Risk Manager rejected {risk['risk_rejections']} signals. Review position sizing.",
                'action': 'Monitor max_risk settings and ensure they align with portfolio risk tolerance'
            })

        # Average confidence
        if perf['avg_confidence'] < 0.60:
            recommendations.append({
                'severity': 'HIGH',
                'category': 'Signal Quality',
                'message': f"Average confidence is low ({perf['avg_confidence']:.1%}). Signal quality may need improvement.",
                'action': 'Review strategy parameters and consider strengthening signal generation logic'
            })

        # Diversification
        if perf['num_strategies'] == 1:
            recommendations.append({
                'severity': 'LOW',
                'category': 'Portfolio Diversity',
                'message': f"Only {perf['num_strategies']} strategy active. Consider diversifying.",
                'action': 'Enable additional strategies to reduce concentration risk'
            })

        return recommendations

    def generate_report(self) -> Dict:
        """Generate complete weekly report"""
        print(f"Generating weekly report for {self.week_start} to {self.week_end}...")

        perf = self.get_performance_summary()
        strat = self.get_strategy_comparison()
        acceptance = self.get_signal_acceptance_rates()
        risk = self.get_risk_analysis()

        report = {
            'metadata': {
                'report_type': 'Weekly Trading Report',
                'week_start': str(self.week_start),
                'week_end': str(self.week_end),
                'generated_at': self.timestamp.isoformat(),
                'version': '1.0'
            },
            'performance_summary': perf,
            'strategy_comparison': strat,
            'signal_acceptance_rates': acceptance,
            'risk_analysis': risk,
        }

        # Add recommendations
        report['recommendations'] = self.get_recommendations(report)

        return report

    def save_report(self, report: Dict, format: str = 'json') -> Path:
        """Save report to file"""
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)

        filename = f"weekly_report_{self.week_start.strftime('%Y%m%d')}_to_{self.week_end.strftime('%Y%m%d')}.{format}"
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
        perf = report['performance_summary']
        strat = report['strategy_comparison']
        risk = report['risk_analysis']
        recs = report['recommendations']

        strategies_rows = ''.join([
            f"""
            <tr>
                <td>{name}</td>
                <td>{data['total_signals']}</td>
                <td>{data['executed']}</td>
                <td>{data['execution_rate']:.1f}%</td>
                <td>${data['executed_return']:.2f}</td>
                <td>${data['total_risk']:.2f}</td>
                <td>{data['risk_reward_ratio']:.2f}</td>
            </tr>
            """
            for name, data in strat.items()
        ])

        recommendations_html = ''.join([
            f"""
            <div class="recommendation {r['severity'].lower()}">
                <strong>{r['category']}</strong>: {r['message']}<br>
                <small>Action: {r['action']}</small>
            </div>
            """
            for r in recs
        ])

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Weekly Trading Report - {perf['week_start']} to {perf['week_end']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; }}
        .metric {{ background: #f9f9f9; padding: 15px; border-left: 4px solid #007bff; }}
        .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .metric-value {{ font-size: 20px; font-weight: bold; color: #333; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #007bff; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        .recommendation {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; background: #f9f9f9; }}
        .recommendation.high {{ border-left-color: #dc3545; }}
        .recommendation.medium {{ border-left-color: #ffc107; }}
        .recommendation.low {{ border-left-color: #28a745; }}
        .recommendation.info {{ border-left-color: #17a2b8; }}
        .timestamp {{ font-size: 12px; color: #999; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Weekly Trading Report</h1>
        <p>{perf['week_start']} to {perf['week_end']}</p>
        <p class="timestamp">Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Performance Summary</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Total Signals</div>
                <div class="metric-value">{perf['total_signals']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Executed</div>
                <div class="metric-value">{perf['executed']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Execution Rate</div>
                <div class="metric-value">{perf['execution_rate']:.1f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Expected Return</div>
                <div class="metric-value">${perf['executed_return']:.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Risk</div>
                <div class="metric-value">${perf['total_risk']:.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Risk/Reward</div>
                <div class="metric-value">{perf['risk_reward_ratio']:.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Avg Confidence</div>
                <div class="metric-value">{perf['avg_confidence']:.1%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Trading Days</div>
                <div class="metric-value">{perf['trading_days']}</div>
            </div>
        </div>

        <h2>Strategy Comparison</h2>
        <table>
            <tr>
                <th>Strategy</th>
                <th>Signals</th>
                <th>Executed</th>
                <th>Rate</th>
                <th>Return</th>
                <th>Risk</th>
                <th>R/R</th>
            </tr>
            {strategies_rows}
        </table>

        <h2>Risk Analysis</h2>
        <p>Risk Status: <strong>{risk['overall_risk_status']}</strong></p>
        <p>Risk Rejections: {risk['risk_rejections']}</p>
        <p>Budget Rejections: {risk['budget_rejections']}</p>
        <p>Max Single Risk: ${risk['max_single_risk']:.2f}</p>

        <h2>Recommendations</h2>
        {recommendations_html if recommendations_html else '<p>No specific recommendations at this time.</p>'}

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
        print("\n" + "=" * 100)
        print(f"WEEKLY TRADING REPORT - {report['metadata']['week_start']} to {report['metadata']['week_end']}")
        print("=" * 100)

        perf = report['performance_summary']

        print("\nPERFORMANCE SUMMARY")
        print("-" * 100)
        print(f"  Total Signals: {perf['total_signals']:<6} Executed: {perf['executed']:<6} Rejected: {perf['rejected']:<6}")
        print(f"  Execution Rate: {perf['execution_rate']:>6.1f}%")
        print(f"  Expected Return (Executed): ${perf['executed_return']:>10,.2f}")
        print(f"  Total Risk: ${perf['total_risk']:>10,.2f}")
        print(f"  Risk/Reward Ratio: {perf['risk_reward_ratio']:>6.2f}")
        print(f"  Avg Confidence: {perf['avg_confidence']:>6.1%}")
        print(f"  Trading Days: {perf['trading_days']:<6} Strategies: {perf['num_strategies']:<6}")

        print("\nSTRATEGY COMPARISON")
        print("-" * 100)
        strat = report['strategy_comparison']
        for name, data in strat.items():
            print(f"  {name:<30} Signals: {data['total_signals']:<4} Exec: {data['executed']:<4} Rate: {data['execution_rate']:>6.1f}% Return: ${data['executed_return']:>10,.2f}")

        print("\nRISK ANALYSIS")
        print("-" * 100)
        risk = report['risk_analysis']
        print(f"  Status: {risk['overall_risk_status']}")
        print(f"  Risk Rejections: {risk['risk_rejections']}")
        print(f"  Budget Rejections: {risk['budget_rejections']}")
        print(f"  Max Single Risk: ${risk['max_single_risk']:.2f}")

        print("\nRECOMMENDATIONS")
        print("-" * 100)
        recs = report['recommendations']
        if recs:
            for i, rec in enumerate(recs, 1):
                print(f"  {i}. [{rec['severity']}] {rec['category']}")
                print(f"     {rec['message']}")
                print(f"     → {rec['action']}\n")
        else:
            print("  No specific recommendations at this time.")

        print("=" * 100)


def main():
    """Main entry point"""
    try:
        # Check for command line arguments
        week_offset = 0
        if len(sys.argv) > 1:
            try:
                week_offset = int(sys.argv[1])
            except ValueError:
                print(f"Invalid week offset: {sys.argv[1]}")
                return 1

        generator = WeeklyReportGenerator(week_offset=week_offset)

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
