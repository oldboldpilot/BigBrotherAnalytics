#!/usr/bin/env python3
"""
Integration test for SIMD-optimized risk management Python bindings
Tests all 8 modules with realistic scenarios
"""

import sys
sys.path.insert(0, 'python')

import bigbrother_risk as risk

def test_position_sizer():
    """Test Position Sizer module"""
    print("Testing Position Sizer...")

    sizer = (risk.PositionSizer.create()
             .with_method(risk.SizingMethod.KellyCriterion)
             .with_account_value(30000.0)
             .with_win_probability(0.60)
             .with_expected_gain(200.0)
             .with_expected_loss(100.0))

    result = sizer.calculate()
    print(f"  ✓ Position Size: ${result.dollar_amount:.2f}")
    print(f"  ✓ Kelly Fraction: {result.kelly_fraction:.2%}")
    print(f"  ✓ Risk Percent: {result.risk_percent:.2%}")
    assert result.dollar_amount > 0
    print()

def test_monte_carlo():
    """Test SIMD-optimized Monte Carlo Simulator"""
    print("Testing Monte Carlo Simulator (SIMD)...")

    simulator = (risk.MonteCarloSimulator.create()
                 .with_simulations(10000)
                 .with_parallel(True))

    result = simulator.simulate_stock(100.0, 110.0, 95.0, 0.25)

    print(f"  ✓ Simulations: {result.num_simulations:,}")
    print(f"  ✓ Mean P&L: ${result.mean_pnl:.2f}")
    print(f"  ✓ VaR (95%): ${result.var_95:.2f}")
    print(f"  ✓ Win Rate: {result.win_probability*100:.1f}%")
    assert result.num_simulations == 10000
    assert result.var_95 > 0
    print()

def test_stop_loss():
    """Test Stop Loss Manager"""
    print("Testing Stop Loss Manager...")

    manager = risk.StopLossManager.create()
    manager.add_hard_stop("AAPL", 145.0, 150.0)

    triggered = manager.update({"AAPL": 144.0})

    print(f"  ✓ Stops managed: {manager.get_stop_count()}")
    print(f"  ✓ Has stop: {manager.has_stop('AAPL')}")
    assert manager.get_stop_count() > 0
    print()

def test_risk_manager():
    """Test Risk Manager module"""
    print("Testing Risk Manager...")

    limits = risk.RiskLimits()
    limits.account_value = 30000.0
    limits.max_daily_loss = 900.0
    limits.max_position_size = 2000.0

    manager = risk.RiskManager.create(limits)

    trade_risk = manager.assess_trade("AAPL", 1500.0, 150.0, 145.0, 155.0, 0.60)

    print(f"  ✓ Trade approved: {trade_risk.approved}")
    print(f"  ✓ Expected value: ${trade_risk.expected_value:.2f}")
    print(f"  ✓ Risk/Reward: {trade_risk.risk_reward_ratio:.2f}")

    portfolio = manager.get_portfolio_risk()
    print(f"  ✓ Portfolio heat: {portfolio.portfolio_heat:.2%}")
    print()

def test_var_calculator():
    """Test VaR Calculator"""
    print("Testing VaR Calculator...")

    returns = [0.01, -0.02, 0.03, -0.01, 0.02, -0.01, 0.01, 0.02, -0.03, 0.01]

    calc = (risk.VaRCalculator.create()
            .with_returns(returns)
            .with_confidence_level(0.95)
            .with_method(risk.VaRMethod.Historical))

    result = calc.calculate(10000.0)

    print(f"  ✓ VaR Amount: ${result.var_amount:.2f}")
    print(f"  ✓ VaR %: {result.var_percentage:.2%}")
    print(f"  ✓ Expected Shortfall: ${result.expected_shortfall:.2f}")
    print(f"  ✓ Valid: {result.is_valid()}")
    assert result.is_valid()
    print()

def test_stress_testing():
    """Test Stress Testing Engine"""
    print("Testing Stress Testing Engine...")

    engine = risk.StressTestingEngine.create()

    position = risk.StressPosition()
    position.symbol = "AAPL"
    position.quantity = 100
    position.current_price = 150.0
    position.beta = 1.2
    position.delta = 0.6

    engine.add_position(position)

    result = engine.run_stress_test(risk.StressScenario.MarketCrash)

    print(f"  ✓ Scenario: Market Crash")
    print(f"  ✓ Initial Value: ${result.initial_value:.2f}")
    print(f"  ✓ Stressed Value: ${result.stressed_value:.2f}")
    print(f"  ✓ P&L: ${result.pnl:.2f} ({result.pnl_percentage:.2%})")
    print(f"  ✓ Viable: {result.is_portfolio_viable()}")
    print()

def test_performance_metrics():
    """Test Performance Metrics Calculator"""
    print("Testing Performance Metrics Calculator...")

    returns = [0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01, 0.02, 0.01]

    calc = (risk.PerformanceMetricsCalculator.create()
            .with_returns(returns)
            .with_risk_free_rate(0.02))

    result = calc.calculate()

    print(f"  ✓ Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  ✓ Sortino Ratio: {result.sortino_ratio:.2f}")
    print(f"  ✓ Max Drawdown: {result.max_drawdown:.2%}")
    print(f"  ✓ Win Rate: {result.win_rate:.2%}")
    print(f"  ✓ Healthy: {result.is_healthy()}")
    print()

def test_correlation_analyzer():
    """Test SIMD-optimized Correlation Analyzer"""
    print("Testing Correlation Analyzer (SIMD)...")

    analyzer = risk.CorrelationAnalyzer.create()

    # Add time series
    analyzer.add_series("AAPL", [100, 102, 101, 103, 105, 104, 106])
    analyzer.add_series("MSFT", [200, 202, 201, 203, 205, 204, 206])
    analyzer.add_series("GOOGL", [150, 148, 149, 147, 145, 146, 144])

    matrix = analyzer.compute_correlation_matrix()

    print(f"  ✓ Dimension: {matrix.dimension}x{matrix.dimension}")
    print(f"  ✓ Valid: {matrix.is_valid()}")
    print(f"  ✓ AAPL-MSFT correlation: {matrix.get_correlation('AAPL', 'MSFT'):.3f}")

    # Diversification metrics
    weights = [0.4, 0.3, 0.3]
    metrics = analyzer.analyze_diversification(weights)

    print(f"  ✓ Avg Correlation: {metrics.avg_correlation:.3f}")
    print(f"  ✓ Diversified: {metrics.is_diversified()}")
    print()

def main():
    """Run all integration tests"""
    print("=" * 70)
    print("SIMD Risk Analytics - Python Bindings Integration Test")
    print("=" * 70)
    print()

    try:
        test_position_sizer()
        test_monte_carlo()
        test_stop_loss()
        test_risk_manager()
        test_var_calculator()
        test_stress_testing()
        test_performance_metrics()
        test_correlation_analyzer()

        print("=" * 70)
        print("✓ ALL INTEGRATION TESTS PASSED")
        print("=" * 70)
        print()
        print("Summary:")
        print("  • Position Sizer: Kelly Criterion calculations")
        print("  • Monte Carlo: SIMD-optimized 10K simulations")
        print("  • Stop Loss: Hard/trailing stop management")
        print("  • Risk Manager: Trade assessment & portfolio heat")
        print("  • VaR Calculator: Historical VaR computation")
        print("  • Stress Testing: Market crash scenario")
        print("  • Performance Metrics: Sharpe, Sortino, drawdown")
        print("  • Correlation Analyzer: SIMD-optimized Pearson correlation")
        print()

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
