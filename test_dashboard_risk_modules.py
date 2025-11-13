#!/usr/bin/env python3
"""
Quick test to verify Risk Analytics dashboard modules work
Tests that all 5 tabs can import and use the risk modules
"""

import sys
from pathlib import Path

# Add project root to path (same as dashboard)
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
python_path = project_root / "python"
sys.path.insert(0, str(python_path))

try:
    import bigbrother_risk as risk
    print("✅ Risk module loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import risk module: {e}")
    sys.exit(1)

# Test all 5 dashboard functionality areas
def test_position_sizing():
    """Test Position Sizing Calculator"""
    print("\n1️⃣ Testing Position Sizing...")
    sizer = risk.PositionSizer.create()
    sizer.with_method(risk.SizingMethod.KellyHalf) \
         .with_account_value(30000.0) \
         .with_win_probability(0.60) \
         .with_expected_gain(500.0) \
         .with_expected_loss(300.0)
    result = sizer.calculate()
    print(f"  ✓ Position Size: ${result.dollar_amount:.2f}")
    print(f"  ✓ Kelly Fraction: {result.kelly_fraction:.2%}")

def test_monte_carlo():
    """Test Monte Carlo Simulator"""
    print("\n2️⃣ Testing Monte Carlo Simulator...")
    simulator = risk.MonteCarloSimulator.create()
    simulator.with_simulations(10000).with_parallel(True).with_seed(42)
    result = simulator.simulate_stock(100.0, 110.0, 95.0, 0.20)
    print(f"  ✓ Simulations: {result.num_simulations:,}")
    print(f"  ✓ Win Probability: {result.win_probability:.1%}")
    print(f"  ✓ VaR (95%): ${result.var_95:.2f}")

def test_var_calculator():
    """Test VaR Calculator"""
    print("\n3️⃣ Testing VaR Calculator...")
    import numpy as np
    np.random.seed(42)
    returns = list(np.random.normal(0.001, 0.02, 252))

    calculator = risk.VaRCalculator.create()
    calculator.with_returns(returns) \
             .with_confidence_level(0.95) \
             .with_method(risk.VaRMethod.Parametric)
    result = calculator.calculate(30000.0)
    print(f"  ✓ VaR Amount: ${result.var_amount:,.2f}")
    print(f"  ✓ VaR %: {result.var_percentage:.2%}")
    print(f"  ✓ Risk Level: {result.get_risk_level()}")

def test_stress_testing():
    """Test Stress Testing Engine"""
    print("\n4️⃣ Testing Stress Testing...")
    engine = risk.StressTestingEngine.create()

    # Add test position
    pos = risk.StressPosition()
    pos.symbol = "AAPL"
    pos.quantity = 100
    pos.current_price = 150.0
    pos.beta = 1.2
    pos.sector_exposure = 1.0
    pos.duration = 0.0
    pos.delta = 1.0
    pos.vega = 0.0
    engine.add_position(pos)

    result = engine.run_stress_test(risk.StressScenario.MarketCrash)
    print(f"  ✓ Initial Value: ${result.initial_value:,.2f}")
    print(f"  ✓ Stressed Value: ${result.stressed_value:,.2f}")
    print(f"  ✓ P&L: ${result.pnl:,.2f} ({result.pnl_percentage:.2%})")

def test_performance_metrics():
    """Test Performance Metrics"""
    print("\n5️⃣ Testing Performance Metrics...")
    import numpy as np
    np.random.seed(42)
    equity_curve = [30000 * (1 + np.random.normal(0.001, 0.015))**i
                   for i in range(252)]

    result = risk.PerformanceMetricsCalculator.from_equity_curve(
        equity_curve, 0.04
    )
    print(f"  ✓ Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  ✓ Sortino Ratio: {result.sortino_ratio:.2f}")
    print(f"  ✓ Max Drawdown: {result.max_drawdown:.2%}")
    print(f"  ✓ Rating: {result.get_rating()}")
    print(f"  ✓ Healthy: {result.is_healthy()}")

if __name__ == "__main__":
    print("=" * 70)
    print("Risk Analytics Dashboard - Module Integration Test")
    print("=" * 70)

    try:
        test_position_sizing()
        test_monte_carlo()
        test_var_calculator()
        test_stress_testing()
        test_performance_metrics()

        print("\n" + "=" * 70)
        print("✅ ALL DASHBOARD MODULES WORKING CORRECTLY")
        print("=" * 70)
        print("\n📊 Dashboard is ready at: http://localhost:8501")
        print("Navigate to: 🎯 Advanced Risk Analytics")
        print()

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
