"""
BigBrotherAnalytics - Advanced Risk Analytics Dashboard

Comprehensive risk management dashboard using C++23 risk modules:
- Position Sizing (Kelly Criterion)
- Monte Carlo Simulation
- VaR Calculator (4 methods)
- Stress Testing (7 scenarios)
- Performance Metrics (Sharpe, Sortino, etc.)
- Correlation Analysis
- Portfolio Risk Management

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-13
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
python_path = project_root / "python"
sys.path.insert(0, str(python_path))

try:
    import bigbrother_risk as risk
    RISK_MODULE_AVAILABLE = True
except ImportError as e:
    st.warning(f"⚠️ Risk management module not available: {e}")
    RISK_MODULE_AVAILABLE = False


def show_position_sizing_calculator():
    """Interactive position sizing calculator using Kelly Criterion"""
    st.subheader("📊 Position Sizing Calculator")

    if not RISK_MODULE_AVAILABLE:
        st.error("Risk management module not loaded")
        return

    col1, col2 = st.columns(2)

    with col1:
        account_value = st.number_input(
            "Account Value ($)",
            min_value=1000.0,
            value=30000.0,
            step=1000.0
        )
        win_prob = st.slider(
            "Win Probability",
            min_value=0.0,
            max_value=1.0,
            value=0.60,
            step=0.01
        )
        expected_gain = st.number_input(
            "Expected Gain ($)",
            min_value=0.0,
            value=500.0,
            step=10.0
        )

    with col2:
        expected_loss = st.number_input(
            "Expected Loss ($)",
            min_value=0.0,
            value=300.0,
            step=10.0
        )
        max_position = st.number_input(
            "Max Position Size ($)",
            min_value=0.0,
            value=2000.0,
            step=100.0
        )
        method = st.selectbox(
            "Sizing Method",
            ["KellyHalf", "KellyCriterion", "KellyQuarter", "FixedPercent"],
            index=0
        )

    if st.button("Calculate Position Size", type="primary"):
        try:
            # Create position sizer
            sizer = risk.PositionSizer.create()
            method_enum = getattr(risk.SizingMethod, method)

            sizer.with_method(method_enum) \
                 .with_account_value(account_value) \
                 .with_win_probability(win_prob) \
                 .with_expected_gain(expected_gain) \
                 .with_expected_loss(expected_loss) \
                 .with_max_position(max_position)

            result = sizer.calculate()

            # Display results
            st.success("✅ Position Size Calculated")

            col1, col2, col3 = st.columns(3)
            col1.metric("Position Size", f"${result.dollar_amount:,.2f}")
            col2.metric("Kelly Fraction", f"{result.kelly_fraction:.2%}")
            col3.metric("Risk %", f"{result.risk_percent:.2%}")

            # Risk/Reward visualization
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Expected Gain', 'Expected Loss', 'Net EV'],
                y=[expected_gain, -expected_loss,
                   win_prob * expected_gain - (1-win_prob) * expected_loss],
                marker_color=['green', 'red', 'blue']
            ))
            fig.update_layout(
                title="Risk/Reward Profile",
                yaxis_title="Amount ($)",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error calculating position size: {e}")


def show_monte_carlo_simulator():
    """Monte Carlo simulation for trade analysis"""
    st.subheader("🎲 Monte Carlo Trade Simulator")

    if not RISK_MODULE_AVAILABLE:
        st.error("Risk management module not loaded")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        entry_price = st.number_input("Entry Price ($)", value=100.0, step=1.0)
        target_price = st.number_input("Target Price ($)", value=110.0, step=1.0)

    with col2:
        stop_price = st.number_input("Stop Loss ($)", value=95.0, step=1.0)
        volatility = st.slider("Volatility", 0.05, 0.50, 0.20, 0.01)

    with col3:
        num_sims = st.selectbox(
            "Simulations",
            [1000, 5000, 10000, 50000, 100000],
            index=2
        )

    if st.button("Run Simulation", type="primary"):
        try:
            with st.spinner(f"Running {num_sims:,} simulations..."):
                simulator = risk.MonteCarloSimulator.create()
                simulator.with_simulations(num_sims) \
                         .with_parallel(True) \
                         .with_seed(42)

                result = simulator.simulate_stock(
                    entry_price, target_price, stop_price, volatility
                )

            st.success(f"✅ Completed {result.num_simulations:,} simulations")

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Win Probability", f"{result.win_probability:.1%}")
            col2.metric("Mean P&L", f"${result.mean_pnl:.2f}")
            col3.metric("VaR (95%)", f"${result.var_95:.2f}")
            col4.metric("CVaR (95%)", f"${result.cvar_95:.2f}")

            # Distribution stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Min P&L", f"${result.min_pnl:.2f}")
            col2.metric("Median P&L", f"${result.median_pnl:.2f}")
            col3.metric("Max P&L", f"${result.max_pnl:.2f}")

            # P&L distribution visualization
            st.subheader("P&L Distribution")
            st.info(f"📊 Based on {num_sims:,} Monte Carlo paths")

        except Exception as e:
            st.error(f"Error running simulation: {e}")


def show_var_calculator(returns_data: Optional[List[float]] = None):
    """Value at Risk calculator with 4 methods"""
    st.subheader("📉 Value at Risk (VaR) Calculator")

    if not RISK_MODULE_AVAILABLE:
        st.error("Risk management module not loaded")
        return

    col1, col2 = st.columns(2)

    with col1:
        portfolio_value = st.number_input(
            "Portfolio Value ($)",
            value=30000.0,
            step=1000.0
        )
        confidence = st.slider(
            "Confidence Level",
            0.90, 0.99, 0.95, 0.01
        )

    with col2:
        method = st.selectbox(
            "VaR Method",
            ["Parametric", "Historical", "MonteCarlo", "Hybrid"],
            index=0
        )

    # Generate sample returns if not provided
    if returns_data is None:
        st.info("Using simulated return data. Connect to live data for real analysis.")
        np.random.seed(42)
        returns_data = list(np.random.normal(0.001, 0.02, 252))  # Daily returns

    if st.button("Calculate VaR", type="primary"):
        try:
            calculator = risk.VaRCalculator.create()
            method_enum = getattr(risk.VaRMethod, method)

            calculator.with_returns(returns_data) \
                     .with_confidence_level(confidence) \
                     .with_method(method_enum)

            result = calculator.calculate(portfolio_value)

            st.success(f"✅ VaR Calculated ({method} method)")

            # VaR metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("VaR Amount", f"${result.var_amount:,.2f}")
            col2.metric("VaR %", f"{result.var_percentage:.2%}")
            col3.metric("Risk Level", result.get_risk_level())

            col1, col2 = st.columns(2)
            col1.metric("Expected Shortfall", f"${result.expected_shortfall:,.2f}")
            col2.metric("Volatility", f"{result.volatility:.2%}")

            # Risk level indicator
            risk_level = result.get_risk_level()
            if risk_level == "HIGH":
                st.error(f"⚠️ HIGH RISK: VaR indicates significant downside exposure")
            elif risk_level == "MEDIUM":
                st.warning(f"⚠️ MEDIUM RISK: Monitor portfolio closely")
            else:
                st.success(f"✅ LOW RISK: VaR within acceptable range")

        except Exception as e:
            st.error(f"Error calculating VaR: {e}")


def show_stress_testing():
    """Portfolio stress testing with multiple scenarios"""
    st.subheader("💥 Portfolio Stress Testing")

    if not RISK_MODULE_AVAILABLE:
        st.error("Risk management module not loaded")
        return

    st.info("Simulate portfolio performance under extreme market conditions")

    # Sample portfolio setup
    with st.expander("📋 Configure Portfolio", expanded=True):
        num_positions = st.slider("Number of Positions", 1, 5, 3)

        positions = []
        for i in range(num_positions):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                symbol = st.text_input(f"Symbol {i+1}", f"STOCK{i+1}", key=f"sym_{i}")
            with col2:
                qty = st.number_input(f"Quantity", 10.0, 1000.0, 100.0, key=f"qty_{i}")
            with col3:
                price = st.number_input(f"Price ($)", 10.0, 500.0, 100.0, key=f"px_{i}")
            with col4:
                beta = st.number_input(f"Beta", 0.5, 2.0, 1.0, 0.1, key=f"beta_{i}")

            positions.append({
                'symbol': symbol,
                'quantity': qty,
                'price': price,
                'beta': beta
            })

    scenarios = st.multiselect(
        "Stress Scenarios",
        ["MarketCrash", "VolatilitySpike", "SectorRotation",
         "InterestRateShock", "BlackSwan", "FlashCrash"],
        default=["MarketCrash", "BlackSwan"]
    )

    if st.button("Run Stress Tests", type="primary"):
        try:
            engine = risk.StressTestingEngine.create()

            # Add positions
            for pos in positions:
                stress_pos = risk.StressPosition()
                stress_pos.symbol = pos['symbol']
                stress_pos.quantity = pos['quantity']
                stress_pos.current_price = pos['price']
                stress_pos.beta = pos['beta']
                stress_pos.sector_exposure = 1.0
                stress_pos.duration = 0.0
                stress_pos.delta = 1.0
                stress_pos.vega = 0.0

                engine.add_position(stress_pos)

            initial_value = engine.get_total_value()
            st.info(f"📊 Portfolio Value: ${initial_value:,.2f}")

            # Run selected scenarios
            results = []
            for scenario_name in scenarios:
                scenario_enum = getattr(risk.StressScenario, scenario_name)
                result = engine.run_stress_test(scenario_enum)
                results.append({
                    'Scenario': scenario_name,
                    'Initial Value': result.initial_value,
                    'Stressed Value': result.stressed_value,
                    'P&L': result.pnl,
                    'P&L %': result.pnl_percentage * 100,
                    'Severity': result.get_severity(),
                    'Viable': '✅' if result.is_portfolio_viable() else '❌'
                })

            # Display results
            results_df = pd.DataFrame(results)
            st.dataframe(
                results_df.style.format({
                    'Initial Value': '${:,.2f}',
                    'Stressed Value': '${:,.2f}',
                    'P&L': '${:,.2f}',
                    'P&L %': '{:.2f}%'
                }),
                use_container_width=True
            )

            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=results_df['Scenario'],
                y=results_df['P&L %'],
                marker_color=['red' if x < 0 else 'green' for x in results_df['P&L %']],
                text=results_df['P&L %'].apply(lambda x: f"{x:.1f}%"),
                textposition='outside'
            ))
            fig.update_layout(
                title="Stress Test Results (P&L %)",
                xaxis_title="Scenario",
                yaxis_title="P&L %",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error running stress tests: {e}")


def show_performance_metrics(equity_curve: Optional[List[float]] = None):
    """Calculate and display performance metrics"""
    st.subheader("📈 Performance Metrics")

    if not RISK_MODULE_AVAILABLE:
        st.error("Risk management module not loaded")
        return

    # Generate sample equity curve if not provided
    if equity_curve is None:
        st.info("Using simulated equity curve. Connect to live data for real analysis.")
        np.random.seed(42)
        equity_curve = [30000 * (1 + np.random.normal(0.001, 0.015))**i
                       for i in range(252)]

    risk_free_rate = st.slider("Risk-Free Rate", 0.0, 0.10, 0.04, 0.01)

    if st.button("Calculate Metrics", type="primary"):
        try:
            result = risk.PerformanceMetricsCalculator.from_equity_curve(
                equity_curve, risk_free_rate
            )

            st.success(f"✅ Performance Rating: {result.get_rating()}")

            # Risk-adjusted returns
            st.subheader("Risk-Adjusted Returns")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
            col2.metric("Sortino Ratio", f"{result.sortino_ratio:.2f}")
            col3.metric("Calmar Ratio", f"{result.calmar_ratio:.2f}")
            col4.metric("Omega Ratio", f"{result.omega_ratio:.2f}")

            # Return metrics
            st.subheader("Return Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Return", f"{result.total_return:.2%}")
            col2.metric("Annual Return", f"{result.annualized_return:.2%}")
            col3.metric("Avg Return", f"{result.average_return:.2%}")

            # Risk metrics
            st.subheader("Risk Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Volatility", f"{result.volatility:.2%}")
            col2.metric("Max Drawdown", f"{result.max_drawdown:.2%}")
            col3.metric("Downside Dev", f"{result.downside_deviation:.2%}")

            # Win/Loss stats
            st.subheader("Win/Loss Statistics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Win Rate", f"{result.win_rate:.2%}")
            col2.metric("Profit Factor", f"{result.profit_factor:.2f}")
            col3.metric("Expectancy", f"${result.expectancy:.2f}")

            # Distribution
            col1, col2 = st.columns(2)
            col1.metric("Skewness", f"{result.skewness:.2f}")
            col2.metric("Kurtosis", f"{result.kurtosis:.2f}")

            # Health indicator
            if result.is_healthy():
                st.success("✅ Portfolio is healthy")
            else:
                st.warning("⚠️ Portfolio needs attention")

        except Exception as e:
            st.error(f"Error calculating metrics: {e}")


def show_risk_analytics_view():
    """Main risk analytics dashboard view"""
    st.title("🎯 Advanced Risk Analytics")
    st.markdown("Powered by C++23 Risk Management Framework with SIMD/MKL acceleration")

    if not RISK_MODULE_AVAILABLE:
        st.error("⚠️ Risk management C++ module not loaded. Build the project first.")
        st.code("ninja -C build risk_management bigbrother_risk")
        return

    # Module availability
    with st.expander("📦 Module Status"):
        st.success("✅ Risk Management Framework Loaded")
        st.info("""
        **Available Modules:**
        - Position Sizer (Kelly Criterion)
        - Monte Carlo Simulator (OpenMP parallel)
        - VaR Calculator (4 methods with MKL)
        - Stress Testing Engine (AVX2 SIMD)
        - Performance Metrics (Sharpe, Sortino, etc.)
        - Correlation Analyzer (MKL)
        - Risk Manager (Portfolio orchestration)
        """)

    # Tabs for different risk tools
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Position Sizing",
        "🎲 Monte Carlo",
        "📉 VaR Calculator",
        "💥 Stress Testing",
        "📈 Performance"
    ])

    with tab1:
        show_position_sizing_calculator()

    with tab2:
        show_monte_carlo_simulator()

    with tab3:
        show_var_calculator()

    with tab4:
        show_stress_testing()

    with tab5:
        show_performance_metrics()

    # Footer
    st.markdown("---")
    st.caption("Risk Analytics powered by BigBrotherAnalytics C++23 Framework")
