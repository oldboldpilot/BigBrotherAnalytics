"""
Circuit Breaker Monitoring Dashboard Component

Displays circuit breaker status for all services in the BigBrotherAnalytics dashboard.
Shows real-time state, statistics, and manual reset controls.
"""

import streamlit as st
import duckdb
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "scripts" / "data_collection"))

try:
    from circuit_breaker_wrapper import get_global_manager, CircuitState
except ImportError:
    st.error("Circuit breaker module not found. Make sure circuit_breaker_wrapper.py is available.")
    get_global_manager = None
    CircuitState = None


def render_circuit_breaker_dashboard():
    """
    Render circuit breaker status dashboard

    Shows:
    - Circuit breaker states (CLOSED, OPEN, HALF_OPEN)
    - Statistics (success rate, failure count, etc.)
    - Manual reset controls
    - State history
    """
    st.header("Circuit Breaker Status")

    if get_global_manager is None:
        st.error("Circuit breaker monitoring unavailable")
        return

    # Get circuit breaker manager
    manager = get_global_manager()

    # Get all circuit breaker statistics
    all_stats = manager.get_all_stats()

    if not all_stats:
        st.info("No circuit breakers registered. Circuit breakers will appear here once initialized.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    total_breakers = len(all_stats)
    open_count = sum(1 for stats in all_stats.values() if stats.state == CircuitState.OPEN)
    half_open_count = sum(1 for stats in all_stats.values() if stats.state == CircuitState.HALF_OPEN)
    closed_count = sum(1 for stats in all_stats.values() if stats.state == CircuitState.CLOSED)

    with col1:
        st.metric("Total Circuits", total_breakers)

    with col2:
        st.metric(
            "🟢 Closed",
            closed_count,
            help="Circuits in normal operation"
        )

    with col3:
        st.metric(
            "🟡 Half-Open",
            half_open_count,
            delta="Testing recovery" if half_open_count > 0 else None,
            help="Circuits testing recovery"
        )

    with col4:
        st.metric(
            "🔴 Open",
            open_count,
            delta="Alert!" if open_count > 0 else None,
            delta_color="inverse",
            help="Circuits in failure mode"
        )

    # Alert if any circuit is open
    if open_count > 0:
        st.error(
            f"⚠️ {open_count} circuit breaker(s) OPEN! "
            "System is using cached data or degraded functionality."
        )

    st.divider()

    # Circuit Breaker Details
    st.subheader("Circuit Breaker Details")

    for name, stats in all_stats.items():
        render_circuit_breaker_card(name, stats, manager)


def render_circuit_breaker_card(name: str, stats, manager):
    """
    Render individual circuit breaker card

    Args:
        name: Circuit breaker name
        stats: CircuitStats object
        manager: CircuitBreakerManager instance
    """
    # State-based styling
    state_colors = {
        CircuitState.CLOSED: "🟢",
        CircuitState.OPEN: "🔴",
        CircuitState.HALF_OPEN: "🟡",
    }

    state_emoji = state_colors.get(stats.state, "⚪")

    with st.expander(f"{state_emoji} **{name}** - {stats.state.value}", expanded=(stats.state != CircuitState.CLOSED)):
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.markdown("### Statistics")
            st.metric("Total Calls", stats.total_calls)
            st.metric("Success Rate", f"{stats.success_rate * 100:.1f}%")
            st.metric("Consecutive Failures", stats.consecutive_failures)

        with col2:
            st.markdown("### Counts")
            st.metric("Successes", stats.success_count)
            st.metric("Failures", stats.failure_count)

            # Show last error if exists
            if stats.last_error:
                st.error(f"Last Error: {stats.last_error}")

        with col3:
            st.markdown("### Controls")

            # Reset button
            if st.button(f"Reset {name}", key=f"reset_{name}", help="Manually reset circuit to CLOSED"):
                breaker = manager.get(name)
                if breaker:
                    breaker.reset()
                    st.success(f"Reset {name} circuit breaker")
                    st.rerun()

            # State badge
            if stats.state == CircuitState.OPEN:
                st.error("Circuit OPEN")
            elif stats.state == CircuitState.HALF_OPEN:
                st.warning("Testing Recovery")
            else:
                st.success("Circuit CLOSED")

        # Timestamps
        st.markdown("### Timing")
        time_col1, time_col2, time_col3 = st.columns(3)

        with time_col1:
            if stats.last_success_time:
                st.text(f"Last Success:\n{stats.last_success_time.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.text("Last Success:\nNever")

        with time_col2:
            if stats.last_failure_time:
                st.text(f"Last Failure:\n{stats.last_failure_time.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.text("Last Failure:\nNever")

        with time_col3:
            if stats.circuit_opened_at:
                st.text(f"Opened At:\n{stats.circuit_opened_at.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.text("Opened At:\nN/A")

        # Success rate visualization
        if stats.total_calls > 0:
            fig = go.Figure(data=[
                go.Bar(
                    x=['Success', 'Failure'],
                    y=[stats.success_count, stats.failure_count],
                    marker_color=['green', 'red']
                )
            ])
            fig.update_layout(
                title=f"{name} Call Distribution",
                yaxis_title="Count",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)


def render_circuit_breaker_history():
    """
    Render circuit breaker state change history

    Shows timeline of state transitions from database logs.
    """
    st.subheader("Circuit Breaker History")

    # Query circuit breaker logs from database
    # Note: This requires a circuit_breaker_events table in DuckDB
    # For now, show placeholder

    st.info("Circuit breaker history tracking will be available after logging is implemented.")

    # Placeholder data
    sample_data = {
        'timestamp': [
            datetime.now() - timedelta(hours=2),
            datetime.now() - timedelta(hours=1),
            datetime.now() - timedelta(minutes=30),
        ],
        'circuit': ['schwab_market_data', 'schwab_market_data', 'schwab_market_data'],
        'from_state': ['CLOSED', 'OPEN', 'HALF_OPEN'],
        'to_state': ['OPEN', 'HALF_OPEN', 'CLOSED'],
        'reason': [
            '5 consecutive failures',
            'Timeout elapsed',
            'Successful test call'
        ]
    }

    df = pd.DataFrame(sample_data)

    if not df.empty:
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


def render_reset_all_button(manager):
    """
    Render reset all circuits button

    Args:
        manager: CircuitBreakerManager instance
    """
    st.divider()

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if st.button(
            "🔄 Reset All Circuits",
            key="reset_all",
            help="Reset all circuit breakers to CLOSED state",
            type="primary"
        ):
            manager.reset_all()
            st.success("All circuit breakers reset to CLOSED")
            st.rerun()


def main():
    """
    Main function for standalone circuit breaker monitoring page
    """
    st.set_page_config(
        page_title="Circuit Breaker Monitor",
        page_icon="⚡",
        layout="wide",
    )

    st.title("⚡ Circuit Breaker Monitoring")
    st.markdown("Real-time monitoring of circuit breaker states across all services")

    # Render dashboard
    render_circuit_breaker_dashboard()

    # Render history
    render_circuit_breaker_history()

    # Reset all button
    if get_global_manager:
        render_reset_all_button(get_global_manager())

    # Auto-refresh
    st.markdown("---")
    st.caption("Dashboard refreshes automatically. Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
