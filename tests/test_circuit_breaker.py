#!/usr/bin/env python3
"""
Circuit Breaker Testing Suite

Tests circuit breaker behavior with simulated failures:
- State transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- Failure threshold enforcement
- Timeout and recovery
- Fallback strategies
- Cache behavior

Usage:
    uv run python tests/test_circuit_breaker.py
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent.parent / "scripts" / "data_collection"))

from circuit_breaker_wrapper import CircuitBreakerWrapper, CircuitState

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestResults:
    """Track test results"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.tests = []

    def record(self, name: str, passed: bool, message: str = ""):
        """Record test result"""
        self.total += 1
        if passed:
            self.passed += 1
            logger.info(f"✅ PASS: {name}")
        else:
            self.failed += 1
            logger.error(f"❌ FAIL: {name} - {message}")

        self.tests.append({
            'name': name,
            'passed': passed,
            'message': message
        })

    def summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("CIRCUIT BREAKER TEST RESULTS")
        print("=" * 80)
        print(f"Total Tests: {self.total}")
        print(f"Passed: {self.passed} ({self.passed / self.total * 100:.1f}%)")
        print(f"Failed: {self.failed} ({self.failed / self.total * 100:.1f}%)")
        print("=" * 80)

        if self.failed > 0:
            print("\nFailed Tests:")
            for test in self.tests:
                if not test['passed']:
                    print(f"  - {test['name']}: {test['message']}")

        return self.failed == 0


def test_circuit_closed_to_open(results: TestResults):
    """Test: Circuit transitions CLOSED -> OPEN after threshold failures"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Circuit CLOSED -> OPEN transition")
    logger.info("=" * 80)

    breaker = CircuitBreakerWrapper("test_closed_to_open", failure_threshold=3, timeout_seconds=5)

    # Verify initial state
    results.record(
        "Initial state is CLOSED",
        breaker.state == CircuitState.CLOSED,
        f"Expected CLOSED, got {breaker.state}"
    )

    # Simulate failures
    def failing_function():
        raise Exception("Simulated failure")

    failure_count = 0
    for i in range(5):
        try:
            breaker.call(failing_function)
        except Exception:
            failure_count += 1
            logger.info(f"Failure {failure_count}/3 recorded")

        # Check if circuit opened after threshold
        if i >= 2:  # After 3 failures (0, 1, 2)
            expected_open = breaker.state == CircuitState.OPEN
            results.record(
                f"Circuit OPEN after {i+1} failures",
                expected_open,
                f"Expected OPEN after {i+1} failures, got {breaker.state}"
            )
            if expected_open:
                break

    # Verify stats
    stats = breaker.get_stats()
    results.record(
        "Consecutive failures tracked correctly",
        stats.consecutive_failures >= 3,
        f"Expected >= 3, got {stats.consecutive_failures}"
    )


def test_circuit_open_fails_fast(results: TestResults):
    """Test: Circuit in OPEN state fails fast"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Circuit OPEN fails fast")
    logger.info("=" * 80)

    breaker = CircuitBreakerWrapper("test_open_fails_fast", failure_threshold=3, timeout_seconds=60)

    # Force circuit to OPEN state
    def failing_function():
        raise Exception("Simulated failure")

    for _ in range(3):
        try:
            breaker.call(failing_function)
        except:
            pass

    # Verify circuit is OPEN
    results.record(
        "Circuit is OPEN after threshold",
        breaker.state == CircuitState.OPEN
    )

    # Try to call - should fail fast without executing function
    call_executed = False

    def test_function():
        nonlocal call_executed
        call_executed = True
        return "success"

    try:
        breaker.call(test_function)
    except Exception as e:
        results.record(
            "Circuit fails fast when OPEN",
            "OPEN" in str(e) and not call_executed,
            f"Function executed: {call_executed}"
        )


def test_circuit_open_to_half_open(results: TestResults):
    """Test: Circuit transitions OPEN -> HALF_OPEN after timeout"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Circuit OPEN -> HALF_OPEN transition")
    logger.info("=" * 80)

    breaker = CircuitBreakerWrapper("test_open_to_half_open", failure_threshold=3, timeout_seconds=2)

    # Force circuit to OPEN
    def failing_function():
        raise Exception("Simulated failure")

    for _ in range(3):
        try:
            breaker.call(failing_function)
        except:
            pass

    results.record(
        "Circuit is OPEN",
        breaker.state == CircuitState.OPEN
    )

    # Wait for timeout
    logger.info("Waiting for timeout (2 seconds)...")
    time.sleep(2.5)

    # Check state - should be HALF_OPEN
    current_state = breaker.state
    results.record(
        "Circuit transitioned to HALF_OPEN after timeout",
        current_state == CircuitState.HALF_OPEN,
        f"Expected HALF_OPEN, got {current_state}"
    )


def test_circuit_half_open_to_closed(results: TestResults):
    """Test: Circuit transitions HALF_OPEN -> CLOSED on success"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Circuit HALF_OPEN -> CLOSED transition")
    logger.info("=" * 80)

    breaker = CircuitBreakerWrapper("test_half_open_to_closed", failure_threshold=3, timeout_seconds=2)

    # Force circuit to OPEN
    def failing_function():
        raise Exception("Simulated failure")

    for _ in range(3):
        try:
            breaker.call(failing_function)
        except:
            pass

    # Wait for HALF_OPEN
    time.sleep(2.5)

    results.record(
        "Circuit is HALF_OPEN",
        breaker.state == CircuitState.HALF_OPEN
    )

    # Execute successful call
    def success_function():
        return "success"

    try:
        result = breaker.call(success_function)
        logger.info(f"Successful call returned: {result}")
    except Exception as e:
        logger.error(f"Unexpected failure: {e}")

    # Check state - should be CLOSED
    results.record(
        "Circuit transitioned to CLOSED after successful call",
        breaker.state == CircuitState.CLOSED,
        f"Expected CLOSED, got {breaker.state}"
    )


def test_circuit_half_open_to_open(results: TestResults):
    """Test: Circuit transitions HALF_OPEN -> OPEN on failure"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Circuit HALF_OPEN -> OPEN transition")
    logger.info("=" * 80)

    breaker = CircuitBreakerWrapper("test_half_open_to_open", failure_threshold=3, timeout_seconds=2)

    # Force circuit to OPEN
    def failing_function():
        raise Exception("Simulated failure")

    for _ in range(3):
        try:
            breaker.call(failing_function)
        except:
            pass

    # Wait for HALF_OPEN
    time.sleep(2.5)

    results.record(
        "Circuit is HALF_OPEN",
        breaker.state == CircuitState.HALF_OPEN
    )

    # Execute failing call
    try:
        breaker.call(failing_function)
    except:
        pass

    # Check state - should be OPEN again
    results.record(
        "Circuit transitioned back to OPEN after failure in HALF_OPEN",
        breaker.state == CircuitState.OPEN,
        f"Expected OPEN, got {breaker.state}"
    )


def test_fallback_strategy(results: TestResults):
    """Test: Fallback strategy works when circuit is OPEN"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Fallback strategy")
    logger.info("=" * 80)

    breaker = CircuitBreakerWrapper("test_fallback", failure_threshold=3, timeout_seconds=60)

    # Force circuit to OPEN
    def failing_function():
        raise Exception("Simulated failure")

    for _ in range(3):
        try:
            breaker.call(failing_function)
        except:
            pass

    results.record(
        "Circuit is OPEN",
        breaker.state == CircuitState.OPEN
    )

    # Test fallback
    def primary_function():
        return "primary"

    def fallback_function():
        return "fallback"

    result = breaker.call_with_fallback(primary_function, fallback_function)

    results.record(
        "Fallback function executed when circuit OPEN",
        result == "fallback",
        f"Expected 'fallback', got '{result}'"
    )


def test_statistics(results: TestResults):
    """Test: Statistics are tracked correctly"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Statistics tracking")
    logger.info("=" * 80)

    breaker = CircuitBreakerWrapper("test_stats", failure_threshold=5, timeout_seconds=60)

    # Execute mix of successes and failures
    def success_function():
        return "success"

    def failing_function():
        raise Exception("Simulated failure")

    # 3 successes
    for _ in range(3):
        try:
            breaker.call(success_function)
        except:
            pass

    # 2 failures
    for _ in range(2):
        try:
            breaker.call(failing_function)
        except:
            pass

    stats = breaker.get_stats()

    results.record(
        "Total calls tracked",
        stats.total_calls == 5,
        f"Expected 5, got {stats.total_calls}"
    )

    results.record(
        "Success count tracked",
        stats.success_count == 3,
        f"Expected 3, got {stats.success_count}"
    )

    results.record(
        "Failure count tracked",
        stats.failure_count == 2,
        f"Expected 2, got {stats.failure_count}"
    )

    results.record(
        "Success rate calculated correctly",
        abs(stats.success_rate - 0.6) < 0.01,  # 3/5 = 0.6
        f"Expected 0.6, got {stats.success_rate}"
    )


def test_decorator_syntax(results: TestResults):
    """Test: Decorator syntax works correctly"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Decorator syntax")
    logger.info("=" * 80)

    breaker = CircuitBreakerWrapper("test_decorator", failure_threshold=5, timeout_seconds=60)

    @breaker.protected
    def decorated_function(x, y):
        return x + y

    # Test successful call
    try:
        result = decorated_function(2, 3)
        results.record(
            "Decorator allows successful calls",
            result == 5,
            f"Expected 5, got {result}"
        )
    except Exception as e:
        results.record(
            "Decorator allows successful calls",
            False,
            f"Unexpected exception: {e}"
        )

    # Test failing call
    @breaker.protected
    def failing_decorated():
        raise ValueError("Test error")

    try:
        failing_decorated()
        results.record(
            "Decorator propagates exceptions",
            False,
            "Exception not propagated"
        )
    except ValueError:
        results.record(
            "Decorator propagates exceptions",
            True
        )


def main():
    """Run all circuit breaker tests"""
    print("\n")
    print("=" * 80)
    print("CIRCUIT BREAKER TEST SUITE")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    results = TestResults()

    try:
        # Run all tests
        test_circuit_closed_to_open(results)
        test_circuit_open_fails_fast(results)
        test_circuit_open_to_half_open(results)
        test_circuit_half_open_to_closed(results)
        test_circuit_half_open_to_open(results)
        test_fallback_strategy(results)
        test_statistics(results)
        test_decorator_syntax(results)

    except Exception as e:
        logger.error(f"Test suite error: {e}")
        results.record("Test Suite Execution", False, str(e))

    # Print summary
    success = results.summary()

    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
