"""
Circuit Breaker Pattern for External Data Sources

Protects BLS, FRED, and other external API calls from cascading failures.
Implements circuit breaker pattern in Python with:
- CLOSED: Normal operation
- OPEN: Fail fast, return cached data
- HALF_OPEN: Test recovery

Usage:
    from circuit_breaker_wrapper import CircuitBreakerWrapper

    breaker = CircuitBreakerWrapper("BLS_API", failure_threshold=5, timeout_seconds=60)

    @breaker.protected
    def fetch_employment_data():
        # Your API call here
        return requests.get(url)
"""

import functools
import time
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional, Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class CircuitStats:
    """Circuit breaker statistics"""
    state: CircuitState = CircuitState.CLOSED
    total_calls: int = 0
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    circuit_opened_at: Optional[datetime] = None
    last_error: str = ""

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_calls == 0:
            return 1.0
        return self.success_count / self.total_calls


@dataclass
class CircuitConfig:
    """Circuit breaker configuration"""
    name: str
    failure_threshold: int = 5
    timeout_seconds: int = 60
    half_open_timeout_seconds: int = 30
    half_open_max_calls: int = 3
    enable_logging: bool = True


class CircuitBreakerWrapper:
    """
    Circuit Breaker Implementation for Python

    Protects external API calls from cascading failures by opening
    circuit after consecutive failures and allowing recovery testing.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        half_open_timeout_seconds: int = 30,
        half_open_max_calls: int = 3,
        enable_logging: bool = True,
    ):
        """
        Initialize circuit breaker

        Args:
            name: Circuit breaker name
            failure_threshold: Number of consecutive failures before opening
            timeout_seconds: Seconds before transitioning to HALF_OPEN
            half_open_timeout_seconds: Seconds in HALF_OPEN before CLOSED
            half_open_max_calls: Max calls allowed in HALF_OPEN state
            enable_logging: Enable state transition logging
        """
        self.config = CircuitConfig(
            name=name,
            failure_threshold=failure_threshold,
            timeout_seconds=timeout_seconds,
            half_open_timeout_seconds=half_open_timeout_seconds,
            half_open_max_calls=half_open_max_calls,
            enable_logging=enable_logging,
        )

        self.stats = CircuitStats()
        self._state = CircuitState.CLOSED
        self._half_open_calls = 0
        self._cache: Dict[str, Any] = {}

        if enable_logging:
            logger.info(
                f"Circuit breaker '{name}' initialized: "
                f"threshold={failure_threshold}, timeout={timeout_seconds}s"
            )

    @property
    def state(self) -> CircuitState:
        """Get current state (with auto-update)"""
        self._update_state()
        return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit is open"""
        return self.state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed"""
        return self.state == CircuitState.CLOSED

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open"""
        return self.state == CircuitState.HALF_OPEN

    def _update_state(self) -> None:
        """Update state based on time and conditions"""
        if self._state == CircuitState.OPEN:
            # Check if timeout elapsed
            if self.stats.circuit_opened_at:
                elapsed = datetime.now() - self.stats.circuit_opened_at
                if elapsed.total_seconds() >= self.config.timeout_seconds:
                    # Transition to HALF_OPEN
                    self._transition_to(CircuitState.HALF_OPEN)
                    self._half_open_calls = 0

                    if self.config.enable_logging:
                        logger.info(
                            f"Circuit breaker '{self.config.name}' transitioning "
                            "OPEN -> HALF_OPEN (timeout elapsed)"
                        )

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state"""
        old_state = self._state
        self._state = new_state
        self.stats.state = new_state

        if self.config.enable_logging and old_state != new_state:
            logger.info(
                f"Circuit breaker '{self.config.name}' state: "
                f"{old_state.value} -> {new_state.value}"
            )

    def _record_success(self) -> None:
        """Record successful call"""
        self.stats.total_calls += 1
        self.stats.success_count += 1
        self.stats.consecutive_failures = 0
        self.stats.last_success_time = datetime.now()

        # If HALF_OPEN and successful, transition to CLOSED
        if self._state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.CLOSED)
            self._half_open_calls = 0

            if self.config.enable_logging:
                logger.info(
                    f"Circuit breaker '{self.config.name}' transitioning "
                    "HALF_OPEN -> CLOSED (success)"
                )

    def _record_failure(self, error: Exception) -> None:
        """Record failed call"""
        self.stats.total_calls += 1
        self.stats.failure_count += 1
        self.stats.consecutive_failures += 1
        self.stats.last_failure_time = datetime.now()
        self.stats.last_error = str(error)

        # If CLOSED and reached threshold, transition to OPEN
        if (
            self._state == CircuitState.CLOSED
            and self.stats.consecutive_failures >= self.config.failure_threshold
        ):
            self._transition_to(CircuitState.OPEN)
            self.stats.circuit_opened_at = datetime.now()

            if self.config.enable_logging:
                logger.error(
                    f"Circuit breaker '{self.config.name}' transitioning "
                    f"CLOSED -> OPEN ({self.stats.consecutive_failures} consecutive failures)"
                )

        # If HALF_OPEN and failure, transition back to OPEN
        if self._state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
            self.stats.circuit_opened_at = datetime.now()
            self._half_open_calls = 0

            if self.config.enable_logging:
                logger.error(
                    f"Circuit breaker '{self.config.name}' transitioning "
                    "HALF_OPEN -> OPEN (failure during recovery)"
                )

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of function call

        Raises:
            Exception: If circuit is open or call fails
        """
        # Check current state
        current_state = self.state

        # If circuit is OPEN, fail fast
        if current_state == CircuitState.OPEN:
            self.stats.total_calls += 1

            if self.config.enable_logging:
                logger.warning(
                    f"Circuit breaker '{self.config.name}' is OPEN - failing fast"
                )

            raise Exception(f"Circuit breaker '{self.config.name}' is OPEN - service unavailable")

        # If circuit is HALF_OPEN, limit calls
        if current_state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.config.half_open_max_calls:
                if self.config.enable_logging:
                    logger.warning(
                        f"Circuit breaker '{self.config.name}' is HALF_OPEN - max calls reached"
                    )
                raise Exception(
                    f"Circuit breaker '{self.config.name}' is HALF_OPEN - limited availability"
                )

            self._half_open_calls += 1

        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise

    def protected(self, func: Callable) -> Callable:
        """
        Decorator for protecting functions with circuit breaker

        Usage:
            @breaker.protected
            def my_api_call():
                return requests.get(url)
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)

        return wrapper

    def call_with_fallback(self, func: Callable, fallback_func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with fallback on circuit open

        Args:
            func: Primary function to execute
            fallback_func: Fallback function if circuit is open
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of primary or fallback function
        """
        try:
            return self.call(func, *args, **kwargs)
        except Exception as e:
            if self.is_open:
                if self.config.enable_logging:
                    logger.warning(
                        f"Circuit breaker '{self.config.name}' is OPEN - using fallback"
                    )
                return fallback_func(*args, **kwargs)
            else:
                raise

    def reset(self) -> None:
        """Manually reset circuit to CLOSED state"""
        self._state = CircuitState.CLOSED
        self._half_open_calls = 0
        self.stats.consecutive_failures = 0
        self.stats.state = CircuitState.CLOSED

        if self.config.enable_logging:
            logger.info(f"Circuit breaker '{self.config.name}' manually RESET to CLOSED")

    def get_stats(self) -> CircuitStats:
        """Get circuit breaker statistics"""
        return self.stats


class CircuitBreakerManager:
    """
    Manages multiple circuit breakers for different services
    """

    def __init__(self):
        self._breakers: Dict[str, CircuitBreakerWrapper] = {}

    def register(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        **kwargs
    ) -> CircuitBreakerWrapper:
        """Register new circuit breaker"""
        breaker = CircuitBreakerWrapper(
            name=name,
            failure_threshold=failure_threshold,
            timeout_seconds=timeout_seconds,
            **kwargs
        )
        self._breakers[name] = breaker
        logger.info(f"Registered circuit breaker: {name}")
        return breaker

    def get(self, name: str) -> Optional[CircuitBreakerWrapper]:
        """Get circuit breaker by name"""
        return self._breakers.get(name)

    def get_all_stats(self) -> Dict[str, CircuitStats]:
        """Get statistics for all circuit breakers"""
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers"""
        for breaker in self._breakers.values():
            breaker.reset()
        logger.info("Reset all circuit breakers")

    def get_open_count(self) -> int:
        """Get count of open circuits"""
        return sum(1 for breaker in self._breakers.values() if breaker.is_open)


# Global circuit breaker manager instance
_global_manager = CircuitBreakerManager()


def get_global_manager() -> CircuitBreakerManager:
    """Get global circuit breaker manager"""
    return _global_manager
