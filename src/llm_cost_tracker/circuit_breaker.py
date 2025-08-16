"""Circuit breaker implementation for resilient service calls."""

import asyncio
import logging
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, calls fail fast
    HALF_OPEN = "half_open"  # Testing if service is back


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    pass


class CircuitBreaker:
    """Circuit breaker implementation with configurable thresholds."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: tuple = (Exception,),
        recovery_timeout: float = 30.0,
        name: str = "circuit_breaker",
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Time to wait before trying again when open
            expected_exception: Exceptions that count as failures
            recovery_timeout: Timeout for half-open state
            name: Name for logging purposes
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.recovery_timeout = recovery_timeout
        self.name = name

        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.next_attempt_time = 0

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        return self.state == CircuitState.OPEN and time.time() >= self.next_attempt_time

    def _on_success(self) -> None:
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        logger.info(f"Circuit breaker '{self.name}' reset to CLOSED")

    def _on_failure(self, exception: Exception) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.next_attempt_time = time.time() + self.timeout
            logger.warning(
                f"Circuit breaker '{self.name}' opened after {self.failure_count} failures. "
                f"Next attempt at {self.next_attempt_time}"
            )

    async def __call__(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""

        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. Next attempt at {self.next_attempt_time}"
                )

        try:
            if self.state == CircuitState.HALF_OPEN:
                # Set timeout for recovery attempt
                result = await asyncio.wait_for(
                    func(*args, **kwargs), timeout=self.recovery_timeout
                )
            else:
                result = await func(*args, **kwargs)

            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure(e)
            raise
        except asyncio.TimeoutError as e:
            self._on_failure(e)
            raise CircuitBreakerError(f"Timeout in circuit breaker '{self.name}'")

    def get_state(self) -> dict:
        """Get current circuit breaker state."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "next_attempt_time": (
                self.next_attempt_time if self.state == CircuitState.OPEN else None
            ),
        }


def circuit_breaker(
    failure_threshold: int = 5,
    timeout: float = 60.0,
    expected_exception: tuple = (Exception,),
    recovery_timeout: float = 30.0,
    name: str = None,
):
    """Decorator to add circuit breaker functionality to async functions."""

    def decorator(func: Callable) -> Callable:
        breaker_name = name or f"{func.__module__}.{func.__name__}"
        breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            timeout=timeout,
            expected_exception=expected_exception,
            recovery_timeout=recovery_timeout,
            name=breaker_name,
        )

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker(func, *args, **kwargs)

        # Attach circuit breaker instance for monitoring
        wrapper._circuit_breaker = breaker
        return wrapper

    return decorator


class CircuitBreakerRegistry:
    """Registry to manage multiple circuit breakers."""

    def __init__(self):
        self.breakers = {}

    def register(self, name: str, breaker: CircuitBreaker) -> None:
        """Register a circuit breaker."""
        self.breakers[name] = breaker

    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self.breakers.get(name)

    def get_all_states(self) -> dict:
        """Get states of all registered circuit breakers."""
        return {name: breaker.get_state() for name, breaker in self.breakers.items()}

    def reset_breaker(self, name: str) -> bool:
        """Reset a specific circuit breaker."""
        if breaker := self.breakers.get(name):
            breaker.failure_count = 0
            breaker.state = CircuitState.CLOSED
            logger.info(f"Manually reset circuit breaker '{name}'")
            return True
        return False

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for name in self.breakers:
            self.reset_breaker(name)


# Global registry instance
circuit_registry = CircuitBreakerRegistry()


# Database-specific circuit breaker
database_circuit_breaker = circuit_breaker(
    failure_threshold=3,
    timeout=30.0,
    expected_exception=(Exception,),
    recovery_timeout=10.0,
    name="database_operations",
)


# External API circuit breaker
api_circuit_breaker = circuit_breaker(
    failure_threshold=5,
    timeout=60.0,
    expected_exception=(Exception,),
    recovery_timeout=15.0,
    name="external_api_calls",
)
