"""
Functional Programming Utilities for rotmd

Provides composable functions, lazy evaluation, and clean abstractions
following functional programming principles.

Key Concepts:
- Immutability: All operations return new objects
- Composition: Functions compose naturally via pipe operator
- Lazy Evaluation: Computations deferred until needed
- Pure Functions: No side effects

Author: Mykyta Bobylyow
Date: 2025
"""

import numpy as np
from typing import Callable, TypeVar, Generic, Optional, Iterator, Any
from functools import wraps, reduce
from dataclasses import dataclass, field
import operator

T = TypeVar('T')
U = TypeVar('U')


# =============================================================================
# Functional Composition
# =============================================================================

def compose(*functions: Callable) -> Callable:
    """
    Compose functions right-to-left: compose(f, g, h)(x) = f(g(h(x)))

    Example:
        >>> add_one = lambda x: x + 1
        >>> double = lambda x: x * 2
        >>> f = compose(double, add_one)
        >>> f(3)  # (3 + 1) * 2 = 8
        8
    """
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def pipe(*functions: Callable) -> Callable:
    """
    Compose functions left-to-right: pipe(f, g, h)(x) = h(g(f(x)))
    """
    return compose(*reversed(functions))


class Pipeline(Generic[T]):
    """
    Chainable pipeline for data transformations.
    """

    def __init__(self, value: T):
        self._value = value

    def map(self, func: Callable[[T], U]) -> 'Pipeline[U]':
        """Apply function to value."""
        return Pipeline(func(self._value))

    def filter(self, predicate: Callable[[T], bool]) -> 'Pipeline[Optional[T]]':
        """Keep value if predicate is True."""
        return Pipeline(self._value if predicate(self._value) else None)

    def tap(self, func: Callable[[T], Any]) -> 'Pipeline[T]':
        """Execute side effect without changing value (for debugging)."""
        func(self._value)
        return self

    @property
    def value(self) -> T:
        """Get final value."""
        return self._value

    def __or__(self, func: Callable[[T], U]) -> 'Pipeline[U]':
        """Pipe operator: pipeline | func."""
        return self.map(func)


# =============================================================================
# Lazy Evaluation
# =============================================================================

@dataclass
class Lazy(Generic[T]):
    """
    Lazy computation - defers evaluation until value is accessed.
    """
    _computation: Callable[[], T]
    _cache: Optional[T] = field(default=None, init=False, repr=False)

    @property
    def value(self) -> T:
        """Evaluate and cache result."""
        if self._cache is None:
            self._cache = self._computation()
        return self._cache

    def map(self, func: Callable[[T], U]) -> 'Lazy[U]':
        """Transform lazy value."""
        return Lazy(lambda: func(self.value))

    def __or__(self, func: Callable[[T], U]) -> 'Lazy[U]':
        """Pipe operator for lazy values."""
        return self.map(func)


def lazy(func: Callable[..., T]) -> Callable[..., Lazy[T]]:
    """
    Decorator to make function return Lazy value.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Lazy[T]:
        return Lazy(lambda: func(*args, **kwargs))
    return wrapper


# =============================================================================
# Memoization
# =============================================================================

def memoize(func: Callable) -> Callable:
    """
    Cache function results (for pure functions only!).
    """
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create hashable key
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    wrapper.cache = cache  # Expose cache for inspection
    wrapper.cache_clear = cache.clear  # Allow manual clearing
    return wrapper


# =============================================================================
# Array Operations (NumPy functional style)
# =============================================================================

def array_map(func: Callable, *arrays: np.ndarray) -> np.ndarray:
    """
    Vectorized map over arrays (more efficient than Python map).
    """
    return np.array([func(*args) for args in zip(*arrays)])


def array_filter(predicate: Callable[[np.ndarray], bool],
                  array: np.ndarray) -> np.ndarray:
    """
    Filter array elements.
    """
    mask = np.array([predicate(x) for x in array])
    return array[mask]


def array_reduce(func: Callable[[Any, Any], Any],
                  array: np.ndarray,
                  initial: Any = None) -> Any:
    """
    Reduce array to single value.
    """
    if initial is None:
        return reduce(func, array)
    return reduce(func, array, initial)


# =============================================================================
# Partial Application & Currying
# =============================================================================

def curry(func: Callable) -> Callable:
    """
    Convert function to curried form (allows partial application).
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get function signature
        import inspect
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        # If all args provided, call function
        if len(args) + len(kwargs) >= len(params):
            return func(*args, **kwargs)

        # Otherwise return partially applied function
        def partial(*more_args, **more_kwargs):
            return wrapper(*(args + more_args), **{**kwargs, **more_kwargs})
        return partial

    return wrapper


# =============================================================================
# Function Transformers
# =============================================================================

def vectorize(func: Callable) -> Callable:
    """
    Make function work on arrays automatically.
    """
    return np.vectorize(func)


def jit_compile(func: Callable) -> Callable:
    """
    JIT compile with numba if available.
    """
    try:
        from numba import jit
        return jit(nopython=True, cache=True)(func)
    except ImportError:
        return func  # Fallback if numba not available


# =============================================================================
# Monadic Operations (Optional/Maybe)
# =============================================================================

@dataclass
class Maybe(Generic[T]):
    """
    Maybe monad for handling optional values functionally.

    Example:
        >>> result = (Maybe.of(data)
        ...     .bind(validate)
        ...     .bind(process)
        ...     .bind(save)
        ...     .or_else(default_value))
    """
    _value: Optional[T]

    @classmethod
    def of(cls, value: T) -> 'Maybe[T]':
        """Wrap value in Maybe."""
        return cls(value)

    @classmethod
    def nothing(cls) -> 'Maybe[T]':
        """Empty Maybe."""
        return cls(None)

    def bind(self, func: Callable[[T], 'Maybe[U]']) -> 'Maybe[U]':
        """
        Monadic bind (flatmap).
        Apply function if value exists, otherwise propagate Nothing.
        """
        if self._value is None:
            return Maybe.nothing()
        return func(self._value)

    def map(self, func: Callable[[T], U]) -> 'Maybe[U]':
        """Apply function to value if exists."""
        if self._value is None:
            return Maybe.nothing()
        return Maybe.of(func(self._value))

    def filter(self, predicate: Callable[[T], bool]) -> 'Maybe[T]':
        """Keep value if predicate is True."""
        if self._value is None or not predicate(self._value):
            return Maybe.nothing()
        return self

    def or_else(self, default: T) -> T:
        """Get value or default."""
        return self._value if self._value is not None else default

    @property
    def is_nothing(self) -> bool:
        """Check if Maybe is empty."""
        return self._value is None

    @property
    def value(self) -> Optional[T]:
        """Get wrapped value."""
        return self._value


# =============================================================================
# Iteration Utilities
# =============================================================================

def take(n: int, iterable: Iterator[T]) -> Iterator[T]:
    """Take first n elements from iterator."""
    for i, item in enumerate(iterable):
        if i >= n:
            break
        yield item


def chunk(n: int, iterable: Iterator[T]) -> Iterator[list[T]]:
    """Split iterator into chunks of size n."""
    chunk_list = []
    for item in iterable:
        chunk_list.append(item)
        if len(chunk_list) == n:
            yield chunk_list
            chunk_list = []
    if chunk_list:
        yield chunk_list


def window(n: int, iterable: Iterator[T]) -> Iterator[tuple[T, ...]]:
    """
    Sliding window over iterator.
    """
    from collections import deque
    it = iter(iterable)
    win = deque(maxlen=n)

    # Fill initial window
    for _ in range(n):
        try:
            win.append(next(it))
        except StopIteration:
            return

    yield tuple(win)

    # Slide window
    for item in it:
        win.append(item)
        yield tuple(win)
