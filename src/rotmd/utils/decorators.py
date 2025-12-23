"""
Functional Decorators for rotmd

Generic decorators to eliminate boilerplate and enable functional patterns.
All decorators are JAX-compatible and support GPU acceleration.

Key decorators:
- @trajectory: Auto-vectorize single-frame functions over trajectories
- @jax_jit: Optional JAX JIT compilation
- @ensure_jax: Automatic NumPy→JAX conversion

Author: Mykyta Bobylyow
Date: 2025
"""

import numpy as np
from functools import wraps
from typing import Callable, TypeVar, Any, Optional

T = TypeVar('T')
U = TypeVar('U')


# =============================================================================
# JAX Import Handling
# =============================================================================

def _has_jax() -> bool:
    """Check if JAX is available."""
    try:
        import jax
        return True
    except ImportError:
        return False


def _get_jax():
    """Get JAX module if available."""
    if _has_jax():
        import jax
        return jax
    return None


# =============================================================================
# Trajectory Vectorization Decorator
# =============================================================================

def trajectory(frame_axis: int = 0, use_jax: bool = True):
    """
    Decorator to vectorize single-frame functions over trajectories.

    Eliminates all *_trajectory() wrapper functions.

    This decorator automatically applies a function that works on single frames
    to entire trajectories using JAX vmap (if available) or NumPy loops.

    Args:
        frame_axis: Axis corresponding to frames (default: 0)
        use_jax: If True, use JAX vmap for vectorization (default: True)

    Returns:
        Decorated function that works on trajectories

    Examples:
        >>> from rotmd.utils.decorators import trajectory
        >>> import numpy as np
        >>> import jax.numpy as jnp
        >>>
        >>> @trajectory(frame_axis=0, use_jax=True)
        >>> def compute_energy(positions, masses):
        ...     '''Compute energy for single frame.'''
        ...     return jnp.sum(positions**2 * masses[:, None])
        >>>
        >>> # Now works on entire trajectory automatically
        >>> pos_traj = np.random.rand(100, 50, 3)  # 100 frames, 50 atoms
        >>> masses = np.ones(50)
        >>> energies = compute_energy(pos_traj, masses)  # Returns (100,)
        >>> print(energies.shape)
        (100,)

    Notes:
        - JAX vmap is 10-100x faster than NumPy loops
        - Falls back to NumPy if JAX not available
        - Function should work on single frame (frame_axis dimension removed)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if use_jax and _has_jax():
                # JAX path: use vmap for automatic vectorization
                from jax import vmap, jit
                from rotmd.core import torch_kernels as tk

                # Convert inputs to JAX arrays
                jax_args = [tkto_jax(arg) if isinstance(arg, np.ndarray) else arg
                           for arg in args]
                jax_kwargs = {k: tkto_jax(v) if isinstance(v, np.ndarray) else v
                             for k, v in kwargs.items()}

                # Create vmapped function
                # Determine which arguments to vmap over
                in_axes = []
                for i, arg in enumerate(jax_args):
                    if hasattr(arg, 'shape') and len(arg.shape) > 0:
                        # Assume first axis is frame axis for array arguments
                        in_axes.append(frame_axis)
                    else:
                        # Scalar or non-array: don't vmap
                        in_axes.append(None)

                # Apply vmap
                vmapped_func = jit(vmap(func, in_axes=tuple(in_axes)))
                result_jax = vmapped_func(*jax_args, **jax_kwargs)

                # Convert back to NumPy
                return tkto_numpy(result_jax)

            else:
                # NumPy fallback: manual loop
                # Assume first argument contains trajectory
                if len(args) == 0:
                    raise ValueError("trajectory decorator requires at least one argument")

                first_arg = args[0]
                if not hasattr(first_arg, 'shape'):
                    raise ValueError("First argument must be array-like")

                n_frames = first_arg.shape[frame_axis]
                results = []

                for i in range(n_frames):
                    # Extract frame from each argument
                    frame_args = []
                    for arg in args:
                        if hasattr(arg, 'shape') and len(arg.shape) > frame_axis:
                            frame_args.append(np.take(arg, i, axis=frame_axis))
                        else:
                            frame_args.append(arg)

                    # Call function on single frame
                    result = func(*frame_args, **kwargs)
                    results.append(result)

                # Stack results
                return np.array(results)

        return wrapper
    return decorator


# =============================================================================
# JAX JIT Compilation Decorator
# =============================================================================

def jax_jit(func: Optional[Callable] = None, *, static_argnums: Optional[tuple] = None):
    """
    Optional JAX JIT compilation decorator.

    Applies JAX JIT compilation if available, otherwise returns original function.

    Args:
        func: Function to decorate
        static_argnums: Tuple of argument indices to treat as static

    Returns:
        JIT-compiled function (or original if JAX unavailable)

    Examples:
        >>> @jax_jit
        >>> def compute_distance(x, y):
        ...     return jnp.sqrt(jnp.sum((x - y)**2))
        >>>
        >>> # First call compiles, subsequent calls are fast
        >>> d = compute_distance(jnp.array([1, 2]), jnp.array([4, 6]))

    Notes:
        - First call has compilation overhead (~100ms-1s)
        - Subsequent calls are 10-100x faster
        - Falls back gracefully if JAX not available
    """
    def decorator(f: Callable) -> Callable:
        if _has_jax():
            from jax import jit
            return jit(f, static_argnums=static_argnums)
        else:
            # No JAX: return original function
            return f

    if func is None:
        # Called with arguments: @jax_jit(static_argnums=(0,))
        return decorator
    else:
        # Called without arguments: @jax_jit
        return decorator(func)


# =============================================================================
# Array Conversion Decorators
# =============================================================================

def ensure_jax(func: Callable) -> Callable:
    """
    Decorator to automatically convert NumPy inputs to JAX arrays.

    Converts all NumPy array arguments to JAX arrays before function call,
    then converts result back to NumPy.

    Args:
        func: Function expecting JAX arrays

    Returns:
        Function that accepts NumPy arrays

    Examples:
        >>> @ensure_jax
        >>> def jax_computation(x, y):
        ...     return jnp.dot(x, y)  # JAX function
        >>>
        >>> # Can call with NumPy arrays
        >>> result = jax_computation(np.array([1, 2]), np.array([3, 4]))
        >>> print(type(result))
        <class 'numpy.ndarray'>
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not _has_jax():
            # No JAX: just call function (may fail)
            return func(*args, **kwargs)

        from rotmd.core import jax_kernels as jk

        # Convert NumPy → JAX
        jax_args = [tkto_jax(arg) if isinstance(arg, np.ndarray) else arg
                   for arg in args]
        jax_kwargs = {k: tkto_jax(v) if isinstance(v, np.ndarray) else v
                     for k, v in kwargs.items()}

        # Call function
        result = func(*jax_args, **jax_kwargs)

        # Convert JAX → NumPy
        if hasattr(result, '__array__'):
            return tkto_numpy(result)
        elif isinstance(result, tuple):
            return tuple(tkto_numpy(r) if hasattr(r, '__array__') else r
                        for r in result)
        else:
            return result

    return wrapper


# =============================================================================
# Performance Timing Decorator
# =============================================================================

def benchmark(func: Callable) -> Callable:
    """
    Decorator to benchmark function execution time.

    Prints execution time after each call. Useful for comparing
    JAX vs NumPy implementations.

    Args:
        func: Function to benchmark

    Returns:
        Wrapped function that prints timing info

    Examples:
        >>> @benchmark
        >>> def slow_computation(n):
        ...     return sum(i**2 for i in range(n))
        >>>
        >>> result = slow_computation(1000000)
        [slow_computation] Execution time: 0.123s
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[{func.__name__}] Execution time: {elapsed:.3f}s")
        return result
    return wrapper


# =============================================================================
# Device Management Decorator
# =============================================================================

def on_device(device: str = 'gpu'):
    """
    Decorator to run function on specific JAX device.

    Args:
        device: 'gpu', 'cpu', or 'tpu'

    Returns:
        Decorator that sets JAX device

    Examples:
        >>> @on_device('gpu')
        >>> def gpu_computation(x):
        ...     return jnp.sum(x**2)
        >>>
        >>> # Runs on GPU if available
        >>> result = gpu_computation(jnp.ones(10000))
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if _has_jax():
                from rotmd.core import jax_kernels as jk
                # Set device
                tkset_device(device)

            return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Batching Decorator
# =============================================================================

def batch_process(chunk_size: int = 1000):
    """
    Decorator to process large arrays in chunks.

    Useful for avoiding GPU memory limits with large trajectories.

    Args:
        chunk_size: Number of frames per chunk

    Returns:
        Decorator that processes in chunks

    Examples:
        >>> @batch_process(chunk_size=1000)
        >>> def process_trajectory(positions):
        ...     # Process all frames
        ...     return compute_energy(positions)
        >>>
        >>> # Automatically chunks if >1000 frames
        >>> large_traj = np.random.rand(50000, 100, 3)
        >>> energies = process_trajectory(large_traj)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) == 0:
                return func(*args, **kwargs)

            first_arg = args[0]
            if not hasattr(first_arg, 'shape'):
                return func(*args, **kwargs)

            n_frames = first_arg.shape[0]

            if n_frames <= chunk_size:
                # Small enough: process all at once
                return func(*args, **kwargs)

            # Process in chunks
            results = []
            for i in range(0, n_frames, chunk_size):
                # Extract chunk from each argument
                chunk_args = []
                for arg in args:
                    if hasattr(arg, 'shape') and arg.shape[0] == n_frames:
                        chunk_args.append(arg[i:i+chunk_size])
                    else:
                        chunk_args.append(arg)

                # Process chunk
                chunk_result = func(*chunk_args, **kwargs)
                results.append(chunk_result)

            # Concatenate results
            return np.concatenate(results, axis=0)

        return wrapper
    return decorator


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'trajectory',
    'jax_jit',
    'ensure_jax',
    'benchmark',
    'on_device',
    'batch_process',
]
