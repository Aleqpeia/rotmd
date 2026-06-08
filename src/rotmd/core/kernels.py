"""
Runtime Selection System for rotmd Computational Kernels

This module dispatches the low-level numerical kernels to a concrete
backend at runtime:

- ``numba`` (default): CPU-optimized with JIT compilation. This is the
  only backend that ships today and is implemented in
  :mod:`rotmd.core.numba_kernels`.
- ``jax`` (planned): GPU/TPU acceleration with automatic
  differentiation. The dispatcher already knows how to load a
  ``rotmd.core.jax_kernels`` module; it simply has not been written yet.

PyTorch was intentionally dropped: the GPU story for this package is
JAX, so we keep exactly one CPU backend and one planned accelerator
backend rather than carrying a third, unused code path.

Usage:
    # Automatic selection (prefers numba)
    from rotmd.core import kernels as K
    result = K.compute_com_batch(positions, masses)

    # Manual selection
    import rotmd.core.kernels as K
    K.set_backend("numba")  # or "jax" once implemented

    # Environment variable
    export ROTMD_BACKEND=numba

Author: Mykyta Bobylyow
Date: 2025
"""

import os

# =============================================================================
# Backend Detection & Selection
# =============================================================================

# Backends the dispatcher knows how to load. Adding "jax" here is the only
# change needed on this side once ``jax_kernels`` exists.
_SUPPORTED_BACKENDS = ("numba", "jax")

_CURRENT_BACKEND = None
_BACKEND_MODULE = None


def _detect_available_backends():
    """Detect which computational backends are importable."""
    available = {}

    try:
        import numba  # noqa: F401

        available["numba"] = True
    except ImportError:
        available["numba"] = False

    try:
        import jax  # noqa: F401

        available["jax"] = True
    except ImportError:
        available["jax"] = False

    return available


def get_available_backends():
    """Get list of available backend names."""
    backends = _detect_available_backends()
    return [name for name, available in backends.items() if available]


def get_backend():
    """Get current active backend name."""
    return _CURRENT_BACKEND


def set_backend(backend: str):
    """
    Set computational backend.

    Args:
        backend: ``"numba"`` or ``"jax"``.

    Raises:
        ImportError: If the requested backend (or its kernel module) is
            not available.
        ValueError: If the backend name is not supported.
    """
    global _CURRENT_BACKEND, _BACKEND_MODULE

    backend = backend.lower()
    available = _detect_available_backends()

    if backend not in _SUPPORTED_BACKENDS:
        supported = ", ".join(repr(b) for b in _SUPPORTED_BACKENDS)
        raise ValueError(f"Invalid backend '{backend}'. Must be one of: {supported}")

    if not available.get(backend, False):
        installed = get_available_backends()
        raise ImportError(
            f"Backend '{backend}' is not available.\n"
            f"Available backends: {installed}\n"
            f"To install {backend}:\n"
            f"  - numba: pip install numba\n"
            f"  - jax: pip install jax"
        )

    # Import the concrete kernel module. The jax backend is not written yet,
    # so loading it raises a clear ImportError rather than failing silently.
    if backend == "numba":
        from . import numba_kernels as kernels
    elif backend == "jax":
        try:
            from . import jax_kernels as kernels
        except ImportError as exc:
            raise ImportError(
                "The 'jax' backend is selected but 'rotmd.core.jax_kernels' "
                "is not implemented yet. Use the 'numba' backend for now."
            ) from exc

    _CURRENT_BACKEND = backend
    _BACKEND_MODULE = kernels

    print(f"rotmd runtime: {backend}")


def _auto_select_backend():
    """
    Automatically select the best available backend.

    Priority:
    1. Environment variable ``ROTMD_BACKEND``
    2. ``numba`` (default for CPU)
    3. ``jax`` (if numba is unavailable)
    """
    env_backend = os.environ.get("ROTMD_BACKEND", "").lower()
    if env_backend:
        try:
            set_backend(env_backend)
            return
        except (ImportError, ValueError) as e:
            print(f"Warning: Could not use ROTMD_BACKEND={env_backend}: {e}")

    available = get_available_backends()

    if not available:
        raise ImportError(
            "No computational backend available!\n"
            "Install at least one of: numba, jax\n"
            "  Recommended: pip install numba"
        )

    # Priority: numba > jax.
    for backend in _SUPPORTED_BACKENDS:
        if backend in available:
            set_backend(backend)
            return


# Initialize backend on module import.
_auto_select_backend()


# =============================================================================
# Proxy Functions - Forward to Active Backend
# =============================================================================


def __getattr__(name):
    """
    Dynamically forward attribute access to the active backend module.

    This allows ``from rotmd.core import kernels as K`` to work
    transparently, with ``K.<kernel>`` resolving to the active backend.
    """
    if _BACKEND_MODULE is None:
        raise RuntimeError("No backend initialized")

    try:
        return getattr(_BACKEND_MODULE, name)
    except AttributeError:
        raise AttributeError(f"Backend '{_CURRENT_BACKEND}' has no attribute '{name}'")


def print_backend_info():
    """Print information about the current and available backends."""
    print("=" * 60)
    print("rotmd Computational Backend Information")
    print("=" * 60)
    print(f"Active backend: {_CURRENT_BACKEND}")
    print(f"Available backends: {get_available_backends()}")

    if _CURRENT_BACKEND == "numba":
        try:
            import numba

            print(f"Numba version: {numba.__version__}")
            print(f"Threading layer: {numba.config.THREADING_LAYER}")
        except Exception:
            pass

    elif _CURRENT_BACKEND == "jax":
        try:
            import jax

            print(f"JAX version: {jax.__version__}")
            print(f"JAX devices: {jax.devices()}")
        except Exception:
            pass

    print("=" * 60)


# =============================================================================
# Module Info
# =============================================================================

__all__ = [
    "set_backend",
    "get_backend",
    "get_available_backends",
    "print_backend_info",
]
