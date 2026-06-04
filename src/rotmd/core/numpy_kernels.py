"""
Runtime Selection System for rotmd Computational Kernels

This module provides automatic runtime selection:
- numba (default): CPU-optimized with JIT compilation
- torch: GPU-accelerated with CUDA support
- jax: TPU/GPU with automatic differentiation (legacy)

The backend is selected automatically based on availability or via environment variable.

Usage:
    # Automatic selection (prefers numba)
    from rotmd.core import kernels as K
    result = K.compute_com_batch(positions, masses)

    # Manual selection
    import rotmd.core.kernels as K
    K.set_backend('torch')  # or 'numba', 'jax'

    # Environment variable
    export ROTMD_BACKEND=torch

Author: Mykyta Bobylyow
Date: 2025
"""

import os
import sys
from typing import Optional

# =============================================================================
# Backend Detection & Selection
# =============================================================================

_CURRENT_BACKEND = None
_BACKEND_MODULE = None


def _detect_available_backends():
    """Detect which computational backends are available."""
    available = {}

    # Check numba
    try:
        import numba

        available["numba"] = True
    except ImportError:
        available["numba"] = False

    # Check PyTorch
    try:
        import torch

        available["torch"] = True
    except ImportError:
        available["torch"] = False

    # Check JAX
    try:
        import jax

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
        backend: 'numba', 'torch', or 'jax'

    Raises:
        ImportError: If requested backend is not available
        ValueError: If backend name is invalid
    """
    global _CURRENT_BACKEND, _BACKEND_MODULE

    backend = backend.lower()
    available = _detect_available_backends()

    if backend not in ["numba", "torch", "jax"]:
        raise ValueError(
            f"Invalid backend '{backend}'. Must be 'numba', 'torch', or 'jax'"
        )

    if not available.get(backend, False):
        installed = get_available_backends()
        raise ImportError(
            f"Backend '{backend}' is not available.\n"
            f"Available backends: {installed}\n"
            f"To install {backend}:\n"
            f"  - numba: pip install numba\n"
            f"  - torch: pip install torch\n"
            f"  - jax: pip install jax"
        )

    # Import the backend module
    if backend == "numba":
        from . import numba_kernels as kernels
    elif backend == "torch":
        from . import torch_kernels as kernels
    elif backend == "jax":
        from . import jax_kernels as kernels

    _CURRENT_BACKEND = backend
    _BACKEND_MODULE = kernels

    print(f"rotmd runtime: {backend}")


def _auto_select_backend():
    """
    Automatically select best available backend.

    Priority:
    1. Environment variable ROTMD_BACKEND
    2. numba (default for CPU)
    3. torch (if numba unavailable)
    4. jax (fallback)
    """
    # Check environment variable
    env_backend = os.environ.get("ROTMD_BACKEND", "").lower()
    if env_backend:
        try:
            set_backend(env_backend)
            return
        except (ImportError, ValueError) as e:
            print(f"Warning: Could not use ROTMD_BACKEND={env_backend}: {e}")

    # Try backends in priority order
    available = get_available_backends()

    if not available:
        raise ImportError(
            "No computational backend available!\n"
            "Install at least one of: numba, torch, jax\n"
            "  Recommended: pip install numba"
        )

    # Priority: numba > torch > jax
    for backend in ["numba", "torch", "jax"]:
        if backend in available:
            set_backend(backend)
            return


# Initialize backend on module import
_auto_select_backend()


# =============================================================================
# Proxy Functions - Forward to Active Backend
# =============================================================================


def __getattr__(name):
    """
    Dynamically forward attribute access to active backend module.

    This allows `from rotmd.core import kernels as K` to work transparently.
    """
    if _BACKEND_MODULE is None:
        raise RuntimeError("No backend initialized")

    try:
        return getattr(_BACKEND_MODULE, name)
    except AttributeError:
        raise AttributeError(f"Backend '{_CURRENT_BACKEND}' has no attribute '{name}'")


def print_backend_info():
    """Print information about current backend and available backends."""
    print("=" * 60)
    print("rotmd Computational Backend Information")
    print("=" * 60)
    print(f"Active backend: {_CURRENT_BACKEND}")
    print(f"Available backends: {get_available_backends()}")

    if _CURRENT_BACKEND == "torch":
        try:
            import torch

            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA devices: {torch.cuda.device_count()}")
        except:
            pass

    elif _CURRENT_BACKEND == "numba":
        try:
            import numba

            print(f"Numba version: {numba.__version__}")
            print(f"Threading layer: {numba.config.THREADING_LAYER}")
        except:
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
