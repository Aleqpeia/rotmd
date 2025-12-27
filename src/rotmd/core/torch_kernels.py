"""
Pure PyTorch Computational Kernels for rotmd

High-performance GPU-accelerated computational kernels using PyTorch.
All functions are optimized for GPU and support automatic differentiation.

This is an OPTIONAL runtime for GPU acceleration. For CPU-only systems,
use numba_kernels.py (the default runtime).

Author: Mykyta Bobylyow
Date: 2025
"""

import numpy as np
from typing import Tuple, Callable, Optional

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise ImportError(
        "PyTorch is not installed. Install with: pip install torch\n"
        "For CPU-only mode, use numba_kernels instead (default runtime)."
    )

# Default to CPU, user can set device
_default_device = torch.device("cpu")

# =============================================================================
# Vector Decomposition
# =============================================================================


def decompose_vector_single(
    vector: torch.Tensor, reference: torch.Tensor  # (3,)  # (3,)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose single vector into parallel and perpendicular components.

    Args:
        vector: 3D vector (3,)
        reference: Reference direction (3,)

    Returns:
        v_parallel, v_perp: (3,) each
    """
    ref_norm = torch.linalg.norm(reference)
    ref_unit = reference / (ref_norm + 1e-10)

    dot_product = torch.dot(vector, ref_unit)
    v_parallel = dot_product * ref_unit
    v_perp = vector - v_parallel

    return v_parallel, v_perp


def decompose_vector_batch(
    vectors: torch.Tensor,  # (n_frames, 3)
    reference_axes: torch.Tensor,  # (n_frames, 3)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch decomposition over frames."""
    return torch.vmap(decompose_vector_single)(vectors, reference_axes)


def decompose_vector_batch_static_ref(
    vectors: torch.Tensor, reference: torch.Tensor  # (n_frames, 3)  # (3,)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch decomposition with static reference."""
    ref_norm = torch.linalg.norm(reference)
    ref_unit = reference / (ref_norm + 1e-10)

    dot_products = torch.matmul(vectors, ref_unit)
    v_parallel = dot_products[:, None] * ref_unit[None, :]
    v_perp = vectors - v_parallel

    return v_parallel, v_perp


# =============================================================================
# Magnitude Computation
# =============================================================================


def compute_magnitude_batch(vectors: torch.Tensor) -> torch.Tensor:
    """
    Compute magnitudes for batch of vectors.

    Args:
        vectors: (n_frames, 3) or (n_frames, n_atoms, 3)

    Returns:
        magnitudes: (n_frames,) or (n_frames, n_atoms)
    """
    return torch.linalg.norm(vectors, dim=-1)


# =============================================================================
# Angular Momentum & Torque
# =============================================================================


def cross_product_single_frame(
    positions: torch.Tensor,  # (n_atoms, 3)
    vectors: torch.Tensor,  # (n_atoms, 3)
    masses: torch.Tensor,  # (n_atoms,)
    com: torch.Tensor,  # (3,)
) -> torch.Tensor:
    """
    Compute Σ m_i (r_i - COM) × v_i for single frame.

    Args:
        positions: Atomic positions (n_atoms, 3)
        vectors: Velocities or forces (n_atoms, 3)
        masses: Atomic masses (n_atoms,)
        com: Center of mass (3,)

    Returns:
        result: Weighted cross product sum (3,)
    """
    r_rel = positions - com
    cross = torch.cross(r_rel, vectors, dim=-1)
    return torch.sum(masses[:, None] * cross, dim=0)


def cross_product_trajectory(
    positions: torch.Tensor,
    vectors: torch.Tensor,
    masses: torch.Tensor,
    com: torch.Tensor,
) -> torch.Tensor:
    """Batch cross products over trajectory."""
    return torch.vmap(lambda p, v, c: cross_product_single_frame(p, v, masses, c))(
        positions, vectors, com
    )


# =============================================================================
# Center of Mass
# =============================================================================


def compute_com_single(
    positions: torch.Tensor, masses: torch.Tensor  # (n_atoms, 3)  # (n_atoms,)
) -> torch.Tensor:
    """Compute center of mass for single frame."""
    total_mass = torch.sum(masses)
    return torch.sum(masses[:, None] * positions, dim=0) / total_mass


def compute_com_batch(
    positions: torch.Tensor,  # (n_frames, n_atoms, 3)
    masses: torch.Tensor,  # (n_atoms,)
) -> torch.Tensor:
    """Batch COM computation."""
    return torch.vmap(lambda p: compute_com_single(p, masses))(positions)


# =============================================================================
# Inertia Tensors
# =============================================================================


def inertia_tensor_single(
    positions: torch.Tensor, masses: torch.Tensor, com: torch.Tensor
) -> torch.Tensor:
    """Compute inertia tensor for single frame."""
    r = positions - com
    r_squared = torch.sum(r**2, dim=1)

    # Diagonal elements
    I_diag = torch.stack(
        [torch.sum(masses * (r_squared - r[:, i] ** 2)) for i in range(3)]
    )

    # Off-diagonal elements
    I_01 = -torch.sum(masses * r[:, 0] * r[:, 1])
    I_02 = -torch.sum(masses * r[:, 0] * r[:, 2])
    I_12 = -torch.sum(masses * r[:, 1] * r[:, 2])

    # Build symmetric matrix
    I = torch.zeros(3, 3, dtype=positions.dtype, device=positions.device)
    I[0, 0], I[1, 1], I[2, 2] = I_diag[0], I_diag[1], I_diag[2]
    I[0, 1] = I[1, 0] = I_01
    I[0, 2] = I[2, 0] = I_02
    I[1, 2] = I[2, 1] = I_12

    return I


def inertia_tensor_batch(
    positions: torch.Tensor, masses: torch.Tensor, com: torch.Tensor
) -> torch.Tensor:
    """Batch inertia tensor computation."""
    return torch.vmap(lambda p, c: inertia_tensor_single(p, masses, c))(positions, com)


def principal_axes_batch(
    inertia_tensors: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute principal axes for batch of inertia tensors."""
    # PyTorch eigh returns (eigenvalues, eigenvectors)
    eigenvalues, eigenvectors = torch.linalg.eigh(inertia_tensors)

    # Sort by eigenvalues (ascending)
    indices = torch.argsort(eigenvalues, dim=-1)

    # Gather sorted eigenvalues and eigenvectors
    moments = torch.gather(eigenvalues, -1, indices)
    # For eigenvectors, need to reorder columns
    axes = torch.stack(
        [eigenvectors[..., indices[i]] for i in range(inertia_tensors.shape[0])]
    )

    return moments, axes


# =============================================================================
# Linear Algebra
# =============================================================================


def solve_linear_batch(
    A: torch.Tensor, b: torch.Tensor  # (n_frames, n, n)  # (n_frames, n)
) -> torch.Tensor:
    """Solve A @ x = b for batch of linear systems."""
    return torch.linalg.solve(A, b)


# =============================================================================
# Time Derivatives
# =============================================================================


def time_derivative_batch(data: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    """Compute time derivative using finite differences."""
    # Central differences
    dt = times[1:] - times[:-1]
    ddata = data[1:] - data[:-1]

    # Pad edges with forward/backward differences
    deriv = torch.zeros_like(data)
    deriv[1:-1] = (data[2:] - data[:-2]) / (times[2:] - times[:-2])[:, None]
    deriv[0] = (data[1] - data[0]) / (times[1] - times[0])
    deriv[-1] = (data[-1] - data[-2]) / (times[-1] - times[-2])

    return deriv


# =============================================================================
# Conversion Utilities
# =============================================================================


def to_torch(array: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert NumPy array to PyTorch tensor."""
    if device is None:
        device = _default_device
    return torch.from_numpy(array).to(device)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to NumPy array."""
    return tensor.cpu().detach().numpy()


# =============================================================================
# Device Management
# =============================================================================


def set_device(device: str = "cpu"):
    """
    Set default PyTorch device.

    Args:
        device: 'cpu', 'cuda', or 'cuda:0', 'cuda:1', etc.
    """
    global _default_device
    if device.startswith("cuda"):
        if torch.cuda.is_available():
            _default_device = torch.device(device)
        else:
            print(f"Warning: CUDA not available, using CPU")
            _default_device = torch.device("cpu")
    else:
        _default_device = torch.device(device)


def get_devices():
    """Get available PyTorch devices."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(torch.device(f"cuda:{i}"))
    return devices


def get_device():
    """Get current default device."""
    return _default_device


# =============================================================================
# Autodiff Functions
# =============================================================================


def angular_momentum_wrt_positions(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    masses: torch.Tensor,
    com: torch.Tensor,
) -> torch.Tensor:
    """Angular momentum with autodiff support."""
    positions.requires_grad_(True)
    return cross_product_single_frame(positions, velocities, masses, com)


def torque_from_energy_gradient(
    theta: float, psi: float, energy_func: Callable
) -> Tuple[float, float]:
    """Compute torque from energy using autodiff."""
    theta_t = torch.tensor(theta, requires_grad=True)
    psi_t = torch.tensor(psi, requires_grad=True)

    energy = energy_func(theta_t, psi_t)
    energy.backward()

    return -theta_t.grad.item(), -psi_t.grad.item()


# =============================================================================
# Batching for Large Trajectories
# =============================================================================


def process_in_chunks(
    func: Callable, data: torch.Tensor, chunk_size: int = 1000, **kwargs
) -> torch.Tensor:
    """Process large trajectory in chunks."""
    n_frames = data.shape[0]
    chunks = []

    for i in range(0, n_frames, chunk_size):
        chunk_data = data[i : i + chunk_size]
        chunk_result = func(chunk_data, **kwargs)
        chunks.append(chunk_result)

    return torch.cat(chunks, dim=0)


# =============================================================================
# Module Info
# =============================================================================

__all__ = [
    # Vector operations
    "decompose_vector_single",
    "decompose_vector_batch",
    "decompose_vector_batch_static_ref",
    "compute_magnitude_batch",
    # Angular momentum & torque
    "cross_product_single_frame",
    "cross_product_trajectory",
    "compute_com_single",
    "compute_com_batch",
    # Inertia tensors
    "inertia_tensor_single",
    "inertia_tensor_batch",
    "principal_axes_batch",
    # Linear algebra
    "solve_linear_batch",
    # Time derivatives
    "time_derivative_batch",
    # Utilities
    "to_torch",
    "to_numpy",
    "set_device",
    "get_devices",
    "get_device",
    "process_in_chunks",
    # Autodiff
    "angular_momentum_wrt_positions",
    "torque_from_energy_gradient",
]
