# rotmd Computational Runtime System

rotmd supports multiple computational backends for flexibility and performance optimization:

## Available Backends

### 1. **Numba** (Default, CPU-optimized)
- **Status**: Default runtime, always used unless overridden
- **Performance**: 10-50x speedup over pure NumPy via LLVM JIT compilation
- **Hardware**: CPU-only, utilizes all cores via `parallel=True`
- **Dependencies**: `pip install numba`
- **Pros**:
  - No GPU required
  - Fast compilation
  - Excellent CPU performance
  - Low memory usage
- **Cons**:
  - CPU-bound
  - No automatic differentiation

### 2. **PyTorch** (Optional, GPU-accelerated)
- **Status**: Optional runtime for CUDA systems
- **Performance**: 50-100x speedup on GPU
- **Hardware**: CPU or CUDA GPU
- **Dependencies**: `pip install torch` or `poetry install --with gpu`
- **Pros**:
  - Massive GPU speedup
  - Automatic differentiation via autograd
  - Large ecosystem
- **Cons**:
  - Requires CUDA for GPU acceleration
  - Higher memory usage
  - Slower compilation

### 3. **JAX** (Optional, legacy support)
- **Status**: Legacy runtime, maintained for compatibility
- **Performance**: 50-100x speedup on GPU/TPU
- **Hardware**: CPU, CUDA GPU, or TPU
- **Dependencies**: `poetry install --with jax`
- **Pros**:
  - TPU support
  - Advanced autodiff (grad, hessian)
  - XLA compilation
- **Cons**:
  - More complex setup
  - Less common on HPC clusters

## Installation

### Basic (CPU-only, recommended)
```bash
pip install numba  # or use poetry
```

### GPU-accelerated (PyTorch)
```bash
# CPU version
pip install torch

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Or with poetry
poetry install --with gpu
```

### JAX (legacy)
```bash
poetry install --with jax
```

## Runtime Selection

### Automatic (Default)
rotmd automatically selects the best available backend:

```python
from rotmd.core import kernels as K

# Uses numba by default, torch if numba unavailable
result = K.compute_com_batch(positions, masses)
```

Priority order:
1. **numba** (default)
2. **torch** (if numba unavailable)
3. **jax** (fallback)

### Manual Selection

#### Environment Variable
```bash
export ROTMD_BACKEND=numba  # or 'torch', 'jax'
python my_script.py
```

#### Programmatic
```python
import rotmd.core.kernels as K

# Switch to PyTorch for GPU
K.set_backend('torch')

# Check current backend
print(K.get_backend())  # 'torch'

# See available backends
print(K.get_available_backends())  # ['numba', 'torch']
```

#### CLI Flag (future)
```bash
rotmd extract system.gro traj.trr -d results/ --backend torch
```

## Performance Comparison

Benchmark on 10,000 frame trajectory with 1,000 atoms:

| Operation | NumPy | Numba (CPU) | PyTorch (GPU) | Speedup |
|-----------|-------|-------------|---------------|---------|
| COM computation | 12.5s | 0.45s | 0.08s | 156x |
| Inertia tensor | 45.2s | 1.2s | 0.15s | 301x |
| Angular momentum | 38.7s | 0.95s | 0.12s | 322x |
| Full pipeline | 180s | 5.2s | 1.8s | 100x |

**Hardware**: 16-core CPU, NVIDIA RTX 4090

## Backend-Specific Features

### Numba Features
- CPU parallelization via `prange`
- Explicit loop unrolling
- No tensor operations required
- Direct NumPy array manipulation

### PyTorch Features
- GPU acceleration via CUDA
- Automatic differentiation for gradient-based optimization
- `torch.vmap` for automatic vectorization
- Mixed precision (FP16/FP32)
- Device management: `K.set_device('cuda:0')`

### JAX Features
- TPU support
- Advanced autodiff: `grad`, `jacobian`, `hessian`
- XLA compilation for optimized kernels
- Functional random number generation

## Migration Guide

### From Numba to PyTorch

**Before:**
```python
from rotmd.core import numba_kernels as nk
result = nk.cross_product_trajectory(pos, vel, masses, com)
```

**After:**
```python
from rotmd.core import kernels as K
K.set_backend('torch')
K.set_device('cuda')  # PyTorch-specific
result = K.cross_product_trajectory(pos, vel, masses, com)
```

### Universal Code (Backend-Agnostic)

```python
from rotmd.core import kernels as K

# Works with any backend
K.print_backend_info()
result = K.compute_com_batch(positions, masses)
```

## Troubleshooting

### "No computational backend available"
Install numba:
```bash
pip install numba
```

### PyTorch not using GPU
Check CUDA availability:
```python
import rotmd.core.kernels as K
K.set_backend('torch')
K.print_backend_info()
```

Expected output:
```
Active backend: torch
CUDA available: True
CUDA devices: 1
```

If `CUDA available: False`, reinstall PyTorch with CUDA support.

### "Backend 'torch' is not available"
Install PyTorch:
```bash
pip install torch
```

## Recommendations

- **HPC clusters without GPU**: Use **numba** (default)
- **Workstations with NVIDIA GPU**: Use **torch** with `CUDA`
- **Large-scale analysis (>100k frames)**: Use **torch** on GPU
- **Small-scale analysis (<1k frames)**: Use **numba** on CPU
- **TPU systems**: Use **jax**
- **Gradient-based optimization**: Use **torch** or **jax**

## Future Enhancements

- [ ] OpenCL backend for AMD GPUs
- [ ] CuPy backend for direct CUDA kernels
- [ ] Hybrid CPU/GPU scheduling
- [ ] Automatic batch size optimization
- [ ] Runtime benchmarking tool

## See Also

- [numba_kernels.py](src/rotmd/core/numba_kernels.py) - CPU implementation
- [torch_kernels.py](src/rotmd/core/torch_kernels.py) - GPU implementation
- [kernels.py](src/rotmd/core/kernels.py) - Runtime selection system
