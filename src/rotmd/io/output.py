"""
Output Handling for Analysis Results

This module provides utilities for saving and loading analysis results
in various formats (JSON, HDF5, NetCDF, NPZ).

Key Features:
- JSON for metadata and small datasets
- HDF5/NetCDF for large arrays
- NPZ for numpy arrays
- Automatic compression
- Metadata preservation

Author: Mykyta Bobylyow
Date: 2025
"""

import json
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np


def save_results_json(results: Dict[str, Any],
                     filename: str,
                     indent: int = 2) -> None:
    """
    Save results to JSON file.

    Args:
        results: Dictionary with analysis results
        filename: Output JSON file path
        indent: Indentation level for readability

    Notes:
        - Converts numpy arrays to lists
        - Good for metadata and small datasets
        - Not suitable for large trajectory data

    Example:
        >>> results = {'D': 0.123, 'friction': 45.6}
        >>> save_results_json(results, 'analysis.json')
    """
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(results)

    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=indent)

    print(f"✓ Saved results to {filename}")


def load_results_json(filename: str) -> Dict[str, Any]:
    """
    Load results from JSON file.

    Args:
        filename: Input JSON file path

    Returns:
        results: Dictionary with analysis results

    Example:
        >>> results = load_results_json('analysis.json')
        >>> print(f"D = {results['D']}")
    """
    with open(filename, 'r') as f:
        results = json.load(f)

    return results


def save_results_npz(results: Dict[str, Any],
                    dirname: str,
                    filename: str,
                    compress: bool = True) -> None:
    """
    Save numpy arrays to NPZ file.

    Args:
        filename: Output .npz file path
        results: Dictionary with numpy arrays
        compress: Use compression (slower but smaller files)

    Notes:
        - Efficient for large numpy arrays
        - Preserves array dtypes
        - Can't store metadata (use JSON separately)

    Example:
        >>> results = {'euler': euler_angles, 'times': times}
        >>> save_results_npz(results, 'trajectories.npz')
    """
    for k, v in results.items():
        try:
            if compress:
                np.savez_compressed(f"{dirname}/{k}_{filename}", v)
            else:
                np.savez(f"{dirname}/{k}_{filename}.npz", v)
        except Exception as e:
            print(f"Could not store {k} of type {type(v)}: \n {e}")
            continue
        

    file_size = Path(filename).stat().st_size / (1024**2)
    print(f"✓ Saved {len(results)} arrays to {filename} ({file_size:.1f} MB)")


def load_results_npz(filename: str) -> Dict[str, np.ndarray]:
    """
    Load numpy arrays from NPZ file.

    Args:
        filename: Input .npz file path

    Returns:
        results: Dictionary with numpy arrays

    Example:
        >>> data = load_results_npz('trajectories.npz')
        >>> euler = data['euler']
    """
    with np.load(filename) as data:
        results = {key: data[key] for key in data.files}

    return results


def save_results_hdf5(results: Dict[str, Any],
                     filename: str,
                     compression: str = 'gzip') -> None:
    """
    Save results to HDF5 file.

    Args:
        results: Dictionary with arrays and metadata
        filename: Output .h5 file path
        compression: Compression method ('gzip', 'lzf', None)

    Notes:
        - Best for large datasets
        - Supports hierarchical organization
        - Can store metadata as attributes
        - Requires h5py: pip install h5py

    Example:
        >>> results = {
        ...     'trajectories/euler': euler_angles,
        ...     'analysis/pmf': pmf_values,
        ...     'metadata': {'temperature': 300.0}
        ... }
        >>> save_results_hdf5(results, 'data.h5')
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for HDF5. Install with: pip install h5py")

    with h5py.File(filename, 'w') as f:
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                # Store array with compression
                f.create_dataset(key, data=value, compression=compression)
            elif isinstance(value, dict):
                # Store metadata as attributes
                grp = f.create_group(key)
                for meta_key, meta_val in value.items():
                    grp.attrs[meta_key] = meta_val
            else:
                # Try to store as dataset
                try:
                    f.create_dataset(key, data=value)
                except:
                    warnings.warn(f"Could not store {key} of type {type(value)}")

    file_size = Path(filename).stat().st_size / (1024**2)
    print(f"✓ Saved results to {filename} ({file_size:.1f} MB)")


def load_results_hdf5(filename: str) -> Dict[str, Any]:
    """
    Load results from HDF5 file.

    Args:
        filename: Input .h5 file path

    Returns:
        results: Dictionary with arrays and metadata

    Example:
        >>> data = load_results_hdf5('data.h5')
        >>> euler = data['trajectories/euler']
        >>> temp = data['metadata']['temperature']
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for HDF5. Install with: pip install h5py")

    results = {}

    with h5py.File(filename, 'r') as f:
        def load_recursive(name, obj):
            if isinstance(obj, h5py.Dataset):
                results[name] = obj[()]
            elif isinstance(obj, h5py.Group):
                # Load attributes as metadata
                if obj.attrs:
                    results[name] = dict(obj.attrs)

        f.visititems(load_recursive)

    return results


def save_trajectory_data(positions: np.ndarray,
                        times: np.ndarray,
                        filename: str,
                        velocities: Optional[np.ndarray] = None,
                        forces: Optional[np.ndarray] = None,
                        masses: Optional[np.ndarray] = None,
                        metadata: Optional[Dict] = None,
                        format: str = 'npz') -> None:
    """
    Save trajectory data in specified format.

    Args:
        positions: (n_frames, n_atoms, 3) positions
        times: (n_frames,) timestamps
        filename: Output file path
        velocities: Optional velocities
        forces: Optional forces
        masses: Optional atomic masses
        metadata: Optional metadata dictionary
        format: Output format ('npz', 'hdf5')

    Example:
        >>> save_trajectory_data(
        ...     positions, times,
        ...     'trajectory.npz',
        ...     velocities=velocities,
        ...     metadata={'temperature': 300.0}
        ... )
    """
    data = {
        'positions': positions,
        'times': times
    }

    if velocities is not None:
        data['velocities'] = velocities

    if forces is not None:
        data['forces'] = forces

    if masses is not None:
        data['masses'] = masses

    if format == 'npz':
        # NPZ doesn't support metadata well, save separately
        save_results_npz(data, filename)

        if metadata is not None:
            meta_file = str(Path(filename).with_suffix('.json'))
            save_results_json(metadata, meta_file)

    elif format == 'hdf5':
        if metadata is not None:
            data['metadata'] = metadata

        save_results_hdf5(data, filename)

    else:
        raise ValueError(f"Unknown format: {format}. Use 'npz' or 'hdf5'")


def load_trajectory_data(filename: str) -> Dict[str, Any]:
    """
    Load trajectory data from file.

    Auto-detects format from extension.

    Args:
        filename: Input file path (.npz or .h5)

    Returns:
        data: Dictionary with trajectory data

    Example:
        >>> data = load_trajectory_data('trajectory.npz')
        >>> positions = data['positions']
        >>> times = data['times']
    """
    path = Path(filename)

    if path.suffix == '.npz':
        data = load_results_npz(filename)

        # Check for metadata file
        meta_file = path.with_suffix('.json')
        if meta_file.exists():
            data['metadata'] = load_results_json(str(meta_file))

    elif path.suffix in ['.h5', '.hdf5']:
        data = load_results_hdf5(filename)

    else:
        raise ValueError(f"Unknown format: {path.suffix}. Use .npz or .h5")

    return data


def export_to_csv(data: Dict[str, np.ndarray],
                 filename: str,
                 flatten: bool = True) -> None:
    """
    Export results to CSV for external analysis.

    Args:
        data: Dictionary with 1D or 2D arrays
        filename: Output .csv file path
        flatten: Flatten multidimensional arrays

    Notes:
        - Good for small datasets
        - Easy to import into Excel, R, etc.
        - Not efficient for large arrays

    Example:
        >>> data = {'time': times, 'theta': euler[:, 1]}
        >>> export_to_csv(data, 'timeseries.csv')
    """
    import csv

    # Prepare data
    if flatten:
        flat_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    flat_data[key] = value
                elif value.ndim == 2:
                    # Split columns
                    for i in range(value.shape[1]):
                        flat_data[f"{key}_{i}"] = value[:, i]
                else:
                    warnings.warn(f"Skipping {key} with {value.ndim} dimensions")
            else:
                flat_data[key] = value
        data = flat_data

    # Check all arrays have same length
    lengths = [len(v) for v in data.values() if isinstance(v, (list, np.ndarray))]
    if len(set(lengths)) > 1:
        raise ValueError("All arrays must have same length for CSV export")

    # Write CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(data.keys())

        # Data rows
        n_rows = lengths[0] if lengths else 0
        for i in range(n_rows):
            row = [data[key][i] if isinstance(data[key], (list, np.ndarray)) else data[key]
                   for key in data.keys()]
            writer.writerow(row)

    print(f"✓ Exported to {filename}")


# ---------------------------------------------------------------------------
# NPZ chunk I/O  (used by the `rotmd extract` / `rotmd merge` CLI)
# ---------------------------------------------------------------------------


def save_npz(path: str | Path, data: dict[str, np.ndarray]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return path


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def merge_npz(inputs: list[Path], output: Path) -> Path:
    if not inputs:
        raise ValueError("at least one input chunk required")

    chunks = [load_npz(p) for p in inputs]

    # Every chunk must have time_ps.
    for c, p in zip(chunks, inputs):
        if "time_ps" not in c:
            raise ValueError(f"{p} has no time_ps")

    # All chunks must share the same set of keys.
    ref_keys = set(chunks[0])
    for c, p in zip(chunks, inputs):
        if set(c) != ref_keys:
            missing = ref_keys - set(c)
            extra = set(c) - ref_keys
            parts = []
            if missing:
                parts.append(f"missing: {sorted(missing)}")
            if extra:
                parts.append(f"extra: {sorted(extra)}")
            raise ValueError(f"key mismatch with {p}: {'; '.join(parts)}")

    # Determine which keys are time-varying (first dim matches time_ps length).
    def _is_time_varying(arr: np.ndarray, n: int) -> bool:
        return arr.ndim >= 1 and arr.shape[0] == n

    # Detect time-invariant keys from the first chunk.
    n0 = len(chunks[0]["time_ps"])
    invariant_keys = {
        k for k in ref_keys if k != "time_ps" and not _is_time_varying(chunks[0][k], n0)
    }

    # Check time monotonicity across chunks.
    all_t = np.concatenate([c["time_ps"] for c in chunks])
    if np.any(np.diff(all_t) <= 0):
        raise ValueError("time_ps is not strictly monotonic across chunks")

    # Check invariant arrays are consistent across chunks.
    for k in invariant_keys:
        ref = chunks[0][k]
        for c in chunks[1:]:
            if c[k].shape != ref.shape or not np.allclose(c[k], ref):
                raise ValueError(
                    f"time-invariant key {k} differs between chunks"
                )

    # Concatenate time-varying arrays.
    merged: dict[str, np.ndarray] = {"time_ps": all_t}
    for k in (ref_keys - {"time_ps"} - invariant_keys):
        merged[k] = np.concatenate([c[k] for c in chunks], axis=0)
    for k in invariant_keys:
        merged[k] = chunks[0][k].copy()

    return save_npz(output, merged)
