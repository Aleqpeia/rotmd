# mdarchive: MD Trajectory Timeseries Archive

## Overview

`mdarchive` is a Python package for storing, organizing, and querying numpy timeseries extracted from molecular dynamics trajectories. It combines Zarr for efficient array storage with SQLite for queryable metadata.

## Design Goals

1. **Concurrent writes** — multiple SLURM jobs can extract and store data simultaneously
2. **Interactive exploration** — query by system, mutation, membrane, observable, time range
3. **HPC-friendly** — no external services, all data on shared filesystem
4. **Efficient storage** — compressed chunked arrays, lazy loading

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SLURM Extraction Jobs                       │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐                         │
│   │ extract  │  │ extract  │  │ extract  │   (concurrent writers)  │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘                         │
└────────┼────────────┼────────────┼──────────────────────────────────┘
         │            │            │
         ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        mdarchive.Archive                            │
│                                                                     │
│  ┌────────────────────┐      ┌─────────────────────────────────┐   │
│  │  SQLite + WAL      │      │         Zarr Store              │   │
│  │  (metadata.db)     │      │   /archive/zarr/                │   │
│  │                    │      │     ├── {uuid1}.zarr            │   │
│  │  Queryable fields: │      │     │     └── data (N,)         │   │
│  │  - system_name     │◄────►│     ├── {uuid2}.zarr            │   │
│  │  - mutation        │      │     └── ...                     │   │
│  │  - membrane        │      │                                 │   │
│  │  - observable_type │      │  Features:                      │   │
│  │  - protein_present │      │  - zstd compression             │   │
│  │  - t_start, t_end  │      │  - chunked (10k frames)         │   │
│  │  - zarr_path (FK)  │      │  - concurrent-safe              │   │
│  └────────────────────┘      └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
         ▲
         │
┌────────┴────────────┐
│  Interactive Client │
│  (Jupyter / REPL)   │
│                     │
│  archive.query(     │
│    mutation="N75K", │
│    membrane="POPC"  │
│  ).load_all()       │
└─────────────────────┘
```

## Directory Structure

```
/mnt/storage_3/home/mbobylyow/pl0270-02/archive/
├── mdarchive/
│   ├── metadata.db          # SQLite database (with .db-wal, .db-shm)
│   └── zarr/                # Zarr array store
│       ├── <uuid1>.zarr/
│       │   └── .zarray
│       ├── <uuid2>.zarr/
│       └── ...
```

## Package Structure

```
mdarchive/
├── __init__.py              # Public API exports
├── config.py                # Configuration (paths, defaults)
├── models.py                # Dataclasses for Timeseries, Metadata
├── db.py                    # SQLite connection and queries
├── storage.py               # Zarr read/write operations
├── archive.py               # High-level Archive class
└── cli.py                   # Optional CLI for maintenance
```

## Data Model

### TimeseriesMetadata

| Field                 | Type           | Required | Description                          |
|-----------------------|----------------|----------|--------------------------------------|
| `id`                  | UUID           | auto     | Unique identifier                    |
| `system_name`         | str            | yes      | e.g., "hpca_n75k_popc"               |
| `replica_id`          | int            | no       | Replica index (0, 1, 2, ...)         |
| `mutation`            | str            | no       | e.g., "WT", "N75K", "D123A"          |
| `membrane_composition`| str            | no       | e.g., "POPC", "POPC:POPS:80:20"      |
| `protein_present`     | bool           | yes      | Whether protein is in system         |
| `protein_name`        | str            | no       | e.g., "hippocalcin"                  |
| `observable_type`     | str            | yes      | e.g., "euler_phi", "tilt_angle"      |
| `observable_unit`     | str            | no       | e.g., "rad", "deg", "nm"             |
| `t_start`             | float          | yes      | Start time in ps                     |
| `t_end`               | float          | yes      | End time in ps                       |
| `dt`                  | float          | yes      | Time step in ps                      |
| `n_frames`            | int            | yes      | Number of frames                     |
| `zarr_path`           | str            | auto     | Relative path to zarr array         |
| `source_trajectory`   | str            | no       | Path to source .xtc/.trr file        |
| `extracted_at`        | datetime       | auto     | Extraction timestamp                 |
| `extra_metadata`      | dict (JSON)    | no       | Arbitrary additional metadata        |

### Timeseries

```python
@dataclass
class Timeseries:
    metadata: TimeseriesMetadata
    data: np.ndarray  # shape (n_frames,) or (n_frames, ndim)
    
    @property
    def time(self) -> np.ndarray:
        """Reconstruct time array from metadata."""
        return np.arange(self.metadata.t_start, 
                         self.metadata.t_end, 
                         self.metadata.dt)
```

## SQLite Schema

```sql
CREATE TABLE timeseries (
    id TEXT PRIMARY KEY,
    
    -- System identification
    system_name TEXT NOT NULL,
    replica_id INTEGER,
    
    -- Biological context
    mutation TEXT,
    membrane_composition TEXT,
    protein_present INTEGER NOT NULL,  -- SQLite boolean
    protein_name TEXT,
    
    -- Observable description
    observable_type TEXT NOT NULL,
    observable_unit TEXT,
    
    -- Time range (picoseconds)
    t_start REAL NOT NULL,
    t_end REAL NOT NULL,
    dt REAL NOT NULL,
    n_frames INTEGER NOT NULL,
    
    -- Storage reference
    zarr_path TEXT NOT NULL UNIQUE,
    
    -- Provenance
    source_trajectory TEXT,
    extracted_at TEXT DEFAULT (datetime('now')),
    extra_metadata TEXT  -- JSON string
);

-- Query indices
CREATE INDEX idx_system ON timeseries(system_name);
CREATE INDEX idx_mutation ON timeseries(mutation);
CREATE INDEX idx_membrane ON timeseries(membrane_composition);
CREATE INDEX idx_observable ON timeseries(observable_type);
CREATE INDEX idx_protein ON timeseries(protein_present, protein_name);
CREATE INDEX idx_time ON timeseries(t_start, t_end);
```

## SQLite Configuration for Concurrency

```python
PRAGMA journal_mode=WAL;      # Write-Ahead Logging for concurrent reads
PRAGMA busy_timeout=30000;    # Wait up to 30s for locks
PRAGMA synchronous=NORMAL;    # Balance durability vs performance
```

WAL mode allows:
- Multiple concurrent readers
- One writer at a time (others queue with busy_timeout)
- Readers don't block writers, writers don't block readers

## Zarr Configuration

```python
compressor = numcodecs.Blosc(
    cname='zstd',           # Good compression ratio
    clevel=3,               # Moderate compression level
    shuffle=Blosc.BITSHUFFLE  # Effective for float arrays
)

chunks = (10000,)  # ~10k frames per chunk, tunable
```

## Public API

### Archive Class

```python
class Archive:
    def __init__(self, root_path: str | Path):
        """
        Initialize archive at given path.
        Creates metadata.db and zarr/ subdirectory if needed.
        """
    
    def store(
        self,
        data: np.ndarray,
        system_name: str,
        observable_type: str,
        t_start: float,
        t_end: float,
        dt: float,
        *,
        mutation: str | None = None,
        membrane_composition: str | None = None,
        protein_present: bool = False,
        protein_name: str | None = None,
        replica_id: int | None = None,
        observable_unit: str | None = None,
        source_trajectory: str | None = None,
        extra_metadata: dict | None = None,
    ) -> str:
        """
        Store timeseries with metadata.
        Returns UUID of stored timeseries.
        Thread-safe for concurrent SLURM jobs.
        """
    
    def query(
        self,
        system_name: str | None = None,
        mutation: str | None = None,
        membrane: str | None = None,
        observable: str | None = None,
        protein_present: bool | None = None,
        protein_name: str | None = None,
        t_min: float | None = None,
        t_max: float | None = None,
    ) -> QueryResult:
        """
        Query timeseries by metadata fields.
        Returns lazy QueryResult for further operations.
        All parameters are optional; omitted = no filter.
        """
    
    def get(self, ts_id: str) -> Timeseries:
        """Retrieve single timeseries by UUID."""
    
    def delete(self, ts_id: str) -> None:
        """Delete timeseries (metadata + zarr array)."""
    
    def list_systems(self) -> list[str]:
        """List all unique system names."""
    
    def list_observables(self) -> list[str]:
        """List all unique observable types."""
```

### QueryResult Class

```python
class QueryResult:
    """Lazy container for query results."""
    
    def __len__(self) -> int:
        """Number of matching timeseries."""
    
    def __iter__(self) -> Iterator[Timeseries]:
        """Iterate over timeseries, loading data lazily."""
    
    def first(self) -> Timeseries | None:
        """Load and return first result, or None if empty."""
    
    def load_all(self) -> list[Timeseries]:
        """Load all matching timeseries into memory."""
    
    def metadata_df(self) -> pd.DataFrame:
        """Return metadata as DataFrame (no array loading)."""
    
    def ids(self) -> list[str]:
        """Return list of matching UUIDs."""
```

## Usage Examples

### Storing from SLURM extraction job

```python
from mdarchive import Archive
import numpy as np

archive = Archive("/mnt/storage_3/home/mbobylyow/pl0270-02/archive/mdarchive")

# After extracting euler angles from trajectory
euler_phi = extract_euler_phi(trajectory)  # np.ndarray

archive.store(
    data=euler_phi,
    system_name="hpca_n75k_popc",
    observable_type="euler_phi",
    observable_unit="rad",
    t_start=0.0,
    t_end=1_000_000.0,  # 1 µs in ps
    dt=10.0,
    mutation="N75K",
    membrane_composition="POPC",
    protein_present=True,
    protein_name="hippocalcin",
    replica_id=1,
    source_trajectory="/path/to/md.xtc",
)
```

### Interactive exploration

```python
from mdarchive import Archive
import matplotlib.pyplot as plt

archive = Archive("/mnt/storage_3/home/mbobylyow/pl0270-02/archive/mdarchive")

# Find all N75K euler_phi timeseries
results = archive.query(mutation="N75K", observable="euler_phi")
print(f"Found {len(results)} timeseries")

# Quick metadata overview
df = results.metadata_df()
print(df[["system_name", "replica_id", "membrane_composition", "n_frames"]])

# Load and plot first one
ts = results.first()
plt.plot(ts.time / 1000, ts.data)  # time in ns
plt.xlabel("Time (ns)")
plt.ylabel("φ (rad)")

# Compare WT vs N75K
wt = archive.query(mutation="WT", observable="euler_phi").load_all()
n75k = archive.query(mutation="N75K", observable="euler_phi").load_all()
```

### Batch analysis

```python
# Stack all replicas for ensemble analysis
results = archive.query(
    system_name="hpca_n75k_popc",
    observable="euler_phi"
)

# Load all and stack
all_data = np.stack([ts.data for ts in results])  # (n_replicas, n_frames)

# Compute ensemble average
mean_phi = np.mean(all_data, axis=0)
std_phi = np.std(all_data, axis=0)
```

## Error Handling

```python
from mdarchive import Archive, TimeseriesNotFoundError

archive = Archive(...)

try:
    ts = archive.get("nonexistent-uuid")
except TimeseriesNotFoundError:
    print("Timeseries not found")

# Query returns empty result (not error) if no matches
results = archive.query(mutation="NONEXISTENT")
assert len(results) == 0
```

## Dependencies

Required:
- `numpy`
- `zarr`
- `numcodecs`

Optional:
- `pandas` (for `metadata_df()`)

All available on Eagle via pip install to `~/.local` or conda environment.

## Installation on HPC

```bash
# In your archive directory or home
pip install --user zarr numcodecs

# Or in extraction job scripts
module load python/3.13.0-gcc-14.2.0
pip install --user zarr numcodecs
```

## Maintenance

### Vacuum database

```python
archive = Archive(...)
archive.vacuum()  # Reclaim space after deletions
```

### Verify integrity

```python
archive = Archive(...)
orphans = archive.check_integrity()
# Returns list of zarr paths without metadata entries (or vice versa)
```

### Export metadata

```python
df = archive.query().metadata_df()
df.to_csv("all_timeseries_metadata.csv")
```

## Future Extensions

- **Multi-dimensional observables**: Store (n_frames, 3) for vector quantities
- **Compression tuning**: Per-observable compression settings
- **Append mode**: Extend existing timeseries with new frames
- **PostgreSQL backend**: Swap db.py for larger deployments
- **Remote access**: Serve via HTTP API for external tools