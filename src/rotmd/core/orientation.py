"""
Protein Orientation Extraction (extract pipeline)

Extract-scope orientation utilities: turn a centered position trajectory into a
per-frame body-frame rotation and its ZYZ Euler angles, plus the membrane tilt
convenience angle written to the chunk schema.

Post-processing of this time series (quaternions, angular displacement,
autocorrelation, angle derivatives) lives in :mod:`rotmd.analysis.orientation`.

Author: Mykyta Bobylyow
Date: 2025
"""

import numpy as np


def rotation_matrix_to_euler_zyz(R: np.ndarray) -> tuple[float, float, float]:
    """
    Convert rotation matrix to Euler angles using ZYZ convention.

    The ZYZ convention is commonly used in molecular dynamics:
    - First rotation φ about lab Z-axis
    - Second rotation θ about new Y-axis (nutation angle)
    - Third rotation ψ about final Z-axis (spin angle)

    Args:
        R: (3, 3) rotation matrix

    Returns:
        (phi, theta, psi): Euler angles in radians
        - phi ∈ [0, 2π]: First rotation about Z
        - theta ∈ [0, π]: Nutation angle
        - psi ∈ [0, 2π]: Spin angle

    Notes:
        - Handles gimbal lock at θ = 0 and θ = π
        - Uses atan2 for proper quadrant handling
    """
    # Extract elements for ZYZ convention
    # R = Rz(φ) Ry(θ) Rz(ψ)
    theta = np.arccos(np.clip(R[2, 2], -1.0, 1.0))

    # Handle gimbal lock
    if np.abs(np.sin(theta)) < 1e-10:
        # θ ≈ 0 or θ ≈ π
        if theta < np.pi / 2:
            # θ ≈ 0: φ + ψ is determined, set ψ = 0
            phi = np.arctan2(R[1, 0], R[0, 0])
            psi = 0.0
        else:
            # θ ≈ π: φ - ψ is determined, set ψ = 0
            phi = np.arctan2(-R[1, 0], R[0, 0])
            psi = 0.0
    else:
        # For R = Rz(phi) Ry(theta) Rz(psi):
        #   R[1,2] = sin(phi) sin(theta),  R[0,2] =  cos(phi) sin(theta)
        #   R[2,1] = sin(theta) sin(psi),  R[2,0] = -sin(theta) cos(psi)
        # With sin(theta) > 0 these give phi and psi directly via atan2.
        phi = np.arctan2(R[1, 2], R[0, 2])
        psi = np.arctan2(R[2, 1], -R[2, 0])

    # Normalize to [0, 2π] for phi and psi, [0, π] for theta
    phi = phi % (2 * np.pi)
    psi = psi % (2 * np.pi)

    return phi, theta, psi


def membrane_tilt_angle(theta: np.ndarray) -> np.ndarray:
    """
    Membrane tilt angle: 0° when the principal axis is collinear with the
    membrane normal, 90° when it lies in the membrane plane — the standard
    membrane-protein convention (tilt measured *from* the normal).

    The ZYZ nutation angle ``theta`` measures the angle between the protein's
    principal axis and the lab z-axis (the membrane normal), and lives in
    ``[0, π]``. We treat the principal axis as an *undirected line* (its
    eigenvector sign is arbitrary, so ``theta`` and ``π − theta`` describe the
    same physical orientation), so this folds the nutation angle into its
    acute branch::

        tilt = min(theta, π − theta)

    which is sign-stable and always in ``[0, π/2]``:

    - ``theta = 0`` or ``π``  (axis along the normal)  → ``tilt = 0``
    - ``theta = π/2``         (axis in the plane)       → ``tilt = π/2`` (90°)

    Args:
        theta: ZYZ nutation angle(s) in radians, any shape.

    Returns:
        Tilt angle(s) in radians, same shape as ``theta``, in ``[0, π/2]``.

    Examples:
        >>> import numpy as np
        >>> np.degrees(membrane_tilt_angle(np.array([0.0, np.pi / 2, np.pi])))
        array([ 0., 90.,  0.])
    """
    return np.minimum(theta, np.pi - theta)


def fold_tilt_and_psi(theta: np.ndarray, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fold ``(theta, psi)`` onto the ``tilt <= pi/2`` branch, shifting ``psi`` to match.

    ``theta`` and ``psi`` are *both* computed from the same headless axis —
    the largest-moment principal axis ``v_c`` — and nothing else: with
    ``R = axes.T`` in :func:`rotation_matrix_to_euler_zyz`,
    ``theta = arccos(v_c_z)`` and ``psi = arctan2(v_c_y, -v_c_x)`` exactly.
    Flipping the arbitrary sign of ``v_c`` (``v_c -> -v_c``) therefore moves
    *both* angles together::

        (theta, psi) -> (pi - theta, psi + pi)

    Folding ``theta`` alone (as :func:`membrane_tilt_angle` does) without also
    shifting ``psi`` on exactly the same frames leaves ``psi`` split into two
    clusters exactly ``pi`` apart — the same physical orientation reported at
    two different psi values depending on which sign the extraction happened
    to assign that frame. This folds both together, so a PMF/friction map
    over ``(tilt, psi)`` shows one connected basin instead of two disconnected
    ones separated by an empty gap.

    Args:
        theta: ZYZ nutation angle(s) in radians, any shape.
        psi: ZYZ spin angle(s) in radians, same shape as ``theta``.

    Returns:
        ``(tilt, psi_folded)``: ``tilt`` as in :func:`membrane_tilt_angle`,
        and ``psi`` shifted by ``pi`` (mod ``2*pi``) wherever ``theta > pi/2``
        needed folding to bring it below ``pi/2``.

    This decides the branch independently per frame from ``theta`` alone,
    which is exactly right for a trajectory that visits one branch and then
    the other in clearly separated stretches. It is the wrong tool wherever
    ``theta`` sits within thermal-noise distance of the ``pi/2`` pivot for an
    extended time — there, this flips on ordinary fluctuation rather than a
    real change of orientation, scattering ``psi`` across the whole circle
    instead of merging it. Use :func:`fold_tilt_and_psi_continuous` for a
    time-ordered trajectory that spends time at or near the pivot.
    """
    theta = np.asarray(theta, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    tilt = np.minimum(theta, np.pi - theta)
    flipped = theta > np.pi / 2
    psi_folded = np.where(flipped, (psi + np.pi) % (2 * np.pi), psi)
    return tilt, psi_folded


def fold_tilt_and_psi_continuous(theta: np.ndarray, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Continuity-aware version of :func:`fold_tilt_and_psi`, for a time-ordered trajectory.

    :func:`fold_tilt_and_psi` decides which branch a frame is on independently
    from ``theta`` alone (``theta > pi/2``?) — correct when the trajectory
    visits one branch, then the other, in clearly separated stretches, but
    wrong wherever ``theta`` sits within thermal-noise distance of the
    ``pi/2`` pivot: a real trajectory spending its whole run within a few
    degrees of the pivot flips branch on ordinary fluctuation almost every
    other frame, scattering ``psi`` across the entire circle instead of
    merging it into one basin.

    This instead reconstructs the headless axis ``v_c`` (see
    :func:`fold_tilt_and_psi`) frame by frame and propagates its sign forward
    in time, flipping only when doing so keeps it closer to the *previous*
    (already-corrected) frame — the same continuity rule
    :func:`extract_orientation_trajectory` already applies to the raw
    eigenvectors, applied here post hoc to the derived angles. It still folds
    onto the ``tilt <= pi/2`` branch at the end, so a trajectory that
    genuinely, physically crosses the pivot will still show a seam there —
    that reflects a real reorientation, not sampling noise.

    Args:
        theta, psi: ZYZ nutation/spin angles in radians, one per frame, in
            time order (do not reorder or pre-filter with a boolean mask that
            skips frames — the continuity step needs consecutive frames).

    Returns:
        ``(tilt, psi_folded)``, same shapes as the inputs.
    """
    theta = np.asarray(theta, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)

    v_c = np.stack(
        [-np.sin(theta) * np.cos(psi), np.sin(theta) * np.sin(psi), np.cos(theta)],
        axis=-1,
    )
    corrected = v_c.copy()
    for i in range(1, len(corrected)):
        if np.dot(corrected[i], corrected[i - 1]) < 0.0:
            corrected[i] = -corrected[i]

    theta_c = np.arccos(np.clip(corrected[:, 2], -1.0, 1.0))
    psi_c = np.arctan2(corrected[:, 1], -corrected[:, 0]) % (2 * np.pi)
    return fold_tilt_and_psi(theta_c, psi_c)


def extract_orientation_trajectory(
    positions: np.ndarray,
    masses: np.ndarray,
        reference_frame: np.ndarray | None = None,
):
    """
    Extract Euler angles trajectory from atomic positions.

    This function computes the protein's orientation at each frame by:
    1. Computing inertia tensor from positions
    2. Finding principal axes (eigenvectors)
    3. Constructing rotation matrix from principal axes
    4. Converting to Euler angles

    Args:
        positions: (n_frames, n_atoms, 3) atomic positions
        masses: (n_atoms,) atomic masses
        reference_frame: Optional (3, 3) reference orientation.
                        If provided, rotations are relative to this frame.
                        If None, uses lab frame.

    Returns:
        euler_angles: (n_frames, 3) array of [phi, theta, psi] in radians
        rotation_matrix: (n_frames, 3, 3) rotation matrices

    Notes:
        - Positions should be centered at origin (use trajectory.load_trajectory with center=True)
        - Principal axes are ordered by eigenvalues (I1 >= I2 >= I3)
        - Handles sign ambiguity in eigenvectors for smooth trajectories

    Examples:
        >>> import numpy as np
        >>> positions = np.random.rand(1000, 100, 3)  # 1000 frames, 100 atoms
        >>> masses = np.ones(100)
        >>> euler, R = extract_orientation_trajectory(positions, masses)
        >>> print(euler.shape, R.shape)
        (1000, 3) (1000, 3, 3)
    """

    # Original NumPy path (for debugging/validation)
    from .inertia import inertia_tensor, principal_axes

    n_frames = len(positions)
    euler_angles = np.zeros((n_frames, 3))
    rotation_matrix = np.zeros((n_frames, 3, 3))
    prev_axes = None

    for i in range(n_frames):
        pos = positions[i]

        # Compute inertia tensor and principal axes
        I = inertia_tensor(pos, masses)
        moments, axes = principal_axes(I)

        # Ensure consistent sign convention across frames
        if prev_axes is not None:
            # Flip axes if they point in opposite direction from previous frame
            for j in range(3):
                if np.dot(axes[:, j], prev_axes[:, j]) < 0:
                    axes[:, j] *= -1

            # The per-axis continuity flips above can flip an odd number of
            # columns, turning the proper rotation from principal_axes() into a
            # reflection (det = -1). rotation_matrix_to_euler_zyz assumes a proper
            # rotation, so restore det = +1 by flipping the largest-moment axis
            # (column 2). This leaves the smallest-moment "spin" axis (column 0),
            # which L/tau/omega are decomposed against, continuous.
            if np.linalg.det(axes) < 0:
                axes[:, 2] *= -1

        prev_axes = axes.copy()

        # Rotation matrix from lab frame to body frame
        R = axes.T  # Principal axes as columns → rotation matrix

        # If reference frame provided, compute relative rotation
        if reference_frame is not None:
            R = R @ reference_frame.T

        # Convert to Euler angles
        phi, theta, psi = rotation_matrix_to_euler_zyz(R)
        euler_angles[i] = [phi, theta, psi]
        rotation_matrix[i] = R

    return euler_angles, rotation_matrix
