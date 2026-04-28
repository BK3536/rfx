"""Stage 1 conformal PEC face-shift for axis-aligned boundaries.

When a PEC domain boundary does not align exactly with a Yee-cell edge, the
standard ``apply_pec`` path zeros tangential E at cell index ``N-1``
(staircase wall at ``N·dx``) rather than at the true physical wall.  This
module computes per-cell fractional-fill factors α ∈ [0, 1] for each
boundary face row so that the update coefficients can be smoothly attenuated:

    ca_tan *= (1 − α)   (decay coefficient)
    cb_tan *= (1 − α)   (curl-drive coefficient)

With α=1 this reproduces the hard-PEC zero (binary staircase), and with
α=0.86 (WR-90 y_hi at dx=1 mm: (22.86-22.0)/1.0) the effective wall moves
from ``23 mm`` to ``22.86 mm``.

Scope — Stage 1: axis-aligned PEC domain faces on uniform grids only.
Stage 2 (curved/rotated PEC, non-uniform grids) is tracked on issue #74.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------

def compute_pec_alpha_for_axis_aligned_face(
    physical_wall: float,
    cell_lo_coords: np.ndarray,
    cell_widths: np.ndarray,
    *,
    face_side: str = "hi",
) -> np.ndarray:
    """Fractional PEC fill α for a 1-D row of cells along one axis.

    For a hi-face (``face_side='hi'``): the PEC wall is at ``physical_wall``
    and the last cell runs from ``cell_lo_coords[-1]`` to
    ``cell_lo_coords[-1] + cell_widths[-1]``.  Interior cells that are
    entirely below (or above for lo-face) the wall get α = 1.  The boundary
    cell gets α = (physical_wall − cell_lo) / cell_width, clamped to [0, 1].
    Cells past the wall (outside the domain) get α = 0.

    Parameters
    ----------
    physical_wall : float
        Physical coordinate of the PEC wall (e.g. ``port.a = 22.86e-3`` m).
    cell_lo_coords : (N,) ndarray
        Physical coordinate of the low edge of each cell (metres).
    cell_widths : (N,) ndarray
        Width of each cell in metres.
    face_side : {'hi', 'lo'}
        Whether this is the +face (``'hi'``) or −face (``'lo'``).

    Returns
    -------
    alpha : (N,) ndarray, dtype float32
        Per-cell fractional fill.  1.0 = fully inside PEC (zeroed), 0.0 =
        fully outside (no change).

    Examples
    --------
    WR-90 y_hi at dx=1 mm, 23 cells (indices 0..22), wall at 22.86 mm:

    >>> cell_lo = np.arange(23) * 1e-3      # 0, 1, …, 22 mm
    >>> widths  = np.full(23, 1e-3)
    >>> alpha = compute_pec_alpha_for_axis_aligned_face(22.86e-3, cell_lo, widths, face_side='hi')
    >>> float(alpha[-1])                    # boundary cell (22..23 mm)
    0.86
    >>> all(alpha[:-1] == 1.0)              # interior cells fully inside PEC
    True
    """
    cell_lo = np.asarray(cell_lo_coords, dtype=np.float64)
    widths = np.asarray(cell_widths, dtype=np.float64)
    cell_hi = cell_lo + widths
    wall = float(physical_wall)

    if face_side == "hi":
        # α = fraction of [cell_lo, cell_hi] that lies inside [0, wall]
        # Interior (cell_hi <= wall): α = 1
        # Boundary (cell_lo < wall < cell_hi): α = (wall - cell_lo) / width
        # Outside (cell_lo >= wall): α = 0
        overlap = np.minimum(cell_hi, wall) - cell_lo
        alpha = np.clip(overlap / widths, 0.0, 1.0)
    else:  # lo face: PEC to the left, domain to the right
        # α = fraction of [cell_lo, cell_hi] that lies inside [wall, +∞)
        overlap = cell_hi - np.maximum(cell_lo, wall)
        alpha = np.clip(overlap / widths, 0.0, 1.0)

    return alpha.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-face alpha computation from waveguide port aperture dimensions
# ---------------------------------------------------------------------------

def compute_boundary_face_alphas_from_port(
    port_a: float,
    port_b: float,
    normal_axis: str,
    grid_dx: float,
    grid_ny: int,
    grid_nz: int,
    grid_nx: int,
    cpml_pad_y: int = 0,
    cpml_pad_z: int = 0,
    cpml_pad_x: int = 0,
) -> dict[str, np.ndarray]:
    """Compute per-face α arrays for the two transverse PEC faces of a
    rectangular waveguide port (x-normal case, WR-90 style).

    For an x-normal port the two PEC-boundary faces are y_hi (aperture y in
    [0, port_a]) and z_hi (aperture z in [0, port_b]).  Cell coordinates are
    derived from the scalar dx and the grid padding (CPML pad).

    Parameters
    ----------
    port_a, port_b : float
        Physical aperture dimensions (metres).
    normal_axis : str
        Port normal axis (``'x'``, ``'y'``, or ``'z'``).  Only ``'x'`` is
        fully wired in Stage 1; other values return empty dict.
    grid_dx : float
        Uniform cell size (metres).
    grid_ny, grid_nz, grid_nx : int
        Total grid dimensions (including CPML padding).
    cpml_pad_y, cpml_pad_z, cpml_pad_x : int
        CPML cell padding on each axis (lo side).

    Returns
    -------
    dict mapping face label (``'y_hi'``, ``'z_hi'``, etc.) to 1-D alpha
    arrays of shape (N_cells_on_face,).  Empty dict when axis not supported.
    """
    dx = float(grid_dx)
    if normal_axis == "x":
        # y_hi face: physical aperture along y in [0, port_a]
        ny_phys = grid_ny - 2 * cpml_pad_y
        cell_lo_y = np.arange(ny_phys) * dx  # physical coords (no CPML offset)
        widths_y = np.full(ny_phys, dx)
        alpha_y = compute_pec_alpha_for_axis_aligned_face(
            port_a, cell_lo_y, widths_y, face_side="hi"
        )

        # z_hi face: physical aperture along z in [0, port_b]
        nz_phys = grid_nz - 2 * cpml_pad_z
        cell_lo_z = np.arange(nz_phys) * dx
        widths_z = np.full(nz_phys, dx)
        alpha_z = compute_pec_alpha_for_axis_aligned_face(
            port_b, cell_lo_z, widths_z, face_side="hi"
        )
        return {"y_hi": alpha_y, "z_hi": alpha_z}

    # Stage 1: y-normal and z-normal ports follow the same pattern but are
    # not needed for cv11 so are deferred — return empty to fall back to
    # binary staircase.
    return {}


# ---------------------------------------------------------------------------
# Stage 2 guard
# ---------------------------------------------------------------------------

def _check_axis_aligned_only(pec_shapes: list) -> None:
    """Raise NotImplementedError for any non-axis-aligned PEC shape.

    Called when ``conformal=True`` is set on a boundary face.  Axis-aligned
    boxes (corner_lo/corner_hi attributes) are allowed.  Anything else
    (spheres, cylinders, rotated boxes, arbitrary CSG) raises.
    """
    for shape in pec_shapes:
        is_box = hasattr(shape, "corner_lo") and hasattr(shape, "corner_hi")
        if not is_box:
            raise NotImplementedError(
                "conformal subpixel for curved/rotated PEC is Stage 2 — "
                "track on issue #74; use mesh-pinning + conformal=False for now. "
                f"Offending shape: {shape!r}"
            )
