"""Perfect Magnetic Conductor (PMC) boundary condition (T7 Phase 2 PR3).

Zeros tangential H-field at boundary faces. The electromagnetic dual
of :mod:`rfx.boundaries.pec`: PEC enforces E_tangential = 0, PMC
enforces H_tangential = 0. PMC is the boundary condition imposed by a
"magnetic wall" — physically rare in practice but essential as a
symmetry-plane image source for structures with mirror symmetry and
as a building block for cavities / waveguides with mixed-BC modes.

For a face normal to axis ``a`` at side ``s ∈ {lo, hi}`` the
tangential H components are the two H components with indices
``≠ a``. On ``x_lo``: Hy[0, :, :] and Hz[0, :, :]. The order point in
the scan body mirrors :func:`rfx.boundaries.pec.apply_pec_faces`
(after the H update, before the next E update).
"""

from __future__ import annotations


def apply_pmc_faces(state, faces: set[str]) -> object:
    """Apply PMC (``H_tan = 0``) on specific boundary faces.

    Parameters
    ----------
    state : FDTDState
    faces : set of str
        Which faces to enforce PMC on. Valid names:
        ``"x_lo"``, ``"x_hi"``, ``"y_lo"``, ``"y_hi"``,
        ``"z_lo"``, ``"z_hi"``.
    """
    if not faces:
        return state
    hx, hy, hz = state.hx, state.hy, state.hz

    if "x_lo" in faces:
        hy = hy.at[0, :, :].set(0.0)
        hz = hz.at[0, :, :].set(0.0)
    if "x_hi" in faces:
        hy = hy.at[-1, :, :].set(0.0)
        hz = hz.at[-1, :, :].set(0.0)
    if "y_lo" in faces:
        hx = hx.at[:, 0, :].set(0.0)
        hz = hz.at[:, 0, :].set(0.0)
    if "y_hi" in faces:
        hx = hx.at[:, -1, :].set(0.0)
        hz = hz.at[:, -1, :].set(0.0)
    if "z_lo" in faces:
        hx = hx.at[:, :, 0].set(0.0)
        hy = hy.at[:, :, 0].set(0.0)
    if "z_hi" in faces:
        hx = hx.at[:, :, -1].set(0.0)
        hy = hy.at[:, :, -1].set(0.0)

    return state._replace(hx=hx, hy=hy, hz=hz)
