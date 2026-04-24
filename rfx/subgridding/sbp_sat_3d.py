"""3D SBP-SAT FDTD subgridding.

The current implementation keeps the same global timestep on the coarse and
fine grids (limited by the fine-grid CFL condition) and uses explicit
restriction/prolongation operators on coarse/fine interface traces.

The original shipped lane was a full-span x/y z slab.  This module now
supports an arbitrary all-PEC refinement box in the low-level runtime while the
higher-level support surface remains governed by the support matrix and docs.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0
from rfx.subgridding.face_ops import (
    EdgeOps,
    ZFaceOps,
    build_edge_ops,
    build_face_ops,
    build_zface_ops,
    prolong_edge,
    prolong_face,
    prolong_zface,
    restrict_edge,
    restrict_face,
    restrict_zface,
)

C0 = 1.0 / np.sqrt(EPS_0 * MU_0)
PHASE1_3D_CFL = 0.99


class SubgridConfig3D(NamedTuple):
    """Configuration for the current all-PEC subgrid lane."""

    nx_c: int
    ny_c: int
    nz_c: int
    dx_c: float
    fi_lo: int
    fi_hi: int
    fj_lo: int
    fj_hi: int
    fk_lo: int
    fk_hi: int
    nx_f: int
    ny_f: int
    nz_f: int
    dx_f: float
    dt: float
    ratio: int
    tau: float
    face_ops: ZFaceOps | None = None


class FaceOrientation(NamedTuple):
    """Face-orientation metadata for box coupling."""

    face: str
    normal_axis: int
    normal_sign: int
    tangential_axes: tuple[int, int]
    tangential_e_components: tuple[str, str]
    tangential_h_components: tuple[str, str]


class EdgeOrientation(NamedTuple):
    """Edge-orientation metadata for box coupling."""

    edge: str
    varying_axis: int
    fixed_faces: tuple[str, str]
    e_component: str
    h_component: str


class CornerOrientation(NamedTuple):
    """Corner-orientation metadata for box coupling."""

    corner: str
    fixed_faces: tuple[str, str, str]


class SubgridState3D(NamedTuple):
    """Field state for the canonical Phase-1 z-slab lane."""

    ex_c: jnp.ndarray
    ey_c: jnp.ndarray
    ez_c: jnp.ndarray
    hx_c: jnp.ndarray
    hy_c: jnp.ndarray
    hz_c: jnp.ndarray
    ex_f: jnp.ndarray
    ey_f: jnp.ndarray
    ez_f: jnp.ndarray
    hx_f: jnp.ndarray
    hy_f: jnp.ndarray
    hz_f: jnp.ndarray
    step: int


FACE_ORIENTATIONS: dict[str, FaceOrientation] = {
    "x_lo": FaceOrientation("x_lo", 0, -1, (1, 2), ("ey", "ez"), ("hy", "hz")),
    "x_hi": FaceOrientation("x_hi", 0, +1, (1, 2), ("ey", "ez"), ("hy", "hz")),
    "y_lo": FaceOrientation("y_lo", 1, -1, (0, 2), ("ex", "ez"), ("hx", "hz")),
    "y_hi": FaceOrientation("y_hi", 1, +1, (0, 2), ("ex", "ez"), ("hx", "hz")),
    "z_lo": FaceOrientation("z_lo", 2, -1, (0, 1), ("ex", "ey"), ("hx", "hy")),
    "z_hi": FaceOrientation("z_hi", 2, +1, (0, 1), ("ex", "ey"), ("hx", "hy")),
}

EDGE_ORIENTATIONS: dict[str, EdgeOrientation] = {
    "x_lo_y_lo": EdgeOrientation("x_lo_y_lo", 2, ("x_lo", "y_lo"), "ez", "hz"),
    "x_lo_y_hi": EdgeOrientation("x_lo_y_hi", 2, ("x_lo", "y_hi"), "ez", "hz"),
    "x_hi_y_lo": EdgeOrientation("x_hi_y_lo", 2, ("x_hi", "y_lo"), "ez", "hz"),
    "x_hi_y_hi": EdgeOrientation("x_hi_y_hi", 2, ("x_hi", "y_hi"), "ez", "hz"),
    "x_lo_z_lo": EdgeOrientation("x_lo_z_lo", 1, ("x_lo", "z_lo"), "ey", "hy"),
    "x_lo_z_hi": EdgeOrientation("x_lo_z_hi", 1, ("x_lo", "z_hi"), "ey", "hy"),
    "x_hi_z_lo": EdgeOrientation("x_hi_z_lo", 1, ("x_hi", "z_lo"), "ey", "hy"),
    "x_hi_z_hi": EdgeOrientation("x_hi_z_hi", 1, ("x_hi", "z_hi"), "ey", "hy"),
    "y_lo_z_lo": EdgeOrientation("y_lo_z_lo", 0, ("y_lo", "z_lo"), "ex", "hx"),
    "y_lo_z_hi": EdgeOrientation("y_lo_z_hi", 0, ("y_lo", "z_hi"), "ex", "hx"),
    "y_hi_z_lo": EdgeOrientation("y_hi_z_lo", 0, ("y_hi", "z_lo"), "ex", "hx"),
    "y_hi_z_hi": EdgeOrientation("y_hi_z_hi", 0, ("y_hi", "z_hi"), "ex", "hx"),
}

CORNER_ORIENTATIONS: dict[str, CornerOrientation] = {
    "x_lo_y_lo_z_lo": CornerOrientation("x_lo_y_lo_z_lo", ("x_lo", "y_lo", "z_lo")),
    "x_lo_y_lo_z_hi": CornerOrientation("x_lo_y_lo_z_hi", ("x_lo", "y_lo", "z_hi")),
    "x_lo_y_hi_z_lo": CornerOrientation("x_lo_y_hi_z_lo", ("x_lo", "y_hi", "z_lo")),
    "x_lo_y_hi_z_hi": CornerOrientation("x_lo_y_hi_z_hi", ("x_lo", "y_hi", "z_hi")),
    "x_hi_y_lo_z_lo": CornerOrientation("x_hi_y_lo_z_lo", ("x_hi", "y_lo", "z_lo")),
    "x_hi_y_lo_z_hi": CornerOrientation("x_hi_y_lo_z_hi", ("x_hi", "y_lo", "z_hi")),
    "x_hi_y_hi_z_lo": CornerOrientation("x_hi_y_hi_z_lo", ("x_hi", "y_hi", "z_lo")),
    "x_hi_y_hi_z_hi": CornerOrientation("x_hi_y_hi_z_hi", ("x_hi", "y_hi", "z_hi")),
}


def _region_shape(config: SubgridConfig3D) -> tuple[int, int, int]:
    return (
        config.fi_hi - config.fi_lo,
        config.fj_hi - config.fj_lo,
        config.fk_hi - config.fk_lo,
    )


def _coarse_box_shape(config: SubgridConfig3D) -> tuple[int, int, int]:
    return _region_shape(config)


def _face_coarse_shape(config: SubgridConfig3D, face: str) -> tuple[int, int]:
    ni, nj, nk = _coarse_box_shape(config)
    axis = FACE_ORIENTATIONS[face].normal_axis
    if axis == 0:
        return (nj, nk)
    if axis == 1:
        return (ni, nk)
    return (ni, nj)


def _get_face_ops(config: SubgridConfig3D, face: str) -> ZFaceOps:
    coarse_shape = _face_coarse_shape(config, face)
    if coarse_shape[0] <= 0 or coarse_shape[1] <= 0:
        raise ValueError(
            f"Invalid face shape for {face}: coarse_shape={coarse_shape}"
        )
    if face.startswith("z") and config.face_ops is not None:
        if config.face_ops.coarse_shape == coarse_shape and config.face_ops.ratio == config.ratio:
            return config.face_ops
    return build_face_ops(coarse_shape, config.ratio, config.dx_c)


def _get_edge_ops(config: SubgridConfig3D, edge: str) -> EdgeOps:
    ni, nj, nk = _coarse_box_shape(config)
    axis = EDGE_ORIENTATIONS[edge].varying_axis
    coarse_size = (ni, nj, nk)[axis]
    if coarse_size <= 0:
        raise ValueError(f"Invalid edge size for {edge}: {coarse_size}")
    return build_edge_ops(coarse_size, config.ratio, config.dx_c)


def validate_subgrid_config_3d(config: SubgridConfig3D) -> None:
    """Validate the current all-PEC box config invariants."""

    if config.ratio <= 1:
        raise ValueError(f"Phase-1 SBP-SAT ratio must be > 1, got {config.ratio}")
    if config.dx_c <= 0 or config.dx_f <= 0:
        raise ValueError("Phase-1 SBP-SAT spacings dx_c and dx_f must be positive")
    expected_dx_f = config.dx_c / config.ratio
    if not np.isclose(config.dx_f, expected_dx_f):
        raise ValueError(
            "Phase-1 SBP-SAT dx_f must equal dx_c / ratio; "
            f"got dx_c={config.dx_c}, ratio={config.ratio}, dx_f={config.dx_f}"
        )
    expected_dt = phase1_3d_dt(config.dx_f)
    if not np.isclose(config.dt, expected_dt):
        raise ValueError(
            "Phase-1 SBP-SAT dt must follow the canonical near-fine-CFL rule; "
            f"got dt={config.dt}, expected {expected_dt}"
        )
    if not (0 <= config.fi_lo < config.fi_hi <= config.nx_c):
        raise ValueError(
            "SBP-SAT requires a non-empty x box inside the coarse grid; "
            f"got fi_lo={config.fi_lo}, fi_hi={config.fi_hi}, nx_c={config.nx_c}"
        )
    if not (0 <= config.fj_lo < config.fj_hi <= config.ny_c):
        raise ValueError(
            "SBP-SAT requires a non-empty y box inside the coarse grid; "
            f"got fj_lo={config.fj_lo}, fj_hi={config.fj_hi}, ny_c={config.ny_c}"
        )
    if not (0 <= config.fk_lo < config.fk_hi <= config.nz_c):
        raise ValueError(
            "SBP-SAT requires a non-empty z box inside the coarse grid; "
            f"got fk_lo={config.fk_lo}, fk_hi={config.fk_hi}, nz_c={config.nz_c}"
        )

    ni, nj, nk = _region_shape(config)
    expected_fine = (ni * config.ratio, nj * config.ratio, nk * config.ratio)
    actual_fine = (config.nx_f, config.ny_f, config.nz_f)
    if actual_fine != expected_fine:
        raise ValueError(
            "SBP-SAT fine shape must match coarse box shape times ratio; "
            f"got {actual_fine}, expected {expected_fine}"
        )
    if config.face_ops is not None:
        if config.face_ops.coarse_shape != (ni, nj):
            raise ValueError(
                "SBP-SAT z-face_ops coarse shape mismatch; "
                f"got {config.face_ops.coarse_shape}, expected {(ni, nj)}"
            )
        if config.face_ops.fine_shape != (config.nx_f, config.ny_f):
            raise ValueError(
                "SBP-SAT z-face_ops fine shape mismatch; "
                f"got {config.face_ops.fine_shape}, expected {(config.nx_f, config.ny_f)}"
            )
        if config.face_ops.ratio != config.ratio:
            raise ValueError(
                "SBP-SAT z-face_ops ratio mismatch; "
                f"got {config.face_ops.ratio}, expected {config.ratio}"
            )


def phase1_3d_dt(dx_f: float) -> float:
    """Return the canonical Phase-1 3D timestep for fine spacing ``dx_f``."""

    return PHASE1_3D_CFL * dx_f / (C0 * np.sqrt(3.0))


def _default_fine_region(shape_c: tuple[int, int, int]) -> tuple[int, int, int, int, int, int]:
    nx_c, ny_c, nz_c = shape_c
    fk_lo = max(1, nz_c // 3)
    fk_hi = min(nz_c - 1, max(fk_lo + 1, 2 * nz_c // 3))
    return (0, nx_c, 0, ny_c, fk_lo, fk_hi)


def _validate_phase1_fine_region(
    shape_c: tuple[int, int, int],
    fine_region: tuple[int, int, int, int, int, int],
) -> tuple[int, int, int, int, int, int]:
    nx_c, ny_c, nz_c = shape_c
    fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi = fine_region
    if not (0 <= fi_lo < fi_hi <= nx_c):
        raise ValueError(f"fine_region={fine_region} has invalid x bounds")
    if not (0 <= fj_lo < fj_hi <= ny_c):
        raise ValueError(f"fine_region={fine_region} has invalid y bounds")
    if not (0 <= fk_lo < fk_hi <= nz_c):
        raise ValueError(f"fine_region={fine_region} has invalid z bounds")
    return fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi


def init_subgrid_3d(
    shape_c: tuple[int, int, int] = (40, 40, 40),
    dx_c: float = 0.003,
    fine_region: tuple[int, int, int, int, int, int] | None = None,
    ratio: int = 3,
    courant: float | None = None,
    tau: float = 0.5,
) -> tuple[SubgridConfig3D, SubgridState3D]:
    """Initialize the current all-PEC subgrid state.

    ``fine_region`` is given in coarse-grid index bounds
    ``(fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi)``. ``courant`` remains a
    compatibility alias for the canonical near-fine-CFL rule and rejects any
    non-canonical value.
    """

    nx_c, ny_c, nz_c = shape_c
    if ratio <= 1:
        raise ValueError(f"ratio must be > 1, got {ratio}")
    if fine_region is None:
        fine_region = _default_fine_region(shape_c)
    fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi = _validate_phase1_fine_region(
        shape_c,
        fine_region,
    )
    dx_f = dx_c / ratio
    if courant is not None and not np.isclose(courant, PHASE1_3D_CFL):
        raise ValueError(
            "Phase-1 init_subgrid_3d uses the canonical near-fine-CFL rule "
            f"only ({PHASE1_3D_CFL}); got courant={courant}"
        )
    dt = phase1_3d_dt(dx_f)

    nx_f = (fi_hi - fi_lo) * ratio
    ny_f = (fj_hi - fj_lo) * ratio
    nz_f = (fk_hi - fk_lo) * ratio

    config = SubgridConfig3D(
        nx_c=nx_c,
        ny_c=ny_c,
        nz_c=nz_c,
        dx_c=dx_c,
        fi_lo=fi_lo,
        fi_hi=fi_hi,
        fj_lo=fj_lo,
        fj_hi=fj_hi,
        fk_lo=fk_lo,
        fk_hi=fk_hi,
        nx_f=nx_f,
        ny_f=ny_f,
        nz_f=nz_f,
        dx_f=dx_f,
        dt=float(dt),
        ratio=ratio,
        tau=tau,
        face_ops=build_zface_ops((fi_hi - fi_lo, fj_hi - fj_lo), ratio, dx_c),
    )
    validate_subgrid_config_3d(config)

    z = lambda s: jnp.zeros(s, dtype=jnp.float32)
    state = SubgridState3D(
        ex_c=z(shape_c),
        ey_c=z(shape_c),
        ez_c=z(shape_c),
        hx_c=z(shape_c),
        hy_c=z(shape_c),
        hz_c=z(shape_c),
        ex_f=z((nx_f, ny_f, nz_f)),
        ey_f=z((nx_f, ny_f, nz_f)),
        ez_f=z((nx_f, ny_f, nz_f)),
        hx_f=z((nx_f, ny_f, nz_f)),
        hy_f=z((nx_f, ny_f, nz_f)),
        hz_f=z((nx_f, ny_f, nz_f)),
        step=0,
    )
    return config, state


def _make_mats(shape):
    from rfx.core.yee import MaterialArrays

    return MaterialArrays(
        eps_r=jnp.ones(shape, dtype=jnp.float32),
        sigma=jnp.zeros(shape, dtype=jnp.float32),
        mu_r=jnp.ones(shape, dtype=jnp.float32),
    )


def _update_h_only(
    ex, ey, ez, hx, hy, hz, dt, dx,
    mats=None,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    pmc_faces: frozenset[str] | None = None,
):
    from rfx.boundaries.pmc import apply_pmc_faces
    from rfx.core.yee import FDTDState, update_h

    shape = ex.shape
    state = FDTDState(
        ex=ex,
        ey=ey,
        ez=ez,
        hx=hx,
        hy=hy,
        hz=hz,
        step=jnp.array(0, dtype=jnp.int32),
    )
    if mats is None:
        mats = _make_mats(shape)
    state = update_h(state, mats, dt, dx, periodic=periodic)
    if pmc_faces:
        state = apply_pmc_faces(state, pmc_faces)
    return state.hx, state.hy, state.hz


def _update_e_only(
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    dt,
    dx,
    mats=None,
    pec_mask=None,
    boundary_axes: str | None = "xyz",
    boundary_faces: frozenset[str] | None = None,
    periodic: tuple[bool, bool, bool] = (False, False, False),
):
    from rfx.boundaries.pec import apply_pec, apply_pec_faces, apply_pec_mask
    from rfx.core.yee import FDTDState, update_e

    shape = ex.shape
    state = FDTDState(
        ex=ex,
        ey=ey,
        ez=ez,
        hx=hx,
        hy=hy,
        hz=hz,
        step=jnp.array(0, dtype=jnp.int32),
    )
    if mats is None:
        mats = _make_mats(shape)
    state = update_e(state, mats, dt, dx, periodic=periodic)
    if boundary_faces is not None:
        state = apply_pec_faces(state, boundary_faces)
    elif boundary_axes:
        state = apply_pec(state, axes=boundary_axes)
    if pec_mask is not None:
        state = apply_pec_mask(state, pec_mask)
    return state.ex, state.ey, state.ez


def sat_penalty_coefficients(ratio: int, tau: float) -> tuple[float, float]:
    """Return the canonical SAT penalty coefficients."""

    alpha_f = tau * ratio / (ratio + 1.0)
    alpha_c = tau * 1.0 / (ratio + 1.0)
    return float(alpha_c), float(alpha_f)

_COMPONENT_INDEX = {
    "ex": 0, "ey": 1, "ez": 2,
    "hx": 0, "hy": 1, "hz": 2,
}


def _face_is_internal(config: SubgridConfig3D, face: str) -> bool:
    if face == "x_lo":
        return config.fi_lo > 0
    if face == "x_hi":
        return config.fi_hi < config.nx_c
    if face == "y_lo":
        return config.fj_lo > 0
    if face == "y_hi":
        return config.fj_hi < config.ny_c
    if face == "z_lo":
        return config.fk_lo > 0
    if face == "z_hi":
        return config.fk_hi < config.nz_c
    raise ValueError(f"unknown face {face!r}")


def _active_faces(config: SubgridConfig3D) -> tuple[str, ...]:
    return tuple(face for face in FACE_ORIENTATIONS if _face_is_internal(config, face))


def _touching_outer_faces(config: SubgridConfig3D) -> frozenset[str]:
    return frozenset(face for face in FACE_ORIENTATIONS if not _face_is_internal(config, face))


def _fixed_face_index(config: SubgridConfig3D, face: str, *, grid: str) -> int:
    if grid == "coarse":
        return {
            "x_lo": config.fi_lo,
            "x_hi": config.fi_hi - 1,
            "y_lo": config.fj_lo,
            "y_hi": config.fj_hi - 1,
            "z_lo": config.fk_lo,
            "z_hi": config.fk_hi - 1,
        }[face]
    return {
        "x_lo": 0,
        "x_hi": config.nx_f - 1,
        "y_lo": 0,
        "y_hi": config.ny_f - 1,
        "z_lo": 0,
        "z_hi": config.nz_f - 1,
    }[face]


def _full_axis_slice(config: SubgridConfig3D, axis: int, *, grid: str):
    if grid == "coarse":
        if axis == 0:
            return slice(config.fi_lo, config.fi_hi)
        if axis == 1:
            return slice(config.fj_lo, config.fj_hi)
        return slice(config.fk_lo, config.fk_hi)
    if axis == 0:
        return slice(0, config.nx_f)
    if axis == 1:
        return slice(0, config.ny_f)
    return slice(0, config.nz_f)


def _face_slice(config: SubgridConfig3D, face: str, *, grid: str) -> tuple:
    axis = FACE_ORIENTATIONS[face].normal_axis
    parts = []
    for ax in range(3):
        if ax == axis:
            parts.append(_fixed_face_index(config, face, grid=grid))
        else:
            parts.append(_full_axis_slice(config, ax, grid=grid))
    return tuple(parts)


def _get_component(fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], name: str) -> jnp.ndarray:
    return fields[_COMPONENT_INDEX[name]]


def _set_component(
    fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    name: str,
    value: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    out = list(fields)
    out[_COMPONENT_INDEX[name]] = value
    return tuple(out)


def extract_tangential_e_face(
    fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
    face: str,
    *,
    grid: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract tangential E traces for one oriented face."""

    sl = _face_slice(config, face, grid=grid)
    comps = FACE_ORIENTATIONS[face].tangential_e_components
    return tuple(_get_component(fields, comp)[sl] for comp in comps)


def extract_tangential_h_face(
    fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
    face: str,
    *,
    grid: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract tangential H traces for one oriented face."""

    sl = _face_slice(config, face, grid=grid)
    comps = FACE_ORIENTATIONS[face].tangential_h_components
    return tuple(_get_component(fields, comp)[sl] for comp in comps)


def scatter_tangential_e_face(
    fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tangential: tuple[jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
    face: str,
    *,
    grid: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scatter tangential E traces back to one oriented face."""

    sl = _face_slice(config, face, grid=grid)
    comps = FACE_ORIENTATIONS[face].tangential_e_components
    out = fields
    for comp, arr in zip(comps, tangential):
        field_arr = _get_component(out, comp)
        out = _set_component(out, comp, field_arr.at[sl].set(arr))
    return out


def scatter_tangential_h_face(
    fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tangential: tuple[jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
    face: str,
    *,
    grid: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scatter tangential H traces back to one oriented face."""

    sl = _face_slice(config, face, grid=grid)
    comps = FACE_ORIENTATIONS[face].tangential_h_components
    out = fields
    for comp, arr in zip(comps, tangential):
        field_arr = _get_component(out, comp)
        out = _set_component(out, comp, field_arr.at[sl].set(arr))
    return out


def _face_interior_masks(coarse_shape: tuple[int, int], ratio: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    coarse_mask = np.zeros(coarse_shape, dtype=np.float32)
    if coarse_shape[0] > 2 and coarse_shape[1] > 2:
        coarse_mask[1:-1, 1:-1] = 1.0

    fine_shape = (coarse_shape[0] * ratio, coarse_shape[1] * ratio)
    fine_mask = np.zeros(fine_shape, dtype=np.float32)
    if coarse_shape[0] > 2 and coarse_shape[1] > 2:
        fine_mask[ratio:-ratio, ratio:-ratio] = 1.0
    return jnp.asarray(coarse_mask), jnp.asarray(fine_mask)


def _apply_sat_pair_face(
    coarse_face: jnp.ndarray,
    fine_face: jnp.ndarray,
    ops: ZFaceOps,
    alpha_c: float,
    alpha_f: float,
    coarse_mask: jnp.ndarray,
    fine_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    coarse_mismatch = restrict_face(fine_face, ops) - coarse_face
    fine_mismatch = prolong_face(coarse_face, ops) - fine_face
    return (
        coarse_face + alpha_c * coarse_mismatch * coarse_mask,
        fine_face + alpha_f * fine_mismatch * fine_mask,
    )


def _edge_slice(config: SubgridConfig3D, edge: str, *, grid: str) -> tuple:
    meta = EDGE_ORIENTATIONS[edge]
    axis = meta.varying_axis
    parts = [None, None, None]
    for ax in range(3):
        if ax == axis:
            parts[ax] = _full_axis_slice(config, ax, grid=grid)
        else:
            face = meta.fixed_faces[0] if FACE_ORIENTATIONS[meta.fixed_faces[0]].normal_axis == ax else meta.fixed_faces[1]
            parts[ax] = _fixed_face_index(config, face, grid=grid)
    return tuple(parts)


def _extract_edge_component(
    fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
    edge: str,
    *,
    grid: str,
    field_type: str,
) -> jnp.ndarray:
    meta = EDGE_ORIENTATIONS[edge]
    comp = meta.e_component if field_type == "e" else meta.h_component
    return _get_component(fields, comp)[_edge_slice(config, edge, grid=grid)]


def _scatter_edge_component(
    fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    values: jnp.ndarray,
    config: SubgridConfig3D,
    edge: str,
    *,
    grid: str,
    field_type: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    meta = EDGE_ORIENTATIONS[edge]
    comp = meta.e_component if field_type == "e" else meta.h_component
    arr = _get_component(fields, comp)
    return _set_component(fields, comp, arr.at[_edge_slice(config, edge, grid=grid)].set(values))


def _edge_interior_masks(coarse_size: int, ratio: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    coarse_mask = np.zeros((coarse_size,), dtype=np.float32)
    if coarse_size > 2:
        coarse_mask[1:-1] = 1.0
    fine_mask = np.zeros((coarse_size * ratio,), dtype=np.float32)
    if coarse_size > 2:
        fine_mask[ratio:-ratio] = 1.0
    return jnp.asarray(coarse_mask), jnp.asarray(fine_mask)


def _apply_sat_pair_edge(
    coarse_line: jnp.ndarray,
    fine_line: jnp.ndarray,
    ops: EdgeOps,
    alpha_c: float,
    alpha_f: float,
    coarse_mask: jnp.ndarray,
    fine_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    coarse_mismatch = restrict_edge(fine_line, ops) - coarse_line
    fine_mismatch = prolong_edge(coarse_line, ops) - fine_line
    return (
        coarse_line + alpha_c * coarse_mismatch * coarse_mask,
        fine_line + alpha_f * fine_mismatch * fine_mask,
    )


def _corner_index(config: SubgridConfig3D, corner: str, *, grid: str) -> tuple[int, int, int]:
    meta = CORNER_ORIENTATIONS[corner]
    return tuple(_fixed_face_index(config, face, grid=grid) for face in meta.fixed_faces)


def _apply_sat_pair_scalar(
    coarse_value: jnp.ndarray,
    fine_value: jnp.ndarray,
    alpha_c: float,
    alpha_f: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return (
        coarse_value + alpha_c * (fine_value - coarse_value),
        fine_value + alpha_f * (coarse_value - fine_value),
    )


def _zero_coarse_overlap_interior(
    fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Suppress strict coarse-grid interior state inside the fine box."""
    fi, fj, fk = config.fi_lo, config.fj_lo, config.fk_lo
    ni, nj, nk = _region_shape(config)
    if ni <= 2 or nj <= 2 or nk <= 2:
        return fields
    interior = (
        slice(fi + 1, fi + ni - 1),
        slice(fj + 1, fj + nj - 1),
        slice(fk + 1, fk + nk - 1),
    )
    return tuple(field.at[interior].set(0.0) for field in fields)


def _active_edges(config: SubgridConfig3D) -> tuple[str, ...]:
    faces = set(_active_faces(config))
    return tuple(
        edge for edge, meta in EDGE_ORIENTATIONS.items()
        if meta.fixed_faces[0] in faces and meta.fixed_faces[1] in faces
    )


def _active_corners(config: SubgridConfig3D) -> tuple[str, ...]:
    faces = set(_active_faces(config))
    return tuple(
        corner for corner, meta in CORNER_ORIENTATIONS.items()
        if all(face in faces for face in meta.fixed_faces)
    )


def apply_sat_h_interfaces(
    coarse_fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    fine_fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Apply SAT coupling to tangential H traces on active faces/edges/corners."""

    alpha_c, alpha_f = sat_penalty_coefficients(config.ratio, config.tau)
    coarse = coarse_fields
    fine = fine_fields
    for face in _active_faces(config):
        ops = _get_face_ops(config, face)
        coarse_mask, fine_mask = _face_interior_masks(ops.coarse_shape, config.ratio)
        hx_c_face, hy_c_face = extract_tangential_h_face(coarse, config, face, grid="coarse")
        hx_f_face, hy_f_face = extract_tangential_h_face(fine, config, face, grid="fine")
        hx_c_face, hx_f_face = _apply_sat_pair_face(
            hx_c_face, hx_f_face, ops, alpha_c, alpha_f, coarse_mask, fine_mask
        )
        hy_c_face, hy_f_face = _apply_sat_pair_face(
            hy_c_face, hy_f_face, ops, alpha_c, alpha_f, coarse_mask, fine_mask
        )
        coarse = scatter_tangential_h_face(coarse, (hx_c_face, hy_c_face), config, face, grid="coarse")
        fine = scatter_tangential_h_face(fine, (hx_f_face, hy_f_face), config, face, grid="fine")

    for edge in _active_edges(config):
        ops = _get_edge_ops(config, edge)
        coarse_mask, fine_mask = _edge_interior_masks(ops.coarse_size, config.ratio)
        hc = _extract_edge_component(coarse, config, edge, grid="coarse", field_type="h")
        hf = _extract_edge_component(fine, config, edge, grid="fine", field_type="h")
        hc, hf = _apply_sat_pair_edge(hc, hf, ops, alpha_c, alpha_f, coarse_mask, fine_mask)
        coarse = _scatter_edge_component(coarse, hc, config, edge, grid="coarse", field_type="h")
        fine = _scatter_edge_component(fine, hf, config, edge, grid="fine", field_type="h")

    for corner in _active_corners(config):
        c_idx = _corner_index(config, corner, grid="coarse")
        f_idx = _corner_index(config, corner, grid="fine")
        for comp in ("hx", "hy", "hz"):
            c_arr = _get_component(coarse, comp)
            f_arr = _get_component(fine, comp)
            c_val, f_val = _apply_sat_pair_scalar(c_arr[c_idx], f_arr[f_idx], alpha_c, alpha_f)
            coarse = _set_component(coarse, comp, c_arr.at[c_idx].set(c_val))
            fine = _set_component(fine, comp, f_arr.at[f_idx].set(f_val))
    return coarse, fine


def apply_sat_e_interfaces(
    coarse_fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    fine_fields: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    config: SubgridConfig3D,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Apply SAT coupling to tangential E traces on active faces/edges/corners."""

    alpha_c, alpha_f = sat_penalty_coefficients(config.ratio, config.tau)
    coarse = coarse_fields
    fine = fine_fields
    for face in _active_faces(config):
        ops = _get_face_ops(config, face)
        coarse_mask, fine_mask = _face_interior_masks(ops.coarse_shape, config.ratio)
        ex_c_face, ey_c_face = extract_tangential_e_face(coarse, config, face, grid="coarse")
        ex_f_face, ey_f_face = extract_tangential_e_face(fine, config, face, grid="fine")
        ex_c_face, ex_f_face = _apply_sat_pair_face(
            ex_c_face, ex_f_face, ops, alpha_c, alpha_f, coarse_mask, fine_mask
        )
        ey_c_face, ey_f_face = _apply_sat_pair_face(
            ey_c_face, ey_f_face, ops, alpha_c, alpha_f, coarse_mask, fine_mask
        )
        coarse = scatter_tangential_e_face(coarse, (ex_c_face, ey_c_face), config, face, grid="coarse")
        fine = scatter_tangential_e_face(fine, (ex_f_face, ey_f_face), config, face, grid="fine")

    for edge in _active_edges(config):
        ops = _get_edge_ops(config, edge)
        coarse_mask, fine_mask = _edge_interior_masks(ops.coarse_size, config.ratio)
        ec = _extract_edge_component(coarse, config, edge, grid="coarse", field_type="e")
        ef = _extract_edge_component(fine, config, edge, grid="fine", field_type="e")
        ec, ef = _apply_sat_pair_edge(ec, ef, ops, alpha_c, alpha_f, coarse_mask, fine_mask)
        coarse = _scatter_edge_component(coarse, ec, config, edge, grid="coarse", field_type="e")
        fine = _scatter_edge_component(fine, ef, config, edge, grid="fine", field_type="e")

    for corner in _active_corners(config):
        c_idx = _corner_index(config, corner, grid="coarse")
        f_idx = _corner_index(config, corner, grid="fine")
        for comp in ("ex", "ey", "ez"):
            c_arr = _get_component(coarse, comp)
            f_arr = _get_component(fine, comp)
            c_val, f_val = _apply_sat_pair_scalar(c_arr[c_idx], f_arr[f_idx], alpha_c, alpha_f)
            coarse = _set_component(coarse, comp, c_arr.at[c_idx].set(c_val))
            fine = _set_component(fine, comp, f_arr.at[f_idx].set(f_val))
    return coarse, fine


def _shared_node_coupling_h_3d(state_c_fields, state_f_fields, config):
    """Compatibility wrapper for legacy imports.

    The current low-level lane uses oriented-face H SAT coupling.
    """

    return apply_sat_h_interfaces(state_c_fields, state_f_fields, config)


def _shared_node_coupling_3d(state_c_fields, state_f_fields, config):
    """Compatibility wrapper for legacy imports.

    The current low-level lane uses oriented-face E SAT coupling.
    """

    return apply_sat_e_interfaces(state_c_fields, state_f_fields, config)


def step_subgrid_3d(
    state: SubgridState3D,
    config: SubgridConfig3D,
    *,
    mats_c=None,
    mats_f=None,
    pec_mask_c=None,
    pec_mask_f=None,
    outer_pec_faces: frozenset[str] = frozenset({"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"}),
    outer_pmc_faces: frozenset[str] = frozenset(),
    periodic: tuple[bool, bool, bool] = (False, False, False),
    fine_periodic: tuple[bool, bool, bool] = (False, False, False),
) -> SubgridState3D:
    """Advance the current all-PEC subgrid lane by one timestep."""

    hx_c, hy_c, hz_c = _update_h_only(
        state.ex_c,
        state.ey_c,
        state.ez_c,
        state.hx_c,
        state.hy_c,
        state.hz_c,
        config.dt,
        config.dx_c,
        mats=mats_c,
        periodic=periodic,
        pmc_faces=outer_pmc_faces,
    )
    hx_f, hy_f, hz_f = _update_h_only(
        state.ex_f,
        state.ey_f,
        state.ez_f,
        state.hx_f,
        state.hy_f,
        state.hz_f,
        config.dt,
        config.dx_f,
        mats=mats_f,
        periodic=fine_periodic,
        pmc_faces=frozenset(face for face in _touching_outer_faces(config) if face in outer_pmc_faces),
    )
    (hx_c, hy_c, hz_c), (hx_f, hy_f, hz_f) = apply_sat_h_interfaces(
        (hx_c, hy_c, hz_c),
        (hx_f, hy_f, hz_f),
        config,
    )
    hx_c, hy_c, hz_c = _zero_coarse_overlap_interior((hx_c, hy_c, hz_c), config)

    ex_c, ey_c, ez_c = _update_e_only(
        state.ex_c,
        state.ey_c,
        state.ez_c,
        hx_c,
        hy_c,
        hz_c,
        config.dt,
        config.dx_c,
        mats=mats_c,
        pec_mask=pec_mask_c,
        boundary_axes=None,
        boundary_faces=outer_pec_faces,
        periodic=periodic,
    )
    ex_f, ey_f, ez_f = _update_e_only(
        state.ex_f,
        state.ey_f,
        state.ez_f,
        hx_f,
        hy_f,
        hz_f,
        config.dt,
        config.dx_f,
        mats=mats_f,
        pec_mask=pec_mask_f,
        boundary_axes=None,
        boundary_faces=frozenset(face for face in _touching_outer_faces(config) if face in outer_pec_faces),
        periodic=fine_periodic,
    )
    (ex_c, ey_c, ez_c), (ex_f, ey_f, ez_f) = apply_sat_e_interfaces(
        (ex_c, ey_c, ez_c),
        (ex_f, ey_f, ez_f),
        config,
    )
    ex_c, ey_c, ez_c = _zero_coarse_overlap_interior((ex_c, ey_c, ez_c), config)

    return SubgridState3D(
        ex_c=ex_c,
        ey_c=ey_c,
        ez_c=ez_c,
        hx_c=hx_c,
        hy_c=hy_c,
        hz_c=hz_c,
        ex_f=ex_f,
        ey_f=ey_f,
        ez_f=ez_f,
        hx_f=hx_f,
        hy_f=hy_f,
        hz_f=hz_f,
        step=state.step + 1,
    )


def compute_energy_3d(state: SubgridState3D, config: SubgridConfig3D) -> float:
    """Total energy with coarse/fine overlap counted once."""

    dv_c = config.dx_c ** 3
    dv_f = config.dx_f ** 3
    fi, fj, fk = config.fi_lo, config.fj_lo, config.fk_lo
    ni, nj, nk = _region_shape(config)

    mask = jnp.ones(state.ex_c.shape, dtype=jnp.bool_)
    mask = mask.at[fi : fi + ni, fj : fj + nj, fk : fk + nk].set(False)

    e_c = (
        float(jnp.sum(jnp.where(mask, state.ex_c**2 + state.ey_c**2 + state.ez_c**2, 0.0)))
        * EPS_0
        * dv_c
        + float(jnp.sum(jnp.where(mask, state.hx_c**2 + state.hy_c**2 + state.hz_c**2, 0.0)))
        * MU_0
        * dv_c
    )
    e_f = (
        float(jnp.sum(state.ex_f**2 + state.ey_f**2 + state.ez_f**2)) * EPS_0 * dv_f
        + float(jnp.sum(state.hx_f**2 + state.hy_f**2 + state.hz_f**2)) * MU_0 * dv_f
    )
    return e_c + e_f
