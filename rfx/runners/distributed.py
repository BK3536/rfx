"""Multi-GPU distributed FDTD runner using jax.pmap.

Uses 1D slab decomposition along the x-axis with ghost cell
exchange via jax.lax.ppermute.  Phase 1 supports PEC boundary,
soft sources, and point probes.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from rfx.core.yee import (
    FDTDState,
    MaterialArrays,
    EPS_0,
    MU_0,
    _shift_fwd,
    _shift_bwd,
)
from rfx.boundaries.pec import apply_pec
from rfx.simulation import (
    SourceSpec,
    ProbeSpec,
    make_source,
    make_j_source,
    make_probe,
)


# ---------------------------------------------------------------------------
# Domain splitting / gathering
# ---------------------------------------------------------------------------

def split_array_x(arr, n_devices, ghost=1):
    """Split a 3D array into N slabs along x with ghost cells.

    Parameters
    ----------
    arr : ndarray, shape (nx, ny, nz)
    n_devices : int
    ghost : int
        Number of ghost cells on each side.

    Returns
    -------
    slabs : ndarray, shape (n_devices, nx_local + 2*ghost, ny, nz)
    """
    nx = arr.shape[0]
    nx_per = nx // n_devices
    slabs = []
    for i in range(n_devices):
        x_start = i * nx_per
        x_end = x_start + nx_per

        # Desired range including ghosts
        want_lo = x_start - ghost
        want_hi = x_end + ghost

        # Clamp to valid array range
        g_lo = max(0, want_lo)
        g_hi = min(nx, want_hi)

        slab_data = arr[g_lo:g_hi]

        # Pad where the desired range exceeds array bounds
        pad_lo = g_lo - want_lo   # > 0 when want_lo < 0
        pad_hi = want_hi - g_hi   # > 0 when want_hi > nx

        if pad_lo > 0 or pad_hi > 0:
            pad_widths = [(pad_lo, pad_hi)] + [(0, 0)] * (arr.ndim - 1)
            slab_data = jnp.pad(slab_data, pad_widths, mode='constant',
                                constant_values=0.0)

        slabs.append(slab_data)
    return jnp.stack(slabs)


def gather_array_x(slabs, ghost=1):
    """Gather slabs back into a single array, stripping ghost cells.

    Parameters
    ----------
    slabs : ndarray, shape (n_devices, nx_local + 2*ghost, ny, nz)
    ghost : int

    Returns
    -------
    arr : ndarray, shape (nx, ny, nz)
    """
    # Strip ghost cells from each slab and concatenate
    inner = slabs[:, ghost:-ghost, :, :]  # (n_devices, nx_per, ny, nz)
    n_devices = inner.shape[0]
    # Reshape: merge device and x dims
    nx_per = inner.shape[1]
    ny = inner.shape[2]
    nz = inner.shape[3]
    return inner.reshape(n_devices * nx_per, ny, nz)


def _split_state(state, n_devices, ghost=1):
    """Split an FDTDState into per-device slabs with ghost cells."""
    return FDTDState(
        ex=split_array_x(state.ex, n_devices, ghost),
        ey=split_array_x(state.ey, n_devices, ghost),
        ez=split_array_x(state.ez, n_devices, ghost),
        hx=split_array_x(state.hx, n_devices, ghost),
        hy=split_array_x(state.hy, n_devices, ghost),
        hz=split_array_x(state.hz, n_devices, ghost),
        step=jnp.broadcast_to(state.step, (n_devices,)),
    )


def _gather_state(state, ghost=1):
    """Gather per-device FDTDState slabs back into a single state."""
    return FDTDState(
        ex=gather_array_x(state.ex, ghost),
        ey=gather_array_x(state.ey, ghost),
        ez=gather_array_x(state.ez, ghost),
        hx=gather_array_x(state.hx, ghost),
        hy=gather_array_x(state.hy, ghost),
        hz=gather_array_x(state.hz, ghost),
        step=state.step[0],
    )


def _split_materials(materials, n_devices, ghost=1):
    """Split MaterialArrays into per-device slabs with ghost cells."""
    return MaterialArrays(
        eps_r=split_array_x(materials.eps_r, n_devices, ghost),
        sigma=split_array_x(materials.sigma, n_devices, ghost),
        mu_r=split_array_x(materials.mu_r, n_devices, ghost),
    )


# ---------------------------------------------------------------------------
# Ghost cell exchange
# ---------------------------------------------------------------------------

def _exchange_component(field, n_devices, axis_name="devices"):
    """Exchange ghost cells for a single field component.

    field : shape (nx_local + 2*ghost, ny, nz)  -- per-device via pmap
    The first and last x-planes are ghost cells.

    After exchange:
    - field[0, :, :] = right-neighbor's field[1, :, :]  (NO -- left neighbor's boundary)

    Convention:
    - ghost[0]  = left ghost  <- should contain left neighbor's rightmost real cell
    - ghost[-1] = right ghost <- should contain right neighbor's leftmost real cell
    - real cells: field[1:-1]
    """
    # The rightmost real cell of each device -> left ghost of right neighbor
    right_boundary = field[-2:-1, :, :]  # last real cell: index -2, shape (1, ny, nz)

    # The leftmost real cell of each device -> right ghost of left neighbor
    left_boundary = field[1:2, :, :]   # first real cell: index 1, shape (1, ny, nz)

    # ppermute: send from device i to device (i+1) % n  (right shift)
    # This sends each device's right_boundary to its right neighbor's left ghost
    perm_right = [(i, (i + 1) % n_devices) for i in range(n_devices)]
    left_ghost_recv = lax.ppermute(right_boundary, axis_name, perm=perm_right)

    # ppermute: send from device i to device (i-1) % n  (left shift)
    # This sends each device's left_boundary to its left neighbor's right ghost
    perm_left = [(i, (i - 1) % n_devices) for i in range(n_devices)]
    right_ghost_recv = lax.ppermute(left_boundary, axis_name, perm=perm_left)

    # Determine device index to mask boundary devices
    device_idx = lax.axis_index(axis_name)

    # For device 0: don't overwrite left ghost (physical boundary, keep zero)
    # For device n-1: don't overwrite right ghost (physical boundary, keep zero)
    left_ghost_val = jnp.where(device_idx > 0,
                               left_ghost_recv,
                               field[0:1, :, :])
    right_ghost_val = jnp.where(device_idx < n_devices - 1,
                                right_ghost_recv,
                                field[-1:, :, :])

    field = field.at[0:1, :, :].set(left_ghost_val)
    field = field.at[-1:, :, :].set(right_ghost_val)
    return field


def _exchange_h_ghosts(state, n_devices, axis_name="devices"):
    """Exchange ghost cells for all H components."""
    return state._replace(
        hx=_exchange_component(state.hx, n_devices, axis_name),
        hy=_exchange_component(state.hy, n_devices, axis_name),
        hz=_exchange_component(state.hz, n_devices, axis_name),
    )


def _exchange_e_ghosts(state, n_devices, axis_name="devices"):
    """Exchange ghost cells for all E components."""
    return state._replace(
        ex=_exchange_component(state.ex, n_devices, axis_name),
        ey=_exchange_component(state.ey, n_devices, axis_name),
        ez=_exchange_component(state.ez, n_devices, axis_name),
    )


# ---------------------------------------------------------------------------
# Local update functions (operate on per-device slab with ghosts)
# ---------------------------------------------------------------------------

def _update_h_local(state, materials, dt, dx):
    """H update on a local slab (including ghost cells).

    Identical to yee.update_h but without jit decorator and always
    non-periodic (ghost cells handle inter-device coupling).
    """
    ex, ey, ez = state.ex, state.ey, state.ez
    mu = materials.mu_r * MU_0

    curl_x = (
        (_shift_fwd(ez, 1) - ez) / dx
        - (_shift_fwd(ey, 2) - ey) / dx
    )
    curl_y = (
        (_shift_fwd(ex, 2) - ex) / dx
        - (_shift_fwd(ez, 0) - ez) / dx
    )
    curl_z = (
        (_shift_fwd(ey, 0) - ey) / dx
        - (_shift_fwd(ex, 1) - ex) / dx
    )

    hx = state.hx - (dt / mu) * curl_x
    hy = state.hy - (dt / mu) * curl_y
    hz = state.hz - (dt / mu) * curl_z

    return state._replace(hx=hx, hy=hy, hz=hz)


def _update_e_local(state, materials, dt, dx):
    """E update on a local slab (including ghost cells).

    Identical to yee.update_e but without jit decorator and always
    non-periodic.
    """
    hx, hy, hz = state.hx, state.hy, state.hz
    eps = materials.eps_r * EPS_0
    sigma = materials.sigma

    sigma_dt_2eps = sigma * dt / (2.0 * eps)
    ca = (1.0 - sigma_dt_2eps) / (1.0 + sigma_dt_2eps)
    cb = (dt / eps) / (1.0 + sigma_dt_2eps)

    curl_x = (
        (hz - _shift_bwd(hz, 1)) / dx
        - (hy - _shift_bwd(hy, 2)) / dx
    )
    curl_y = (
        (hx - _shift_bwd(hx, 2)) / dx
        - (hz - _shift_bwd(hz, 0)) / dx
    )
    curl_z = (
        (hy - _shift_bwd(hy, 0)) / dx
        - (hx - _shift_bwd(hx, 1)) / dx
    )

    ex = ca * state.ex + cb * curl_x
    ey = ca * state.ey + cb * curl_y
    ez = ca * state.ez + cb * curl_z

    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)


def _apply_pec_local(state, n_devices, nx_local_with_ghost, axis_name="devices"):
    """Apply PEC boundary on a local slab.

    - y and z PEC: always applied (all devices own the full y/z extent).
    - x PEC: only device 0 applies x-lo, only device N-1 applies x-hi.
      These operate on the first/last REAL cell (index ghost and
      nx_local+ghost-1), not the ghost cell itself.
    """
    device_idx = lax.axis_index(axis_name)
    ghost = 1

    ex, ey, ez = state.ex, state.ey, state.ez

    # Y-axis PEC (all devices)
    ex = ex.at[:, 0, :].set(0.0)
    ex = ex.at[:, -1, :].set(0.0)
    ez = ez.at[:, 0, :].set(0.0)
    ez = ez.at[:, -1, :].set(0.0)

    # Z-axis PEC (all devices)
    ex = ex.at[:, :, 0].set(0.0)
    ex = ex.at[:, :, -1].set(0.0)
    ey = ey.at[:, :, 0].set(0.0)
    ey = ey.at[:, :, -1].set(0.0)

    # X-axis PEC: only at physical boundaries
    # Device 0: x-lo PEC at real cell index ghost (=1)
    # We zero tangential components (ey, ez) at x=0 of the global domain,
    # which is index `ghost` (first real cell) of device 0.
    is_first = (device_idx == 0)
    ey_xlo = jnp.where(is_first, 0.0, ey[ghost, :, :])
    ez_xlo = jnp.where(is_first, 0.0, ez[ghost, :, :])
    ey = ey.at[ghost, :, :].set(ey_xlo)
    ez = ez.at[ghost, :, :].set(ez_xlo)

    # Device N-1: x-hi PEC at last real cell
    is_last = (device_idx == n_devices - 1)
    last_real = nx_local_with_ghost - 1 - ghost  # last real cell index
    ey_xhi = jnp.where(is_last, 0.0, ey[last_real, :, :])
    ez_xhi = jnp.where(is_last, 0.0, ez[last_real, :, :])
    ey = ey.at[last_real, :, :].set(ey_xhi)
    ez = ez.at[last_real, :, :].set(ez_xhi)

    return state._replace(ex=ex, ey=ey, ez=ez)


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

def run_distributed(sim, *, n_steps, devices=None, **kwargs):
    """Run FDTD simulation distributed across multiple devices.

    Uses 1D slab decomposition along the x-axis.  Phase 1 supports
    PEC boundary, soft sources, and point probes only.

    Parameters
    ----------
    sim : Simulation
        The Simulation instance.
    n_steps : int
        Number of timesteps.
    devices : list of jax.Device or None
        If None, use all available devices.

    Returns
    -------
    Result
    """
    from rfx.api import Result

    if devices is None:
        devices = jax.devices()
    n_devices = len(devices)

    # Build grid and materials (full domain)
    grid = sim._build_grid()
    base_materials, debye_spec, lorentz_spec, pec_mask, pec_shapes = (
        sim._assemble_materials(grid)
    )
    materials = base_materials

    nx, ny, nz = grid.shape
    if nx % n_devices != 0:
        raise ValueError(
            f"Grid nx={nx} is not evenly divisible by {n_devices} devices. "
            f"Adjust domain size or dx so that nx is a multiple of n_devices."
        )

    nx_per = nx // n_devices
    ghost = 1
    nx_local = nx_per + 2 * ghost

    # Build sources and probes on the full grid
    sources = []
    probes = []
    for pe in sim._ports:
        if pe.impedance == 0.0:
            if sim._boundary == "cpml":
                sources.append(make_j_source(grid, pe.position, pe.component,
                                             pe.waveform, n_steps, materials))
            else:
                sources.append(make_source(grid, pe.position, pe.component,
                                           pe.waveform, n_steps))
    for pe in sim._probes:
        probes.append(make_probe(grid, pe.position, pe.component))

    # Map source/probe global indices to (device_id, local_index)
    # Source mapping
    src_device_ids = []
    src_local_specs = []  # (local_i, j, k, component, waveform)
    for s in sources:
        dev_id = s.i // nx_per
        local_i = (s.i % nx_per) + ghost  # offset by ghost
        src_device_ids.append(dev_id)
        src_local_specs.append((local_i, s.j, s.k, s.component))

    # Probe mapping
    prb_device_ids = []
    prb_local_specs = []
    for p in probes:
        dev_id = p.i // nx_per
        local_i = (p.i % nx_per) + ghost
        prb_device_ids.append(dev_id)
        prb_local_specs.append((local_i, p.j, p.k, p.component))

    # Precompute source waveform matrix: (n_steps, n_sources)
    if sources:
        src_waveforms = jnp.stack([s.waveform for s in sources], axis=-1)
    else:
        src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    # Replicate waveforms to all devices: (n_devices, n_steps, n_sources)
    src_waveforms_rep = jnp.broadcast_to(
        src_waveforms[None, :, :],
        (n_devices, n_steps, src_waveforms.shape[-1]),
    )

    # Build per-device source mask: (n_devices, n_sources) bool
    # device d only injects source s if src_device_ids[s] == d
    src_device_mask = jnp.array(
        [[1.0 if src_device_ids[s] == d else 0.0
          for s in range(len(sources))]
         for d in range(n_devices)],
        dtype=jnp.float32,
    ) if sources else jnp.zeros((n_devices, 0), dtype=jnp.float32)

    # Build per-device probe mask similarly
    prb_device_mask = jnp.array(
        [[1.0 if prb_device_ids[p] == d else 0.0
          for p in range(len(probes))]
         for d in range(n_devices)],
        dtype=jnp.float32,
    ) if probes else jnp.zeros((n_devices, 0), dtype=jnp.float32)

    # Split domain into per-device slabs
    from rfx.core.yee import init_state
    full_state = init_state(grid.shape)
    state_slabs = _split_state(full_state, n_devices, ghost)
    materials_slabs = _split_materials(materials, n_devices, ghost)

    # Static metadata captured by closure
    n_src = len(sources)
    n_prb = len(probes)

    # Build the pmap'd step function
    @partial(jax.pmap, axis_name="devices", devices=devices)
    def distributed_scan(state_slab, materials_slab, step_indices,
                         src_waveforms_dev, src_mask, prb_mask):
        """Scan body over timesteps on one device."""

        def step_fn(carry, xs):
            _step_idx, src_vals = xs
            st = carry

            # 1. H update (local)
            st = _update_h_local(st, materials_slab, dt, dx)

            # 2. Exchange H ghost cells
            st = _exchange_h_ghosts(st, n_devices, "devices")

            # 3. E update (local)
            st = _update_e_local(st, materials_slab, dt, dx)

            # 4. Exchange E ghost cells
            st = _exchange_e_ghosts(st, n_devices, "devices")

            # 5. PEC boundaries
            st = _apply_pec_local(st, n_devices, nx_local, "devices")

            # 6. Source injection (only on owning device)
            for idx_s in range(n_src):
                li, lj, lk, lc = src_local_specs[idx_s]
                val = src_vals[idx_s] * src_mask[idx_s]
                field = getattr(st, lc)
                field = field.at[li, lj, lk].add(val)
                st = st._replace(**{lc: field})

            # 7. Probe sampling (only on owning device)
            samples = []
            for idx_p in range(n_prb):
                li, lj, lk, lc = prb_local_specs[idx_p]
                val = getattr(st, lc)[li, lj, lk] * prb_mask[idx_p]
                samples.append(val)

            if samples:
                probe_out = jnp.stack(samples)
            else:
                probe_out = jnp.zeros(0, dtype=jnp.float32)

            return st, probe_out

        xs = (step_indices, src_waveforms_dev)
        final_state, probe_ts = lax.scan(step_fn, state_slab, xs)
        return final_state, probe_ts

    dt = grid.dt
    dx = grid.dx

    # Prepare scan inputs: replicate step indices across devices
    step_indices = jnp.arange(n_steps, dtype=jnp.int32)
    step_indices_rep = jnp.broadcast_to(
        step_indices[None, :], (n_devices, n_steps)
    )

    # Run the distributed simulation
    final_state_slabs, probe_ts_all = distributed_scan(
        state_slabs,
        materials_slabs,
        step_indices_rep,
        src_waveforms_rep,
        src_device_mask,
        prb_device_mask,
    )

    # Gather final state
    final_state = _gather_state(final_state_slabs, ghost)

    # Aggregate probe time series: sum across devices
    # probe_ts_all: (n_devices, n_steps, n_probes)
    # Each probe is non-zero only on its owning device, so sum works
    if n_prb > 0:
        time_series = jnp.sum(probe_ts_all, axis=0)  # (n_steps, n_probes)
    else:
        time_series = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    return Result(
        state=final_state,
        time_series=time_series,
        s_params=None,
        freqs=None,
        dt=grid.dt,
        freq_range=(sim._freq_max / 10, sim._freq_max, sim._boundary),
    )
