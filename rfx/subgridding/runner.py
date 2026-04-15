"""Compatibility wrapper for the canonical JIT subgridded runner."""

from __future__ import annotations

from rfx.subgridding.jit_runner import run_subgridded_jit


def run_subgridded(
    grid_c,
    mats_c,
    grid_f,
    mats_f,
    subgrid_config,
    n_steps,
    *,
    pec_mask_c=None,
    pec_mask_f=None,
    sources_f=None,
    probe_indices_f=None,
    probe_components=None,
    cpml_axes="xyz",
):
    """Compatibility wrapper around the canonical single-stepper JIT path."""

    del grid_f, cpml_axes
    result = run_subgridded_jit(
        grid_c,
        mats_c,
        mats_f,
        subgrid_config,
        n_steps,
        pec_mask_c=pec_mask_c,
        pec_mask_f=pec_mask_f,
        sources_f=sources_f,
        probe_indices_f=probe_indices_f,
        probe_components=probe_components,
    )
    return {
        "state_c": result.state_c,
        "state_f": result.state_f,
        "time_series": result.time_series,
        "config": result.config,
        "dt": result.dt,
    }
