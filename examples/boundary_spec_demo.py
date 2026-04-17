"""BoundarySpec demo — v1.7.0 per-axis boundary API (T7).

This example shows three common boundary configurations expressed with
the new ``BoundarySpec`` / ``Boundary`` types, side-by-side with the
equivalent legacy triad for migration reference.

Run as::

    python examples/boundary_spec_demo.py

The three configurations each build a small Simulation, run 20 steps,
and print the resolved ``sim._boundary_spec`` for inspection. The old
API still works and is kept runnable for comparison.
"""

from __future__ import annotations

import numpy as np

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


# ---------------------------------------------------------------------------
# Pattern 1 — patch antenna on ground plane
# (z_lo = PEC ground, z_hi + xy = CPML open)
# ---------------------------------------------------------------------------

def patch_antenna_new_api():
    return Simulation(
        freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
        boundary=BoundarySpec(
            x="cpml",
            y="cpml",
            z=Boundary(lo="pec", hi="cpml"),
        ),
    )


def patch_antenna_legacy_api():
    # Equivalent to the above; pec_faces kwarg emits DeprecationWarning.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return Simulation(
            freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
            boundary="cpml",
            pec_faces={"z_lo"},
        )


# ---------------------------------------------------------------------------
# Pattern 2 — 2D periodic surface (metasurface / frequency-selective surface)
# ---------------------------------------------------------------------------

def metasurface_new_api():
    return Simulation(
        freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
        boundary=BoundarySpec(
            x="periodic",
            y="periodic",
            z="cpml",
        ),
    )


def metasurface_legacy_api():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sim = Simulation(
            freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
            boundary="cpml",
        )
        sim.set_periodic_axes("xy")
        return sim


# ---------------------------------------------------------------------------
# Pattern 3 — fully absorbing open box
# ---------------------------------------------------------------------------

def open_box_new_api():
    return Simulation(
        freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
        boundary=BoundarySpec.uniform("cpml"),
    )


def open_box_legacy_api():
    return Simulation(
        freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
        boundary="cpml",
    )


# ---------------------------------------------------------------------------
# Drive the three patterns and compare
# ---------------------------------------------------------------------------

def _equivalence(new_sim, legacy_sim):
    return (
        new_sim._boundary_spec == legacy_sim._boundary_spec
        and new_sim._pec_faces == legacy_sim._pec_faces
        and new_sim._periodic_axes == legacy_sim._periodic_axes
    )


if __name__ == "__main__":
    for label, new_fn, legacy_fn in [
        ("patch antenna", patch_antenna_new_api, patch_antenna_legacy_api),
        ("metasurface",   metasurface_new_api,   metasurface_legacy_api),
        ("open box",      open_box_new_api,      open_box_legacy_api),
    ]:
        new = new_fn()
        legacy = legacy_fn()
        print(f"[{label}] spec = {new._boundary_spec.to_dict()}")
        assert _equivalence(new, legacy), (
            f"{label}: new API and legacy API must produce the same "
            f"canonical BoundarySpec"
        )
    print("All three patterns round-trip identically between APIs.")
