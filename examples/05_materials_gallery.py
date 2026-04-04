"""Example 5: Materials Gallery

Demonstrates different material types in a PEC cavity:
  - Vacuum (eps_r=1.0)
  - FR4 substrate (eps_r=4.4)
  - Lossy dielectric (eps_r=2.5, sigma=0.5 S/m)
  - Debye dispersive material (water at 20 degrees C)

Each sub-simulation uses a small PEC cavity with a point source.
SnapshotSpec captures Ez at the z-mid plane every 50 steps.
The 2x2 figure shows the Ez field pattern at ~30% through the run
for each material, with physical mm coordinates on axes.

Save: examples/05_materials_gallery.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import Simulation, Box, GaussianPulse
from rfx.materials.debye import DebyePole
from rfx.simulation import SnapshotSpec

# ---------------------------------------------------------------------------
# Cavity parameters (shared across all sub-simulations)
# ---------------------------------------------------------------------------

FREQ_MAX = 10e9          # 10 GHz upper band
CAVITY_SIZE = 0.02       # 20 mm cube cavity
DX = 5e-4                # 0.5 mm cell size
N_STEPS = 300            # enough steps to show field pattern
SOURCE_POS = (CAVITY_SIZE * 0.35, CAVITY_SIZE * 0.35, CAVITY_SIZE * 0.50)
PROBE_POS  = (CAVITY_SIZE * 0.65, CAVITY_SIZE * 0.65, CAVITY_SIZE * 0.50)

SNAP_INTERVAL = 50       # capture every 50 steps -> 6 frames total

# ---------------------------------------------------------------------------
# Material definitions
# ---------------------------------------------------------------------------

MATERIALS = [
    {
        "label": "Vacuum\n(eps_r=1.0)",
        "kwargs": {},
        "lib_name": "vacuum",
    },
    {
        "label": "FR4\n(eps_r=4.4)",
        "kwargs": {"eps_r": 4.4, "sigma": 0.025},
        "lib_name": None,
    },
    {
        "label": "Lossy Dielectric\n(eps_r=2.5, sigma=0.5 S/m)",
        "kwargs": {"eps_r": 2.5, "sigma": 0.5},
        "lib_name": None,
    },
    {
        "label": "Water 20 C\n(Debye, eps_inf=4.9)",
        "kwargs": {
            "eps_r": 4.9,
            "sigma": 0.0,
            "debye_poles": [DebyePole(delta_eps=74.1, tau=8.3e-12)],
        },
        "lib_name": None,
    },
]


def run_cavity(mat_def, nz_mid):
    """Build and run a small PEC cavity with the given filling material.

    Uses SnapshotSpec to capture Ez slices at the z-mid plane every
    SNAP_INTERVAL steps.  Returns (result, snap_interval) so the caller
    can pick the desired frame.
    """
    sim = Simulation(
        freq_max=FREQ_MAX,
        domain=(CAVITY_SIZE, CAVITY_SIZE, CAVITY_SIZE),
        boundary="pec",
        dx=DX,
    )

    if mat_def["lib_name"] is not None:
        sim.add(
            Box((0, 0, 0), (CAVITY_SIZE, CAVITY_SIZE, CAVITY_SIZE)),
            material=mat_def["lib_name"],
        )
    else:
        sim.add_material("fill", **mat_def["kwargs"])
        sim.add(
            Box((0, 0, 0), (CAVITY_SIZE, CAVITY_SIZE, CAVITY_SIZE)),
            material="fill",
        )

    sim.add_source(SOURCE_POS, "ez",
                   waveform=GaussianPulse(f0=FREQ_MAX / 2, bandwidth=0.8))
    sim.add_probe(PROBE_POS, "ez")

    snap = SnapshotSpec(
        interval=SNAP_INTERVAL,
        components=("ez",),
        slice_axis=2,
        slice_index=nz_mid,
    )
    result = sim.run(n_steps=N_STEPS, snapshot=snap)
    return result


# ---------------------------------------------------------------------------
# Determine grid z-mid index once (shared geometry)
# ---------------------------------------------------------------------------

_sim_ref = Simulation(
    freq_max=FREQ_MAX,
    domain=(CAVITY_SIZE, CAVITY_SIZE, CAVITY_SIZE),
    boundary="pec",
    dx=DX,
)
_grid = _sim_ref._build_grid()
NZ_MID = _grid.nz // 2

# ---------------------------------------------------------------------------
# Run all four simulations
# ---------------------------------------------------------------------------

print("Running 4 material sub-simulations...")
results = []
for mat in MATERIALS:
    print(f"  {mat['label'].replace(chr(10), ' ')} ...")
    results.append(run_cavity(mat, NZ_MID))
print("Done.\n")

# ---------------------------------------------------------------------------
# Build 2x2 visualization grid
# ---------------------------------------------------------------------------

# Physical extent for imshow (mm)
extent_mm = [0, CAVITY_SIZE * 1e3, 0, CAVITY_SIZE * 1e3]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(
    "Ez Field Slice — Material Gallery (PEC Cavity, z-mid plane, ~30% through run)",
    fontsize=13,
)

for ax, result, mat in zip(axes.ravel(), results, MATERIALS):
    # result.snapshots["ez"] shape: (n_captures, nx, ny)
    snaps = np.asarray(result.snapshots["ez"])

    # Pick frame at ~30% through the simulation
    # n_captures = N_STEPS // SNAP_INTERVAL = 6 frames (steps 50,100,...,300)
    n_captures = snaps.shape[0]
    frame_idx = max(0, int(round(n_captures * 0.30)) - 1)
    slc = snaps[frame_idx]          # (nx, ny)
    slc_plot = slc.T                # imshow expects (row=ny, col=nx)

    vmax = float(np.max(np.abs(slc_plot))) or 1.0
    im = ax.imshow(
        slc_plot,
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
        extent=extent_mm,
    )
    fig.colorbar(im, ax=ax, label="Ez (V/m)", fraction=0.046, pad=0.04)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    step_shown = (frame_idx + 1) * SNAP_INTERVAL
    ax.set_title(f"{mat['label']}\n(step {step_shown}/{N_STEPS})", fontsize=10)

    # Mark source and probe positions
    ax.plot(SOURCE_POS[0] * 1e3, SOURCE_POS[1] * 1e3,
            "g^", markersize=8, label="source")
    ax.plot(PROBE_POS[0] * 1e3, PROBE_POS[1] * 1e3,
            "rs", markersize=8, label="probe")
    ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
out_path = "examples/05_materials_gallery.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
