"""SBP-SAT subgridding JIT performance benchmark.

Compares three configurations on a PEC cavity:
  1. Uniform coarse grid  (dx = dx_coarse)
  2. Uniform fine grid    (dx = dx_coarse / ratio)
  3. Subgridded           (coarse + local z-refinement at ratio:1)

Reports: grid shape, total cells, Mcells/s, time per step, overhead ratio.

The subgridded approach should be faster than uniform-fine while
capturing fine-scale physics in the refinement region.
"""

import time
import numpy as np
import jax

from rfx import Simulation
from rfx.grid import C0

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FREQ_MAX = 5e9
DOMAIN = (0.06, 0.06, 0.06)  # 60 mm PEC cavity
N_STEPS = 500
RATIO = 2                     # subgrid refinement ratio
DX_COARSE = C0 / FREQ_MAX / 20  # lambda/20 at freq_max

# Source / probe at cavity centre
CENTER = tuple(d / 2 for d in DOMAIN)

# Refinement covers the lower half of z (arbitrary choice)
Z_REFINE = (0.0, DOMAIN[2] / 2)


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def benchmark(label: str, sim: Simulation, n_steps: int = N_STEPS):
    """Run simulation and measure throughput after JIT warmup."""

    # --- Warmup (triggers JIT compilation) ---
    print(f"  [{label}] JIT warmup (10 steps) ...")
    t_jit0 = time.perf_counter()
    warmup_result = sim.run(n_steps=10)
    warmup_result.time_series.block_until_ready()
    t_jit1 = time.perf_counter()
    jit_time = t_jit1 - t_jit0
    print(f"  [{label}] JIT warmup done in {jit_time:.2f}s")

    # --- Timed run ---
    print(f"  [{label}] Running {n_steps} steps ...")
    t0 = time.perf_counter()
    result = sim.run(n_steps=n_steps)
    result.time_series.block_until_ready()
    elapsed = time.perf_counter() - t0

    # Determine grid shape and cell count
    grid = sim._build_grid()
    shape = grid.shape
    total_cells = shape[0] * shape[1] * shape[2]

    # For subgridded sims the effective cell count includes fine region
    if sim._refinement is not None:
        ref = sim._refinement
        ratio = ref["ratio"]
        cpml = grid.cpml_layers
        fk_lo = max(int(round(Z_REFINE[0] / grid.dx)) + cpml, cpml)
        fk_hi = min(int(round(Z_REFINE[1] / grid.dx)) + cpml + 1, grid.nz - cpml)
        fi_lo, fi_hi = cpml, grid.nx - cpml
        fj_lo, fj_hi = cpml, grid.ny - cpml
        nx_f = (fi_hi - fi_lo) * ratio
        ny_f = (fj_hi - fj_lo) * ratio
        nz_f = (fk_hi - fk_lo) * ratio
        fine_cells = nx_f * ny_f * nz_f
        total_cells_eff = total_cells + fine_cells
        shape_info = f"coarse {shape} + fine ({nx_f},{ny_f},{nz_f})"
    else:
        total_cells_eff = total_cells
        fine_cells = 0
        shape_info = f"{shape}"

    mcells_per_sec = (total_cells_eff * n_steps) / elapsed / 1e6
    ms_per_step = elapsed / n_steps * 1000

    print(f"\n  {label}:")
    print(f"    Grid        : {shape_info}")
    print(f"    Total cells : {total_cells_eff:,}")
    print(f"    Time        : {elapsed:.2f}s ({ms_per_step:.2f} ms/step)")
    print(f"    Throughput  : {mcells_per_sec:.1f} Mcells/s")
    print(f"    JIT warmup  : {jit_time:.2f}s")

    return {
        "elapsed": elapsed,
        "mcells_per_sec": mcells_per_sec,
        "ms_per_step": ms_per_step,
        "total_cells": total_cells_eff,
        "jit_time": jit_time,
        "shape_info": shape_info,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    backend = jax.default_backend()
    devices = jax.devices()
    print(f"JAX backend : {backend}")
    print(f"Devices     : {devices}")
    print(f"dx_coarse   : {DX_COARSE*1e3:.3f} mm")
    print(f"dx_fine     : {DX_COARSE/RATIO*1e3:.3f} mm")
    print(f"Ratio       : {RATIO}")
    print(f"N_STEPS     : {N_STEPS}")
    print()

    # ---- 1. Uniform coarse ----
    print("=" * 60)
    print("1. Uniform coarse grid")
    print("=" * 60)
    sim_coarse = Simulation(
        freq_max=FREQ_MAX, domain=DOMAIN, boundary="pec", dx=DX_COARSE,
    )
    sim_coarse.add_source(CENTER, component="ez")
    sim_coarse.add_probe(CENTER, component="ez")
    r_coarse = benchmark("Uniform coarse", sim_coarse)

    # ---- 2. Uniform fine (dx = dx_coarse / ratio) ----
    print()
    print("=" * 60)
    print("2. Uniform fine grid (dx = dx_coarse / ratio)")
    print("=" * 60)
    dx_fine = DX_COARSE / RATIO
    sim_fine = Simulation(
        freq_max=FREQ_MAX, domain=DOMAIN, boundary="pec", dx=dx_fine,
    )
    sim_fine.add_source(CENTER, component="ez")
    sim_fine.add_probe(CENTER, component="ez")
    r_fine = benchmark("Uniform fine", sim_fine)

    # ---- 3. Subgridded (coarse + local z-refinement) ----
    print()
    print("=" * 60)
    print(f"3. Subgridded (coarse + {RATIO}:1 z-refinement)")
    print("=" * 60)
    sim_sub = Simulation(
        freq_max=FREQ_MAX, domain=DOMAIN, boundary="pec", dx=DX_COARSE,
    )
    sim_sub.add_source(CENTER, component="ez")
    sim_sub.add_probe(CENTER, component="ez")
    sim_sub.add_refinement(z_range=Z_REFINE, ratio=RATIO)
    r_sub = benchmark("Subgridded", sim_sub)

    # ---- Summary ----
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Config':<25s}  {'Cells':>12s}  {'Mcells/s':>10s}  {'ms/step':>9s}  {'JIT (s)':>8s}")
    print("-" * 70)
    for label, r in [("Uniform coarse", r_coarse),
                     ("Uniform fine", r_fine),
                     (f"Subgridded ({RATIO}:1)", r_sub)]:
        print(f"{label:<25s}  {r['total_cells']:>12,}  "
              f"{r['mcells_per_sec']:>10.1f}  "
              f"{r['ms_per_step']:>9.2f}  "
              f"{r['jit_time']:>8.2f}")

    overhead_vs_coarse = r_sub["elapsed"] / r_coarse["elapsed"]
    speedup_vs_fine = r_fine["elapsed"] / r_sub["elapsed"]
    cell_ratio = r_sub["total_cells"] / r_fine["total_cells"]

    print()
    print(f"Subgrid overhead vs coarse : {overhead_vs_coarse:.2f}x")
    print(f"Subgrid speedup vs fine    : {speedup_vs_fine:.2f}x")
    print(f"Subgrid cell ratio vs fine : {cell_ratio:.2f}x ({r_sub['total_cells']:,} vs {r_fine['total_cells']:,})")

    return {
        "overhead_vs_coarse": overhead_vs_coarse,
        "speedup_vs_fine": speedup_vs_fine,
        "coarse_mcells": r_coarse["mcells_per_sec"],
        "fine_mcells": r_fine["mcells_per_sec"],
        "sub_mcells": r_sub["mcells_per_sec"],
    }


if __name__ == "__main__":
    main()
