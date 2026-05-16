#!/usr/bin/env python3
"""Crossval 12: disjoint 3-D subgrid prototype diagnostic.

This example exercises the disjoint subgrid prototype, which is **research-only**
and **not long-time energy-stable**. It is not a public
``Simulation.add_refinement`` feature claim and it is not a Meep/openEMS
cross-solver pass. The short-window gate below is a research diagnostic, *not* a
"COMPLETE" or production-ready status: the disjoint topology still grows energy
over long integration windows and remains outside the validated production
envelope. Guarded one-sided production subgrid crossval evidence lives in
``scripts/subgrid_external_crossval_audit.py`` and related guarded-envelope
artifacts; this disjoint prototype remains research-only.

Run:
    python examples/crossval/12_subgrid_disjoint_prototype.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scripts.stage2_disjoint_full_physics_gate import run_gate


def main() -> int:
    result = run_gate(n_steps=100)
    print(
        "Crossval 12 disjoint subgrid prototype: short-window research gate "
        "PASS (research-only; NOT a COMPLETE/production status — the disjoint "
        "topology is not long-time energy-stable)"
    )
    print(f"  energy max ratio:   {result.max_energy_ratio:.6f}x")
    print(f"  face signal min:    {result.face_signal_min:.6e}")
    print(f"  coarse hole max:    {result.coarse_hole_max:.6e}")
    print(f"  AD gradient:        {result.ad_grad:.6e}")
    print(f"  cell savings:       {result.cell_savings_factor:.2f}x")
    print(f"  allocated savings:  {result.allocated_cell_savings_factor:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
