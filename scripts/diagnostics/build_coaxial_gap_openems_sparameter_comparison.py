#!/usr/bin/env python3
"""Build a generic comparator artifact for the coaxial gap diagnostic.

This shard compares the rfx low-level ``add_coaxial_port`` material/source gap
diagnostic against an analogous openEMS PEC-box ``AddLumpedPort`` setup.  The
claim is intentionally narrow: it is E4-enabling evidence for the current
single-cell coaxial gap diagnostic path, not a calibrated coaxial TEM
reference-plane S-parameter validation and not a promoted high-level API.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from compare_sparameter_reference import (
    _repo_path,
    compare_sparameter_datasets,
    load_sparameter_dataset,
)
from generate_coaxial_gap_vi_dump import generate_coaxial_gap_vi_dump
from rfx import load_port_vi_dump_npz
from rfx.sources.coaxial_port import SMA_OUTER_RADIUS, SMA_PIN_RADIUS


DEFAULT_FREQS_HZ = np.asarray([3.0e9, 5.0e9, 7.0e9], dtype=float)


def _ensure_openems_numpy_compat() -> None:
    # openEMS v0.0.35 still refers to deprecated NumPy aliases.
    for name, value in {"float": float, "int": int, "complex": complex}.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def _oneport_from_s11(s11: np.ndarray) -> np.ndarray:
    s11 = np.asarray(s11, dtype=np.complex128)
    s_params = np.zeros((1, 1, s11.size), dtype=np.complex128)
    s_params[0, 0, :] = s11
    return s_params


def _run_openems_gap_reference(*, sim_dir: Path, n_steps: int) -> np.ndarray:
    """Run the analogous openEMS PEC-box gap model and return one-port S11."""
    _ensure_openems_numpy_compat()
    from CSXCAD.CSXCAD import ContinuousStructure
    from openEMS.openEMS import openEMS

    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    sim_dir.mkdir(parents=True, exist_ok=True)

    domain_m = (0.020, 0.020, 0.020)
    dx_m = 0.5e-3
    unit = 1.0e-3  # openEMS geometry coordinates in mm.

    fdtd = openEMS(NrTS=n_steps, EndCriteria=0)
    fdtd.SetGaussExcite(5.0e9, 4.0e9)
    fdtd.SetBoundaryCond(["PEC"] * 6)
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(unit)
    for axis, length_m in zip("xyz", domain_m):
        n_cells = int(round(length_m / dx_m))
        mesh.AddLine(axis, np.linspace(0.0, length_m / unit, n_cells + 1))

    center_mm = [10.0, 10.0]
    pin_z_min_mm = 10.0
    gap_z_mm = 15.0
    ptfe = csx.AddMaterial("ptfe", epsilon=2.1)
    ptfe.AddCylinder(
        [center_mm[0], center_mm[1], pin_z_min_mm],
        [center_mm[0], center_mm[1], gap_z_mm],
        radius=SMA_OUTER_RADIUS / unit,
        priority=1,
    )
    pin = csx.AddMetal("pin")
    pin.AddCylinder(
        [center_mm[0], center_mm[1], pin_z_min_mm],
        [center_mm[0], center_mm[1], gap_z_mm],
        radius=SMA_PIN_RADIUS / unit,
        priority=10,
    )

    # rfx measures the single z-directed gap cell at the port wall coordinate.
    # The openEMS lumped port is a one-cell edge extending outward from the same
    # coordinate.  This intentionally compares the gap diagnostic, not a TEM
    # line reference plane.
    port = fdtd.AddLumpedPort(
        1,
        50.0,
        [center_mm[0], center_mm[1], gap_z_mm],
        [center_mm[0], center_mm[1], gap_z_mm + dx_m / unit],
        "z",
        excite=1.0,
    )

    fdtd.Run(str(sim_dir), verbose=0, cleanup=True)
    port.CalcPort(str(sim_dir), DEFAULT_FREQS_HZ)
    return np.asarray(port.uf_ref / port.uf_inc, dtype=np.complex128)


def build_coaxial_gap_openems_comparison_from_s11(
    *,
    freqs_hz: np.ndarray,
    rfx_s11: np.ndarray,
    openems_s11: np.ndarray,
    output_dir: Path,
    n_steps: int = 200,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_npz = output_dir / "coaxial_gap_rfx_candidate_sparams.npz"
    reference_npz = output_dir / "coaxial_gap_openems_reference_sparams.npz"
    np.savez(
        candidate_npz,
        freqs_hz=np.asarray(freqs_hz, dtype=float),
        s_params=_oneport_from_s11(rfx_s11),
    )
    np.savez(
        reference_npz,
        freqs_hz=np.asarray(freqs_hz, dtype=float),
        s_params=_oneport_from_s11(openems_s11),
    )

    payload = compare_sparameter_datasets(
        load_sparameter_dataset(candidate_npz),
        load_sparameter_dataset(reference_npz),
        terms="S11",
        comparison_mode="magnitude",
        max_abs_tol=0.20,
        mean_abs_tol=0.15,
        max_mag_abs_tol=0.05,
        mean_mag_abs_tol=0.03,
    )
    payload["claim"] = "coaxial gap diagnostic S11 magnitude comparison against openEMS"
    payload["claim_scope"] = (
        "narrow single-cell coaxial gap diagnostic using rfx low-level "
        "coaxial material/source helpers and an analogous openEMS AddLumpedPort "
        "PEC-box setup; not calibrated coaxial TEM reference-plane E5"
    )
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["n_steps"] = int(n_steps)
    output_json = output_dir / "coaxial_gap_openems_generic_sparameter_comparison.json"
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_coaxial_gap_openems_generic_comparison(
    output_dir: Path,
    *,
    n_steps: int = 200,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_path = output_dir / "coaxial_gap_vi_dump.npz"
    summary_path = output_dir / "coaxial_gap_vi_summary.json"
    replay_path = output_dir / "coaxial_gap_vi_replay.json"
    generate_coaxial_gap_vi_dump(
        dump_path=dump_path,
        summary_json=summary_path,
        replay_json=replay_path,
        n_steps=n_steps,
    )
    dump = load_port_vi_dump_npz(dump_path)
    rfx_s11 = np.asarray(dump.production_smatrix[0, 0, :], dtype=np.complex128)
    openems_s11 = _run_openems_gap_reference(
        sim_dir=output_dir / "openems_coaxial_gap_tmp",
        n_steps=n_steps,
    )
    payload = build_coaxial_gap_openems_comparison_from_s11(
        freqs_hz=np.asarray(dump.freqs, dtype=float),
        rfx_s11=rfx_s11,
        openems_s11=openems_s11,
        output_dir=output_dir,
        n_steps=n_steps,
    )
    payload["rfx_gap_vi_dump"] = str(dump_path)
    payload["rfx_gap_vi_summary"] = str(summary_path)
    payload["rfx_gap_vi_replay"] = str(replay_path)
    output_json = output_dir / "coaxial_gap_openems_generic_sparameter_comparison.json"
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=".omx/physics-gate/latest-coaxial-gap-openems-generic-comparison",
    )
    parser.add_argument("--n-steps", type=int, default=200)
    args = parser.parse_args(argv)

    payload = build_coaxial_gap_openems_generic_comparison(
        _repo_path(args.output_dir),
        n_steps=args.n_steps,
    )
    print(
        "status={status} max_mag_abs_diff={max_mag_abs_diff:.6g}".format(
            status=payload["status"],
            max_mag_abs_diff=payload["summary"]["max_mag_abs_diff"],
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
