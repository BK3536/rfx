#!/usr/bin/env python3
"""Build a broad freq-band/DFT envelope artifact for the coaxial gap lane.

This sweeps the rfx coaxial gap diagnostic and the analogous openEMS
``AddLumpedPort`` PEC-box reference across multiple frequency bands and DFT
window lengths, producing one envelope JSON for the
``broad_e5_envelope_artifacts`` slot of
``port_external_reference_requirements``.

The geometry is held fixed at the SMA / 0.5 mm Yee setup that the M35 narrow
comparator validated; this script extends that one-band point to three bands
(low / mid / high) plus one extra DFT-window length to demonstrate that the
gap diagnostic converges with both frequency and time-integration.

This artifact is **broad-E4 enabling** on the gap diagnostic lane only. It is
**not** a calibrated coaxial TEM reference-plane S-parameter promotion: the
M67 distributed TEM plane-source prototype is the path forward for the public
``add_coaxial_port`` API and is tracked separately.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import jax.numpy as jnp
import numpy as np

from compare_sparameter_reference import (
    _repo_path,
    compare_sparameter_datasets,
    load_sparameter_dataset,
)
from rfx.core.yee import init_materials
from rfx.grid import Grid
from rfx.probes.probes import extract_s_matrix
from rfx.sources.coaxial_port import (
    CoaxialPort,
    SMA_OUTER_RADIUS,
    SMA_PIN_RADIUS,
    setup_coaxial_port,
)
from rfx.sources.sources import GaussianPulse, LumpedPort


@dataclass(frozen=True)
class CoaxialGapEnvelopeCase:
    name: str
    freqs_hz: tuple[float, ...]
    n_steps: int
    pulse_f0_hz: float
    pulse_bandwidth: float
    max_mag_abs_tol: float
    mean_mag_abs_tol: float


# Geometry held fixed at the M35 baseline. Vary the freq band + DFT length to
# demonstrate the gap diagnostic converges across the SMA-supported range.
DEFAULT_CASES: tuple[CoaxialGapEnvelopeCase, ...] = (
    CoaxialGapEnvelopeCase(
        name="low_band_n200",
        freqs_hz=(1.0e9, 1.5e9, 2.0e9),
        n_steps=200,
        pulse_f0_hz=1.5e9,
        pulse_bandwidth=0.8,
        max_mag_abs_tol=0.10,
        mean_mag_abs_tol=0.06,
    ),
    CoaxialGapEnvelopeCase(
        name="mid_band_n200",
        freqs_hz=(3.0e9, 5.0e9, 7.0e9),
        n_steps=200,
        pulse_f0_hz=5.0e9,
        pulse_bandwidth=0.8,
        max_mag_abs_tol=0.10,
        mean_mag_abs_tol=0.06,
    ),
    CoaxialGapEnvelopeCase(
        name="mid_band_n500",
        freqs_hz=(3.0e9, 5.0e9, 7.0e9),
        n_steps=500,
        pulse_f0_hz=5.0e9,
        pulse_bandwidth=0.8,
        max_mag_abs_tol=0.10,
        mean_mag_abs_tol=0.06,
    ),
)


_DOMAIN_M: tuple[float, float, float] = (0.020, 0.020, 0.020)
_DX_M: float = 0.5e-3


def _ensure_openems_numpy_compat() -> None:
    for name, value in {"float": float, "int": int, "complex": complex}.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def _oneport_from_s11(s11: np.ndarray) -> np.ndarray:
    s11 = np.asarray(s11, dtype=np.complex128)
    s_params = np.zeros((1, 1, s11.size), dtype=np.complex128)
    s_params[0, 0, :] = s11
    return s_params


def _git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _run_rfx_gap(case: CoaxialGapEnvelopeCase) -> np.ndarray:
    """Run the rfx coaxial gap diagnostic and return one-port S11."""
    grid = Grid(freq_max=10.0e9, domain=_DOMAIN_M, dx=_DX_M, cpml_layers=0)
    materials = init_materials(grid.shape)
    pulse = GaussianPulse(
        f0=case.pulse_f0_hz, bandwidth=case.pulse_bandwidth, amplitude=1.0
    )
    coax = CoaxialPort(
        position=(0.010, 0.010, 0.015),
        face="top",
        pin_length=5.0e-3,
        pin_radius=SMA_PIN_RADIUS,
        outer_radius=SMA_OUTER_RADIUS,
        impedance=50.0,
        excitation=pulse,
    )
    materials = setup_coaxial_port(grid, coax, materials)

    # Drive the gap with a single-cell lumped port co-located with the coax
    # gap; this is exactly the M35 gap-diagnostic configuration extended to
    # the case's freq band/DFT length.
    gap_port = LumpedPort(
        position=(0.010, 0.010, 0.015),
        component="ez",
        impedance=50.0,
        excitation=pulse,
    )
    freqs_arr = jnp.asarray(case.freqs_hz, dtype=jnp.float32)
    s = extract_s_matrix(
        grid,
        materials,
        [gap_port],
        freqs_arr,
        n_steps=int(case.n_steps),
        boundary="pec",
    )
    return np.asarray(s[0, 0, :], dtype=np.complex128)


def _run_openems_gap(case: CoaxialGapEnvelopeCase, sim_dir: Path) -> np.ndarray:
    _ensure_openems_numpy_compat()
    from CSXCAD.CSXCAD import ContinuousStructure
    from openEMS.openEMS import openEMS

    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    sim_dir.mkdir(parents=True, exist_ok=True)

    unit = 1.0e-3
    fdtd = openEMS(NrTS=case.n_steps, EndCriteria=0)
    fdtd.SetGaussExcite(case.pulse_f0_hz, case.pulse_bandwidth * case.pulse_f0_hz)
    fdtd.SetBoundaryCond(["PEC"] * 6)
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(unit)
    for axis, length_m in zip("xyz", _DOMAIN_M):
        n_cells = int(round(length_m / _DX_M))
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
    port = fdtd.AddLumpedPort(
        1,
        50.0,
        [center_mm[0], center_mm[1], gap_z_mm],
        [center_mm[0], center_mm[1], gap_z_mm + _DX_M / unit],
        "z",
        excite=1.0,
    )
    fdtd.Run(str(sim_dir), verbose=0, cleanup=True)
    port.CalcPort(str(sim_dir), np.asarray(case.freqs_hz, dtype=float))
    return np.asarray(port.uf_ref / port.uf_inc, dtype=np.complex128)


def _build_case_payload(
    case: CoaxialGapEnvelopeCase, output_dir: Path
) -> dict[str, Any]:
    rfx_s11 = _run_rfx_gap(case)
    oem_s11 = _run_openems_gap(case, output_dir / f"_openems_tmp_{case.name}")
    case_dir = output_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    rfx_npz = case_dir / "rfx_gap_candidate_sparams.npz"
    oem_npz = case_dir / "openems_gap_reference_sparams.npz"
    np.savez(rfx_npz, freqs_hz=np.asarray(case.freqs_hz, dtype=float), s_params=_oneport_from_s11(rfx_s11))
    np.savez(oem_npz, freqs_hz=np.asarray(case.freqs_hz, dtype=float), s_params=_oneport_from_s11(oem_s11))

    payload = compare_sparameter_datasets(
        load_sparameter_dataset(rfx_npz),
        load_sparameter_dataset(oem_npz),
        terms="S11",
        comparison_mode="magnitude",
        max_abs_tol=1.0,
        mean_abs_tol=1.0,
        max_mag_abs_tol=case.max_mag_abs_tol,
        mean_mag_abs_tol=case.mean_mag_abs_tol,
    )
    payload["case"] = asdict(case)
    payload["case_artifacts"] = {
        "rfx_npz": str(rfx_npz.relative_to(output_dir)),
        "openems_npz": str(oem_npz.relative_to(output_dir)),
    }
    return payload


def build_coaxial_gap_openems_broad_envelope(
    output_dir: Path,
    cases: tuple[CoaxialGapEnvelopeCase, ...] = DEFAULT_CASES,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_payloads: list[dict[str, Any]] = []
    overall_max = 0.0
    overall_mean = 0.0
    fail_count = 0

    for case in cases:
        payload = _build_case_payload(case, output_dir)
        case_payloads.append(payload)
        s = payload.get("summary", {})
        overall_max = max(overall_max, float(s.get("max_mag_abs_diff", 0.0)))
        overall_mean = max(
            overall_mean, float(s.get("mean_mag_abs_diff_max_over_terms", 0.0))
        )
        if payload.get("status") != "passed":
            fail_count += 1

    envelope_payload: dict[str, Any] = {
        "schema": "rfx.port_external_envelope",
        "schema_version": 1,
        "claim": (
            "coaxial gap diagnostic broad freq-band/DFT envelope vs openEMS"
        ),
        "claim_scope": (
            "uniform Yee SMA-geometry coaxial gap diagnostic (rfx low-level "
            "add_coaxial_port material/source helpers) vs openEMS PEC-box "
            "AddLumpedPort across three (freq band, DFT length) cases at the "
            "fixed M35 dx=0.5 mm baseline; broad-E4 enabling envelope on the "
            "gap diagnostic lane only, not a promoted add_coaxial_port "
            "S-parameter API or calibrated TEM reference-plane E5"
        ),
        "evidence_level": "E4-broad-enabling",
        "status": "passed" if fail_count == 0 else "failed",
        "case_count": len(case_payloads),
        "fail_count": fail_count,
        "envelope_summary": {
            "max_mag_abs_diff_across_cases": overall_max,
            "max_mean_mag_abs_diff_across_cases": overall_mean,
            "freq_range_hz": [
                float(min(min(c.freqs_hz) for c in cases)),
                float(max(max(c.freqs_hz) for c in cases)),
            ],
            "n_steps_grid": sorted({c.n_steps for c in cases}),
            "case_names": [c.name for c in cases],
            "geometry_dx_m": _DX_M,
            "geometry_domain_m": list(_DOMAIN_M),
        },
        "cases": case_payloads,
        "commit_hash": _git_commit_short(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    output_json = output_dir / "coaxial_gap_openems_broad_envelope.json"
    output_json.write_text(json.dumps(envelope_payload, indent=2, sort_keys=True) + "\n")
    return envelope_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=".omx/physics-gate/2026-05-10-m71-coaxial-gap-openems-broad-envelope",
    )
    parser.add_argument("--keep-openems-tmp", action="store_true")
    args = parser.parse_args(argv)

    output_dir = _repo_path(args.output_dir)
    payload = build_coaxial_gap_openems_broad_envelope(output_dir, DEFAULT_CASES)
    if not args.keep_openems_tmp:
        for case in DEFAULT_CASES:
            tmp = output_dir / f"_openems_tmp_{case.name}"
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)

    summary = payload.get("envelope_summary", {})
    print(
        "status={status} case_count={n} fail_count={f} "
        "max_mag_abs_diff_across_cases={m:.6g}".format(
            status=payload["status"],
            n=payload["case_count"],
            f=payload["fail_count"],
            m=summary.get("max_mag_abs_diff_across_cases", 0.0),
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
