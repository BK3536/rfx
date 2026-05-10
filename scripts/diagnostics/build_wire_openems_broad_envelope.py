#!/usr/bin/env python3
"""Build a broad mesh/length envelope artifact for the wire-port openEMS lane.

This sweeps a small two-port PEC-cavity geometry across mesh density and wire
length, runs both rfx ``add_port(extent=...)`` (WirePort) and openEMS
``AddLumpedPort`` with matching start/stop coordinates, and aggregates the
per-case magnitude comparisons into a single envelope JSON suitable for the
``broad_e5_envelope_artifacts`` slot of ``port_external_reference_requirements``.

This artifact is **broad-E4 enabling** (multi-case external comparison) plus an
explicit mesh/length envelope. It is **not** a calibrated absolute wire-port E5
proof on its own; the per-case magnitude tolerance is intentionally loose
(``max_mag_abs_diff <= 0.20``) because the wire-port mid-cell convention has
known mismatch with openEMS's lumped-port multi-cell averaging. The envelope
documents the size of that mismatch across mesh; broad calibrated E5 still
requires resolving the wire-port absolute calibration convention.
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
from rfx.probes.probes import extract_s_matrix_wire
from rfx.sources.sources import GaussianPulse, WirePort


@dataclass(frozen=True)
class WireEnvelopeCase:
    name: str
    domain_m: tuple[float, float, float]
    dx_m: float
    freq_max_hz: float
    freqs_hz: tuple[float, ...]
    port1_start_m: tuple[float, float, float]
    port1_end_m: tuple[float, float, float]
    port2_start_m: tuple[float, float, float]
    port2_end_m: tuple[float, float, float]
    port_impedance_ohm: float
    pulse_f0_hz: float
    pulse_bandwidth: float
    num_periods: float
    max_mag_abs_tol: float
    mean_mag_abs_tol: float


# Small but meaningful envelope: vary mesh density and wire length.
# All wires are z-oriented PEC feeds in a PEC cavity; ports are uncoupled
# so |S11|, |S22| dominate and reciprocity acts as a free invariant check.
DEFAULT_CASES: tuple[WireEnvelopeCase, ...] = (
    WireEnvelopeCase(
        name="medium_short_wire_dx2mm",
        domain_m=(0.030, 0.020, 0.012),
        dx_m=2.0e-3,
        freq_max_hz=2.0e9,
        freqs_hz=(0.8e9, 1.0e9, 1.2e9, 1.5e9, 1.8e9),
        port1_start_m=(0.010, 0.010, 0.004),
        port1_end_m=(0.010, 0.010, 0.008),
        port2_start_m=(0.020, 0.010, 0.004),
        port2_end_m=(0.020, 0.010, 0.008),
        port_impedance_ohm=50.0,
        pulse_f0_hz=1.0e9,
        pulse_bandwidth=0.8,
        num_periods=40.0,
        max_mag_abs_tol=0.20,
        mean_mag_abs_tol=0.10,
    ),
    WireEnvelopeCase(
        name="medium_tall_wire_dx2mm",
        domain_m=(0.030, 0.020, 0.016),
        dx_m=2.0e-3,
        freq_max_hz=2.0e9,
        freqs_hz=(0.8e9, 1.0e9, 1.2e9, 1.5e9, 1.8e9),
        port1_start_m=(0.010, 0.010, 0.004),
        port1_end_m=(0.010, 0.010, 0.012),
        port2_start_m=(0.020, 0.010, 0.004),
        port2_end_m=(0.020, 0.010, 0.012),
        port_impedance_ohm=50.0,
        pulse_f0_hz=1.0e9,
        pulse_bandwidth=0.8,
        num_periods=40.0,
        max_mag_abs_tol=0.25,
        mean_mag_abs_tol=0.12,
    ),
    WireEnvelopeCase(
        name="fine_tall_wire_dx1mm",
        domain_m=(0.030, 0.020, 0.016),
        dx_m=1.0e-3,
        freq_max_hz=2.0e9,
        freqs_hz=(0.8e9, 1.0e9, 1.2e9, 1.5e9, 1.8e9),
        port1_start_m=(0.010, 0.010, 0.004),
        port1_end_m=(0.010, 0.010, 0.012),
        port2_start_m=(0.020, 0.010, 0.004),
        port2_end_m=(0.020, 0.010, 0.012),
        port_impedance_ohm=50.0,
        pulse_f0_hz=1.0e9,
        pulse_bandwidth=0.8,
        num_periods=40.0,
        max_mag_abs_tol=0.20,
        mean_mag_abs_tol=0.10,
    ),
)


def _ensure_openems_numpy_compat() -> None:
    for name, value in {"float": float, "int": int, "complex": complex}.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def _two_port_from_terms(s11: np.ndarray, s21: np.ndarray) -> np.ndarray:
    s11 = np.asarray(s11, dtype=np.complex128)
    s21 = np.asarray(s21, dtype=np.complex128)
    s_params = np.zeros((2, 2, s11.size), dtype=np.complex128)
    s_params[0, 0, :] = s11
    s_params[1, 0, :] = s21
    # Drive port 1 only; mirror the measured terms on the other column for a
    # valid 2-port shape, since the comparator only reads S11/S21.
    s_params[1, 1, :] = s11
    s_params[0, 1, :] = s21
    return s_params


def _run_rfx_case(case: WireEnvelopeCase) -> np.ndarray:
    grid = Grid(
        freq_max=case.freq_max_hz,
        domain=case.domain_m,
        dx=case.dx_m,
        cpml_layers=0,
    )
    materials = init_materials(grid.shape)
    pulse = GaussianPulse(
        f0=case.pulse_f0_hz,
        bandwidth=case.pulse_bandwidth,
        amplitude=1.0,
    )
    ports = [
        WirePort(case.port1_start_m, case.port1_end_m, "ez", case.port_impedance_ohm, pulse),
        WirePort(case.port2_start_m, case.port2_end_m, "ez", case.port_impedance_ohm, pulse),
    ]
    freqs = jnp.asarray(case.freqs_hz, dtype=jnp.float32)
    return np.asarray(
        extract_s_matrix_wire(
            grid,
            materials,
            ports,
            freqs,
            n_steps=grid.num_timesteps(case.num_periods),
            boundary="pec",
        ),
        dtype=np.complex128,
    )


def _run_openems_case(case: WireEnvelopeCase, sim_dir: Path) -> np.ndarray:
    _ensure_openems_numpy_compat()
    from CSXCAD.CSXCAD import ContinuousStructure
    from openEMS.openEMS import openEMS

    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    sim_dir.mkdir(parents=True, exist_ok=True)

    unit = 1.0e-3
    grid = Grid(
        freq_max=case.freq_max_hz,
        domain=case.domain_m,
        dx=case.dx_m,
        cpml_layers=0,
    )
    fdtd = openEMS(NrTS=grid.num_timesteps(case.num_periods), EndCriteria=0)
    fdtd.SetGaussExcite(case.pulse_f0_hz, case.pulse_bandwidth * case.pulse_f0_hz)
    fdtd.SetBoundaryCond(["PEC"] * 6)

    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(unit)
    for axis, length_m in zip("xyz", case.domain_m):
        n_cells = int(round(length_m / case.dx_m))
        mesh.AddLine(axis, np.linspace(0.0, length_m / unit, n_cells + 1))

    def mm(point_m: tuple[float, float, float]) -> list[float]:
        return [float(v / unit) for v in point_m]

    p1_lo = mm(case.port1_start_m)
    p1_hi = mm(case.port1_end_m)
    p2_lo = mm(case.port2_start_m)
    p2_hi = mm(case.port2_end_m)
    port1 = fdtd.AddLumpedPort(
        1, case.port_impedance_ohm, p1_lo, p1_hi, "z", excite=1.0,
    )
    port2 = fdtd.AddLumpedPort(
        2, case.port_impedance_ohm, p2_lo, p2_hi, "z", excite=0.0,
    )

    fdtd.Run(str(sim_dir), verbose=0, cleanup=True)
    freqs = np.asarray(case.freqs_hz, dtype=float)
    port1.CalcPort(str(sim_dir), freqs)
    port2.CalcPort(str(sim_dir), freqs)
    s11 = np.asarray(port1.uf_ref / port1.uf_inc, dtype=np.complex128)
    s21 = np.asarray(port2.uf_ref / port1.uf_inc, dtype=np.complex128)
    return _two_port_from_terms(s11, s21)


def _git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _build_case_payload(
    case: WireEnvelopeCase,
    rfx_sparams: np.ndarray,
    openems_sparams: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    case_dir = output_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    candidate_npz = case_dir / "rfx_wire_candidate_sparams.npz"
    reference_npz = case_dir / "openems_wire_reference_sparams.npz"
    np.savez(candidate_npz, freqs_hz=np.asarray(case.freqs_hz, dtype=float), s_params=rfx_sparams)
    np.savez(reference_npz, freqs_hz=np.asarray(case.freqs_hz, dtype=float), s_params=openems_sparams)

    payload = compare_sparameter_datasets(
        load_sparameter_dataset(candidate_npz),
        load_sparameter_dataset(reference_npz),
        terms="S11,S21",
        comparison_mode="magnitude",
        max_abs_tol=1.0,
        mean_abs_tol=1.0,
        max_mag_abs_tol=case.max_mag_abs_tol,
        mean_mag_abs_tol=case.mean_mag_abs_tol,
    )
    payload["case"] = asdict(case)
    payload["case_artifacts"] = {
        "rfx_npz": str(candidate_npz.relative_to(output_dir)),
        "openems_npz": str(reference_npz.relative_to(output_dir)),
    }
    return payload


def build_wire_openems_broad_envelope(
    output_dir: Path,
    cases: tuple[WireEnvelopeCase, ...] = DEFAULT_CASES,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_payloads: list[dict[str, Any]] = []
    overall_max = 0.0
    overall_mean = 0.0
    fail_count = 0

    for case in cases:
        rfx_sparams = _run_rfx_case(case)
        openems_sparams = _run_openems_case(case, output_dir / f"_openems_tmp_{case.name}")
        payload = _build_case_payload(case, rfx_sparams, openems_sparams, output_dir)
        case_payloads.append(payload)
        s = payload.get("summary", {})
        overall_max = max(overall_max, float(s.get("max_mag_abs_diff", 0.0)))
        overall_mean = max(overall_mean, float(s.get("mean_mag_abs_diff", 0.0)))
        if payload.get("status") != "passed":
            fail_count += 1

    envelope_status = "passed" if fail_count == 0 else "failed"
    envelope_payload: dict[str, Any] = {
        "schema": "rfx.port_external_envelope",
        "schema_version": 1,
        "claim": (
            "wire-port broad mesh/length envelope vs openEMS magnitude "
            "comparison across multiple PEC-cavity two-port cases"
        ),
        "claim_scope": (
            "uniform Yee PEC-cavity two-port wire-port (rfx add_port extent=...) "
            "compared against openEMS AddLumpedPort with matching start/stop "
            "coordinates; mesh dx in [1e-3, 2e-3] m, wire length in [4e-3, 8e-3] "
            "m, frequency in [0.8e9, 1.8e9] Hz; broad-E4 enabling envelope, not "
            "calibrated absolute wire-port E5"
        ),
        "evidence_level": "E4-broad-enabling",
        "status": envelope_status,
        "case_count": len(case_payloads),
        "fail_count": fail_count,
        "envelope_summary": {
            "max_mag_abs_diff_across_cases": overall_max,
            "max_mean_mag_abs_diff_across_cases": overall_mean,
            "mesh_range_m": [
                min(c.dx_m for c in cases),
                max(c.dx_m for c in cases),
            ],
            "wire_length_range_m": [
                min(abs(c.port1_end_m[2] - c.port1_start_m[2]) for c in cases),
                max(abs(c.port1_end_m[2] - c.port1_start_m[2]) for c in cases),
            ],
            "freq_range_hz": [
                min(min(c.freqs_hz) for c in cases),
                max(max(c.freqs_hz) for c in cases),
            ],
        },
        "cases": case_payloads,
        "commit_hash": _git_commit_short(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    output_json = output_dir / "wire_openems_broad_envelope.json"
    output_json.write_text(json.dumps(envelope_payload, indent=2, sort_keys=True) + "\n")
    return envelope_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=".omx/physics-gate/2026-05-10-m68-wire-openems-broad-envelope",
        help="Output directory (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--keep-openems-tmp",
        action="store_true",
        help="Keep openEMS scratch directories after each case (default: removed).",
    )
    args = parser.parse_args(argv)

    output_dir = _repo_path(args.output_dir)
    payload = build_wire_openems_broad_envelope(output_dir, DEFAULT_CASES)

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
