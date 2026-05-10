#!/usr/bin/env python3
"""Build a generic comparator artifact for a narrow lumped/openEMS case.

This runs a small two-port PEC-cavity geometry with rfx single-cell lumped ports
and an analogous openEMS ``AddLumpedPort`` setup.  The comparison is intentionally
limited and magnitude-mode only: it is useful E4-enabling evidence for the
single-cell lumped-port infrastructure, not broad calibrated-port E5.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
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
from rfx.sources.sources import GaussianPulse, LumpedPort


@dataclass(frozen=True)
class LumpedOpenEMSCase:
    name: str = "coarse_two_port_pec_box"
    domain_m: tuple[float, float, float] = (0.030, 0.020, 0.015)
    dx_m: float = 5.0e-3
    freq_max_hz: float = 2.0e9
    freqs_hz: tuple[float, ...] = (0.8e9, 1.0e9, 1.2e9, 1.5e9, 1.8e9)
    port1_pos_m: tuple[float, float, float] = (0.010, 0.010, 0.005)
    port2_pos_m: tuple[float, float, float] = (0.020, 0.010, 0.005)
    port_impedance_ohm: float = 50.0
    pulse_f0_hz: float = 1.0e9
    pulse_bandwidth: float = 0.8
    num_periods: float = 60.0


DEFAULT_CASE = LumpedOpenEMSCase()


def _parse_float_tuple(value: str, *, expected_len: int, name: str) -> tuple[float, ...]:
    parts = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if len(parts) != expected_len:
        raise ValueError(f"{name} must have {expected_len} comma-separated floats, got {value!r}")
    return parts


def _case_from_args(args: argparse.Namespace) -> LumpedOpenEMSCase:
    return LumpedOpenEMSCase(
        name=args.case_name,
        domain_m=_parse_float_tuple(args.domain_m, expected_len=3, name="--domain-m"),
        dx_m=args.dx_m,
        freq_max_hz=args.freq_max_hz,
        freqs_hz=_parse_float_tuple(args.freqs_hz, expected_len=len(args.freqs_hz.split(",")), name="--freqs-hz"),
        port1_pos_m=_parse_float_tuple(args.port1_pos_m, expected_len=3, name="--port1-pos-m"),
        port2_pos_m=_parse_float_tuple(args.port2_pos_m, expected_len=3, name="--port2-pos-m"),
        port_impedance_ohm=args.port_impedance_ohm,
        pulse_f0_hz=args.pulse_f0_hz,
        pulse_bandwidth=args.pulse_bandwidth,
        num_periods=args.num_periods,
    )


def _ensure_openems_numpy_compat() -> None:
    # openEMS v0.0.35 still refers to deprecated NumPy aliases.
    for name, value in {"float": float, "int": int, "complex": complex}.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def _two_port_from_terms(
    s11: np.ndarray,
    s21: np.ndarray,
) -> np.ndarray:
    s11 = np.asarray(s11, dtype=np.complex128)
    s21 = np.asarray(s21, dtype=np.complex128)
    if s11.shape != s21.shape:
        raise ValueError(f"S11 and S21 shapes differ: {s11.shape} != {s21.shape}")
    s_params = np.zeros((2, 2, s11.size), dtype=np.complex128)
    s_params[0, 0, :] = s11
    s_params[1, 0, :] = s21
    # This shard drives port 1 only.  Mirror the measured terms so the artifact
    # has a valid 2-port shape, but compare only S11/S21.
    s_params[1, 1, :] = s11
    s_params[0, 1, :] = s21
    return s_params


def _run_rfx_case(case: LumpedOpenEMSCase) -> np.ndarray:
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
        LumpedPort(case.port1_pos_m, "ez", case.port_impedance_ohm, pulse),
        LumpedPort(case.port2_pos_m, "ez", case.port_impedance_ohm, pulse),
    ]
    freqs = jnp.asarray(case.freqs_hz, dtype=jnp.float32)
    return np.asarray(
        extract_s_matrix(
            grid,
            materials,
            ports,
            freqs,
            n_steps=grid.num_timesteps(case.num_periods),
            boundary="pec",
        ),
        dtype=np.complex128,
    )


def _run_openems_case(case: LumpedOpenEMSCase, sim_dir: Path) -> np.ndarray:
    _ensure_openems_numpy_compat()
    from CSXCAD.CSXCAD import ContinuousStructure
    from openEMS.openEMS import openEMS

    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    sim_dir.mkdir(parents=True, exist_ok=True)

    unit = 1.0e-3  # openEMS geometry coordinates in mm.
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

    dz_mm = case.dx_m / unit
    p1 = mm(case.port1_pos_m)
    p2 = mm(case.port2_pos_m)
    p1_stop = [p1[0], p1[1], p1[2] + dz_mm]
    p2_stop = [p2[0], p2[1], p2[2] + dz_mm]
    port1 = fdtd.AddLumpedPort(
        1,
        case.port_impedance_ohm,
        p1,
        p1_stop,
        "z",
        excite=1.0,
    )
    port2 = fdtd.AddLumpedPort(
        2,
        case.port_impedance_ohm,
        p2,
        p2_stop,
        "z",
        excite=0.0,
    )

    fdtd.Run(str(sim_dir), verbose=0, cleanup=True)
    freqs = np.asarray(case.freqs_hz, dtype=float)
    port1.CalcPort(str(sim_dir), freqs)
    port2.CalcPort(str(sim_dir), freqs)
    s11 = np.asarray(port1.uf_ref / port1.uf_inc, dtype=np.complex128)
    s21 = np.asarray(port2.uf_ref / port1.uf_inc, dtype=np.complex128)
    return _two_port_from_terms(s11, s21)


def build_lumped_openems_comparison_from_sparams(
    *,
    freqs_hz: np.ndarray,
    rfx_sparams: np.ndarray,
    openems_sparams: np.ndarray,
    output_dir: Path,
    case: LumpedOpenEMSCase = DEFAULT_CASE,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_npz = output_dir / "lumped_rfx_candidate_sparams.npz"
    reference_npz = output_dir / "lumped_openems_reference_sparams.npz"
    np.savez(candidate_npz, freqs_hz=np.asarray(freqs_hz, dtype=float), s_params=rfx_sparams)
    np.savez(
        reference_npz,
        freqs_hz=np.asarray(freqs_hz, dtype=float),
        s_params=openems_sparams,
    )

    payload = compare_sparameter_datasets(
        load_sparameter_dataset(candidate_npz),
        load_sparameter_dataset(reference_npz),
        terms="S11,S21",
        comparison_mode="magnitude",
        max_abs_tol=1.0,
        mean_abs_tol=1.0,
        max_mag_abs_tol=0.13,
        mean_mag_abs_tol=0.07,
    )
    payload["claim"] = "single-cell lumped-port S11/S21 magnitude comparison against openEMS"
    payload["claim_scope"] = (
        "narrow two-port PEC-cavity single-cell lumped-port comparison using "
        "rfx LumpedPort and openEMS AddLumpedPort; not broad calibrated lumped-port E5"
    )
    payload["case"] = asdict(case)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()

    output_json = output_dir / "lumped_openems_generic_sparameter_comparison.json"
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_lumped_openems_generic_comparison(
    output_dir: Path,
    case: LumpedOpenEMSCase = DEFAULT_CASE,
) -> dict[str, Any]:
    rfx_sparams = _run_rfx_case(case)
    openems_sparams = _run_openems_case(case, output_dir / "openems_lumped_tmp")
    return build_lumped_openems_comparison_from_sparams(
        freqs_hz=np.asarray(case.freqs_hz, dtype=float),
        rfx_sparams=rfx_sparams,
        openems_sparams=openems_sparams,
        output_dir=output_dir,
        case=case,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=".omx/physics-gate/latest-lumped-openems-generic-comparison",
    )
    parser.add_argument("--case-name", default=DEFAULT_CASE.name)
    parser.add_argument(
        "--domain-m",
        default=",".join(str(value) for value in DEFAULT_CASE.domain_m),
        help="Comma-separated domain dimensions in meters.",
    )
    parser.add_argument("--dx-m", type=float, default=DEFAULT_CASE.dx_m)
    parser.add_argument("--freq-max-hz", type=float, default=DEFAULT_CASE.freq_max_hz)
    parser.add_argument(
        "--freqs-hz",
        default=",".join(str(value) for value in DEFAULT_CASE.freqs_hz),
        help="Comma-separated comparison frequencies in Hz.",
    )
    parser.add_argument(
        "--port1-pos-m",
        default=",".join(str(value) for value in DEFAULT_CASE.port1_pos_m),
        help="Comma-separated port-1 position in meters.",
    )
    parser.add_argument(
        "--port2-pos-m",
        default=",".join(str(value) for value in DEFAULT_CASE.port2_pos_m),
        help="Comma-separated port-2 position in meters.",
    )
    parser.add_argument(
        "--port-impedance-ohm",
        type=float,
        default=DEFAULT_CASE.port_impedance_ohm,
    )
    parser.add_argument("--pulse-f0-hz", type=float, default=DEFAULT_CASE.pulse_f0_hz)
    parser.add_argument(
        "--pulse-bandwidth",
        type=float,
        default=DEFAULT_CASE.pulse_bandwidth,
    )
    parser.add_argument("--num-periods", type=float, default=DEFAULT_CASE.num_periods)
    args = parser.parse_args(argv)

    payload = build_lumped_openems_generic_comparison(
        _repo_path(args.output_dir),
        _case_from_args(args),
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
