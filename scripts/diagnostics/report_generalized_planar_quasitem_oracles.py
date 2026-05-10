#!/usr/bin/env python3
"""Report quasi-TEM analytic oracles for planned generalized planar ports.

This is a planning/qualification artifact for future stripline, CPW, and
microstrip-launch style ports.  It computes closed-form or quasi-static
``Z0``/``beta`` references for representative geometries and checks basic
physical invariants.  It does not imply that a generalized planar-port API is
implemented, replay-validated, externally cross-validated, or E5-promoted.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from rfx.grid import C0


@dataclass(frozen=True)
class PlanarOracleCase:
    family: str
    geometry: dict[str, float]
    eps_r: float
    freqs_hz: tuple[float, ...] = (1.0e9, 5.0e9, 10.0e9)


DEFAULT_CASES = (
    PlanarOracleCase(
        family="microstrip",
        geometry={"width_m": 2.8e-3, "height_m": 1.6e-3},
        eps_r=4.2,
    ),
    PlanarOracleCase(
        family="symmetric_stripline",
        geometry={"width_m": 1.4e-3, "plate_spacing_m": 3.2e-3},
        eps_r=3.0,
    ),
    PlanarOracleCase(
        family="coplanar_waveguide",
        geometry={"strip_width_m": 2.0e-3, "slot_gap_m": 0.35e-3},
        eps_r=3.5,
    ),
    PlanarOracleCase(
        family="microstrip_to_coax_launch_proxy",
        geometry={"width_m": 2.4e-3, "height_m": 1.2e-3, "launch_pin_radius_m": 0.5e-3},
        eps_r=3.0,
    ),
)
MIDBAND_FREQ_HZ = 5.0e9


@dataclass(frozen=True)
class PlanarSweepSpec:
    family: str
    sweep_axis: str
    values: tuple[float, ...]
    fixed_geometry: dict[str, float]
    eps_r: float
    fixed_value_label: str


def _ellipk_agm(k: float) -> float:
    if not 0.0 <= k < 1.0:
        raise ValueError(f"elliptic modulus must be in [0, 1), got {k}")
    a = 1.0
    b = math.sqrt(1.0 - k * k)
    for _ in range(32):
        a_next = 0.5 * (a + b)
        b_next = math.sqrt(a * b)
        if abs(a_next - b_next) < 1e-15:
            a = a_next
            break
        a, b = a_next, b_next
    return math.pi / (2.0 * a)


def microstrip_quasitem(width_m: float, height_m: float, eps_r: float) -> tuple[float, float]:
    if width_m <= 0 or height_m <= 0 or eps_r <= 0:
        raise ValueError("microstrip dimensions and eps_r must be positive")
    u = width_m / height_m
    eps_eff = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 / math.sqrt(1.0 + 12.0 / u)
    if u <= 1.0:
        z0 = 60.0 / math.sqrt(eps_eff) * math.log(8.0 / u + 0.25 * u)
    else:
        z0 = 120.0 * math.pi / (math.sqrt(eps_eff) * (u + 1.393 + 0.667 * math.log(u + 1.444)))
    return z0, eps_eff


def symmetric_stripline_quasitem(width_m: float, plate_spacing_m: float, eps_r: float) -> tuple[float, float]:
    if width_m <= 0 or plate_spacing_m <= 0 or eps_r <= 0:
        raise ValueError("stripline dimensions and eps_r must be positive")
    u = width_m / plate_spacing_m
    z0 = 30.0 * math.pi / (math.sqrt(eps_r) * (u + 0.441))
    return z0, eps_r


def cpw_quasitem(strip_width_m: float, slot_gap_m: float, eps_r: float) -> tuple[float, float]:
    if strip_width_m <= 0 or slot_gap_m <= 0 or eps_r <= 0:
        raise ValueError("CPW dimensions and eps_r must be positive")
    k = strip_width_m / (strip_width_m + 2.0 * slot_gap_m)
    kp = math.sqrt(1.0 - k * k)
    eps_eff = (eps_r + 1.0) / 2.0
    z0 = 30.0 * math.pi / math.sqrt(eps_eff) * (_ellipk_agm(kp) / _ellipk_agm(k))
    return z0, eps_eff


def _evaluate_case(case: PlanarOracleCase) -> dict[str, Any]:
    if case.family in {"microstrip", "microstrip_to_coax_launch_proxy"}:
        z0, eps_eff = microstrip_quasitem(
            case.geometry["width_m"],
            case.geometry["height_m"],
            case.eps_r,
        )
    elif case.family == "symmetric_stripline":
        z0, eps_eff = symmetric_stripline_quasitem(
            case.geometry["width_m"],
            case.geometry["plate_spacing_m"],
            case.eps_r,
        )
    elif case.family == "coplanar_waveguide":
        z0, eps_eff = cpw_quasitem(
            case.geometry["strip_width_m"],
            case.geometry["slot_gap_m"],
            case.eps_r,
        )
    else:
        raise ValueError(f"unsupported family: {case.family}")

    freqs = np.asarray(case.freqs_hz, dtype=float)
    beta = 2.0 * np.pi * freqs * math.sqrt(eps_eff) / C0
    phase_velocity = 2.0 * np.pi * freqs / beta
    checks = {
        "finite_positive_z0": bool(np.isfinite(z0) and z0 > 0.0),
        "finite_positive_eps_eff": bool(np.isfinite(eps_eff) and eps_eff > 0.0),
        "z0_plausible_rf_range": bool(10.0 <= z0 <= 150.0),
        "beta_monotonic": bool(np.all(np.diff(beta) > 0.0)),
        "phase_velocity_matches_eps_eff": bool(
            np.max(np.abs(phase_velocity - C0 / math.sqrt(eps_eff))) < 1e-6
        ),
    }
    return {
        "case": asdict(case),
        "z0_ohm": float(z0),
        "eps_eff": float(eps_eff),
        "beta_rad_per_m": beta.tolist(),
        "phase_velocity_m_per_s": phase_velocity.tolist(),
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
    }


def _quasitem_for_family(
    family: str,
    geometry: dict[str, float],
    eps_r: float,
) -> tuple[float, float]:
    if family in {"microstrip", "microstrip_to_coax_launch_proxy"}:
        return microstrip_quasitem(geometry["width_m"], geometry["height_m"], eps_r)
    if family == "symmetric_stripline":
        return symmetric_stripline_quasitem(
            geometry["width_m"],
            geometry["plate_spacing_m"],
            eps_r,
        )
    if family == "coplanar_waveguide":
        return cpw_quasitem(
            geometry["strip_width_m"],
            geometry["slot_gap_m"],
            eps_r,
        )
    raise ValueError(f"unsupported family: {family}")


def _beta_at_midband(eps_eff: float) -> float:
    return float(2.0 * math.pi * MIDBAND_FREQ_HZ * math.sqrt(eps_eff) / C0)


def _evaluate_sweep(spec: PlanarSweepSpec) -> dict[str, Any]:
    rows = []
    for value in spec.values:
        geometry = dict(spec.fixed_geometry)
        eps_r = spec.eps_r
        if spec.sweep_axis == "eps_r":
            eps_r = value
        else:
            geometry[spec.sweep_axis] = value
        z0, eps_eff = _quasitem_for_family(spec.family, geometry, eps_r)
        rows.append(
            {
                "value": float(value),
                "geometry": geometry,
                "eps_r": float(eps_r),
                "z0_ohm": float(z0),
                "eps_eff": float(eps_eff),
                "beta_at_5ghz_rad_per_m": _beta_at_midband(eps_eff),
            }
        )

    z0_values = np.asarray([row["z0_ohm"] for row in rows], dtype=float)
    beta_values = np.asarray([row["beta_at_5ghz_rad_per_m"] for row in rows], dtype=float)
    finite_positive = bool(
        np.all(np.isfinite(z0_values))
        and np.all(z0_values > 0.0)
        and np.all(np.isfinite(beta_values))
        and np.all(beta_values > 0.0)
    )
    expected_z0_decreases = spec.sweep_axis in {
        "width_m",
        "strip_width_m",
        "eps_r",
    }
    z0_trend_ok = (
        bool(np.all(np.diff(z0_values) < 0.0))
        if expected_z0_decreases
        else True
    )
    beta_trend_ok = (
        bool(np.all(np.diff(beta_values) > 0.0))
        if spec.sweep_axis == "eps_r"
        else True
    )
    checks = {
        "finite_positive_z0_and_beta": finite_positive,
        "z0_expected_monotonic_decrease": z0_trend_ok,
        "beta_expected_monotonic_increase": beta_trend_ok,
    }
    return {
        "family": spec.family,
        "sweep_axis": spec.sweep_axis,
        "fixed_value_label": spec.fixed_value_label,
        "rows": rows,
        "z0_ohm_start": float(z0_values[0]),
        "z0_ohm_end": float(z0_values[-1]),
        "beta_at_5ghz_start": float(beta_values[0]),
        "beta_at_5ghz_end": float(beta_values[-1]),
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
    }


def evaluate_generalized_planar_quasitem_sweeps() -> dict[str, Any]:
    specs = (
        PlanarSweepSpec(
            family="microstrip",
            sweep_axis="width_m",
            values=(1.2e-3, 2.8e-3, 5.6e-3),
            fixed_geometry={"width_m": 2.8e-3, "height_m": 1.6e-3},
            eps_r=4.2,
            fixed_value_label="height_m=1.6e-3, eps_r=4.2",
        ),
        PlanarSweepSpec(
            family="microstrip",
            sweep_axis="eps_r",
            values=(2.2, 4.2, 6.0),
            fixed_geometry={"width_m": 2.8e-3, "height_m": 1.6e-3},
            eps_r=4.2,
            fixed_value_label="width_m=2.8e-3, height_m=1.6e-3",
        ),
        PlanarSweepSpec(
            family="symmetric_stripline",
            sweep_axis="width_m",
            values=(0.8e-3, 1.4e-3, 2.8e-3),
            fixed_geometry={"width_m": 1.4e-3, "plate_spacing_m": 3.2e-3},
            eps_r=3.0,
            fixed_value_label="plate_spacing_m=3.2e-3, eps_r=3.0",
        ),
        PlanarSweepSpec(
            family="symmetric_stripline",
            sweep_axis="eps_r",
            values=(2.2, 3.0, 5.0),
            fixed_geometry={"width_m": 1.4e-3, "plate_spacing_m": 3.2e-3},
            eps_r=3.0,
            fixed_value_label="width_m=1.4e-3, plate_spacing_m=3.2e-3",
        ),
        PlanarSweepSpec(
            family="coplanar_waveguide",
            sweep_axis="strip_width_m",
            values=(1.0e-3, 2.0e-3, 4.0e-3),
            fixed_geometry={"strip_width_m": 2.0e-3, "slot_gap_m": 0.35e-3},
            eps_r=3.5,
            fixed_value_label="slot_gap_m=0.35e-3, eps_r=3.5",
        ),
        PlanarSweepSpec(
            family="coplanar_waveguide",
            sweep_axis="eps_r",
            values=(2.2, 3.5, 6.0),
            fixed_geometry={"strip_width_m": 2.0e-3, "slot_gap_m": 0.35e-3},
            eps_r=3.5,
            fixed_value_label="strip_width_m=2.0e-3, slot_gap_m=0.35e-3",
        ),
        PlanarSweepSpec(
            family="microstrip_to_coax_launch_proxy",
            sweep_axis="width_m",
            values=(1.2e-3, 2.4e-3, 4.8e-3),
            fixed_geometry={
                "width_m": 2.4e-3,
                "height_m": 1.2e-3,
                "launch_pin_radius_m": 0.5e-3,
            },
            eps_r=3.0,
            fixed_value_label="height_m=1.2e-3, eps_r=3.0, launch_pin_radius_m=0.5e-3",
        ),
        PlanarSweepSpec(
            family="microstrip_to_coax_launch_proxy",
            sweep_axis="eps_r",
            values=(2.2, 3.0, 5.0),
            fixed_geometry={
                "width_m": 2.4e-3,
                "height_m": 1.2e-3,
                "launch_pin_radius_m": 0.5e-3,
            },
            eps_r=3.0,
            fixed_value_label="width_m=2.4e-3, height_m=1.2e-3, launch_pin_radius_m=0.5e-3",
        ),
    )
    rows = [_evaluate_sweep(spec) for spec in specs]
    status = "passed" if all(row["status"] == "passed" for row in rows) else "failed"
    return {
        "status": status,
        "evidence_level": "E2-planning-envelope-template",
        "claim_scope": (
            "generalized planar-port quasi-TEM parameter-sweep envelope template "
            "only; no implemented generalized planar-port API, no raw V/I replay, "
            "no external solver reference, and no E5 promotion"
        ),
        "sweep_count": len(rows),
        "row_count": sum(len(row["rows"]) for row in rows),
        "families": sorted({row["family"] for row in rows}),
        "rows": rows,
    }


def evaluate_generalized_planar_quasitem_oracles(
    cases: tuple[PlanarOracleCase, ...] = DEFAULT_CASES,
) -> dict[str, Any]:
    rows = [_evaluate_case(case) for case in cases]
    status = "passed" if all(row["status"] == "passed" for row in rows) else "failed"
    z0_values = [row["z0_ohm"] for row in rows]
    sweeps = evaluate_generalized_planar_quasitem_sweeps()
    status = "passed" if status == "passed" and sweeps["status"] == "passed" else "failed"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "evidence_level": "E2-planning",
        "claim_scope": (
            "generalized planar-port quasi-TEM analytic oracle/checklist only; "
            "no implemented generalized planar-port API, no raw V/I replay, no "
            "external solver reference, and no E5 promotion"
        ),
        "families": [case.family for case in cases],
        "case_count": len(rows),
        "sweep_count": sweeps["sweep_count"],
        "sweep_row_count": sweeps["row_count"],
        "min_z0_ohm": float(min(z0_values)),
        "max_z0_ohm": float(max(z0_values)),
        "rows": rows,
        "sweeps": sweeps,
        "completion_decision": (
            "Do not claim generalized planar-port support or broad E5 until an "
            "implementation, dump replay, external reference, and envelope exist."
        ),
    }


def _write_report(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "generalized_planar_quasitem_oracle_report.json"
    md_path = output_dir / "generalized_planar_quasitem_oracle_report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Generalized planar quasi-TEM oracle report",
        "",
        f"- status: `{payload['status']}`",
        f"- evidence_level: `{payload['evidence_level']}`",
        f"- case_count: `{payload['case_count']}`",
        f"- sweep_count: `{payload['sweep_count']}`",
        f"- sweep_row_count: `{payload['sweep_row_count']}`",
        f"- min_z0_ohm: `{payload['min_z0_ohm']:.6g}`",
        f"- max_z0_ohm: `{payload['max_z0_ohm']:.6g}`",
        "",
        payload["claim_scope"],
        "",
        "| Family | Z0 (ohm) | eps_eff | Status |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| `{family}` | `{z0:.6g}` | `{eps:.6g}` | `{status}` |".format(
                family=row["case"]["family"],
                z0=row["z0_ohm"],
                eps=row["eps_eff"],
                status=row["status"],
            )
        )
    lines.append(f"\nDecision: {payload['completion_decision']}\n")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = evaluate_generalized_planar_quasitem_oracles()
    _write_report(payload, args.output_dir)
    print(f"status={payload['status']} case_count={payload['case_count']}")
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
