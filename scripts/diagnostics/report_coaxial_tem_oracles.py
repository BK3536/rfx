#!/usr/bin/env python3
"""Report analytic TEM reference oracles for the coaxial-port helper.

This is a diagnostic E2-formula artifact. It validates the exposed closed-form
coaxial TEM helpers (Z0, L'/C', beta, and load reflection) for the default
SMA/PTFE geometry plus a small geometry/eps_r sweep. It does **not** promote
`add_coaxial_port(...)` to a high-level S-parameter API; no calibrated FDTD TEM
V/I extraction or external cross-solver coax evidence is claimed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rfx.core.yee import EPS_0, MU_0
from rfx.grid import C0
from rfx.sources.coaxial_port import (
    PTFE_EPS_R,
    SMA_OUTER_RADIUS,
    SMA_PIN_RADIUS,
    coaxial_load_reflection,
    coaxial_tem_capacitance_per_m,
    coaxial_tem_characteristic_impedance,
    coaxial_tem_inductance_per_m,
    coaxial_tem_phase_constant,
    coaxial_tem_reference_plane_s11,
    coaxial_tem_reference_plane_vi,
    coaxial_tem_reference_plane_vi_from_cartesian_plane,
)


FREQS_HZ = np.asarray([1.0e9, 2.0e9, 3.0e9, 4.0e9], dtype=np.float64)
GEOMETRY_SWEEP_CASES = (
    {
        "name": "sma_ptfe_default",
        "inner_radius_m": SMA_PIN_RADIUS,
        "outer_radius_m": SMA_OUTER_RADIUS,
        "eps_r": PTFE_EPS_R,
    },
    {
        "name": "air_50ohm_like",
        "inner_radius_m": 0.50e-3,
        "outer_radius_m": 1.15e-3,
        "eps_r": 1.0,
    },
    {
        "name": "compact_high_eps",
        "inner_radius_m": 0.35e-3,
        "outer_radius_m": 1.40e-3,
        "eps_r": 4.0,
    },
    {
        "name": "wide_low_eps",
        "inner_radius_m": 0.80e-3,
        "outer_radius_m": 3.20e-3,
        "eps_r": 1.5,
    },
)


def _evaluate_reference_plane_extractor(
    *,
    inner_radius: float,
    outer_radius: float,
    eps_r: float,
) -> dict:
    z0 = coaxial_tem_characteristic_impedance(inner_radius, outer_radius, eps_r)
    radial_positions = np.linspace(inner_radius, outer_radius, 4097)
    h_sample_radius = 0.5 * (inner_radius + outer_radius)
    phi = np.linspace(0.0, 2.0 * np.pi, 73, endpoint=False)
    gamma_cases = {
        "matched": 0.0 + 0.0j,
        "short_like": -0.75 + 0.05j,
        "open_like": 0.70 - 0.10j,
        "complex_load": -0.20 + 0.35j,
    }
    incident_voltage = {
        "matched": 1.0 + 0.0j,
        "short_like": 0.9 - 0.1j,
        "open_like": 0.8 + 0.2j,
        "complex_load": 0.7 - 0.3j,
    }
    rows = []
    max_voltage_rel_error = 0.0
    max_current_rel_error = 0.0
    max_s11_abs_diff = 0.0
    for name, gamma in gamma_cases.items():
        v_inc = incident_voltage[name]
        voltage = v_inc * (1.0 + gamma)
        current = v_inc * (1.0 - gamma) / z0
        e_radial = voltage / (radial_positions * np.log(outer_radius / inner_radius))
        h_phi = np.full(phi.shape, current / (2.0 * np.pi * h_sample_radius), dtype=np.complex128)
        extracted = coaxial_tem_reference_plane_vi(
            radial_positions,
            e_radial,
            h_phi,
            h_sample_radius_m=h_sample_radius,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            eps_r=eps_r,
        )
        extracted_s11 = complex(
            coaxial_tem_reference_plane_s11(extracted.voltage, extracted.current, z0)
        )
        voltage_error = abs(complex(extracted.voltage) - voltage)
        current_error = abs(complex(extracted.current) - current)
        voltage_rel_error = voltage_error / max(abs(voltage), 1e-30)
        current_rel_error = current_error / max(abs(current), 1e-30)
        s11_abs_diff = abs(extracted_s11 - gamma)
        max_voltage_rel_error = max(max_voltage_rel_error, float(voltage_rel_error))
        max_current_rel_error = max(max_current_rel_error, float(current_rel_error))
        max_s11_abs_diff = max(max_s11_abs_diff, float(s11_abs_diff))
        rows.append(
            {
                "name": name,
                "status": "passed" if s11_abs_diff <= 3e-7 else "failed",
                "expected_s11": {"re": float(gamma.real), "im": float(gamma.imag)},
                "extracted_s11": {
                    "re": float(extracted_s11.real),
                    "im": float(extracted_s11.imag),
                },
                "voltage_rel_error": float(voltage_rel_error),
                "current_rel_error": float(current_rel_error),
                "s11_abs_diff": float(s11_abs_diff),
            }
        )
    status = "passed" if all(row["status"] == "passed" for row in rows) else "failed"
    return {
        "status": status,
        "evidence_level": "E2-calibration-oracle",
        "claim_scope": (
            "synthetic coaxial TEM reference-plane V/I extraction oracle only; "
            "no FDTD field sampling and no external full-wave coax reference"
        ),
        "sample_count_radial": int(radial_positions.size),
        "sample_count_azimuthal": int(phi.size),
        "h_sample_radius_m": float(h_sample_radius),
        "max_voltage_rel_error": float(max_voltage_rel_error),
        "max_current_rel_error": float(max_current_rel_error),
        "max_s11_abs_diff": float(max_s11_abs_diff),
        "rows": rows,
    }


def _evaluate_cartesian_plane_adapter(
    *,
    inner_radius: float,
    outer_radius: float,
    eps_r: float,
) -> dict:
    z0 = coaxial_tem_characteristic_impedance(inner_radius, outer_radius, eps_r)
    span = 1.15 * outer_radius
    u = np.linspace(-span, span, 601)
    v = np.linspace(-span, span, 601)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    rr = np.hypot(uu, vv)
    cos_phi = np.divide(uu, rr, out=np.zeros_like(rr), where=rr > 0.0)
    sin_phi = np.divide(vv, rr, out=np.zeros_like(rr), where=rr > 0.0)
    nonzero_r = rr > 0.0
    gamma = np.asarray([0.0 + 0.0j, -0.20 + 0.35j, 0.40 - 0.15j])
    incident_voltage = np.asarray([1.0 + 0.0j, 0.7 - 0.15j, 0.6 + 0.1j])
    voltage = incident_voltage * (1.0 + gamma)
    current = incident_voltage * (1.0 - gamma) / z0
    e_radial = np.zeros((gamma.size,) + rr.shape, dtype=np.complex128)
    h_phi = np.zeros_like(e_radial)
    e_radial[:, nonzero_r] = voltage[:, None] / (
        rr[nonzero_r][None, :] * np.log(outer_radius / inner_radius)
    )
    h_phi[:, nonzero_r] = current[:, None] / (2.0 * np.pi * rr[nonzero_r][None, :])
    extracted = coaxial_tem_reference_plane_vi_from_cartesian_plane(
        u,
        v,
        e_radial * cos_phi,
        e_radial * sin_phi,
        -h_phi * sin_phi,
        h_phi * cos_phi,
        center_u_m=0.0,
        center_v_m=0.0,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        eps_r=eps_r,
        radial_positions_m=np.linspace(inner_radius, outer_radius, 257),
        h_sample_radius_m=0.5 * (inner_radius + outer_radius),
        azimuthal_angles_rad=np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False),
    )
    observed_s11 = np.asarray(
        coaxial_tem_reference_plane_s11(extracted.vi.voltage, extracted.vi.current, z0),
        dtype=np.complex128,
    )
    s11_abs_diff = np.abs(observed_s11 - gamma)
    voltage_rel_error = np.abs(extracted.vi.voltage - voltage) / np.maximum(np.abs(voltage), 1e-30)
    current_rel_error = np.abs(extracted.vi.current - current) / np.maximum(np.abs(current), 1e-30)
    rows = []
    for idx, expected in enumerate(gamma):
        rows.append(
            {
                "index": int(idx),
                "status": "passed" if s11_abs_diff[idx] <= 2e-3 else "failed",
                "expected_s11": {"re": float(expected.real), "im": float(expected.imag)},
                "extracted_s11": {
                    "re": float(observed_s11[idx].real),
                    "im": float(observed_s11[idx].imag),
                },
                "s11_abs_diff": float(s11_abs_diff[idx]),
                "voltage_rel_error": float(voltage_rel_error[idx]),
                "current_rel_error": float(current_rel_error[idx]),
            }
        )
    return {
        "status": "passed" if all(row["status"] == "passed" for row in rows) else "failed",
        "evidence_level": "E2-calibration-oracle",
        "claim_scope": (
            "synthetic Cartesian field-plane to coaxial TEM V/I adapter oracle "
            "only; no real FDTD plane dump and no external full-wave coax reference"
        ),
        "plane_shape": [int(u.size), int(v.size)],
        "radial_sample_count": 257,
        "azimuthal_sample_count": 64,
        "max_voltage_rel_error": float(np.max(voltage_rel_error)),
        "max_current_rel_error": float(np.max(current_rel_error)),
        "max_s11_abs_diff": float(np.max(s11_abs_diff)),
        "rows": rows,
    }


def _closed_form_z0(inner_radius: float, outer_radius: float, eps_r: float) -> float:
    return float(np.sqrt(MU_0 / (EPS_0 * eps_r)) * np.log(outer_radius / inner_radius) / (2.0 * np.pi))


def _evaluate_one_geometry(
    *,
    case_name: str,
    inner_radius: float = SMA_PIN_RADIUS,
    outer_radius: float = SMA_OUTER_RADIUS,
    eps_r: float = PTFE_EPS_R,
    freqs_hz: np.ndarray = FREQS_HZ,
    z0_expected_ohm: float | None = None,
    z0_tolerance_ohm: float | None = None,
) -> dict:
    c_per_m = coaxial_tem_capacitance_per_m(inner_radius, outer_radius, eps_r)
    l_per_m = coaxial_tem_inductance_per_m(inner_radius, outer_radius)
    z0 = coaxial_tem_characteristic_impedance(inner_radius, outer_radius, eps_r)
    z0_from_lc = float(np.sqrt(l_per_m / c_per_m))
    z0_closed_form = _closed_form_z0(inner_radius, outer_radius, eps_r)
    velocity_from_lc = float(1.0 / np.sqrt(l_per_m * c_per_m))
    velocity_expected = float(C0 / np.sqrt(eps_r))
    beta = np.asarray(coaxial_tem_phase_constant(freqs_hz, eps_r), dtype=np.float64)
    beta_expected = 2.0 * np.pi * freqs_hz / velocity_expected

    gamma_cases = {
        "matched": complex(coaxial_load_reflection(z0, z0)),
        "short": complex(coaxial_load_reflection(0.0, z0)),
        "open": complex(coaxial_load_reflection(np.inf, z0)),
        "resistor_25ohm": complex(coaxial_load_reflection(25.0, z0)),
        "resistor_75ohm": complex(coaxial_load_reflection(75.0, z0)),
    }
    gamma_expected = {
        "matched": 0.0 + 0.0j,
        "short": -1.0 + 0.0j,
        "open": 1.0 + 0.0j,
        "resistor_25ohm": (25.0 - z0) / (25.0 + z0),
        "resistor_75ohm": (75.0 - z0) / (75.0 + z0),
    }
    gamma_rows = []
    for name, gamma in gamma_cases.items():
        expected = gamma_expected[name]
        gamma_rows.append(
            {
                "name": name,
                "gamma": {"re": float(gamma.real), "im": float(gamma.imag)},
                "expected": {"re": float(expected.real), "im": float(expected.imag)},
                "abs_diff": float(abs(gamma - expected)),
                "status": "passed" if abs(gamma - expected) <= 1e-12 else "failed",
            }
        )

    lc_error = abs(z0 - z0_from_lc)
    closed_form_error = abs(z0 - z0_closed_form)
    velocity_rel_error = abs(velocity_from_lc - velocity_expected) / velocity_expected
    beta_rel_error = float(np.max(np.abs(beta - beta_expected) / np.maximum(np.abs(beta_expected), 1e-30)))

    checks = [
        {
            "name": "z0_equals_sqrt_l_over_c",
            "value": float(z0),
            "expected": float(z0_from_lc),
            "abs_error": float(lc_error),
            "limit": 1e-12,
            "status": "passed" if lc_error <= 1e-12 else "failed",
        },
        {
            "name": "z0_matches_closed_form_ln_b_over_a",
            "value": float(z0),
            "expected": float(z0_closed_form),
            "abs_error": float(closed_form_error),
            "limit": 5e-8,
            "status": "passed" if closed_form_error <= 5e-8 else "failed",
        },
        {
            "name": "velocity_equals_c_over_sqrt_epsr",
            "value": float(velocity_from_lc),
            "expected": float(velocity_expected),
            "relative_error": float(velocity_rel_error),
            "limit": 1e-9,
            "status": "passed" if velocity_rel_error <= 1e-9 else "failed",
        },
        {
            "name": "beta_matches_lossless_tem",
            "max_relative_error": beta_rel_error,
            "limit": 5e-7,
            "status": "passed" if beta_rel_error <= 5e-7 else "failed",
        },
    ]
    if z0_expected_ohm is not None:
        z0_error = abs(z0 - z0_expected_ohm)
        limit = float(z0_tolerance_ohm if z0_tolerance_ohm is not None else 0.0)
        checks.insert(
            0,
            {
                "name": f"{case_name}_z0_nominal_gate",
                "value": float(z0),
                "expected": float(z0_expected_ohm),
                "abs_error": float(z0_error),
                "limit": limit,
                "status": "passed" if z0_error <= limit else "failed",
            },
        )
    status = "passed" if all(row["status"] == "passed" for row in checks + gamma_rows) else "failed"
    return {
        "name": case_name,
        "status": status,
        "geometry": {
            "inner_radius_m": float(inner_radius),
            "outer_radius_m": float(outer_radius),
            "eps_r": float(eps_r),
        },
        "z0_ohm": float(z0),
        "z0_closed_form_ohm": float(z0_closed_form),
        "capacitance_per_m_f": float(c_per_m),
        "inductance_per_m_h": float(l_per_m),
        "velocity_m_per_s": float(velocity_from_lc),
        "freqs_hz": freqs_hz.tolist(),
        "beta_rad_per_m": beta.tolist(),
        "checks": checks,
        "reflection_cases": gamma_rows,
    }


def evaluate_coaxial_tem_oracles(
    *,
    inner_radius: float = SMA_PIN_RADIUS,
    outer_radius: float = SMA_OUTER_RADIUS,
    eps_r: float = PTFE_EPS_R,
    freqs_hz: np.ndarray = FREQS_HZ,
    z0_expected_ohm: float = 48.6,
    z0_tolerance_ohm: float = 1.5,
) -> dict:
    primary = _evaluate_one_geometry(
        case_name="sma_ptfe_default",
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        eps_r=eps_r,
        freqs_hz=freqs_hz,
        z0_expected_ohm=z0_expected_ohm,
        z0_tolerance_ohm=z0_tolerance_ohm,
    )
    sweep = []
    for case in GEOMETRY_SWEEP_CASES:
        sweep.append(
            _evaluate_one_geometry(
                case_name=str(case["name"]),
                inner_radius=float(case["inner_radius_m"]),
                outer_radius=float(case["outer_radius_m"]),
                eps_r=float(case["eps_r"]),
                freqs_hz=freqs_hz,
                z0_expected_ohm=(
                    z0_expected_ohm
                    if case["name"] == "sma_ptfe_default"
                    else None
                ),
                z0_tolerance_ohm=z0_tolerance_ohm,
            )
        )
    max_sweep_z0_abs_diff = max(
        max(
            check.get("abs_error", 0.0)
            for check in case["checks"]
            if check["name"] == "z0_matches_closed_form_ln_b_over_a"
        )
        for case in sweep
    )
    max_sweep_beta_rel_error = max(
        max(
            check.get("max_relative_error", 0.0)
            for check in case["checks"]
            if check["name"] == "beta_matches_lossless_tem"
        )
        for case in sweep
    )
    status = (
        "passed"
        if primary["status"] == "passed" and all(case["status"] == "passed" for case in sweep)
        else "failed"
    )
    reference_plane = _evaluate_reference_plane_extractor(
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        eps_r=eps_r,
    )
    cartesian_adapter = _evaluate_cartesian_plane_adapter(
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        eps_r=eps_r,
    )
    if reference_plane["status"] != "passed":
        status = "failed"
    if cartesian_adapter["status"] != "passed":
        status = "failed"
    return {
        **primary,
        "status": status,
        "evidence_level": "E2-calibration-oracle",
        "claim_scope": (
            "coaxial TEM analytic formula/reference helper, geometry sweep, "
            "synthetic reference-plane V/I extraction oracle, and synthetic "
            "Cartesian field-plane adapter oracle only; no FDTD-calibrated "
            "coaxial S-parameter API or external full-wave coax reference"
        ),
        "reference_plane_extractor": reference_plane,
        "cartesian_plane_adapter": cartesian_adapter,
        "geometry_sweep": sweep,
        "geometry_sweep_case_count": len(sweep),
        "max_sweep_z0_abs_diff": float(max_sweep_z0_abs_diff),
        "max_sweep_beta_rel_error": float(max_sweep_beta_rel_error),
    }


def _write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# Coaxial TEM analytic oracle report",
        "",
        f"- status: `{payload['status']}`",
        f"- claim_scope: {payload['claim_scope']}",
        f"- z0_ohm: `{payload['z0_ohm']:.6g}`",
        f"- capacitance_per_m_f: `{payload['capacitance_per_m_f']:.6g}`",
        f"- inductance_per_m_h: `{payload['inductance_per_m_h']:.6g}`",
        f"- velocity_m_per_s: `{payload['velocity_m_per_s']:.6g}`",
        f"- geometry_sweep_case_count: `{payload['geometry_sweep_case_count']}`",
        f"- max_sweep_z0_abs_diff: `{payload['max_sweep_z0_abs_diff']:.6g}`",
        f"- max_sweep_beta_rel_error: `{payload['max_sweep_beta_rel_error']:.6g}`",
        "- reference_plane_extractor_max_s11_abs_diff: "
        f"`{payload['reference_plane_extractor']['max_s11_abs_diff']:.6g}`",
        "- cartesian_plane_adapter_max_s11_abs_diff: "
        f"`{payload['cartesian_plane_adapter']['max_s11_abs_diff']:.6g}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Value | Expected/limit |",
        "|---|---:|---:|---:|",
    ]
    for check in payload["checks"]:
        value = check.get("value", check.get("max_relative_error"))
        expected = check.get("expected", check.get("limit"))
        lines.append(
            f"| `{check['name']}` | `{check['status']}` | `{value:.6g}` | `{expected:.6g}` |"
        )
    lines.extend(["", "## Reflection cases", "", "| Case | Status | Γ | Expected | Abs diff |", "|---|---:|---:|---:|---:|"])
    for row in payload["reflection_cases"]:
        gamma = row["gamma"]
        expected = row["expected"]
        lines.append(
            "| "
            f"`{row['name']}` | `{row['status']}` | "
            f"`{gamma['re']:.6g}{gamma['im']:+.6g}j` | "
            f"`{expected['re']:.6g}{expected['im']:+.6g}j` | "
            f"`{row['abs_diff']:.6g}` |"
        )
    lines.extend(
        [
            "",
            "## Synthetic reference-plane V/I extractor",
            "",
            "| Case | Status | Extracted S11 | Expected S11 | Abs diff |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["reference_plane_extractor"]["rows"]:
        observed = row["extracted_s11"]
        expected = row["expected_s11"]
        lines.append(
            "| "
            f"`{row['name']}` | `{row['status']}` | "
            f"`{observed['re']:.6g}{observed['im']:+.6g}j` | "
            f"`{expected['re']:.6g}{expected['im']:+.6g}j` | "
            f"`{row['s11_abs_diff']:.6g}` |"
        )
    lines.extend(
        [
            "",
            "## Synthetic Cartesian plane adapter",
            "",
            "| Index | Status | Extracted S11 | Expected S11 | Abs diff |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["cartesian_plane_adapter"]["rows"]:
        observed = row["extracted_s11"]
        expected = row["expected_s11"]
        lines.append(
            "| "
            f"`{row['index']}` | `{row['status']}` | "
            f"`{observed['re']:.6g}{observed['im']:+.6g}j` | "
            f"`{expected['re']:.6g}{expected['im']:+.6g}j` | "
            f"`{row['s11_abs_diff']:.6g}` |"
        )
    lines.extend(
        [
            "",
            "## Geometry sweep",
            "",
            "| Case | Status | inner a (m) | outer b (m) | eps_r | Z0 (ohm) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for case in payload["geometry_sweep"]:
        geom = case["geometry"]
        lines.append(
            f"| `{case['name']}` | `{case['status']}` | "
            f"`{geom['inner_radius_m']:.6g}` | `{geom['outer_radius_m']:.6g}` | "
            f"`{geom['eps_r']:.6g}` | `{case['z0_ohm']:.6g}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = evaluate_coaxial_tem_oracles()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "coaxial_tem_oracle_report.json"
    md_path = args.output_dir / "coaxial_tem_oracle_report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_markdown(payload, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"status={payload['status']} z0_ohm={payload['z0_ohm']:.6g}")
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
