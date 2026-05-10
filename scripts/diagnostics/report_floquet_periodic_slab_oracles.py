#!/usr/bin/env python3
"""Report analytic homogeneous-slab oracles for Floquet reference planning.

This is an E2 analytic reference for the zero-order, broadside Floquet
periodic-cell limit: a laterally uniform, lossless dielectric slab between two
air half-spaces.  It deliberately does not call rfx FDTD, RCWA, or an external
solver, and it does not promote ``add_floquet_port(...)`` to a public
S-parameter API.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


C0 = 299_792_458.0
FREQS_HZ = np.asarray([5.0e9, 10.0e9, 15.0e9], dtype=np.float64)
FIXED_SLAB_CASES = (
    ("air_reference_4mm", 1.0, 0.004),
    ("zero_thickness_er225", 2.25, 0.0),
    ("er225_5mm", 2.25, 0.005),
    ("er400_3mm", 4.0, 0.003),
    ("er625_2mm", 6.25, 0.002),
)
DESIGN_FREQ_HZ = 10.0e9


def _complex_dict(value: complex) -> dict[str, float]:
    return {"re": float(value.real), "im": float(value.imag)}


def slab_sparams_normal_incidence(
    *,
    freq_hz: float,
    eps_r: float,
    thickness_m: float,
    n_left: float = 1.0,
    n_right: float = 1.0,
) -> tuple[complex, complex]:
    """Return analytic normal-incidence slab reflection/transmission amplitudes.

    The amplitudes are referenced at the two slab interfaces.  The expression
    is the standard Fabry-Perot sum for a lossless homogeneous layer.
    """

    if freq_hz <= 0.0:
        raise ValueError(f"freq_hz must be positive, got {freq_hz!r}")
    if eps_r <= 0.0:
        raise ValueError(f"eps_r must be positive, got {eps_r!r}")
    if thickness_m < 0.0:
        raise ValueError(f"thickness_m must be nonnegative, got {thickness_m!r}")
    n_slab = math.sqrt(eps_r)
    k0 = 2.0 * math.pi * freq_hz / C0
    phase = k0 * n_slab * thickness_m
    exp_2j_phase = np.exp(2j * phase)
    r_left = (n_left - n_slab) / (n_left + n_slab)
    r_right = (n_slab - n_right) / (n_slab + n_right)
    t_left = 2.0 * n_left / (n_left + n_slab)
    t_right = 2.0 * n_slab / (n_slab + n_right)
    denom = 1.0 + r_left * r_right * exp_2j_phase
    reflection = (r_left + r_right * exp_2j_phase) / denom
    transmission = t_left * t_right * np.exp(1j * phase) / denom
    return complex(reflection), complex(transmission)


def airy_reflectance_symmetric_air_slab(*, freq_hz: float, eps_r: float, thickness_m: float) -> float:
    """Closed-form Airy reflectance for a symmetric air/slab/air etalon."""

    n_slab = math.sqrt(eps_r)
    interface_reflectance = ((1.0 - n_slab) / (1.0 + n_slab)) ** 2
    if interface_reflectance == 1.0:
        return 1.0
    phase = 2.0 * math.pi * freq_hz * n_slab * thickness_m / C0
    coefficient = 4.0 * interface_reflectance / (1.0 - interface_reflectance) ** 2
    return float(
        coefficient * math.sin(phase) ** 2
        / (1.0 + coefficient * math.sin(phase) ** 2)
    )


def evaluate_floquet_periodic_slab_oracles(
    *,
    freqs_hz: np.ndarray = FREQS_HZ,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    max_power_balance_abs_error = 0.0
    max_airy_reflectance_abs_diff = 0.0
    max_air_or_zero_reflection_abs = 0.0
    max_halfwave_reflection_abs = 0.0

    cases = list(FIXED_SLAB_CASES)
    for eps_r in (2.25, 4.0, 6.25):
        n_slab = math.sqrt(eps_r)
        halfwave_thickness = C0 / DESIGN_FREQ_HZ / n_slab / 2.0
        cases.append((f"halfwave_er{eps_r:g}_at_10ghz", eps_r, halfwave_thickness))

    for case_name, eps_r, thickness_m in cases:
        is_air_or_zero = eps_r == 1.0 or thickness_m == 0.0
        is_halfwave = case_name.startswith("halfwave_")
        for freq_hz in freqs_hz:
            reflection, transmission = slab_sparams_normal_incidence(
                freq_hz=float(freq_hz),
                eps_r=float(eps_r),
                thickness_m=float(thickness_m),
            )
            reflectance = abs(reflection) ** 2
            transmittance = abs(transmission) ** 2
            power_balance_abs_error = abs(reflectance + transmittance - 1.0)
            airy_reflectance = airy_reflectance_symmetric_air_slab(
                freq_hz=float(freq_hz),
                eps_r=float(eps_r),
                thickness_m=float(thickness_m),
            )
            airy_reflectance_abs_diff = abs(reflectance - airy_reflectance)
            max_power_balance_abs_error = max(
                max_power_balance_abs_error,
                power_balance_abs_error,
            )
            max_airy_reflectance_abs_diff = max(
                max_airy_reflectance_abs_diff,
                airy_reflectance_abs_diff,
            )
            if is_air_or_zero:
                max_air_or_zero_reflection_abs = max(
                    max_air_or_zero_reflection_abs,
                    abs(reflection),
                )
            if is_halfwave and math.isclose(float(freq_hz), DESIGN_FREQ_HZ):
                max_halfwave_reflection_abs = max(
                    max_halfwave_reflection_abs,
                    abs(reflection),
                )
            rows.append(
                {
                    "case": case_name,
                    "freq_hz": float(freq_hz),
                    "eps_r": float(eps_r),
                    "thickness_m": float(thickness_m),
                    "reflection": _complex_dict(reflection),
                    "transmission": _complex_dict(transmission),
                    "reflectance": float(reflectance),
                    "transmittance": float(transmittance),
                    "power_balance_abs_error": float(power_balance_abs_error),
                    "airy_reflectance": float(airy_reflectance),
                    "airy_reflectance_abs_diff": float(airy_reflectance_abs_diff),
                }
            )

    checks = [
        {
            "name": "lossless_power_balance",
            "value": float(max_power_balance_abs_error),
            "limit": 1e-12,
            "status": "passed" if max_power_balance_abs_error <= 1e-12 else "failed",
        },
        {
            "name": "fabry_perot_matches_airy_reflectance",
            "value": float(max_airy_reflectance_abs_diff),
            "limit": 1e-12,
            "status": "passed" if max_airy_reflectance_abs_diff <= 1e-12 else "failed",
        },
        {
            "name": "air_or_zero_thickness_has_zero_reflection",
            "value": float(max_air_or_zero_reflection_abs),
            "limit": 1e-12,
            "status": "passed" if max_air_or_zero_reflection_abs <= 1e-12 else "failed",
        },
        {
            "name": "design_frequency_halfwave_has_zero_reflection",
            "value": float(max_halfwave_reflection_abs),
            "limit": 1e-12,
            "status": "passed" if max_halfwave_reflection_abs <= 1e-12 else "failed",
        },
    ]
    status = "passed" if all(check["status"] == "passed" for check in checks) else "failed"
    return {
        "status": status,
        "evidence_level": "E2-analytic",
        "claim": "Floquet broadside homogeneous dielectric-slab analytic oracle",
        "claim_scope": (
            "broadside zero-order Floquet periodic homogeneous-slab analytic "
            "reference over frequency/eps_r/thickness cases; no rfx FDTD "
            "candidate, no RCWA/external full-wave comparison, and no promoted "
            "Floquet S-parameter API"
        ),
        "design_freq_hz": float(DESIGN_FREQ_HZ),
        "freqs_hz": [float(freq) for freq in freqs_hz],
        "case_count": len(cases),
        "row_count": len(rows),
        "checks": checks,
        "max_power_balance_abs_error": float(max_power_balance_abs_error),
        "max_airy_reflectance_abs_diff": float(max_airy_reflectance_abs_diff),
        "max_air_or_zero_reflection_abs": float(max_air_or_zero_reflection_abs),
        "max_halfwave_reflection_abs": float(max_halfwave_reflection_abs),
        "rows": rows,
        "completion_decision": (
            "Do not claim Floquet E4/E5 from this artifact. It is an analytic "
            "non-empty slab oracle that narrows one E2 blocker only."
        ),
    }


def _write_report(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "floquet_periodic_slab_oracle_report.json"
    md_path = output_dir / "floquet_periodic_slab_oracle_report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Floquet periodic slab analytic oracle report",
        "",
        f"- status: `{payload['status']}`",
        f"- evidence_level: `{payload['evidence_level']}`",
        f"- case_count: `{payload['case_count']}`",
        f"- row_count: `{payload['row_count']}`",
        f"- max_power_balance_abs_error: `{payload['max_power_balance_abs_error']:.6g}`",
        f"- max_airy_reflectance_abs_diff: `{payload['max_airy_reflectance_abs_diff']:.6g}`",
        f"- max_air_or_zero_reflection_abs: `{payload['max_air_or_zero_reflection_abs']:.6g}`",
        f"- max_halfwave_reflection_abs: `{payload['max_halfwave_reflection_abs']:.6g}`",
        "",
        payload["claim_scope"],
        "",
        f"Decision: {payload['completion_decision']}",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".omx/physics-gate/latest-floquet-periodic-slab-oracles"),
    )
    args = parser.parse_args(argv)
    payload = evaluate_floquet_periodic_slab_oracles()
    _write_report(payload, args.output_dir)
    print(
        "status={status} max_power_balance_abs_error={max_power_balance_abs_error:.6g}".format(
            **payload
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
