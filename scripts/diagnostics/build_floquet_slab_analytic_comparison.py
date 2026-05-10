#!/usr/bin/env python3
"""Build an rfx-FDTD vs analytic slab comparison for Floquet diagnostics.

This diagnostic runs a narrow broadside TE Floquet excitation against a
laterally uniform dielectric slab, records raw Ex/Hy DFT planes in rfx, and
recomputes an approximate one-port reflection coefficient from the raw phasors
with a tiny independent NumPy post-processor.  The reference is the analytic
normal-incidence homogeneous-slab reflection magnitude.

The result is useful E2/E3-enabling evidence for the non-empty broadside slab
Floquet diagnostic lane.  It is deliberately not RCWA/external E4 evidence, not
a calibrated reference-plane S-parameter validation, not a scan/polarization
envelope, and not a promoted ``add_floquet_port(...)`` S-parameter API.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from compare_sparameter_reference import (
    _repo_path,
    compare_sparameter_datasets,
    load_sparameter_dataset,
)
from report_floquet_periodic_slab_oracles import slab_sparams_normal_incidence
from rfx import Box, Simulation


ETA0_OHM = 376.730313668
DEFAULT_FREQS_HZ = np.asarray([3.0e9, 4.0e9, 5.0e9], dtype=float)


def _oneport_from_s11(s11: np.ndarray) -> np.ndarray:
    s11 = np.asarray(s11, dtype=np.complex128)
    s_params = np.zeros((1, 1, s11.size), dtype=np.complex128)
    s_params[0, 0, :] = s11
    return s_params


def _complex_dict(value: complex) -> dict[str, float]:
    return {"re": float(value.real), "im": float(value.imag)}


def _mean_plane_by_freq(field_dft: np.ndarray, n_freqs: int) -> np.ndarray:
    field_dft = np.asarray(field_dft, dtype=np.complex128)
    if field_dft.shape[0] != n_freqs:
        raise ValueError(
            f"DFT plane first axis must contain {n_freqs} frequencies; "
            f"got shape {field_dft.shape}"
        )
    return np.mean(field_dft.reshape(n_freqs, -1), axis=1)


def _decompose_broadside_te_s11(
    *,
    ex_dft: np.ndarray,
    hy_dft: np.ndarray,
    freqs_hz: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return raw-plane broadside TE S11 from Ex/Hy without production helper.

    For a +z TE plane wave in air, ``Hy = Ex / eta0``.  With a reflected -z
    component, ``Hy = (E_forward - E_backward) / eta0``.  Therefore the raw
    plane phasors give ``E_forward=(Ex+eta0*Hy)/2`` and
    ``E_backward=(Ex-eta0*Hy)/2``.  This is intentionally a diagnostic plane
    decomposition, not a calibrated Floquet-port reference-plane extractor.
    """

    freqs = np.asarray(freqs_hz, dtype=float)
    ex_avg = _mean_plane_by_freq(ex_dft, freqs.size)
    hy_avg = _mean_plane_by_freq(hy_dft, freqs.size)
    forward = 0.5 * (ex_avg + ETA0_OHM * hy_avg)
    backward = 0.5 * (ex_avg - ETA0_OHM * hy_avg)
    min_forward_abs = float(np.min(np.abs(forward)))
    if min_forward_abs <= 1e-15:
        raise RuntimeError(
            "raw Floquet slab DFT plane produced near-zero forward amplitude; "
            f"min_abs={min_forward_abs:.6g}"
        )
    s11 = backward / forward
    if not np.all(np.isfinite(s11)):
        raise RuntimeError("raw Floquet slab DFT decomposition produced non-finite S11")
    summary = {
        "eta0_ohm": float(ETA0_OHM),
        "max_abs_ex_dft": float(np.max(np.abs(ex_dft))),
        "max_abs_hy_dft": float(np.max(np.abs(hy_dft))),
        "max_abs_ex_plane_mean": float(np.max(np.abs(ex_avg))),
        "max_abs_hy_plane_mean": float(np.max(np.abs(hy_avg))),
        "min_abs_forward_amplitude": min_forward_abs,
        "max_abs_backward_amplitude": float(np.max(np.abs(backward))),
        "candidate_s11_magnitudes": [float(abs(value)) for value in s11],
    }
    return s11, summary


def analytic_slab_s11(
    *,
    freqs_hz: np.ndarray,
    eps_r: float,
    thickness_m: float,
) -> np.ndarray:
    values: list[complex] = []
    for freq_hz in np.asarray(freqs_hz, dtype=float):
        reflection, _transmission = slab_sparams_normal_incidence(
            freq_hz=float(freq_hz),
            eps_r=float(eps_r),
            thickness_m=float(thickness_m),
        )
        values.append(reflection)
    return np.asarray(values, dtype=np.complex128)


def build_floquet_slab_comparison_from_s11(
    *,
    freqs_hz: np.ndarray,
    candidate_s11: np.ndarray,
    reference_s11: np.ndarray,
    output_dir: Path,
    n_steps: int = 240,
    case: dict[str, Any] | None = None,
    raw_dft_decomposition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    freqs = np.asarray(freqs_hz, dtype=float)
    candidate_npz = output_dir / "floquet_slab_rfx_candidate_sparams.npz"
    reference_npz = output_dir / "floquet_slab_analytic_reference_sparams.npz"
    np.savez(candidate_npz, freqs_hz=freqs, s_params=_oneport_from_s11(candidate_s11))
    np.savez(reference_npz, freqs_hz=freqs, s_params=_oneport_from_s11(reference_s11))

    payload = compare_sparameter_datasets(
        load_sparameter_dataset(candidate_npz),
        load_sparameter_dataset(reference_npz),
        terms="S11",
        comparison_mode="magnitude",
        max_abs_tol=1.0,
        mean_abs_tol=1.0,
        max_mag_abs_tol=0.07,
        mean_mag_abs_tol=0.04,
    )
    payload["evidence_level"] = "E2/E3-enabling"
    payload["claim"] = (
        "Floquet broadside dielectric-slab rfx FDTD vs analytic reflection "
        "magnitude comparison"
    )
    payload["claim_scope"] = (
        "narrow broadside homogeneous-slab Floquet diagnostic: rfx real-FDTD "
        "raw Ex/Hy DFT-plane decomposition compared to analytic normal-incidence "
        "slab reflection magnitude; not RCWA/external E4 evidence, not a "
        "calibrated reference-plane S-parameter validation, not a scan/"
        "polarization envelope, and not a promoted Floquet S-parameter API"
    )
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["n_steps"] = int(n_steps)
    payload["case"] = case or {}
    payload["raw_dft_decomposition"] = raw_dft_decomposition or {}
    payload["analytic_reference"] = [
        {
            "freq_hz": float(freq),
            "s11": _complex_dict(complex(value)),
            "s11_magnitude": float(abs(value)),
        }
        for freq, value in zip(freqs, np.asarray(reference_s11, dtype=np.complex128), strict=True)
    ]
    payload["completion_decision"] = (
        "Do not claim Floquet E4/E5 from this artifact. It exercises a "
        "non-empty rfx FDTD slab diagnostic against an analytic oracle only."
    )

    output_json = output_dir / "floquet_slab_analytic_sparameter_comparison.json"
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_floquet_slab_analytic_comparison(
    output_dir: Path,
    *,
    n_steps: int = 240,
    freqs_hz: np.ndarray = DEFAULT_FREQS_HZ,
    domain_m: tuple[float, float, float] = (0.016, 0.016, 0.034),
    dx_m: float = 0.002,
    eps_r: float = 2.25,
    slab_z_m: tuple[float, float] = (0.014, 0.018),
    source_z_m: float = 0.006,
    probe_z_m: float = 0.010,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    freqs = np.asarray(freqs_hz, dtype=float)
    thickness_m = float(slab_z_m[1] - slab_z_m[0])
    if thickness_m <= 0.0:
        raise ValueError(f"slab thickness must be positive, got {thickness_m!r}")

    sim = Simulation(domain=domain_m, dx=dx_m, freq_max=8.0e9)
    sim.add_material("floquet_slab", eps_r=eps_r)
    sim.add(
        Box((0.0, 0.0, slab_z_m[0]), (domain_m[0], domain_m[1], slab_z_m[1])),
        material="floquet_slab",
    )
    sim.add_floquet_port(
        source_z_m,
        axis="z",
        scan_theta=0.0,
        scan_phi=0.0,
        polarization="te",
        f0=4.0e9,
        bandwidth=0.8,
        amplitude=1.0,
    )
    sim.add_dft_plane_probe(
        axis="z",
        coordinate=probe_z_m,
        component="ex",
        freqs=freqs,
        name="ex_plane",
    )
    sim.add_dft_plane_probe(
        axis="z",
        coordinate=probe_z_m,
        component="hy",
        freqs=freqs,
        name="hy_plane",
    )
    result = sim.run(n_steps=n_steps)
    if result.dft_planes is None:
        raise RuntimeError("Floquet slab diagnostic produced no DFT planes")

    ex_dft = np.asarray(result.dft_planes["ex_plane"].accumulator, dtype=np.complex128)
    hy_dft = np.asarray(result.dft_planes["hy_plane"].accumulator, dtype=np.complex128)
    candidate_s11, raw_summary = _decompose_broadside_te_s11(
        ex_dft=ex_dft,
        hy_dft=hy_dft,
        freqs_hz=freqs,
    )
    reference_s11 = analytic_slab_s11(
        freqs_hz=freqs,
        eps_r=eps_r,
        thickness_m=thickness_m,
    )

    case = {
        "domain_m": [float(value) for value in domain_m],
        "dx_m": float(dx_m),
        "eps_r": float(eps_r),
        "slab_z_m": [float(value) for value in slab_z_m],
        "slab_thickness_m": thickness_m,
        "source_z_m": float(source_z_m),
        "probe_z_m": float(probe_z_m),
        "freqs_hz": [float(value) for value in freqs],
        "n_steps": int(n_steps),
        "grid_shape": list(result.grid.shape),
        "dt_s": float(result.dt),
    }
    payload = build_floquet_slab_comparison_from_s11(
        freqs_hz=freqs,
        candidate_s11=candidate_s11,
        reference_s11=reference_s11,
        output_dir=output_dir,
        n_steps=n_steps,
        case=case,
        raw_dft_decomposition=raw_summary,
    )
    payload["candidate_s11"] = [
        _complex_dict(complex(value)) for value in np.asarray(candidate_s11, dtype=np.complex128)
    ]
    output_json = output_dir / "floquet_slab_analytic_sparameter_comparison.json"
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=".omx/physics-gate/latest-floquet-slab-analytic-comparison",
    )
    parser.add_argument("--n-steps", type=int, default=240)
    args = parser.parse_args(argv)

    payload = build_floquet_slab_analytic_comparison(
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
