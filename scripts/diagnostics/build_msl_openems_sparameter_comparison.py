#!/usr/bin/env python3
"""Build a generic comparator artifact for the stored MSL openEMS reference.

The historical MSL/openEMS smoke check has a family-specific JSON schema.  This
adapter converts that evidence into the generic
``compare_sparameter_reference.py`` schema so the broad-E5 external-reference
manifest can verify it uniformly with other port families.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from compare_sparameter_reference import (
    _repo_path,
    compare_sparameter_datasets,
    load_sparameter_dataset,
)


DEFAULT_REFERENCE = "docs/research_notes/msl_thru_openems_80um.json"
DEFAULT_RFX_COMPARISON = (
    ".omx/physics-gate/vessl-2026-05-09-m3/"
    "msl_openems_reference/msl_openems_comparison.json"
)


def _complex_array(values: list[list[float]]) -> np.ndarray:
    return np.asarray([complex(real, imag) for real, imag in values], dtype=np.complex128)


def _twoport_from_s11_s21(s11: np.ndarray, s21: np.ndarray) -> np.ndarray:
    if s11.shape != s21.shape:
        raise ValueError(f"s11/s21 shapes differ: {s11.shape} vs {s21.shape}")
    s_params = np.zeros((2, 2, s11.size), dtype=np.complex128)
    s_params[0, 0, :] = s11
    s_params[1, 0, :] = s21
    # The stored smoke reference is used only for S11/S21.  Fill the reciprocal
    # terms for completeness, but the comparator is invoked with --terms S11,S21.
    s_params[0, 1, :] = s21
    s_params[1, 1, :] = s11
    return s_params


def build_msl_openems_generic_comparison(
    reference_path: Path,
    rfx_comparison_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    rfx_comparison = json.loads(rfx_comparison_path.read_text(encoding="utf-8"))

    reference_freqs_hz = np.asarray(reference["freqs_ghz"], dtype=float) * 1e9
    reference_s = _twoport_from_s11_s21(
        _complex_array(reference["s11"]),
        _complex_array(reference["s21"]),
    )
    candidate_freqs_hz = np.asarray(rfx_comparison["rfx_freqs_hz"], dtype=float)
    candidate_s = _twoport_from_s11_s21(
        _complex_array(rfx_comparison["rfx_s11"]),
        _complex_array(rfx_comparison["rfx_s21"]),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    reference_npz = output_dir / "msl_openems_reference_sparams.npz"
    candidate_npz = output_dir / "msl_rfx_candidate_sparams.npz"
    np.savez(reference_npz, freqs_hz=reference_freqs_hz, s_params=reference_s)
    np.savez(candidate_npz, freqs_hz=candidate_freqs_hz, s_params=candidate_s)

    payload = compare_sparameter_datasets(
        load_sparameter_dataset(candidate_npz),
        load_sparameter_dataset(reference_npz),
        terms="S11,S21",
        comparison_mode="magnitude",
        f_lo_hz=3.0e9,
        f_hi_hz=4.5e9,
        max_abs_tol=0.08,
        mean_abs_tol=0.04,
        max_mag_abs_tol=0.08,
        mean_mag_abs_tol=0.04,
    )
    payload["claim"] = "MSL thru-line generic S-parameter comparison against stored openEMS"
    payload["claim_scope"] = (
        "narrow stored-reference MSL thru smoke check over 3.0--4.5 GHz; "
        "not broad all-mode MSL E5"
    )
    payload["source_reference_artifact"] = str(reference_path)
    payload["source_rfx_comparison_artifact"] = str(rfx_comparison_path)

    output_json = output_dir / "msl_openems_generic_sparameter_comparison.json"
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--rfx-comparison", default=DEFAULT_RFX_COMPARISON)
    parser.add_argument(
        "--output-dir",
        default=".omx/physics-gate/latest-msl-openems-generic-comparison",
    )
    args = parser.parse_args(argv)

    payload = build_msl_openems_generic_comparison(
        _repo_path(args.reference),
        _repo_path(args.rfx_comparison),
        _repo_path(args.output_dir),
    )
    print(
        "status={status} max_abs_diff={max_abs_diff:.6g}".format(
            status=payload["status"],
            max_abs_diff=payload["summary"]["max_abs_diff"],
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
