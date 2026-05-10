#!/usr/bin/env python3
"""Build a generic comparator artifact for crossval05 patch wire-port S11.

The crossval05 patch antenna is a probe-fed geometry.  In rfx it uses
``add_port(..., extent=...)``, i.e. the WirePort extractor path.  The comparison
is intentionally narrow and magnitude-mode only: the script is useful E4-
enabling evidence for the probe-fed patch resonance lane, not broad absolute
wire-port S-parameter calibration.
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


DEFAULT_CROSSVAL_JSON = (
    ".omx/physics-gate/2026-05-09-m32-patch-wire-openems/"
    "crossval05_patch_openems_rfx.json"
)


def _complex_array(values: list[list[float]]) -> np.ndarray:
    return np.asarray([complex(real, imag) for real, imag in values], dtype=np.complex128)


def _oneport_from_s11(s11: np.ndarray) -> np.ndarray:
    s_params = np.zeros((1, 1, s11.size), dtype=np.complex128)
    s_params[0, 0, :] = s11
    return s_params


def build_patch_wire_openems_generic_comparison(
    crossval_json: Path,
    output_dir: Path,
) -> dict[str, Any]:
    crossval = json.loads(crossval_json.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_npz = output_dir / "patch_wire_rfx_candidate_sparams.npz"
    reference_npz = output_dir / "patch_wire_openems_reference_sparams.npz"

    np.savez(
        candidate_npz,
        freqs_hz=np.asarray(crossval["rfx_freqs_hz"], dtype=float),
        s_params=_oneport_from_s11(_complex_array(crossval["rfx_s11"])),
    )
    np.savez(
        reference_npz,
        freqs_hz=np.asarray(crossval["openems_freqs_hz"], dtype=float),
        s_params=_oneport_from_s11(_complex_array(crossval["openems_s11"])),
    )

    payload = compare_sparameter_datasets(
        load_sparameter_dataset(candidate_npz),
        load_sparameter_dataset(reference_npz),
        terms="S11",
        comparison_mode="magnitude",
        f_lo_hz=1.5e9,
        f_hi_hz=3.4e9,
        max_abs_tol=1.0,
        mean_abs_tol=1.0,
        max_mag_abs_tol=0.06,
        mean_mag_abs_tol=0.04,
    )
    payload["claim"] = "probe-fed patch wire-port S11 magnitude comparison against OpenEMS"
    payload["claim_scope"] = (
        "narrow crossval05 patch resonance lane using rfx add_port(..., "
        "extent=...) and OpenEMS lumped port; not broad absolute wire-port E5"
    )
    payload["source_crossval_json"] = str(crossval_json)
    payload["resonance_metrics"] = {
        "rfx_vs_openems_harminv_pct": crossval["rfx_vs_openems_harminv_pct"],
        "rfx_internal_pct": crossval["rfx_internal_pct"],
        "rfx_vs_analytic_pct": crossval["rfx_vs_analytic_pct"],
        "rfx_s11_passive": crossval["rfx_s11_passive"],
        "rfx_s11_max_abs": crossval["rfx_s11_max_abs"],
    }

    output_json = output_dir / "patch_wire_openems_generic_sparameter_comparison.json"
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crossval-json", default=DEFAULT_CROSSVAL_JSON)
    parser.add_argument(
        "--output-dir",
        default=".omx/physics-gate/latest-patch-wire-openems-generic-comparison",
    )
    args = parser.parse_args(argv)

    payload = build_patch_wire_openems_generic_comparison(
        _repo_path(args.crossval_json),
        _repo_path(args.output_dir),
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
