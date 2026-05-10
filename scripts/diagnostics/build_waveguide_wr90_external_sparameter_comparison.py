#!/usr/bin/env python3
"""Build a generic comparator artifact from the WR90 cv11 external tables.

The cv11 waveguide gate prints a four-way diagnostic table:
``rfx | MEEP_r4 | OpenEMS_r4 | Palace_r_h2``.  This adapter extracts the slab
S11/S21 rfx and Palace columns and runs the generic S-parameter comparator in
magnitude mode.  It is a narrow E4-enabling artifact for the documented WR90
slab envelope, not broad waveguide-port E5.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from compare_sparameter_reference import (
    _repo_path,
    compare_sparameter_datasets,
    load_sparameter_dataset,
)


DEFAULT_CV11_STDOUT = ".omx/physics-gate/2026-05-09-m4-cv11-waveguide-wr90.stdout.txt"

_TABLE_START_RE = re.compile(r"^\[4way (?P<geom>\S+) (?P<comp>S\d\d)\]")
_VALUE_RE = re.compile(r"(?P<mag>[0-9.]+)@\s*(?P<deg>[+-]?[0-9.]+)d")


def _parse_mag_phase(value: str) -> complex:
    match = _VALUE_RE.search(value)
    if not match:
        raise ValueError(f"cannot parse magnitude/phase cell: {value!r}")
    mag = float(match.group("mag"))
    deg = float(match.group("deg"))
    return mag * np.exp(1j * np.deg2rad(deg))


def extract_4way_series(
    text: str,
    *,
    geom: str = "slab",
    comp: str = "S21",
    reference_column: str = "Palace_r_h2",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    in_table = False
    frequencies_ghz: list[float] = []
    rfx_values: list[complex] = []
    reference_values: list[complex] = []
    reference_index_by_name = {
        "MEEP_r4": 2,
        "OpenEMS_r4": 3,
        "Palace_r_h2": 4,
    }
    if reference_column not in reference_index_by_name:
        raise ValueError(f"unsupported reference column: {reference_column!r}")
    reference_index = reference_index_by_name[reference_column]

    for line in text.splitlines():
        start = _TABLE_START_RE.match(line)
        if start:
            in_table = start.group("geom") == geom and start.group("comp") == comp
            continue
        if not in_table:
            continue
        if line.startswith("[summary") or not line.strip():
            break
        if "|" not in line or "----" in line or "f_GHz" in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 5:
            continue
        try:
            frequencies_ghz.append(float(parts[0]))
        except ValueError:
            continue
        rfx_values.append(_parse_mag_phase(parts[1]))
        reference_values.append(_parse_mag_phase(parts[reference_index]))

    if not frequencies_ghz:
        raise ValueError(f"no rows parsed for [4way {geom} {comp}]")
    return (
        np.asarray(frequencies_ghz, dtype=float) * 1e9,
        np.asarray(rfx_values, dtype=np.complex128),
        np.asarray(reference_values, dtype=np.complex128),
    )


def _twoport_from_s11_s21(s11: np.ndarray, s21: np.ndarray) -> np.ndarray:
    if s11.shape != s21.shape:
        raise ValueError(f"s11/s21 shapes differ: {s11.shape} vs {s21.shape}")
    s_params = np.zeros((2, 2, s11.size), dtype=np.complex128)
    s_params[0, 0, :] = s11
    s_params[1, 0, :] = s21
    s_params[0, 1, :] = s21
    s_params[1, 1, :] = s11
    return s_params


def build_waveguide_wr90_generic_comparison(
    cv11_stdout: Path,
    output_dir: Path,
    *,
    reference_column: str = "Palace_r_h2",
) -> dict[str, Any]:
    text = cv11_stdout.read_text(encoding="utf-8")
    f_s11, rfx_s11, ref_s11 = extract_4way_series(
        text,
        geom="slab",
        comp="S11",
        reference_column=reference_column,
    )
    f_s21, rfx_s21, ref_s21 = extract_4way_series(
        text,
        geom="slab",
        comp="S21",
        reference_column=reference_column,
    )
    if not np.array_equal(f_s11, f_s21):
        raise ValueError("S11/S21 frequency grids differ in cv11 stdout")

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_npz = output_dir / "wr90_rfx_candidate_sparams.npz"
    reference_npz = output_dir / "wr90_external_reference_sparams.npz"
    np.savez(candidate_npz, freqs_hz=f_s11, s_params=_twoport_from_s11_s21(rfx_s11, rfx_s21))
    np.savez(reference_npz, freqs_hz=f_s11, s_params=_twoport_from_s11_s21(ref_s11, ref_s21))

    payload = compare_sparameter_datasets(
        load_sparameter_dataset(candidate_npz),
        load_sparameter_dataset(reference_npz),
        terms="S11,S21",
        comparison_mode="magnitude",
        max_abs_tol=0.30,
        mean_abs_tol=0.20,
        max_mag_abs_tol=0.08,
        mean_mag_abs_tol=0.05,
    )
    payload["claim"] = (
        f"WR90 slab waveguide-port S-parameter magnitude comparison against {reference_column}"
    )
    payload["claim_scope"] = (
        "narrow rectangular-waveguide WR90 slab external-reference comparison; "
        "not branch/multimode/nonuniform broad waveguide E5"
    )
    payload["source_cv11_stdout"] = str(cv11_stdout)
    payload["external_reference_column"] = reference_column

    output_json = output_dir / "wr90_external_generic_sparameter_comparison.json"
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv11-stdout", default=DEFAULT_CV11_STDOUT)
    parser.add_argument("--reference-column", default="Palace_r_h2")
    parser.add_argument(
        "--output-dir",
        default=".omx/physics-gate/latest-waveguide-wr90-generic-comparison",
    )
    args = parser.parse_args(argv)

    payload = build_waveguide_wr90_generic_comparison(
        _repo_path(args.cv11_stdout),
        _repo_path(args.output_dir),
        reference_column=args.reference_column,
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
