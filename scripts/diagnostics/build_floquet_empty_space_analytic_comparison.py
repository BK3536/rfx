#!/usr/bin/env python3
"""Build an analytic-null comparison artifact for the Floquet diagnostic.

The reference is the broadside TE empty-space result: no structure is present,
so the specular reflection coefficient should be zero.  The candidate is the
current rfx real-FDTD DFT-plane modal replay diagnostic.  This is E2/E3
evidence for the narrow empty-space Floquet modal diagnostic lane only; it is
not RCWA/external full-wave evidence and does not promote a public Floquet
S-parameter API.
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
from generate_floquet_modal_field_dump import generate_floquet_modal_field_dump


def _oneport_from_s11(s11: np.ndarray) -> np.ndarray:
    s11 = np.asarray(s11, dtype=np.complex128)
    s_params = np.zeros((1, 1, s11.size), dtype=np.complex128)
    s_params[0, 0, :] = s11
    return s_params


def build_floquet_empty_space_comparison_from_s11(
    *,
    freqs_hz: np.ndarray,
    candidate_s11: np.ndarray,
    output_dir: Path,
    n_steps: int = 120,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_npz = output_dir / "floquet_empty_space_rfx_candidate_sparams.npz"
    reference_npz = output_dir / "floquet_empty_space_analytic_reference_sparams.npz"
    freqs = np.asarray(freqs_hz, dtype=float)
    np.savez(candidate_npz, freqs_hz=freqs, s_params=_oneport_from_s11(candidate_s11))
    np.savez(reference_npz, freqs_hz=freqs, s_params=_oneport_from_s11(np.zeros_like(freqs, dtype=np.complex128)))

    payload = compare_sparameter_datasets(
        load_sparameter_dataset(candidate_npz),
        load_sparameter_dataset(reference_npz),
        terms="S11",
        comparison_mode="complex",
        max_abs_tol=0.07,
        mean_abs_tol=0.06,
        max_mag_abs_tol=0.07,
        mean_mag_abs_tol=0.06,
    )
    payload["evidence_level"] = "E2/E3-enabling"
    payload["claim"] = "Floquet broadside TE empty-space analytic-null S11 comparison"
    payload["claim_scope"] = (
        "narrow empty-space Floquet modal diagnostic: rfx real-FDTD DFT-plane "
        "modal replay compared to analytic zero reflection; not RCWA/external "
        "full-wave evidence and not a promoted Floquet S-parameter API"
    )
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["n_steps"] = int(n_steps)
    output_json = output_dir / "floquet_empty_space_analytic_sparameter_comparison.json"
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_floquet_empty_space_analytic_comparison(
    output_dir: Path,
    *,
    n_steps: int = 120,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = output_dir / "floquet_modal_field_dump"
    summary = generate_floquet_modal_field_dump(dump_dir, n_steps=n_steps)
    dump_path = Path(summary["dump_path"])
    if not dump_path.is_absolute():
        dump_path = _repo_path(dump_path)
    with np.load(dump_path, allow_pickle=False) as data:
        freqs_hz = np.asarray(data["freqs_hz"], dtype=float)
        candidate_s11 = np.asarray(data["helper_s"], dtype=np.complex128)
    payload = build_floquet_empty_space_comparison_from_s11(
        freqs_hz=freqs_hz,
        candidate_s11=candidate_s11,
        output_dir=output_dir,
        n_steps=n_steps,
    )
    payload["source_field_dump"] = str(dump_path)
    payload["source_replay_json"] = summary["replay_json"]
    output_json = output_dir / "floquet_empty_space_analytic_sparameter_comparison.json"
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=".omx/physics-gate/latest-floquet-empty-space-analytic-comparison",
    )
    parser.add_argument("--n-steps", type=int, default=120)
    args = parser.parse_args(argv)

    payload = build_floquet_empty_space_analytic_comparison(
        _repo_path(args.output_dir),
        n_steps=args.n_steps,
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
