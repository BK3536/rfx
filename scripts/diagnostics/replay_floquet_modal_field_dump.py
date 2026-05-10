#!/usr/bin/env python3
"""Replay a Floquet specular-TE modal dump without calling rfx.floquet helpers.

The dump schema is intentionally small: a z-normal TE diagnostic stores raw Ex
and Hy complex DFT planes plus the helper-produced modal S vector. This script
recomputes the specular forward/backward decomposition from those raw fields
using only NumPy and the free-space constants, then compares against the stored
helper result.

This is diagnostic replay infrastructure only. It does not promote
`add_floquet_port(...)` to a high-level Simulation S-parameter API.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

EPS_0 = 8.8541878128e-12
MU_0 = 1.25663706212e-6


def replay_floquet_modal_dump(dump_path: Path, *, atol: float = 1e-6) -> dict:
    with np.load(dump_path, allow_pickle=False) as data:
        ex_dft = np.asarray(data["ex_dft"], dtype=np.complex128)
        hy_dft = np.asarray(data["hy_dft"], dtype=np.complex128)
        freqs_hz = np.asarray(data["freqs_hz"], dtype=np.float64)
        theta_deg = float(np.asarray(data["theta_deg"]))
        helper_s = np.asarray(data["helper_s"], dtype=np.complex128)
        helper_forward = np.asarray(data["helper_forward"], dtype=np.complex128)
        helper_backward = np.asarray(data["helper_backward"], dtype=np.complex128)

    if ex_dft.shape != hy_dft.shape:
        raise ValueError(f"Ex/Hy shape mismatch: {ex_dft.shape} != {hy_dft.shape}")
    if ex_dft.ndim != 3:
        raise ValueError(f"expected DFT planes shaped (n_freqs, nu, nv), got {ex_dft.shape}")
    if ex_dft.shape[0] != freqs_hz.shape[0]:
        raise ValueError("DFT plane frequency axis does not match freqs_hz")

    e_avg = np.mean(ex_dft, axis=(1, 2))
    h_avg = np.mean(hy_dft, axis=(1, 2))
    eta0 = math.sqrt(MU_0 / EPS_0)
    cos_theta = max(math.cos(math.radians(theta_deg)), 1e-10)
    eta_te = eta0 / cos_theta
    replay_forward = (e_avg + eta_te * h_avg) / 2.0
    replay_backward = (e_avg - eta_te * h_avg) / 2.0
    replay_s = replay_backward / np.where(np.abs(replay_forward) > 1e-30, replay_forward, 1e-30)

    max_s_abs_diff = float(np.max(np.abs(replay_s - helper_s)))
    max_forward_abs_diff = float(np.max(np.abs(replay_forward - helper_forward)))
    max_backward_abs_diff = float(np.max(np.abs(replay_backward - helper_backward)))
    finite = bool(
        np.all(np.isfinite(ex_dft))
        and np.all(np.isfinite(hy_dft))
        and np.all(np.isfinite(replay_s))
    )
    status = "passed" if finite and max_s_abs_diff <= atol else "failed"
    return {
        "status": status,
        "dump_path": str(dump_path),
        "claim_scope": (
            "Floquet real FDTD DFT-plane modal replay diagnostic only; no promoted "
            "high-level Floquet S-parameter API"
        ),
        "freqs_hz": freqs_hz.tolist(),
        "theta_deg": theta_deg,
        "plane_shape": list(ex_dft.shape),
        "eta_te_ohm": float(eta_te),
        "max_s_abs_diff": max_s_abs_diff,
        "max_forward_abs_diff": max_forward_abs_diff,
        "max_backward_abs_diff": max_backward_abs_diff,
        "atol": float(atol),
        "finite": finite,
        "max_abs_ex_dft": float(np.max(np.abs(ex_dft))),
        "max_abs_hy_dft": float(np.max(np.abs(hy_dft))),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_path", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args(argv)

    payload = replay_floquet_modal_dump(args.dump_path, atol=args.atol)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"status={payload['status']} max_s_abs_diff={payload['max_s_abs_diff']:.6g}")
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
