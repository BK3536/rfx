#!/usr/bin/env python3
"""Generate a real-FDTD Floquet DFT-plane dump plus independent replay report.

This diagnostic runs a small z-normal TE Floquet excitation, records raw Ex/Hy
DFT planes, stores the current helper decomposition, and invokes
`replay_floquet_modal_field_dump.py` to independently replay the modal S vector
from the raw fields. It is evidence for modal bookkeeping and dump/replay
plumbing only; it does not promote `add_floquet_port(...)` to a public
S-parameter calculator.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from rfx import Simulation
from rfx.floquet import FloquetDFTAccumulator, extract_floquet_modes


def generate_floquet_modal_field_dump(
    output_dir: Path,
    *,
    n_steps: int = 120,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_path = output_dir / "floquet_modal_field_dump.npz"
    replay_json = output_dir / "floquet_modal_replay.json"

    freqs_hz = np.asarray([3.0e9, 4.0e9, 5.0e9], dtype=np.float64)
    domain = (0.016, 0.016, 0.024)
    dx = 0.002
    theta_deg = 0.0
    phi_deg = 0.0
    probe_z = 0.012

    sim = Simulation(domain=domain, dx=dx, freq_max=8.0e9)
    sim.add_floquet_port(
        0.006,
        axis="z",
        scan_theta=theta_deg,
        scan_phi=phi_deg,
        polarization="te",
        f0=4.0e9,
        bandwidth=0.8,
        amplitude=1.0,
    )
    sim.add_dft_plane_probe(axis="z", coordinate=probe_z, component="ex", freqs=freqs_hz, name="ex_plane")
    sim.add_dft_plane_probe(axis="z", coordinate=probe_z, component="hy", freqs=freqs_hz, name="hy_plane")
    result = sim.run(n_steps=n_steps)

    if result.dft_planes is None:
        raise RuntimeError("Floquet diagnostic produced no DFT planes")
    ex_dft = np.asarray(result.dft_planes["ex_plane"].accumulator, dtype=np.complex64)
    hy_dft = np.asarray(result.dft_planes["hy_plane"].accumulator, dtype=np.complex64)
    zeros = jnp.zeros_like(jnp.asarray(ex_dft))
    acc = FloquetDFTAccumulator(
        e_tang1_dft=jnp.asarray(ex_dft),
        e_tang2_dft=zeros,
        h_tang1_dft=zeros,
        h_tang2_dft=jnp.asarray(hy_dft),
    )
    helper = extract_floquet_modes(
        acc,
        dx=dx,
        Lx=domain[0],
        Ly=domain[1],
        freqs=jnp.asarray(freqs_hz, dtype=jnp.float32),
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        n_modes=1,
    )
    helper_s = np.asarray(helper["S"][0, :], dtype=np.complex64)
    helper_forward = np.asarray(helper["forward_amplitude"], dtype=np.complex64)
    helper_backward = np.asarray(helper["backward_amplitude"], dtype=np.complex64)

    np.savez_compressed(
        dump_path,
        schema_version=np.asarray(1, dtype=np.int32),
        claim_scope=np.asarray("floquet_modal_diagnostic_no_promoted_api"),
        ex_dft=ex_dft,
        hy_dft=hy_dft,
        freqs_hz=freqs_hz,
        theta_deg=np.asarray(theta_deg, dtype=np.float64),
        phi_deg=np.asarray(phi_deg, dtype=np.float64),
        dx_m=np.asarray(dx, dtype=np.float64),
        domain_m=np.asarray(domain, dtype=np.float64),
        probe_z_m=np.asarray(probe_z, dtype=np.float64),
        dt_s=np.asarray(result.dt, dtype=np.float64),
        n_steps=np.asarray(n_steps, dtype=np.int32),
        grid_shape=np.asarray(result.grid.shape, dtype=np.int32),
        helper_s=helper_s,
        helper_forward=helper_forward,
        helper_backward=helper_backward,
    )

    replay_script = Path(__file__).with_name("replay_floquet_modal_field_dump.py")
    completed = subprocess.run(
        [sys.executable, str(replay_script), str(dump_path), "--output-json", str(replay_json)],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Floquet modal replay failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    replay = json.loads(replay_json.read_text(encoding="utf-8"))
    summary = {
        "status": replay["status"],
        "dump_path": str(dump_path),
        "replay_json": str(replay_json),
        "n_steps": int(n_steps),
        "grid_shape": list(result.grid.shape),
        "freqs_hz": freqs_hz.tolist(),
        "max_s_abs_diff": replay["max_s_abs_diff"],
        "max_abs_ex_dft": replay["max_abs_ex_dft"],
        "max_abs_hy_dft": replay["max_abs_hy_dft"],
        "claim_scope": replay["claim_scope"],
    }
    summary_json = output_dir / "floquet_modal_field_dump_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {dump_path}")
    print(f"wrote {replay_json}")
    print(f"wrote {summary_json}")
    print(f"status={summary['status']} max_s_abs_diff={summary['max_s_abs_diff']:.6g}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-steps", type=int, default=120)
    args = parser.parse_args(argv)
    summary = generate_floquet_modal_field_dump(args.output_dir, n_steps=args.n_steps)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
