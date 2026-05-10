#!/usr/bin/env python3
"""Prototype a TEM-capable coaxial plane source scaffold.

M66 fixed the annular outer-shell material stamp but the existing public
``add_coaxial_port`` helper still injects an axial/normal field component. This
diagnostic does **not** replace that public helper.  It builds an explicit
distributed transverse E/M source on one coaxial cross-section to prove the
remaining source-side blocker is mechanically addressable before attempting a
promoted coaxial port API.

The result is an internal prototype only: same-code FDTD + same-code replay,
no external solver, no matched/open/short/load fixture, and not broad E5.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.yee import init_materials
from rfx.grid import Grid
from rfx.probes.probes import init_dft_plane_probe
from rfx.simulation import MagneticSourceSpec, SourceSpec, run
from rfx.sources.coaxial_port import (
    PTFE_EPS_R,
    SMA_OUTER_RADIUS,
    SMA_PIN_RADIUS,
    CoaxialPort,
    _coaxial_port_geometry,
    coaxial_tem_characteristic_impedance,
    coaxial_tem_reference_plane_s11,
    coaxial_tem_reference_plane_vi_from_cartesian_plane,
    setup_coaxial_port,
)
from rfx.sources.sources import GaussianPulse


FREQS_HZ = np.asarray([3.0e9, 5.0e9, 7.0e9], dtype=np.float64)
SIGNAL_FLOOR = 1e-12
MAX_INTERNAL_S11_ABS = 0.5


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def classify_tem_plane_source_prototype(
    *,
    s11,
    max_abs_voltage: float,
    max_abs_current: float,
    signal_floor: float = SIGNAL_FLOOR,
    max_internal_s11_abs: float = MAX_INTERNAL_S11_ABS,
) -> dict:
    values = np.asarray(s11, dtype=np.complex128)
    max_abs_s11 = float(np.max(np.abs(values))) if values.size else float("nan")
    blockers: list[str] = []
    if not np.all(np.isfinite(values)):
        blockers.append("non-finite S11 values")
    if max_abs_voltage <= signal_floor:
        blockers.append(
            f"TEM prototype voltage below signal floor: {max_abs_voltage:.3e} <= {signal_floor:.3e}"
        )
    if max_abs_current <= signal_floor:
        blockers.append(
            f"TEM prototype current below signal floor: {max_abs_current:.3e} <= {signal_floor:.3e}"
        )
    if not np.isfinite(max_abs_s11) or max_abs_s11 > max_internal_s11_abs:
        blockers.append(
            f"prototype internal |S11| smoke limit failed: {max_abs_s11:.3e} > {max_internal_s11_abs:.3e}"
        )
    status = "passed" if not blockers else "blocked"
    return {
        "status": status,
        "evidence_level": (
            "E3-diagnostic-prototype"
            if status == "passed"
            else "E3-diagnostic-prototype-blocked"
        ),
        "claim_scope": (
            "distributed coaxial TEM plane-source prototype only; not the public "
            "add_coaxial_port API, not external full-wave evidence, and not broad E5"
        ),
        "max_abs_s11": max_abs_s11,
        "max_internal_s11_abs": float(max_internal_s11_abs),
        "signal_floor": float(signal_floor),
        "max_abs_voltage": float(max_abs_voltage),
        "max_abs_current": float(max_abs_current),
        "blockers": blockers,
    }


def _build_tem_plane_sources(
    *,
    grid: Grid,
    port: CoaxialPort,
    n_steps: int,
    plane_index: int,
    field_scale: float,
    magnetic_ratio: float,
) -> tuple[list[SourceSpec], list[MagneticSourceSpec], int]:
    times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    waveform = jax.vmap(port.excitation)(times)
    z0 = coaxial_tem_characteristic_impedance(
        SMA_PIN_RADIUS,
        SMA_OUTER_RADIUS,
        PTFE_EPS_R,
    )
    log_ratio = np.log(SMA_OUTER_RADIUS / SMA_PIN_RADIUS)
    electric_sources: list[SourceSpec] = []
    magnetic_sources: list[MagneticSourceSpec] = []
    source_cell_count = 0
    for i in range(grid.nx):
        x = (i - grid.pad_x_lo) * grid.dx
        for j in range(grid.ny):
            y = (j - grid.pad_y_lo) * grid.dx
            du = x - port.position[0]
            dv = y - port.position[1]
            radius = float(np.hypot(du, dv))
            if not (
                SMA_PIN_RADIUS + 0.25 * grid.dx
                <= radius
                <= SMA_OUTER_RADIUS - 0.25 * grid.dx
            ):
                continue
            cos_phi = du / radius
            sin_phi = dv / radius
            e_radial = field_scale / (radius * log_ratio)
            h_phi = (
                field_scale
                * magnetic_ratio
                * (1.0 / z0)
                / (2.0 * np.pi * radius)
            )
            source_cell_count += 1
            if abs(cos_phi) > 1e-12:
                electric_sources.append(
                    SourceSpec(
                        i=i,
                        j=j,
                        k=plane_index,
                        component="ex",
                        waveform=(e_radial * cos_phi * waveform).astype(jnp.float32),
                    )
                )
            if abs(sin_phi) > 1e-12:
                electric_sources.append(
                    SourceSpec(
                        i=i,
                        j=j,
                        k=plane_index,
                        component="ey",
                        waveform=(e_radial * sin_phi * waveform).astype(jnp.float32),
                    )
                )
            if abs(sin_phi) > 1e-12:
                magnetic_sources.append(
                    MagneticSourceSpec(
                        i=i,
                        j=j,
                        k=plane_index,
                        component="hx",
                        waveform=(-h_phi * sin_phi * waveform).astype(jnp.float32),
                    )
                )
            if abs(cos_phi) > 1e-12:
                magnetic_sources.append(
                    MagneticSourceSpec(
                        i=i,
                        j=j,
                        k=plane_index,
                        component="hy",
                        waveform=(h_phi * cos_phi * waveform).astype(jnp.float32),
                    )
                )
    return electric_sources, magnetic_sources, source_cell_count


def run_tem_plane_source_prototype(
    *,
    output_dir: Path,
    n_steps: int = 160,
    field_scale: float = 1.0e4,
    magnetic_ratio: float = 0.1,
) -> dict:
    grid = Grid(
        freq_max=10.0e9,
        domain=(0.020, 0.020, 0.020),
        dx=0.5e-3,
        cpml_layers=0,
    )
    port = CoaxialPort(
        position=(0.010, 0.010, 0.015),
        face="top",
        pin_length=5e-3,
        pin_radius=SMA_PIN_RADIUS,
        outer_radius=SMA_OUTER_RADIUS,
        impedance=50.0,
        excitation=GaussianPulse(f0=5.0e9, bandwidth=0.8, amplitude=1.0),
    )
    materials = setup_coaxial_port(grid, port, init_materials(grid.shape))
    axis, _direction, _component, pin_center, _pin_tip, _gap_idx = _coaxial_port_geometry(
        grid,
        port,
    )
    if axis != "z":
        raise ValueError("this prototype currently expects the default z-axis coaxial port")
    plane_index = grid.position_to_index(pin_center)[2]
    electric_sources, magnetic_sources, source_cell_count = _build_tem_plane_sources(
        grid=grid,
        port=port,
        n_steps=n_steps,
        plane_index=plane_index,
        field_scale=field_scale,
        magnetic_ratio=magnetic_ratio,
    )
    freqs = jnp.asarray(FREQS_HZ, dtype=jnp.float32)
    dft_planes = [
        init_dft_plane_probe(
            axis=2,
            index=plane_index,
            component=component,
            freqs=freqs,
            grid_shape=grid.shape,
            dft_total_steps=n_steps,
        )
        for component in ("ex", "ey", "hx", "hy")
    ]
    result = run(
        grid,
        materials,
        n_steps,
        boundary="pec",
        sources=electric_sources,
        mag_sources=magnetic_sources,
        dft_planes=dft_planes,
        return_state=False,
    )
    if result.dft_planes is None:
        raise RuntimeError("TEM plane-source prototype produced no DFT planes")
    plane_by_component = {
        probe.component: np.asarray(probe.accumulator, dtype=np.complex128)
        for probe in result.dft_planes
    }
    u_coords = (np.arange(grid.nx, dtype=np.float64) - grid.pad_x_lo) * grid.dx
    v_coords = (np.arange(grid.ny, dtype=np.float64) - grid.pad_y_lo) * grid.dx
    extracted = coaxial_tem_reference_plane_vi_from_cartesian_plane(
        u_coords,
        v_coords,
        plane_by_component["ex"],
        plane_by_component["ey"],
        plane_by_component["hx"],
        plane_by_component["hy"],
        center_u_m=port.position[0],
        center_v_m=port.position[1],
        inner_radius=SMA_PIN_RADIUS,
        outer_radius=SMA_OUTER_RADIUS,
        eps_r=PTFE_EPS_R,
    )
    z0 = coaxial_tem_characteristic_impedance(
        SMA_PIN_RADIUS,
        SMA_OUTER_RADIUS,
        PTFE_EPS_R,
    )
    s11 = np.asarray(
        coaxial_tem_reference_plane_s11(
            extracted.vi.voltage,
            extracted.vi.current,
            z0,
        ),
        dtype=np.complex128,
    )
    max_abs_voltage = float(np.max(np.abs(extracted.vi.voltage)))
    max_abs_current = float(np.max(np.abs(extracted.vi.current)))
    classification = classify_tem_plane_source_prototype(
        s11=s11,
        max_abs_voltage=max_abs_voltage,
        max_abs_current=max_abs_current,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_path = output_dir / "coaxial_tem_plane_source_prototype_dump.npz"
    np.savez_compressed(
        dump_path,
        freqs_hz=FREQS_HZ,
        plane_voltage=extracted.vi.voltage,
        plane_current=extracted.vi.current,
        plane_s11=s11,
        ex_dft=plane_by_component["ex"],
        ey_dft=plane_by_component["ey"],
        hx_dft=plane_by_component["hx"],
        hy_dft=plane_by_component["hy"],
    )
    rows = [
        {
            "freq_hz": float(freq),
            "s11": {
                "re": float(s11[idx].real),
                "im": float(s11[idx].imag),
                "abs": float(abs(s11[idx])),
            },
            "voltage_abs": float(abs(extracted.vi.voltage[idx])),
            "current_abs": float(abs(extracted.vi.current[idx])),
        }
        for idx, freq in enumerate(FREQS_HZ)
    ]
    return {
        **classification,
        "family": "coaxial_port",
        "diagnostic": "coaxial_tem_plane_source_prototype",
        "commit_hash": _git_commit(),
        "artifact": str(dump_path),
        "n_steps": int(n_steps),
        "field_scale": float(field_scale),
        "magnetic_ratio": float(magnetic_ratio),
        "source_cell_count": int(source_cell_count),
        "electric_source_count": len(electric_sources),
        "magnetic_source_count": len(magnetic_sources),
        "plane_index": int(plane_index),
        "plane_z_m": float(pin_center[2]),
        "tem_z0_ohm": float(z0),
        "freqs_hz": FREQS_HZ.tolist(),
        "max_abs_plane_field_dft": {
            name: float(np.max(np.abs(values)))
            for name, values in plane_by_component.items()
        },
        "rows": rows,
    }


def _write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# Coaxial TEM plane-source prototype",
        "",
        f"- status: `{payload['status']}`",
        f"- evidence_level: `{payload['evidence_level']}`",
        f"- claim_scope: {payload['claim_scope']}",
        f"- artifact: `{payload['artifact']}`",
        f"- source_cell_count: `{payload['source_cell_count']}`",
        f"- electric_source_count: `{payload['electric_source_count']}`",
        f"- magnetic_source_count: `{payload['magnetic_source_count']}`",
        f"- max_abs_s11: `{payload['max_abs_s11']:.6g}`",
        f"- max_abs_voltage: `{payload['max_abs_voltage']:.6g}`",
        f"- max_abs_current: `{payload['max_abs_current']:.6g}`",
        "",
        "## Frequency rows",
        "",
        "| f (Hz) | |S11| | |V| | |I| |",
        "|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| `{row['freq_hz']:.6g}` | `{row['s11']['abs']:.6g}` | "
            f"`{row['voltage_abs']:.6g}` | `{row['current_abs']:.6g}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-steps", type=int, default=160)
    parser.add_argument("--field-scale", type=float, default=1.0e4)
    parser.add_argument("--magnetic-ratio", type=float, default=0.1)
    args = parser.parse_args(argv)

    payload = run_tem_plane_source_prototype(
        output_dir=args.output_dir,
        n_steps=args.n_steps,
        field_scale=args.field_scale,
        magnetic_ratio=args.magnetic_ratio,
    )
    json_path = args.output_dir / "coaxial_tem_plane_source_prototype.json"
    md_path = args.output_dir / "coaxial_tem_plane_source_prototype.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_markdown(payload, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"status={payload['status']} max_abs_s11={payload['max_abs_s11']:.6g}")
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
