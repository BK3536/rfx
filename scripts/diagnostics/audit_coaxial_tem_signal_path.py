#!/usr/bin/env python3
"""Audit structural blockers for a coaxial TEM reference-plane signal path.

M63/M64 proved that the current real-FDTD DFT-plane replay path can capture
Cartesian planes but does not see usable transverse TEM V/I.  This audit checks
the helper geometry/source semantics that must be fixed before a calibrated
coaxial TEM S-parameter lane can be promoted:

* the current source component is axial/normal to the coax reference plane,
  while TEM V/I extraction needs transverse E/H fields on that plane;
* the default material stamp currently has no PEC annular outer-conductor
  shell at the nominal outer radius in the sampled line section.

The report is a blocker diagnostic only, not E4/E5 physics evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rfx.core.yee import init_materials
from rfx.grid import Grid
from rfx.sources.coaxial_port import (
    PEC_SIGMA,
    SMA_OUTER_RADIUS,
    SMA_PIN_RADIUS,
    CoaxialPort,
    _coaxial_port_geometry,
    setup_coaxial_port,
)
from rfx.sources.sources import GaussianPulse


AXIS_TO_NORMAL_COMPONENT = {"x": "ex", "y": "ey", "z": "ez"}
AXIS_TO_TANGENTIAL_E = {
    "x": ("ey", "ez"),
    "y": ("ez", "ex"),
    "z": ("ex", "ey"),
}
AXIS_TO_TANGENTIAL_H = {
    "x": ("hy", "hz"),
    "y": ("hz", "hx"),
    "z": ("hx", "hy"),
}


def _default_port() -> CoaxialPort:
    return CoaxialPort(
        position=(0.010, 0.010, 0.015),
        face="top",
        pin_length=5e-3,
        pin_radius=SMA_PIN_RADIUS,
        outer_radius=SMA_OUTER_RADIUS,
        impedance=50.0,
        excitation=GaussianPulse(f0=5.0e9, bandwidth=0.8, amplitude=1.0),
    )


def _plane_radius_grid(grid: Grid, *, center_xy: tuple[float, float]) -> np.ndarray:
    x = (np.arange(grid.nx, dtype=np.float64) - grid.pad_x_lo) * grid.dx
    y = (np.arange(grid.ny, dtype=np.float64) - grid.pad_y_lo) * grid.dx
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.hypot(xx - center_xy[0], yy - center_xy[1])


def audit_default_coaxial_tem_signal_path() -> dict:
    grid = Grid(
        freq_max=10.0e9,
        domain=(0.020, 0.020, 0.020),
        dx=0.5e-3,
        cpml_layers=0,
    )
    base_materials = init_materials(grid.shape)
    port = _default_port()
    stamped = setup_coaxial_port(grid, port, base_materials)
    axis, _direction, component, pin_center, pin_tip, gap_idx = _coaxial_port_geometry(
        grid,
        port,
    )
    normal_component = AXIS_TO_NORMAL_COMPONENT[axis]
    source_is_normal = component == normal_component

    # Inspect a plane through the middle of the current coax helper line.
    # For the default top port this is a z-normal plane and the coax center is
    # at (x, y) = port.position[:2].  Cells close to the nominal outer radius
    # should contain PEC if an annular outer-conductor shell is stamped.
    mid_idx = grid.position_to_index(pin_center)
    radius = _plane_radius_grid(grid, center_xy=(port.position[0], port.position[1]))
    shell_band = np.abs(radius - port.outer_radius) <= 0.51 * grid.dx
    dielectric_annulus = (radius > port.pin_radius + 0.51 * grid.dx) & (
        radius < port.outer_radius - 0.51 * grid.dx
    )
    sigma_plane = np.asarray(stamped.sigma[:, :, mid_idx[2]], dtype=np.float64)
    eps_plane = np.asarray(stamped.eps_r[:, :, mid_idx[2]], dtype=np.float64)
    shell_pec_count = int(np.count_nonzero(shell_band & (sigma_plane >= 0.5 * PEC_SIGMA)))
    shell_cell_count = int(np.count_nonzero(shell_band))
    dielectric_count = int(np.count_nonzero(dielectric_annulus & (eps_plane > 1.0)))

    checks = [
        {
            "name": "source_component_is_transverse_to_tem_reference_plane",
            "status": "failed" if source_is_normal else "passed",
            "axis": axis,
            "source_component": component,
            "normal_component": normal_component,
            "expected_tangential_e_components": list(AXIS_TO_TANGENTIAL_E[axis]),
            "expected_tangential_h_components": list(AXIS_TO_TANGENTIAL_H[axis]),
            "message": (
                "current source component is normal/axial to the TEM reference plane"
                if source_is_normal
                else "source component is not normal to the TEM reference plane"
            ),
        },
        {
            "name": "outer_conductor_shell_has_pec_cells",
            "status": "passed" if shell_pec_count > 0 else "failed",
            "shell_pec_cell_count": shell_pec_count,
            "shell_band_cell_count": shell_cell_count,
            "message": (
                "no PEC annular shell cells found near the nominal outer radius"
                if shell_pec_count == 0
                else "PEC shell cells found near the nominal outer radius"
            ),
        },
        {
            "name": "dielectric_annulus_is_present",
            "status": "passed" if dielectric_count > 0 else "failed",
            "dielectric_annulus_cell_count": dielectric_count,
        },
    ]
    blockers = [
        f"{check['name']}: {check.get('message', 'failed')}"
        for check in checks
        if check["status"] != "passed"
    ]
    status = "blocked" if blockers else "passed"
    return {
        "status": status,
        "evidence_level": (
            "structural-diagnostic"
            if status == "passed"
            else "structural-diagnostic-blocked"
        ),
        "claim_scope": (
            "coaxial TEM signal-path structural audit only; no FDTD S-parameter "
            "accuracy claim, no external solver comparison, and not broad E5"
        ),
        "family": "coaxial_port",
        "diagnostic": "coaxial_tem_signal_path_audit",
        "grid_shape": list(grid.shape),
        "dx_m": float(grid.dx),
        "port_face": port.face,
        "axis": axis,
        "source_component": component,
        "gap_index": list(gap_idx),
        "pin_center_m": list(pin_center),
        "pin_tip_m": list(pin_tip),
        "inspection_plane_index": int(mid_idx[2]),
        "checks": checks,
        "blockers": blockers,
        "next_required_fix": (
            "replace or augment the current probe-style axial source/material "
            "stamp with a TEM-capable coaxial feed path: annular outer conductor "
            "boundary plus transverse/radial excitation and calibrated V/I plane"
        ),
    }


def _write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# Coaxial TEM signal-path structural audit",
        "",
        f"- status: `{payload['status']}`",
        f"- evidence_level: `{payload['evidence_level']}`",
        f"- claim_scope: {payload['claim_scope']}",
        f"- next_required_fix: {payload['next_required_fix']}",
        "",
        "## Checks",
        "",
        "| Check | Status | Message |",
        "|---|---:|---|",
    ]
    for check in payload["checks"]:
        lines.append(
            f"| `{check['name']}` | `{check['status']}` | "
            f"{check.get('message', '')} |"
        )
    lines.extend(["", "## Blockers", ""])
    if payload["blockers"]:
        lines.extend(f"- {item}" for item in payload["blockers"])
    else:
        lines.append("- none for this structural diagnostic")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = audit_default_coaxial_tem_signal_path()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "coaxial_tem_signal_path_audit.json"
    md_path = args.output_dir / "coaxial_tem_signal_path_audit.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_markdown(payload, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"status={payload['status']} blocker_count={len(payload['blockers'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
