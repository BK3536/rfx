#!/usr/bin/env python3
"""Emit a VESSL parallel execution plan for the lumped/openEMS sweep.

This is orchestration infrastructure for slow M47-style external comparisons.
It writes one case-specific VESSL Run YAML per default lumped/openEMS case and
an aggregation command that combines the per-case JSON artifacts without
rerunning openEMS locally.  The plan is not physics evidence by itself and must
not be counted as E4/E5 until the generated jobs actually run and their case
artifacts are aggregated.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from build_lumped_openems_sweep_comparison import DEFAULT_SWEEP_CASES


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ".omx/physics-gate/vessl-2026-05-10-lumped-openems-parallel"
DEFAULT_CLUSTER = "remilab-c0"
DEFAULT_PRESET = "gpu-rtx4090"
DEFAULT_IMAGE = "nvcr.io/nvidia/jax:24.10-py3"


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _yaml_text(
    *,
    case: dict[str, str],
    output_root: str,
    cluster: str,
    preset: str,
    image: str,
) -> str:
    case_name = case["case_name"]
    return f"""name: rfx-lumped-openems-{case_name.replace('_', '-')}
description: "Parallel case shard for M47 lumped/openEMS PEC-box sweep: {case_name}"
tags: [rfx, physics-gate, port-external, lumped, openems, sparameter-validation]
resources:
  cluster: {cluster}
  preset: {preset}
image: {image}
env:
  PYTHONUNBUFFERED: "1"
  XLA_PYTHON_CLIENT_PREALLOCATE: "false"
  HDF5_USE_FILE_LOCKING: "FALSE"
  LANG: "C.UTF-8"
mount:
  /root/workspace/: volume://remilab-fs/personal-workspaces/
run: |-
  set -euo pipefail
  cd /root/workspace/byungkwan-workspace/research/rfx
  git config --global --add safe.directory /root/workspace/byungkwan-workspace/research/rfx || true
  echo "=== rfx lumped/openEMS parallel case: {case_name} ==="
  git status --short || true
  echo "=== install openEMS runtime dependencies ==="
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq openems python3-openems
  echo "=== make repo + apt openEMS bindings importable without editable install ==="
  # The VESSL JAX image has previously failed `pip install -e .[dev]` because
  # its build frontend/backend combination did not expose PEP-660 editable
  # installs.  Run directly from the mounted repository instead, and append the
  # apt-provided openEMS bindings after normal site-packages so Ubuntu's older
  # NumPy does not shadow the JAX image runtime.
  python -m pip install -q "scipy>=1.11" "h5py>=3.8" "matplotlib>=3.7"
  export RFX_REPO_ROOT="/root/workspace/byungkwan-workspace/research/rfx"
  export RFX_SYSTEM_SITE_DIR="/usr/lib/python3/dist-packages"
  export RFX_SITECUSTOMIZE_DIR="/tmp/rfx-openems-sitecustomize"
  mkdir -p "$RFX_SITECUSTOMIZE_DIR"
  python -c "from pathlib import Path; import os; Path(os.environ['RFX_SITECUSTOMIZE_DIR'], 'sitecustomize.py').write_text('import os\\nimport sys\\n\\nsystem_site = os.environ.get(\\\"RFX_SYSTEM_SITE_DIR\\\")\\nif system_site and system_site not in sys.path:\\n    sys.path.append(system_site)\\n')"
  export PYTHONPATH="$RFX_SITECUSTOMIZE_DIR:$RFX_REPO_ROOT:${{PYTHONPATH:-}}"
  python -c "import numpy, rfx; from CSXCAD.CSXCAD import ContinuousStructure; from openEMS.openEMS import openEMS; print('dependency_probe=passed', 'numpy', numpy.__version__, 'rfx', getattr(rfx, '__version__', 'unknown')); _ = (ContinuousStructure, openEMS)"
  output_root="${{RFX_PORT_EXTERNAL_OUTPUT_ROOT:-{output_root}}}"
  out="$output_root/lumped_openems_sweep_cases/{case_name}"
  mkdir -p "$out"
  python scripts/diagnostics/check_external_solver_dependencies.py \
    --output-dir "$out/dependencies"
  python scripts/diagnostics/build_lumped_openems_sparameter_comparison.py \
    --output-dir "$out" \
    --case-name "{case_name}" \
    --port1-pos-m "{case['port1_pos_m']}" \
    --port2-pos-m "{case['port2_pos_m']}"
"""


def build_lumped_openems_parallel_plan(
    *,
    output_dir: Path,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    cluster: str = DEFAULT_CLUSTER,
    preset: str = DEFAULT_PRESET,
    image: str = DEFAULT_IMAGE,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_dir = output_dir / "vessl_lumped_openems_cases"
    yaml_dir.mkdir(parents=True, exist_ok=True)

    case_rows: list[dict[str, Any]] = []
    for case in DEFAULT_SWEEP_CASES:
        case_name = case["case_name"]
        yaml_path = yaml_dir / f"lumped_openems_{case_name}.yaml"
        yaml_path.write_text(
            _yaml_text(
                case=case,
                output_root=output_root,
                cluster=cluster,
                preset=preset,
                image=image,
            ),
            encoding="utf-8",
        )
        case_artifact = (
            f"{output_root}/lumped_openems_sweep_cases/{case_name}/"
            "lumped_openems_generic_sparameter_comparison.json"
        )
        case_rows.append(
            {
                "case_name": case_name,
                "port1_pos_m": case["port1_pos_m"],
                "port2_pos_m": case["port2_pos_m"],
                "yaml_path": _display(yaml_path),
                "launch_command": f"vessl run create -f {_display(yaml_path)}",
                "expected_case_artifact": case_artifact,
            }
        )

    aggregation_command = (
        "python scripts/diagnostics/build_lumped_openems_sweep_comparison.py "
        f"--case-artifact-root {output_root}/lumped_openems_sweep_cases "
        f"--output-dir {output_root}/lumped_openems_sweep_aggregate"
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "evidence_level": "orchestration-only",
        "claim_scope": (
            "VESSL parallelization plan for lumped/openEMS slow sweep cases; "
            "not physics evidence, not E4/E5, and not proof the jobs ran"
        ),
        "case_count": len(case_rows),
        "output_root": output_root,
        "cluster": cluster,
        "preset": preset,
        "image": image,
        "cases": case_rows,
        "aggregation_command": aggregation_command,
        "completion_decision": (
            "Do not claim broad E5 from this plan. Launch the case YAMLs, collect "
            "their JSON artifacts, run the aggregation command, then rerun the "
            "external-reference and RF-infra E5 audits."
        ),
    }
    json_path = output_dir / "lumped_openems_parallel_plan.json"
    md_path = output_dir / "lumped_openems_parallel_plan.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Lumped/openEMS parallel VESSL plan",
        "",
        f"- status: `{payload['status']}`",
        f"- evidence_level: `{payload['evidence_level']}`",
        f"- case_count: `{payload['case_count']}`",
        f"- output_root: `{payload['output_root']}`",
        "",
        payload["claim_scope"],
        "",
        "## Launch commands",
        "",
    ]
    for row in case_rows:
        lines.append(f"- `{row['launch_command']}`")
    lines.extend(
        [
            "",
            "## Aggregate after all case jobs finish",
            "",
            "```bash",
            aggregation_command,
            "```",
            "",
            f"Decision: {payload['completion_decision']}",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=".omx/physics-gate/latest-lumped-openems-parallel-plan",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cluster", default=DEFAULT_CLUSTER)
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    args = parser.parse_args(argv)
    payload = build_lumped_openems_parallel_plan(
        output_dir=_repo_path(args.output_dir),
        output_root=args.output_root,
        cluster=args.cluster,
        preset=args.preset,
        image=args.image,
    )
    print(
        "status={status} case_count={case_count} output_root={output_root}".format(
            **payload
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
