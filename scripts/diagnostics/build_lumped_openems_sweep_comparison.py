#!/usr/bin/env python3
"""Build a multi-case lumped/openEMS comparison sweep artifact.

This wrapper runs the existing single-case lumped/openEMS comparator in fresh
Python subprocesses for several small PEC-box geometries.  The subprocess
boundary is intentional: openEMS Python bindings are not robust to repeated
in-process setup/teardown in this environment.

The result is broader E4-enabling evidence than the single M33 case, but it is
still a narrow PEC-box infrastructure sweep, not broad calibrated lumped-port
E5 over matched/open/short/load families.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from compare_sparameter_reference import _repo_path


SINGLE_CASE_SCRIPT = Path(__file__).with_name(
    "build_lumped_openems_sparameter_comparison.py"
)
DEFAULT_SWEEP_CASES = (
    {
        "case_name": "coarse_two_port_pec_box",
        "port1_pos_m": "0.010,0.010,0.005",
        "port2_pos_m": "0.020,0.010,0.005",
    },
    {
        "case_name": "yshift_two_port_pec_box",
        "port1_pos_m": "0.010,0.005,0.005",
        "port2_pos_m": "0.020,0.015,0.005",
    },
    {
        "case_name": "wide_two_port_pec_box",
        "port1_pos_m": "0.005,0.010,0.005",
        "port2_pos_m": "0.025,0.010,0.005",
    },
)


def _case_by_name(cases: tuple[dict[str, str], ...]) -> dict[str, dict[str, str]]:
    return {case["case_name"]: case for case in cases}


def select_sweep_cases(
    *,
    only_case: str | None = None,
    cases: tuple[dict[str, str], ...] = DEFAULT_SWEEP_CASES,
) -> tuple[dict[str, str], ...]:
    """Return all sweep cases or one named case for parallel shard execution."""

    if only_case is None:
        return cases
    by_name = _case_by_name(cases)
    if only_case not in by_name:
        valid = ", ".join(sorted(by_name))
        raise ValueError(f"unknown lumped/openEMS sweep case {only_case!r}; valid: {valid}")
    return (by_name[only_case],)


def single_case_command(case: dict[str, str], output_dir: Path) -> list[str]:
    """Command that runs one M47 lumped/openEMS case without the sweep wrapper."""

    return [
        sys.executable,
        str(SINGLE_CASE_SCRIPT),
        "--output-dir",
        str(output_dir),
        "--case-name",
        case["case_name"],
        "--port1-pos-m",
        case["port1_pos_m"],
        "--port2-pos-m",
        case["port2_pos_m"],
    ]


def _summary_metric(payload: dict[str, Any], *names: str) -> float:
    summary = payload["summary"]
    for name in names:
        if name in summary:
            return float(summary[name])
    raise KeyError(f"summary lacks any of {names!r}")


def summarize_lumped_openems_sweep(
    case_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    passed_cases = [payload for payload in case_payloads if payload.get("status") == "passed"]
    max_case_max_mag_abs_diff = max(
        _summary_metric(payload, "max_mag_abs_diff") for payload in case_payloads
    )
    max_case_mean_mag_abs_diff = max(
        _summary_metric(
            payload,
            "mean_mag_abs_diff",
            "mean_mag_abs_diff_max_over_terms",
        )
        for payload in case_payloads
    )
    status = "passed" if len(passed_cases) == len(case_payloads) else "failed"
    return {
        "status": status,
        "evidence_level": "E4-enabling",
        "claim": "multi-case lumped-port S11/S21 magnitude comparison against openEMS",
        "claim_scope": (
            "narrow three-case PEC-box sweep using rfx LumpedPort and openEMS "
            "AddLumpedPort at matched 50-ohm ports; broader than the single M33 "
            "fixture but still not broad calibrated lumped-port E5 over "
            "matched/open/short/load cases"
        ),
        "case_count": len(case_payloads),
        "passed_case_count": len(passed_cases),
        "max_case_max_mag_abs_diff": max_case_max_mag_abs_diff,
        "max_case_mean_mag_abs_diff": max_case_mean_mag_abs_diff,
        "case_payloads": case_payloads,
        "completion_decision": (
            "Do not claim broad lumped-port E5 from this artifact. It is a "
            "multi-fixture E4-enabling PEC-box sweep only."
        ),
    }


def _run_single_case(output_dir: Path, case: dict[str, str]) -> dict[str, Any]:
    case_dir = output_dir / case["case_name"]
    command = single_case_command(case, case_dir)
    completed = subprocess.run(
        command,
        cwd=_repo_path("."),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    json_path = case_dir / "lumped_openems_generic_sparameter_comparison.json"
    payload: dict[str, Any]
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "status": "failed",
            "summary": {
                "max_mag_abs_diff": float("inf"),
                "mean_mag_abs_diff": float("inf"),
            },
        }
    payload["case_name"] = case["case_name"]
    payload["comparison_artifact"] = str(json_path)
    payload["subprocess"] = {
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "command": command,
    }
    if completed.returncode != 0 and payload.get("status") == "passed":
        payload["status"] = "failed"
        payload["subprocess_error"] = "single-case comparator returned nonzero"
    return payload


def _case_payload_from_artifact(case: dict[str, str], artifact_root: Path) -> dict[str, Any]:
    json_path = artifact_root / case["case_name"] / "lumped_openems_generic_sparameter_comparison.json"
    if not json_path.exists():
        return {
            "status": "failed",
            "summary": {
                "max_mag_abs_diff": float("inf"),
                "mean_mag_abs_diff": float("inf"),
            },
            "case_name": case["case_name"],
            "comparison_artifact": str(json_path),
            "artifact_error": "missing parallel case artifact",
        }
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["case_name"] = case["case_name"]
    payload["comparison_artifact"] = str(json_path)
    payload["parallel_artifact_replay"] = True
    return payload


def build_lumped_openems_sweep_comparison_from_artifacts(
    *,
    case_artifact_root: Path,
    output_dir: Path,
    cases: tuple[dict[str, str], ...] = DEFAULT_SWEEP_CASES,
) -> dict[str, Any]:
    """Aggregate per-case artifacts produced by parallel VESSL jobs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    case_payloads = [
        _case_payload_from_artifact(case, case_artifact_root)
        for case in cases
    ]
    payload = summarize_lumped_openems_sweep(case_payloads)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["case_artifact_root"] = str(case_artifact_root)
    payload["child_comparison_artifacts"] = [
        case_payload["comparison_artifact"] for case_payload in case_payloads
    ]
    payload["execution_mode"] = "parallel_artifact_aggregation"
    output_json = output_dir / "lumped_openems_sweep_sparameter_comparison.json"
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_lumped_openems_sweep_comparison(
    output_dir: Path,
    *,
    cases: tuple[dict[str, str], ...] = DEFAULT_SWEEP_CASES,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_payloads = [_run_single_case(output_dir, case) for case in cases]
    payload = summarize_lumped_openems_sweep(case_payloads)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["child_comparison_artifacts"] = [
        case_payload["comparison_artifact"] for case_payload in case_payloads
    ]
    payload["execution_mode"] = "serial_subprocess_sweep"
    output_json = output_dir / "lumped_openems_sweep_sparameter_comparison.json"
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=".omx/physics-gate/latest-lumped-openems-sweep-comparison",
    )
    parser.add_argument(
        "--only-case",
        help=(
            "Run only one named default sweep case. Useful for VESSL parallel "
            "case shards; aggregate later with --case-artifact-root."
        ),
    )
    parser.add_argument(
        "--case-artifact-root",
        help=(
            "Aggregate existing per-case artifacts from this root instead of "
            "running openEMS locally. Expected layout: ROOT/<case_name>/"
            "lumped_openems_generic_sparameter_comparison.json."
        ),
    )
    args = parser.parse_args(argv)

    selected_cases = select_sweep_cases(only_case=args.only_case)
    if args.case_artifact_root:
        payload = build_lumped_openems_sweep_comparison_from_artifacts(
            case_artifact_root=_repo_path(args.case_artifact_root),
            output_dir=_repo_path(args.output_dir),
            cases=selected_cases,
        )
    else:
        payload = build_lumped_openems_sweep_comparison(
            _repo_path(args.output_dir),
            cases=selected_cases,
        )
    print(
        "status={status} passed_case_count={passed_case_count}/{case_count} "
        "max_case_max_mag_abs_diff={max_case_max_mag_abs_diff:.6g}".format(**payload)
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
