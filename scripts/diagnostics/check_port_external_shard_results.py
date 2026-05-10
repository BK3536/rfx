#!/usr/bin/env python3
"""Aggregate broad-E5 port external-reference shard result JSON files.

The broad-E5 manifest can prove that every port family has a launchable shard,
but YAML presence is not execution evidence.  This checker inspects the JSON
reports emitted by ``report_port_external_reference_shard.py`` and keeps the
all-port E5 gate blocked until every required family has a present, valid, and
``status=passed`` result.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from check_port_external_references import (
    DEFAULT_SUPPORT_MATRIX,
    _display,
    _repo_path,
    build_external_reference_audit,
)


DEFAULT_RESULT_ROOT = ".omx/physics-gate/vessl-2026-05-09-port-external"


def _read_json_if_present(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as exc:
        return None, f"invalid_json: {exc}"


def _result_path_for(
    row: dict[str, Any],
    result_root: Path,
    directory_layout: str,
) -> Path:
    if directory_layout == "family":
        directory = row["family"]
    elif directory_layout == "shard_id":
        directory = row["recommended_vessl_shard_id"]
    else:  # pragma: no cover - argparse choices should prevent this.
        raise ValueError(f"unsupported directory layout: {directory_layout!r}")
    return result_root / directory / f"{row['family']}_external_reference_shard.json"


def _evaluate_one_result(
    row: dict[str, Any],
    result_root: Path,
    directory_layout: str,
) -> dict[str, Any]:
    path = _result_path_for(row, result_root, directory_layout)
    payload, read_error = _read_json_if_present(path)
    blockers: list[str] = []
    if read_error:
        blockers.append(read_error)
        result_status = "missing" if read_error == "missing" else "invalid"
        payload_status = ""
    else:
        assert payload is not None
        payload_status = str(payload.get("status", ""))
        if payload.get("family") != row["family"]:
            blockers.append(
                f"result family is {payload.get('family')!r}, expected {row['family']!r}"
            )
        if payload.get("recommended_vessl_shard_id") != row["recommended_vessl_shard_id"]:
            blockers.append(
                "result recommended_vessl_shard_id does not match the manifest"
            )
        if payload_status != "passed":
            blockers.append(f"result status is {payload_status!r}, not 'passed'")
        blockers.extend(str(item) for item in payload.get("blockers", []))
        result_status = "passed" if payload_status == "passed" and not blockers else "blocked"
        if any(blocker.startswith("result family") for blocker in blockers):
            result_status = "invalid"
        if any(
            blocker == "result recommended_vessl_shard_id does not match the manifest"
            for blocker in blockers
        ):
            result_status = "invalid"

    return {
        "family": row["family"],
        "recommended_vessl_shard_id": row["recommended_vessl_shard_id"],
        "result_json": _display(path),
        "exists": path.exists(),
        "payload_status": payload_status,
        "status": result_status,
        "blockers": blockers,
    }


def build_shard_result_audit(
    manifest_path: Path,
    support_matrix_path: Path,
    result_root: Path,
    directory_layout: str = "shard_id",
) -> dict[str, Any]:
    reference_audit = build_external_reference_audit(manifest_path, support_matrix_path)
    required_rows = [
        row for row in reference_audit["requirements"] if row["required_for_e5"]
    ]
    results = [
        _evaluate_one_result(row, result_root, directory_layout)
        for row in required_rows
    ]
    missing = [row for row in results if row["status"] == "missing"]
    invalid = [row for row in results if row["status"] == "invalid"]
    blocked = [row for row in results if row["status"] == "blocked"]
    passed = [row for row in results if row["status"] == "passed"]
    result_coverage_status = "passed" if not missing and not invalid else "failed"
    status = (
        "passed"
        if result_coverage_status == "passed" and len(passed) == len(required_rows)
        else "blocked"
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": _display(manifest_path),
        "support_matrix": _display(support_matrix_path),
        "result_root": _display(result_root),
        "directory_layout": directory_layout,
        "reference_manifest_status": reference_audit["status"],
        "result_coverage_status": result_coverage_status,
        "status": status,
        "required_family_count": len(required_rows),
        "present_result_count": sum(1 for row in results if row["exists"]),
        "passed_family_count": len(passed),
        "blocked_family_count": len(blocked),
        "missing_result_count": len(missing),
        "invalid_result_count": len(invalid),
        "results": results,
        "incomplete": [row for row in results if row["status"] != "passed"],
        "completion_decision": (
            "Do not call update_goal: broad-E5 external-reference shard results "
            "are missing, invalid, or blocked."
            if status != "passed"
            else "All required port external-reference shard results passed; "
            "still run the full RF-infra E5 audit before update_goal."
        ),
    }


def _write_report(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "port_external_shard_result_audit.json"
    md_path = output_dir / "port_external_shard_result_audit.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Port external-reference shard result audit",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- status: `{payload['status']}`",
        f"- reference_manifest_status: `{payload['reference_manifest_status']}`",
        f"- result_coverage_status: `{payload['result_coverage_status']}`",
        f"- result_root: `{payload['result_root']}`",
        f"- directory_layout: `{payload['directory_layout']}`",
        f"- required_family_count: `{payload['required_family_count']}`",
        f"- present_result_count: `{payload['present_result_count']}`",
        f"- passed_family_count: `{payload['passed_family_count']}`",
        f"- blocked_family_count: `{payload['blocked_family_count']}`",
        f"- missing_result_count: `{payload['missing_result_count']}`",
        f"- invalid_result_count: `{payload['invalid_result_count']}`",
        "",
        "| Family | Status | Payload status | Result JSON |",
        "|---|---:|---:|---|",
    ]
    for row in payload["results"]:
        lines.append(
            f"| `{row['family']}` | `{row['status']}` | "
            f"`{row['payload_status']}` | `{row['result_json']}` |"
        )
    if payload["incomplete"]:
        lines.extend(["", "## Incomplete shard results", ""])
        for row in payload["incomplete"]:
            lines.append(f"### `{row['family']}`")
            for blocker in row["blockers"]:
                lines.append(f"- {blocker}")
            lines.append("")
    lines.append(f"\nDecision: {payload['completion_decision']}\n")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {_display(json_path)}")
    print(f"wrote {_display(md_path)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="scripts/diagnostics/port_external_reference_requirements.json",
    )
    parser.add_argument("--support-matrix", default=DEFAULT_SUPPORT_MATRIX)
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--directory-layout",
        choices=["shard_id", "family"],
        default="shard_id",
        help="Whether result directories are named by recommended shard ID or family.",
    )
    parser.add_argument(
        "--output-dir",
        default=".omx/physics-gate/latest-port-external-shard-result-audit",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit 2 unless every required port external shard result passed.",
    )
    args = parser.parse_args(argv)

    payload = build_shard_result_audit(
        _repo_path(args.manifest),
        _repo_path(args.support_matrix),
        _repo_path(args.result_root),
        args.directory_layout,
    )
    _write_report(payload, _repo_path(args.output_dir))
    print(
        "status={status} passed_family_count={passed} incomplete_count={incomplete}".format(
            status=payload["status"],
            passed=payload["passed_family_count"],
            incomplete=len(payload["incomplete"]),
        )
    )
    if payload["result_coverage_status"] != "passed":
        return 1
    if args.require_complete and payload["status"] != "passed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
