#!/usr/bin/env python3
"""Verify the rfx slow physics-gate VESSL shard manifest.

This is a lightweight discipline check: every slow/release group should have an
independent VESSL YAML, an output artifact location, a last verified run ID, and
(optional) result JSON that can be interpreted with the same coverage semantics
as `run_physics_gate.py`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_physics_gate import (
    REPO_ROOT,
    aggregate_coverage_status,
    coverage_metadata,
    gate_groups_by_id,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text_if_exists(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_result(shard: dict[str, Any]) -> dict[str, Any] | None:
    result_path = _repo_path(str(shard["result_json"]))
    if not result_path.exists():
        return None
    payload = _read_json(result_path)
    results = payload.get("results", [])
    if len(results) != 1:
        raise SystemExit(f"expected one result in {result_path}, found {len(results)}")
    result = dict(results[0])
    group = gate_groups_by_id()[str(shard["group_id"])]
    stdout = _read_text_if_exists(_repo_path(str(result.get("stdout_path", ""))))
    metadata = coverage_metadata(
        group,
        execution_status=str(result.get("status", "failed")),
        stdout=stdout,
        vessl_run_ids=[str(shard["last_verified_run_id"])],
    )
    result.update(metadata)
    result["result_json"] = _display(result_path)
    return result


def _check_shard(shard: dict[str, Any], *, verify_results: bool) -> dict[str, Any]:
    group_id = str(shard.get("group_id", ""))
    groups = gate_groups_by_id()
    errors: list[str] = []
    warnings: list[str] = []

    if group_id not in groups:
        errors.append("group_id is not registered in run_physics_gate.py")
    elif not group_id.startswith("slow_"):
        errors.append("VESSL shard manifest should contain only slow_* groups")

    yaml_path = _repo_path(str(shard.get("yaml_path", "")))
    yaml_text = _read_text_if_exists(yaml_path)
    if not yaml_text:
        errors.append("yaml_path is missing")
    else:
        required_snippets = [
            "scripts/diagnostics/run_physics_gate.py",
            f"--group {group_id}",
            "RFX_PHYSICS_GATE_OUTPUT_ROOT",
            "--vessl-run-id",
        ]
        for snippet in required_snippets:
            if snippet not in yaml_text:
                errors.append(f"YAML missing snippet: {snippet}")

    if not shard.get("last_verified_run_id"):
        errors.append("last_verified_run_id is missing")

    result = None
    if verify_results:
        result = _load_result(shard)
        if result is None:
            errors.append("result_json is missing")
        elif result.get("status") != "passed":
            errors.append(f"result status is {result.get('status')}")

    return {
        "group_id": group_id,
        "yaml_path": str(shard.get("yaml_path", "")),
        "result_json": str(shard.get("result_json", "")),
        "last_verified_run_id": str(shard.get("last_verified_run_id", "")),
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "result": result,
    }


def _write_report(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "vessl_physics_shard_check.json"
    md_path = output_dir / "vessl_physics_shard_check.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# VESSL physics shard discipline check",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- status: `{payload['status']}`",
        f"- coverage_status: `{payload['coverage_status']}`",
        f"- require_full_coverage: `{payload['require_full_coverage']}`",
        f"- shard_count: `{len(payload['shards'])}`",
        "",
        "| Group | Status | Coverage | Run ID | YAML | Result JSON |",
        "|---|---:|---:|---:|---|---|",
    ]
    for shard in payload["shards"]:
        result = shard.get("result") or {}
        lines.append(
            "| "
            f"`{shard['group_id']}` | `{shard['status']}` | "
            f"`{result.get('coverage_status', 'not_checked')}` | "
            f"`{shard['last_verified_run_id']}` | "
            f"`{shard['yaml_path']}` | `{shard['result_json']}` |"
        )
    failed = [shard for shard in payload["shards"] if shard["errors"]]
    if failed:
        lines.extend(["", "## Errors", ""])
        for shard in failed:
            for error in shard["errors"]:
                lines.append(f"- `{shard['group_id']}`: {error}")
    lines.append("")
    json_rel = _display(json_path)
    md_rel = _display(md_path)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {json_rel}")
    print(f"wrote {md_rel}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="scripts/diagnostics/physics_gate_vessl_shards.json",
        help="VESSL shard manifest JSON.",
    )
    parser.add_argument(
        "--verify-results",
        action="store_true",
        help="Also read each shard result JSON and recompute coverage status.",
    )
    parser.add_argument(
        "--require-full-coverage",
        action="store_true",
        help=(
            "Exit 2 unless verified shard results aggregate to full coverage. "
            "Use this for release/E5 completion gates; it intentionally treats "
            "blocked, skipped, xfailed, and not-checked coverage as incomplete."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=".omx/physics-gate/latest-vessl-shard-check",
        help="Directory for check JSON/Markdown artifacts.",
    )
    args = parser.parse_args(argv)

    if args.require_full_coverage and not args.verify_results:
        raise SystemExit("--require-full-coverage requires --verify-results")

    manifest = _read_json(_repo_path(args.manifest))
    shards = [
        _check_shard(shard, verify_results=args.verify_results)
        for shard in manifest.get("shards", [])
    ]
    result_entries = [shard["result"] for shard in shards if shard.get("result")]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": args.manifest,
        "status": "passed" if shards and all(s["status"] == "passed" for s in shards) else "failed",
        "coverage_status": aggregate_coverage_status(result_entries) if result_entries else "not_checked",
        "require_full_coverage": bool(args.require_full_coverage),
        "shards": shards,
    }
    _write_report(payload, (REPO_ROOT / args.output_dir).resolve())
    print(f"status={payload['status']} coverage_status={payload['coverage_status']}")
    if payload["status"] != "passed":
        return 1
    if args.require_full_coverage and payload["coverage_status"] != "full":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
