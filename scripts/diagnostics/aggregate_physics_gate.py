#!/usr/bin/env python3
"""Aggregate rfx physics-gate result JSON without hiding claim blockers.

This script accepts one or more `physics_gate_results.json` files produced by
`scripts/diagnostics/run_physics_gate.py`.  It also upgrades older result JSONs
by reading their raw pytest stdout and recomputing coverage fields such as
`coverage_status`, skip counts, strict xfail counts, and blocked claims.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_physics_gate import (
    REPO_ROOT,
    GateGroup,
    aggregate_coverage_status,
    coverage_metadata,
    gate_groups_by_id,
    parse_vessl_run_ids,
)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"missing result JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc


def _read_artifact_text(path_value: str) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _group_for_result(result: dict[str, Any]) -> GateGroup:
    groups = gate_groups_by_id()
    group_id = str(result.get("group_id", "unknown"))
    if group_id in groups:
        return groups[group_id]
    return GateGroup(
        group_id=group_id,
        description=str(result.get("description", "unknown gate group")),
        tests=tuple(str(test) for test in result.get("tests", [])),
        claim_level=str(result.get("claim_level", "E0")),
        coverage_scope="not_claims_bearing",
        blocked_claims=(
            {
                "claim": f"unknown physics-gate group {group_id}",
                "evidence_level": "E0",
                "reason": "group is not registered in run_physics_gate.py",
            },
        ),
    )


def _merge_run_ids(*values: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for run_ids in values:
        for run_id in run_ids:
            if run_id and run_id not in seen:
                seen.add(run_id)
                merged.append(run_id)
    return merged


def _enrich_result(
    result: dict[str, Any],
    *,
    source_json: Path,
    cli_vessl_run_ids: list[str],
) -> dict[str, Any]:
    group = _group_for_result(result)
    enriched = dict(result)
    stdout = _read_artifact_text(str(result.get("stdout_path", "")))
    run_ids = _merge_run_ids(
        [str(run_id) for run_id in result.get("vessl_run_ids", [])],
        cli_vessl_run_ids,
    )
    metadata = coverage_metadata(
        group,
        execution_status=str(result.get("status", "failed")),
        stdout=stdout,
        vessl_run_ids=run_ids,
    )
    enriched.update(metadata)
    try:
        enriched["result_json"] = str(source_json.relative_to(REPO_ROOT))
    except ValueError:
        enriched["result_json"] = str(source_json)
    return enriched


def _aggregate_payload(
    results: list[dict[str, Any]],
    *,
    notes: list[str],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "status": (
            "passed"
            if results and all(result.get("status") == "passed" for result in results)
            else "failed"
        ),
        "coverage_status": aggregate_coverage_status(results),
        "required_skip_count": sum(int(r.get("required_skip_count", 0)) for r in results),
        "optional_skip_count": sum(int(r.get("optional_skip_count", 0)) for r in results),
        "strict_xfail_count": sum(int(r.get("strict_xfail_count", 0)) for r in results),
        "vessl_run_ids": sorted(
            {
                run_id
                for result in results
                for run_id in result.get("vessl_run_ids", [])
            }
        ),
        "validated_claims": [
            claim for result in results for claim in result.get("validated_claims", [])
        ],
        "blocked_claims": [
            claim for result in results for claim in result.get("blocked_claims", [])
        ],
        "notes": notes,
        "results": results,
    }


def _write_reports(payload: dict[str, Any], output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{prefix}.json"
    md_path = output_dir / f"{prefix}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Aggregate rfx physics gate",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- overall_status: `{payload['status']}`",
        f"- coverage_status: `{payload['coverage_status']}`",
        f"- required_skip_count: `{payload['required_skip_count']}`",
        f"- optional_skip_count: `{payload['optional_skip_count']}`",
        f"- strict_xfail_count: `{payload['strict_xfail_count']}`",
        f"- vessl_run_ids: `{', '.join(payload['vessl_run_ids']) or 'none'}`",
        "",
        "| Group | Status | Coverage | E-level | Req. skips | Opt. skips | Xfails | Result JSON |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in payload["results"]:
        lines.append(
            "| "
            f"`{result.get('group_id')}` | `{result.get('status')}` | "
            f"`{result.get('coverage_status')}` | `{result.get('claim_level')}` | "
            f"{result.get('required_skip_count', 0)} | "
            f"{result.get('optional_skip_count', 0)} | "
            f"{result.get('strict_xfail_count', 0)} | "
            f"`{result.get('result_json')}` |"
        )

    if payload["blocked_claims"]:
        lines.extend(["", "## Blocked or unpromoted claims", ""])
        for claim in payload["blocked_claims"]:
            lines.append(
                "- "
                f"`{claim.get('evidence_level', '?')}` "
                f"{claim.get('claim', 'claim')}: "
                f"{claim.get('reason', 'no reason recorded')}"
            )

    if payload["notes"]:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in payload["notes"])

    lines.extend(
        [
            "",
            "Execution `passed` is not equivalent to full claim coverage. Required "
            "skips and strict xfails remain visible in `coverage_status` and "
            "`blocked_claims`.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {_display_path(json_path)}")
    print(f"wrote {_display_path(md_path)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_json",
        nargs="*",
        help="physics_gate_results.json files to aggregate.",
    )
    parser.add_argument(
        "--result-json",
        action="append",
        dest="result_json_flags",
        default=[],
        help="physics_gate_results.json file to aggregate. May be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        default=".omx/physics-gate/latest-aggregate",
        help="Directory for aggregate JSON/Markdown artifacts.",
    )
    parser.add_argument(
        "--prefix",
        default="physics_gate_aggregate",
        help="Output file prefix without extension.",
    )
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="Add an aggregate note. May be repeated.",
    )
    parser.add_argument(
        "--vessl-run-id",
        action="append",
        default=[],
        help="Record a VESSL run id as GROUP=ID or a bare ID for all groups.",
    )
    parser.add_argument(
        "--require-full-coverage",
        action="store_true",
        help="Exit non-zero unless aggregate coverage_status is full.",
    )
    args = parser.parse_args(argv)

    paths = [Path(p) for p in [*args.result_json_flags, *args.result_json]]
    if not paths:
        raise SystemExit("provide at least one physics_gate_results.json path")

    raw_payloads = [(path, _read_json(path)) for path in paths]
    group_ids = [
        str(result.get("group_id"))
        for _, payload in raw_payloads
        for result in payload.get("results", [])
    ]
    selected_groups = [
        _group_for_result({"group_id": group_id, "tests": []}) for group_id in group_ids
    ]
    cli_run_ids = parse_vessl_run_ids(
        args.vessl_run_id,
        selected_groups=selected_groups,
    )

    results: list[dict[str, Any]] = []
    for path, payload in raw_payloads:
        source_json = (REPO_ROOT / path).resolve() if not path.is_absolute() else path
        for result in payload.get("results", []):
            group_id = str(result.get("group_id"))
            results.append(
                _enrich_result(
                    result,
                    source_json=source_json,
                    cli_vessl_run_ids=cli_run_ids.get(group_id, []),
                )
            )

    payload = _aggregate_payload(results, notes=list(args.note))
    _write_reports(payload, (REPO_ROOT / args.output_dir).resolve(), args.prefix)
    print(
        f"overall_status={payload['status']} "
        f"coverage_status={payload['coverage_status']} "
        f"required_skips={payload['required_skip_count']} "
        f"xfails={payload['strict_xfail_count']}"
    )
    if args.require_full_coverage and payload["coverage_status"] != "full":
        print(
            "aggregate coverage is not full; refusing full physics-validation claim",
            file=sys.stderr,
        )
        return 2
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
