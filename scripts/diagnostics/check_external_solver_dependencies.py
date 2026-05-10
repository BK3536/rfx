#!/usr/bin/env python3
"""Audit external solver dependencies needed for RF/port E4/E5 evidence.

This is not physics evidence by itself.  It records whether the current
execution environment can run the external-solver/reference lanes that the
all-port E5 campaign depends on.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import io
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODULES = {
    "meep": "MEEP Python API for FDTD external cross-validation",
    "openEMS.openEMS": "openEMS Python API for FDTD external cross-validation",
    "CSXCAD.CSXCAD": "CSXCAD geometry API required by openEMS",
    "rcwa": "RCWA package for Floquet/periodic-cell references",
    "S4": "S4 RCWA package alternative",
    "s4": "S4 RCWA package alternative",
}

DEFAULT_COMMANDS = {
    "meep": "MEEP executable",
    "openEMS": "openEMS executable",
    "AppCSXCAD": "CSXCAD viewer/helper executable",
    "palace": "Palace executable for FEM waveguide/coax references",
    "mpirun": "MPI runner for external solvers",
}

CAPABILITIES = {
    "meep_crossval": {
        "requires_any_module": ["meep"],
        "requires_any_command": ["meep"],
    },
    "openems_crossval": {
        "requires_any_module": ["openEMS.openEMS"],
        "requires_all_modules": ["CSXCAD.CSXCAD"],
        "requires_any_command": ["openEMS"],
    },
    "palace_crossval": {
        "requires_any_command": ["palace"],
    },
    "rcwa_floquet": {
        "requires_any_module": ["rcwa", "S4", "s4"],
    },
}


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _module_probe(name: str) -> dict[str, Any]:
    stderr_capture = io.StringIO()
    stdout_capture = io.StringIO()
    try:
        with (
            contextlib.redirect_stderr(stderr_capture),
            contextlib.redirect_stdout(stdout_capture),
        ):
            spec = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError) as exc:
        return {
            "available": False,
            "find_spec_available": False,
            "import_checked": False,
            "import_error": f"{type(exc).__name__}: {exc}",
            "import_stderr": stderr_capture.getvalue()[-4000:],
            "import_stdout": stdout_capture.getvalue()[-4000:],
        }
    if spec is None:
        return {
            "available": False,
            "find_spec_available": False,
            "import_checked": False,
            "import_error": "module spec not found",
            "import_stderr": stderr_capture.getvalue()[-4000:],
            "import_stdout": stdout_capture.getvalue()[-4000:],
        }
    try:
        with contextlib.redirect_stderr(stderr_capture), contextlib.redirect_stdout(stdout_capture):
            importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - depends on local solver wheels
        return {
            "available": False,
            "find_spec_available": True,
            "import_checked": True,
            "import_error": f"{type(exc).__name__}: {exc}",
            "import_stderr": stderr_capture.getvalue()[-4000:],
            "import_stdout": stdout_capture.getvalue()[-4000:],
        }
    return {
        "available": True,
        "find_spec_available": True,
        "import_checked": True,
        "import_error": None,
        "import_stderr": stderr_capture.getvalue()[-4000:],
        "import_stdout": stdout_capture.getvalue()[-4000:],
    }


def build_dependency_audit() -> dict[str, Any]:
    module_results = {}
    for name, purpose in DEFAULT_MODULES.items():
        module_results[name] = _module_probe(name)
        module_results[name]["purpose"] = purpose
    command_results = {
        name: {
            "available": shutil.which(name) is not None,
            "path": shutil.which(name),
            "purpose": purpose,
        }
        for name, purpose in DEFAULT_COMMANDS.items()
    }

    capabilities: dict[str, dict[str, Any]] = {}
    for name, requirements in CAPABILITIES.items():
        any_modules = requirements.get("requires_any_module", [])
        all_modules = requirements.get("requires_all_modules", [])
        any_commands = requirements.get("requires_any_command", [])
        module_any_ok = (
            True
            if not any_modules
            else any(module_results[module]["available"] for module in any_modules)
        )
        module_all_ok = all(
            module_results[module]["available"] for module in all_modules
        )
        command_any_ok = (
            True
            if not any_commands
            else any(command_results[command]["available"] for command in any_commands)
        )
        status = "available" if module_any_ok and module_all_ok and command_any_ok else "blocked"
        blockers: list[str] = []
        if not module_any_ok:
            errors = {
                module: module_results[module].get("import_error")
                for module in any_modules
                if not module_results[module]["available"]
            }
            blockers.append(f"missing/import-failed one of modules: {any_modules}; errors={errors}")
        if not module_all_ok:
            missing = [m for m in all_modules if not module_results[m]["available"]]
            errors = {module: module_results[module].get("import_error") for module in missing}
            blockers.append(f"missing/import-failed required modules: {missing}; errors={errors}")
        if not command_any_ok:
            blockers.append(f"missing one of commands: {any_commands}")
        capabilities[name] = {
            "status": status,
            "requirements": requirements,
            "blockers": blockers,
        }

    available_count = sum(1 for item in capabilities.values() if item["status"] == "available")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if available_count else "blocked",
        "claim_scope": (
            "dependency availability only; not E4/E5 physics evidence and not "
            "a substitute for external comparison artifacts"
        ),
        "module_results": module_results,
        "command_results": command_results,
        "capabilities": capabilities,
        "available_capability_count": available_count,
        "blocked_capability_count": len(capabilities) - available_count,
        "completion_decision": (
            "External solver dependencies are partially available; run the "
            "family-specific comparison shards before any E4/E5 promotion."
        ),
    }


def _write_report(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "external_solver_dependency_audit.json"
    md_path = output_dir / "external_solver_dependency_audit.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# External solver dependency audit",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- status: `{payload['status']}`",
        f"- available_capability_count: `{payload['available_capability_count']}`",
        f"- blocked_capability_count: `{payload['blocked_capability_count']}`",
        "",
        payload["claim_scope"],
        "",
        "## Capabilities",
        "",
        "| Capability | Status | Blockers |",
        "|---|---:|---|",
    ]
    for name, item in payload["capabilities"].items():
        blockers = "; ".join(item["blockers"]) if item["blockers"] else "—"
        lines.append(f"| `{name}` | `{item['status']}` | {blockers} |")
    lines.extend(["", "## Commands", "", "| Command | Available | Path |", "|---|---:|---|"])
    for name, item in payload["command_results"].items():
        lines.append(f"| `{name}` | `{item['available']}` | `{item['path']}` |")
    lines.extend(
        [
            "",
            "## Python modules",
            "",
            "| Module | Available | find_spec | Import error |",
            "|---|---:|---:|---|",
        ]
    )
    for name, item in payload["module_results"].items():
        error = item.get("import_error") or "—"
        lines.append(
            f"| `{name}` | `{item['available']}` | "
            f"`{item.get('find_spec_available')}` | {error} |"
        )
    lines.append(f"\nDecision: {payload['completion_decision']}\n")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {_display(json_path)}")
    print(f"wrote {_display(md_path)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=".omx/physics-gate/latest-external-solver-dependency-audit",
    )
    args = parser.parse_args(argv)

    payload = build_dependency_audit()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    _write_report(payload, output_dir)
    print(
        "status={status} available_capability_count={available} blocked_capability_count={blocked}".format(
            status=payload["status"],
            available=payload["available_capability_count"],
            blocked=payload["blocked_capability_count"],
        )
    )
    return 0 if payload["available_capability_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
