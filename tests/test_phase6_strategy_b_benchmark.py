"""Phase VI Strategy B benchmark harness tests."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "phase6_strategy_b_workflow_benchmark.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase6_strategy_b_workflow_benchmark", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase6_strategy_b_benchmark_schema_and_runtime_rows():
    module = _load_module()

    report = module.build_phase6_report(budget_gb=24.0, checkpoint_every=1000)

    assert report["schema_version"] == 1
    assert report["benchmark_contract"] == "phase_vi_strategy_b_workflow_runtime"
    assert report["preserves_phase_iii_gate0_contract"] is True

    rows = report["results"]
    assert rows
    required = set(module.REQUIRED_RESULT_FIELDS)
    assert all(required <= set(row) for row in rows)

    families = {row["objective_family"] for row in rows}
    assert {
        "source_probe",
        "cpml_topology_probe_energy",
        "one_excited_one_passive_port_proxy",
    } <= families

    for row in rows:
        assert row["runtime_support_status"] == "implemented"
        assert row["correctness_status"] == "passed"
        assert row["correctness_metric"] <= 1e-5
        assert len(row["grid_shape"]) == 3
        assert math.prod(row["grid_shape"]) == row["cell_count"]
        assert row["n_steps"] > 0
        assert row["checkpoint_every"] == 1000
        assert row["runtime_s"] > 0
        assert row["estimated_strategy_b_memory_gb"] > 0
        assert row["estimated_strategy_b_memory_gb"] < row["estimated_strategy_a_memory_gb"]
        assert isinstance(row["pass"], bool)
        assert row["reason"]


def test_phase6_strategy_b_benchmark_cli_emits_machine_readable_json():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--indent", "0"],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)

    assert report["contract_status"] == "phase_vi_runtime_evidence"
    assert report["summary"]["rows_with_required_fields"] == report["summary"]["result_rows"]
    assert report["summary"]["runtime_supported_cases"] == 3
