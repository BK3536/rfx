"""Phase III Strategy B Gate 0 benchmark-contract tests."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "phase3_strategy_b_benchmark.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase3_strategy_b_benchmark", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase3_strategy_b_benchmark_schema_and_estimator_contract():
    module = _load_module()

    report = module.build_contract_report(budget_gb=24.0, checkpoint_every=1000)

    assert report["schema_version"] == 1
    assert report["benchmark_contract"] == "phase_iii_strategy_b_gate0"
    assert report["disabled_optional_cases"] == ["ntff_directivity_smoke"]

    rows = report["results"]
    assert rows
    required = set(module.REQUIRED_RESULT_FIELDS)
    assert all(required <= set(row) for row in rows)

    roles = {row["ladder_role"] for row in rows}
    assert {"primary", "secondary", "tertiary"} <= roles
    assert "optional" not in roles

    objective_families = {row["objective_family"] for row in rows}
    assert {
        "source_probe_optimize",
        "cpml_topology_probe_energy",
        "one_excited_one_passive_port_proxy",
    } <= objective_families

    strategies = {row["strategy_name"] for row in rows}
    assert strategies == set(module.STRATEGIES)

    for row in rows:
        assert len(row["grid_shape"]) == 3
        assert math.prod(row["grid_shape"]) == row["cell_count"]
        assert row["n_steps"] > 0
        assert row["estimated_memory_gb"] > 0
        assert row["runtime_s"] >= 0
        assert row["correctness_status"] == "not_evaluated_gate0_contract_only"
        assert isinstance(row["pass"], bool)
        assert row["reason"]

    primary = [row for row in rows if row["case_id"] == "source_probe_optimize_patch"]
    by_strategy = {row["strategy_name"]: row for row in primary}
    assert by_strategy["strategy_b_segmented_checkpoint"]["checkpoint_every"] == 1000
    assert (
        by_strategy["strategy_b_segmented_checkpoint"]["estimated_memory_gb"]
        < by_strategy["strategy_a_replay_trace"]["estimated_memory_gb"]
    )


def test_phase3_strategy_b_benchmark_cli_emits_machine_readable_json():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--indent", "0"],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)

    assert report["contract_status"] == "gate0_schema_locked"
    assert report["summary"]["rows_with_required_fields"] == report["summary"]["result_rows"]
    assert report["summary"]["enabled_cases"] == 3
