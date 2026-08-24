"""Analyze paired residual-driven interface coarse-space experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


BASE_MODE = "local_sparse_schur_sparse"
PRIMARY_MODE = "interface_residual_snapshot_pod_r4_sparse"
RESIDUAL_LIMIT = 1.0e-8


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def row_key(row: Dict[str, Any]) -> Tuple[int, int]:
    return int(row["circuit_id"]), int(row["selected_step_index"])


def strict_success(row: Dict[str, Any]) -> bool:
    residual = row.get("true_rel_residual")
    return bool(
        row.get("status") == "ok"
        and row.get("chosen_success")
        and residual is not None
        and float(residual) < RESIDUAL_LIMIT
    )


def numbers(values: Iterable[Any]) -> List[float]:
    output = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if value == value and abs(value) != float("inf"):
            output.append(value)
    return sorted(output)


def distribution(values: Iterable[Any]) -> Dict[str, Any]:
    values = numbers(values)
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "q1": None,
            "q3": None,
            "min": None,
            "max": None,
        }

    def percentile(fraction: float) -> float:
        position = fraction * float(len(values) - 1)
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - float(lower)
        return float(
            values[lower] * (1.0 - weight) + values[upper] * weight
        )

    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "q1": percentile(0.25),
        "q3": percentile(0.75),
        "min": float(values[0]),
        "max": float(values[-1]),
    }


def paired_comparison(
    baseline_rows: Sequence[Dict[str, Any]],
    candidate_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    baseline = {row_key(row): row for row in baseline_rows}
    candidate = {row_key(row): row for row in candidate_rows}
    keys = sorted(set(baseline) & set(candidate))
    successful = [
        key
        for key in keys
        if strict_success(baseline[key]) and strict_success(candidate[key])
    ]

    def relative_changes(field: str) -> List[float]:
        output = []
        for key in successful:
            base_value = baseline[key].get(field)
            candidate_value = candidate[key].get(field)
            if base_value is None or candidate_value is None:
                continue
            output.append(
                100.0
                * (float(candidate_value) - float(base_value))
                / max(abs(float(base_value)), 1.0e-30)
            )
        return output

    def win_tie_loss(field: str) -> Dict[str, int]:
        output = {"win": 0, "tie": 0, "loss": 0}
        for key in successful:
            base_value = baseline[key].get(field)
            candidate_value = candidate[key].get(field)
            if base_value is None or candidate_value is None:
                continue
            candidate_value = float(candidate_value)
            base_value = float(base_value)
            if candidate_value < base_value:
                output["win"] += 1
            elif candidate_value > base_value:
                output["loss"] += 1
            else:
                output["tie"] += 1
        return output

    baseline_success = {
        row_key(row) for row in baseline_rows if strict_success(row)
    }
    candidate_success = {
        row_key(row) for row in candidate_rows if strict_success(row)
    }
    fallback_count = sum(
        row.get("coarse_enabled") is False for row in candidate_rows
    )
    no_fallback_success = sum(
        strict_success(row) and row.get("coarse_enabled") is not False
        for row in candidate_rows
    )
    return {
        "paired_task_count": len(keys),
        "paired_strict_success_count": len(successful),
        "baseline_strict_success_count": len(baseline_success),
        "candidate_strict_success_count": len(candidate_success),
        "strict_success_sets_equal": baseline_success == candidate_success,
        "candidate_success_missing_keys": [
            list(key) for key in sorted(baseline_success - candidate_success)
        ],
        "candidate_success_extra_keys": [
            list(key) for key in sorted(candidate_success - baseline_success)
        ],
        "fallback_count": int(fallback_count),
        "no_fallback_strict_success_count": int(no_fallback_success),
        "no_fallback_strict_success_rate": float(
            no_fallback_success / max(len(candidate_rows), 1)
        ),
        "iteration_relative_change_percent": distribution(
            relative_changes("gmres_iterations")
        ),
        "iteration_win_tie_loss": win_tie_loss("gmres_iterations"),
        "total_time_relative_change_percent": distribution(
            relative_changes("total_wall_time")
        ),
        "total_time_win_tie_loss": win_tie_loss("total_wall_time"),
        "peak_rss_relative_change_percent": distribution(
            relative_changes("peak_rss_kb")
        ),
        "peak_rss_win_tie_loss": win_tie_loss("peak_rss_kb"),
        "candidate_true_relative_residual": distribution(
            candidate[key].get("true_rel_residual") for key in successful
        ),
    }


def storage_comparison(
    baseline_rows: Sequence[Dict[str, Any]],
    candidate_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    output = paired_comparison(baseline_rows, candidate_rows)
    baseline = {row_key(row): row for row in baseline_rows}
    candidate = {row_key(row): row for row in candidate_rows}
    keys = sorted(set(baseline) & set(candidate))

    def savings(field: str) -> Dict[str, Any]:
        values = []
        for key in keys:
            base_value = baseline[key].get(field)
            candidate_value = candidate[key].get(field)
            if base_value is None or candidate_value is None:
                continue
            values.append(
                100.0
                * (
                    1.0
                    - float(candidate_value)
                    / max(float(base_value), 1.0e-30)
                )
            )
        return distribution(values)

    output["interface_storage_saving_percent"] = savings(
        "interface_retained_bytes"
    )
    output["accounted_preconditioner_storage_saving_percent"] = savings(
        "accounted_preconditioner_retained_bytes"
    )
    output["peak_rss_saving_percent"] = savings("peak_rss_kb")
    return output


def group_mode(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    output: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        output.setdefault(str(row["mode"]), []).append(row)
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def combine_outputs(
    root: Path,
    sources: Sequence[Tuple[str, float, Path]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_all: List[Dict[str, Any]] = []
    normalized_all: List[Dict[str, Any]] = []
    planned_all: List[Dict[str, Any]] = []
    for stage, budget, source in sources:
        for row in read_jsonl(source / "raw_rows.jsonl"):
            raw_all.append(
                {
                    **row,
                    "experiment_stage": stage,
                    "local_schur_budget_multiplier": budget,
                }
            )
        for row in read_jsonl(source / "solver_outcome.jsonl"):
            normalized_all.append(
                {
                    **row,
                    "experiment_stage": stage,
                    "local_schur_budget_multiplier": budget,
                }
            )
        planned = json.loads(
            (source / "planned_tasks.json").read_text(encoding="utf-8")
        )
        for task in planned["tasks"]:
            planned_all.append(
                {
                    **task,
                    "experiment_stage": stage,
                    "local_schur_budget_multiplier": budget,
                }
            )
    write_jsonl(root / "raw_rows.jsonl", raw_all)
    write_jsonl(root / "solver_outcome.jsonl", normalized_all)
    (root / "planned_tasks.json").write_text(
        json.dumps(
            {"task_count": len(planned_all), "tasks": planned_all},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    preferred = [
        "experiment_stage",
        "local_schur_budget_multiplier",
        "mode",
        "circuit_id",
        "selected_step_index",
        "matrix_size",
        "nnz_A",
        "gmres_iterations",
        "true_rel_residual",
        "chosen_success",
        "total_wall_time",
        "peak_rss_kb",
        "interface_retained_bytes",
        "coarse_retained_bytes",
        "accounted_preconditioner_retained_bytes",
        "coarse_actual_rank",
        "coarse_enabled",
        "coarse_guard_accepted",
        "coarse_guard_ratio",
        "coarse_fallback_reason",
    ]
    remaining = sorted(
        set().union(*(row.keys() for row in normalized_all)) - set(preferred)
    )
    with (root / "result_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=preferred + remaining,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(normalized_all)
    return raw_all, normalized_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--workpoint-manifest", required=True)
    parser.add_argument("--snapshot-file", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    sources = [
        ("main_formula", 2.0, root / "main_formula"),
        (
            "storage_substitution",
            0.5,
            root / "storage_substitution" / "budget_0_5",
        ),
        (
            "storage_substitution",
            1.0,
            root / "storage_substitution" / "budget_1_0",
        ),
    ]
    raw_all, normalized_all = combine_outputs(root, sources)
    main_rows = read_jsonl(root / "main_formula" / "solver_outcome.jsonl")
    main_modes = group_mode(main_rows)
    baseline = main_modes[BASE_MODE]
    main_analysis = {
        mode: paired_comparison(baseline, rows)
        for mode, rows in sorted(main_modes.items())
        if mode != BASE_MODE
    }
    storage_analysis: Dict[str, Any] = {}
    for _, budget, source in sources[1:]:
        modes = group_mode(read_jsonl(source / "solver_outcome.jsonl"))
        storage_analysis[str(budget)] = {
            "candidate_vs_budget_2_baseline": storage_comparison(
                baseline, modes[PRIMARY_MODE]
            ),
            "candidate_vs_same_budget_baseline": paired_comparison(
                modes[BASE_MODE], modes[PRIMARY_MODE]
            ),
        }

    primary = main_analysis[PRIMARY_MODE]
    iteration_change = primary[
        "iteration_relative_change_percent"
    ]["median"]
    time_change = primary["total_time_relative_change_percent"]["median"]
    algorithm_criteria = {
        "strict_success_rate_not_lower": (
            primary["candidate_strict_success_count"]
            >= primary["baseline_strict_success_count"]
        ),
        "no_fallback_success_rate_at_least_95_percent": (
            primary["no_fallback_strict_success_rate"] >= 0.95
        ),
        "median_iteration_reduction_at_least_10_percent": (
            iteration_change is not None and iteration_change <= -10.0
        ),
        "median_total_time_reduction_at_least_10_percent": (
            time_change is not None and time_change <= -10.0
        ),
        "all_success_residuals_below_1e-8": (
            primary["candidate_true_relative_residual"]["max"]
            is not None
            and primary["candidate_true_relative_residual"]["max"]
            < RESIDUAL_LIMIT
        ),
        "no_unrecorded_fallback": primary["fallback_count"] == 0,
    }
    algorithm_criteria["all_satisfied"] = all(
        algorithm_criteria.values()
    )

    memory_criteria: Dict[str, Any] = {}
    for budget, item in storage_analysis.items():
        comparison = item["candidate_vs_budget_2_baseline"]
        iteration_median = comparison[
            "iteration_relative_change_percent"
        ]["median"]
        interface_saving = comparison[
            "interface_storage_saving_percent"
        ]["median"]
        accounted_saving = comparison[
            "accounted_preconditioner_storage_saving_percent"
        ]["median"]
        criteria = {
            "strict_success_rate_not_lower": (
                comparison["candidate_strict_success_count"]
                >= comparison["baseline_strict_success_count"]
            ),
            "median_iterations_not_higher": (
                iteration_median is not None and iteration_median <= 0.0
            ),
            "interface_storage_saving_at_least_50_percent": (
                interface_saving is not None and interface_saving >= 50.0
            ),
            "accounted_preconditioner_storage_saving_at_least_50_percent": (
                accounted_saving is not None and accounted_saving >= 50.0
            ),
        }
        criteria["all_satisfied"] = all(criteria.values())
        memory_criteria[budget] = criteria

    paired = {
        "schema_version": 1,
        "strict_success_definition": (
            "status=ok, chosen_success=true, full true relative residual<1e-8"
        ),
        "main_formula": main_analysis,
        "storage_substitution": storage_analysis,
    }
    (root / "paired_analysis.json").write_text(
        json.dumps(paired, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "task_count": len(normalized_all),
        "raw_task_count": len(raw_all),
        "strict_success_count": sum(
            strict_success(row) for row in normalized_all
        ),
        "timeout_count": sum(
            bool(row.get("timeout_hit")) for row in normalized_all
        ),
        "main_formula_task_count": len(main_rows),
        "algorithm_criteria": algorithm_criteria,
        "memory_criteria": memory_criteria,
        "research_decision": (
            "advance_shared_basis_dictionary_and_mode_selection"
            if algorithm_criteria["all_satisfied"]
            and any(
                value["all_satisfied"] for value in memory_criteria.values()
            )
            else "stop_interface_low_rank_route_and_prioritize_partition_edge_selection_scale_coverage"
        ),
        "memory_claim_scope": (
            "preconditioner_incremental_storage_and_process_peak_rss_reported_separately"
        ),
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root.parents[1],
        text=True,
    ).strip()
    manifest = Path(args.workpoint_manifest).resolve()
    snapshot = Path(args.snapshot_file).resolve()
    metadata = {
        "schema_version": 1,
        "code_commit": commit,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "logical_cpu_count": os.cpu_count(),
        "target_workers": 200,
        "actual_main_workers": 128,
        "actual_storage_workers": 40,
        "actual_worker_reduction_reason": (
            "independent_python_processes_local_factors_and_concurrent_io"
        ),
        "timeout_per_task_seconds": 120,
        "blas_threads_per_task": 1,
        "task_count": len(normalized_all),
        "workpoint_manifest": str(manifest),
        "workpoint_manifest_sha256": sha256(manifest),
        "snapshot_file": str(snapshot),
        "snapshot_file_sha256": sha256(snapshot),
    }
    (root / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    hash_paths = [
        root / "run_metadata.json",
        root / "planned_tasks.json",
        root / "raw_rows.jsonl",
        root / "solver_outcome.jsonl",
        root / "result_table.csv",
        root / "paired_analysis.json",
        root / "summary.json",
        snapshot,
        root / "snapshots" / "post_schwarz_error_manifest.json",
    ]
    (root / "SHA256SUMS.txt").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(root)}\n"
            for path in hash_paths
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
