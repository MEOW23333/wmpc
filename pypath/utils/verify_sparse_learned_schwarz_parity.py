#!/usr/bin/env python3
"""?????? 10x10 ????? V1 ????????????

?????? GMRES?????????????? 0 ??????
Newton ???????????????????????????
???????????????????????????????
?????
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pypath.preconditioner.block_schwarz import (  # noqa: E402
    BlockPlanConfig,
    BlockSchwarzPlan,
    build_block_schwarz_plan,
)
from pypath.preconditioner.linear_system_contract import (  # noqa: E402
    INITIAL_GUESS_MODE_RHSOLD,
    INITIAL_GUESS_MODES,
    compute_initial_residual,
    resolve_initial_guess,
)
from pypath.preconditioner.learned_schwarz import (  # noqa: E402
    build_learned_schwarz_sample,
)
from pypath.preconditioner.sparse_learned_schwarz import (  # noqa: E402
    SparseLearnedSchwarzV1Preconditioner,
    extract_learned_schwarz_blocks,
    load_learned_schwarz_v1_model,
)
from pypath.utils.ngspice_utils import (  # noqa: E402
    read_J_sparse,
    read_continuation_step,
)
from pypath.utils.run_sparse_solver_benchmark import (  # noqa: E402
    _read_time_v,
    _resolve_netlist,
)
from pypath.utils.sparse_gmres_prototype import _find_steps, _matrix_stats  # noqa: E402


DEFAULT_TRAJECTORY_DIR = "experiments/core_block_repro_10x10/aggregator/trajectory"
DEFAULT_NETLIST_DIR = "experiments/core_block_repro_10x10/aggregator/generated_netlists"
DEFAULT_CHECKPOINT = (
    "experiments/core_block_repro_10x10/learned_schwarz/"
    "smoke_norm_v1/learned_schwarz_v1.pt"
)
DEFAULT_WORK_DIR = (
    "pals_data/runs/learned_schwarz_sparse_memory_validation_20260814/stage1"
)
THREAD_ENV = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _repo_path(raw_path: str, label: str) -> Path:
    """??????? PALS ???????"""
    candidate = Path(str(raw_path)).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} ???? PALS ????{resolved}") from exc
    return resolved


def _relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _feature_vector(value: Any, matrix_size: int) -> np.ndarray:
    """???????????????????????????"""
    try:
        array = np.asarray([] if value is None else value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        array = np.zeros(0, dtype=np.float64)
    if array.shape[0] != int(matrix_size):
        return np.zeros(int(matrix_size), dtype=np.float64)
    return array


def _required_vector(value: Any, matrix_size: int, label: str) -> np.ndarray:
    array = np.asarray([] if value is None else value, dtype=np.float64).reshape(-1)
    if array.shape[0] != int(matrix_size):
        raise ValueError(
            f"{label} ?? {array.shape[0]} ????? {matrix_size} ???"
        )
    return array


def _probe_vector(case: Dict[str, Any], source: str) -> np.ndarray:
    if source == "rhsnew":
        return np.asarray(case["linear_rhs"], dtype=np.float64)
    if source == "initial_residual":
        return np.asarray(case["initial_residual"], dtype=np.float64)
    if source == "ones":
        return np.ones(int(case["matrix"].shape[0]), dtype=np.float64)
    raise ValueError(f"?? probe_source?{source}")


def _block_sizes(blocks: Iterable[np.ndarray]) -> Dict[str, Any]:
    sizes = [int(np.asarray(rows).shape[0]) for rows in blocks]
    if not sizes:
        return {"count": 0, "min": 0, "median": 0.0, "max": 0, "sum": 0}
    return {
        "count": int(len(sizes)),
        "min": int(min(sizes)),
        "median": float(np.median(np.asarray(sizes, dtype=np.float64))),
        "max": int(max(sizes)),
        "sum": int(sum(sizes)),
    }


def _plan_summary(plan: BlockSchwarzPlan) -> Dict[str, Any]:
    return {
        "block_mode": str(plan.block_mode),
        "candidate_block_count": int(plan.candidate_block_count),
        "block_count": int(len(plan.blocks)),
        "skipped_block_count": int(plan.skipped_block_count),
        "covered_rows": int(np.count_nonzero(plan.covered_mask)),
        "uncovered_rows": int(plan.covered_mask.shape[0] - np.count_nonzero(plan.covered_mask)),
        "coverage_ratio": float(plan.coverage_ratio),
        "max_block_size": int(plan.max_block_size),
        "total_block_nnz": int(plan.total_block_nnz),
        "block_sizes": _block_sizes(plan.blocks),
    }


def _output_summary(values: np.ndarray) -> Dict[str, Any]:
    output = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "size": int(output.shape[0]),
        "finite": bool(np.all(np.isfinite(output))),
        "l2_norm": float(np.linalg.norm(output)),
        "linf_norm": float(np.max(np.abs(output))) if output.size else 0.0,
    }


def _load_case(args: argparse.Namespace) -> Dict[str, Any]:
    trajectory_dir = _repo_path(args.trajectory_dir, "trajectory_dir")
    steps = _find_steps(str(trajectory_dir), int(args.circuit_id), max_steps=1000000)
    if args.positive_gmin_only:
        steps = [step for step in steps if float(step.get("gmin_val", 0.0)) > 0.0]
    if not steps:
        raise FileNotFoundError(
            f"????? {int(args.circuit_id)} ???? Newton ??{trajectory_dir}"
        )
    selected_step_index = int(args.step_offset)
    if selected_step_index < 0 or selected_step_index >= len(steps):
        raise IndexError("selected step offset is outside the available steps")
    step = dict(steps[selected_step_index])
    step_path = _repo_path(str(step["step_path"]), "step_path")
    jacobian_path = _repo_path(str(step["jacobian_path"]), "jacobian_path")

    if args.netlist_path:
        netlist_path = _repo_path(args.netlist_path, "netlist_path")
    else:
        netlist_dir = _repo_path(args.netlist_dir, "netlist_dir")
        netlist_path = _repo_path(
            _resolve_netlist(
                SimpleNamespace(netlist_path="", netlist_dir=str(netlist_dir)),
                int(args.circuit_id),
            ),
            "resolved_netlist_path",
        )
    if not netlist_path.is_file():
        raise FileNotFoundError(f"??????{netlist_path}")

    checkpoint_path = _repo_path(
        args.learned_schwarz_checkpoint,
        "learned_schwarz_checkpoint",
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"??? learned Schwarz V1 ????{checkpoint_path}")

    matrix = read_J_sparse(str(jacobian_path), matrix_format="csr")
    matrix = matrix.tocsr().real.astype(np.float64, copy=False)
    raw_matrix_nnz = int(matrix.nnz)
    gmin_value = float(step.get("gmin_val", 0.0))
    if not bool(args.apply_gmin_diagonal):
        raise ValueError("schema v4 verification requires the gmin diagonal")
    if gmin_value != 0.0:
        matrix = matrix + sp.eye(matrix.shape[0], dtype=matrix.dtype, format="csr") * gmin_value
    matrix = matrix.tocsr()
    matrix.sort_indices()
    if matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Jacobian ???????")

    payload = read_continuation_step(str(step_path))
    linear_rhs = _required_vector(
        payload.get("rhsnew"), int(matrix.shape[0]), "rhsnew"
    )
    initial_guess = resolve_initial_guess(
        rhsold=payload.get("rhsold"),
        matrix_size=int(matrix.shape[0]),
        initial_guess_mode=str(args.initial_guess_mode),
    )
    initial_residual = compute_initial_residual(
        effective_matrix=matrix,
        linear_rhs=linear_rhs,
        initial_guess=initial_guess,
    )
    node_map = payload.get("node_map")
    if not isinstance(node_map, dict) or not node_map:
        raise ValueError("continuation step ???? node_map????????")

    matrix_info = _matrix_stats(matrix)
    matrix_info["raw_matrix_nnz"] = raw_matrix_nnz
    matrix_info["gmin_diagonal_applied"] = bool(gmin_value != 0.0)
    matrix_info["gmin_value"] = gmin_value
    matrix_info["initial_guess_mode"] = str(args.initial_guess_mode)
    matrix_info["initial_residual_l2_norm"] = float(np.linalg.norm(initial_residual))
    return {
        "matrix": matrix,
        "linear_rhs": linear_rhs,
        "initial_guess": initial_guess,
        "initial_residual": initial_residual,
        "node_map": node_map,
        "checkpoint_path": checkpoint_path,
        "netlist_path": netlist_path,
        "matrix_info": matrix_info,
        "step": {
            "circuit_id": int(step["circuit_id"]),
            "selected_step_index": selected_step_index,
            "first_recorded_newton_step": bool(selected_step_index == 0),
            "positive_gmin_only": bool(args.positive_gmin_only),
            "time": float(step["time"]),
            "gmin_val": gmin_value,
            "iteration": int(step["iteration"]),
            "step_path": _relative_path(step_path),
            "jacobian_path": _relative_path(jacobian_path),
            "netlist_path": _relative_path(netlist_path),
            "node_map_size": int(len(node_map)),
        },
    }


def _build_dense_v1(
    case: Dict[str, Any],
    args: argparse.Namespace,
    model: torch.nn.Module,
) -> Tuple[Dict[str, Any], np.ndarray, BlockSchwarzPlan]:
    """??? dense V1 ????????? apply ???"""
    matrix_sparse = case["matrix"]
    dense_matrix = matrix_sparse.toarray()
    plan = build_block_schwarz_plan(
        matrix=dense_matrix,
        node_map=case["node_map"],
        netlist_path=str(case["netlist_path"]),
        config=BlockPlanConfig(
            block_mode="cell_core_plus_onehop_boundary",
            max_block_size=int(args.max_block_size),
            min_block_size=int(args.min_block_size),
            max_blocks=int(args.max_blocks),
            uncovered_row_policy="row_sum",
        ),
    )
    sample = build_learned_schwarz_sample(
        matrix=dense_matrix,
        plan=plan,
        linear_rhs=case["linear_rhs"],
        initial_residual=case["initial_residual"],
        gmin=float(case["step"]["gmin_val"]),
        dtype=torch.float64,
    )
    probe = _probe_vector(case, args.probe_source)
    with torch.no_grad():
        output = model.apply(
            sample,
            torch.as_tensor(probe, dtype=torch.float64),
        ).detach().cpu().numpy()

    with torch.no_grad():
        dense_parameters = model.predict_parameters(sample)
    dense_shift = {
        'local_shift_contract': 'block_inf_norm_relative_floor_v1',
        'local_shift_floor_relative': 1e-6,
        'block_lambda_pred': dense_parameters['lambda_pred'].detach().cpu().numpy().tolist(),
        'block_lambda_floor': dense_parameters['lambda_floor'].detach().cpu().numpy().tolist(),
        'block_lambda_effective': dense_parameters['lambdas'].detach().cpu().numpy().tolist(),
    }
    dense_global_bytes = int(
        int(matrix_sparse.shape[0])
        * int(matrix_sparse.shape[1])
        * np.dtype(np.float64).itemsize
    )
    return (
        {
            "implementation": "learned_schwarz_v1_dense",
            "blocks": _plan_summary(plan),
            "output": _output_summary(output),
            "memory": {
                "dense_v1_global_matrix_bytes": dense_global_bytes,
                "dense_v1_global_matrix_mib": float(dense_global_bytes / (1024 ** 2)),
                "comparison_basis": "?? float64 ??????????????????",
            },
            'local_shift': dense_shift,
        },
        np.asarray(output, dtype=np.float64),
        plan,
    )


def _build_sparse_v1(
    case: Dict[str, Any],
    args: argparse.Namespace,
    model: torch.nn.Module,
) -> Tuple[Dict[str, Any], np.ndarray, List[np.ndarray], List[Dict[str, Any]]]:
    """?????? V1 ???????????????"""
    matrix = case["matrix"]
    blocks, candidates, block_debug = extract_learned_schwarz_blocks(
        node_map=case["node_map"],
        netlist_path=str(case["netlist_path"]),
        max_block_size=int(args.max_block_size),
        min_block_size=int(args.min_block_size),
        max_blocks=int(args.max_blocks),
        matrix_size=int(matrix.shape[0]),
    )
    preconditioner = SparseLearnedSchwarzV1Preconditioner(
        matrix=matrix,
        blocks=blocks,
        block_candidates=candidates,
        model=model,
        linear_rhs=case["linear_rhs"],
        initial_residual=case["initial_residual"],
        gmin=float(case["step"]["gmin_val"]),
    )
    probe = _probe_vector(case, args.probe_source)
    output = preconditioner.apply(probe)
    core = preconditioner.metadata()
    lu_matrix_bytes = int(sum(factor[0].nbytes for factor in preconditioner.factors))
    lu_pivot_bytes = int(sum(factor[1].nbytes for factor in preconditioner.factors))
    local_lu_bytes = int(lu_matrix_bytes + lu_pivot_bytes)
    memory = {
        "dense_v1_global_matrix_bytes": int(core["dense_v1_global_matrix_bytes"]),
        "sparse_retained_estimated_bytes": int(core["sparse_retained_estimated_bytes"]),
        "sparse_peak_estimated_bytes": int(core["sparse_peak_estimated_bytes"]),
        "estimated_memory_saved_bytes": int(core["estimated_memory_saved_bytes"]),
        "estimated_peak_memory_saving_ratio": float(core["estimated_peak_memory_saving_ratio"]),
        "memory_saving_target_over_50pct": bool(core["memory_saving_target_over_50pct"]),
        "local_lu_factor_bytes": local_lu_bytes,
        "local_lu_matrix_bytes": lu_matrix_bytes,
        "local_lu_pivot_bytes": lu_pivot_bytes,
        "max_local_dense_bytes_during_setup": int(preconditioner.max_local_dense_bytes),
        "conservative_peak_formula": "sparse_retained_estimated_bytes + 2 * max_local_dense_bytes_during_setup",
    }
    sparse_shift = {
        'local_shift_contract': core['local_shift_contract'],
        'local_shift_floor_relative': core['local_shift_floor_relative'],
        'all_candidate_blocks_factorized': core['all_candidate_blocks_factorized'],
        'block_lambda_pred': core['block_lambda_pred'],
        'block_lambda_floor': core['block_lambda_floor'],
        'block_lambda_effective': core['block_lambda_effective'],
    }
    return (
        {
            "implementation": "learned_schwarz_v1_sparse",
            "no_global_dense_materialization": bool(core["no_global_dense_materialization"]),
            "block_debug": block_debug,
            "blocks": {
                "candidate_block_count": int(core["candidate_block_count"]),
                "block_count": int(core["block_count"]),
                "skipped_block_count": int(core["skipped_block_count"]),
                "skipped_block_reasons": dict(core["skipped_block_reasons"]),
                "covered_rows": int(core["covered_rows"]),
                "uncovered_rows": int(core["uncovered_rows"]),
                "coverage_ratio": float(core["coverage_ratio"]),
                "max_block_size": int(core["max_block_size"]),
                "total_block_nnz": int(core["total_block_nnz"]),
                "block_sizes": _block_sizes(preconditioner.blocks),
            },
            "output": _output_summary(output),
            "memory": memory,
            'local_shift': sparse_shift,
        },
        np.asarray(output, dtype=np.float64),
        blocks,
        candidates,
    )


def _role_signature(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {
            str(key): str(value)
            for key, value in sorted(
                dict(candidate.get("row_role_by_index") or {}).items(),
                key=lambda item: str(item[0]),
            )
        }
        for candidate in candidates
    ]


def _block_contract(
    dense_plan: BlockSchwarzPlan,
    sparse_blocks: Sequence[np.ndarray],
    sparse_candidates: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    dense_rows = [tuple(int(value) for value in rows.tolist()) for rows in dense_plan.blocks]
    sparse_rows = [
        tuple(int(value) for value in np.asarray(rows, dtype=np.int64).tolist())
        for rows in sparse_blocks
    ]
    rows_match = dense_rows == sparse_rows
    roles_match = _role_signature(dense_plan.block_candidates) == _role_signature(
        sparse_candidates
    )
    return {
        "dense_block_count": int(len(dense_rows)),
        "sparse_candidate_block_count": int(len(sparse_rows)),
        "row_sequences_match": bool(rows_match),
        "row_role_maps_match": bool(roles_match),
        "matches": bool(rows_match and roles_match),
    }


def _case_header(case: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    checkpoint = case["checkpoint_path"]
    return {
        "schema_version": 1,
        "stage": "stage1_sparse_learned_schwarz_memory_validation",
        "gmres_executed": False,
        "step": dict(case["step"]),
        "matrix": dict(case["matrix_info"]),
        "checkpoint": {
            "path": _relative_path(checkpoint),
            "sha256": _sha256(checkpoint),
            "bytes": int(checkpoint.stat().st_size),
        },
        "configuration": {
            "max_block_size": int(args.max_block_size),
            "min_block_size": int(args.min_block_size),
            "max_blocks": int(args.max_blocks),
            "probe_source": str(args.probe_source),
            "probe_l2_norm": float(np.linalg.norm(_probe_vector(case, args.probe_source))),
            "initial_guess_mode": str(args.initial_guess_mode),
            "relative_error_tolerance": float(args.relative_error_tolerance),
            "memory_saving_target_ratio": float(args.memory_saving_target_ratio),
        },
    }


def _run_child(args: argparse.Namespace, child_mode: str) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        case = _load_case(args)
        report = _case_header(case, args)
        report["mode"] = str(child_mode)
        model = load_learned_schwarz_v1_model(
            str(case["checkpoint_path"]),
            initial_guess_mode=str(args.initial_guess_mode),
        )

        if child_mode == "baseline":
            report["baseline"] = {
                "implementation": "shared_case_and_model_only",
                "preconditioner_constructed": False,
                "model_parameter_bytes": int(
                    sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
                ),
                "model_buffer_bytes": int(
                    sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
                ),
            }
            report["ok"] = True
        elif child_mode == "dense":
            dense, _, _ = _build_dense_v1(case, args, model)
            report["dense"] = dense
            report["ok"] = bool(dense["output"]["finite"])
        elif child_mode == "sparse":
            sparse, _, _, _ = _build_sparse_v1(case, args, model)
            report["sparse"] = sparse
            report["ok"] = bool(sparse["output"]["finite"])
        elif child_mode == "parity":
            dense, dense_output, dense_plan = _build_dense_v1(case, args, model)
            sparse, sparse_output, sparse_blocks, sparse_candidates = _build_sparse_v1(
                case,
                args,
                model,
            )
            difference = dense_output - sparse_output
            dense_norm = float(np.linalg.norm(dense_output))
            absolute_error = float(np.linalg.norm(difference))
            relative_error = float(absolute_error / max(dense_norm, 1e-30))
            contract = _block_contract(dense_plan, sparse_blocks, sparse_candidates)
            dense_shift = dict(dense.get('local_shift') or {})
            sparse_shift = dict(sparse.get('local_shift') or {})
            shift_keys = ('block_lambda_pred', 'block_lambda_floor', 'block_lambda_effective')
            shift_parameter_matches = bool(
                all(
                    np.array_equal(
                        np.asarray(dense_shift.get(key, []), dtype=np.float64),
                        np.asarray(sparse_shift.get(key, []), dtype=np.float64),
                    )
                    for key in shift_keys
                )
            )
            final_blocks_match = bool(
                sparse['blocks']['candidate_block_count'] == sparse['blocks']['block_count']
                and sparse['blocks']['skipped_block_count'] == 0
                and sparse_shift.get('all_candidate_blocks_factorized') is True
            )
            contract.update({
                'shift_parameter_matches': shift_parameter_matches,
                'final_blocks_match': final_blocks_match,
            })
            contract['matches'] = bool(
                contract['matches'] and shift_parameter_matches and final_blocks_match
            )
            relative_error_pass = bool(
                np.isfinite(relative_error)
                and relative_error <= float(args.relative_error_tolerance)
            )
            report["dense"] = dense
            report["sparse"] = sparse
            report["parity"] = {
                "probe_source": str(args.probe_source),
                "absolute_l2_error": absolute_error,
                "relative_l2_error": relative_error,
                "max_absolute_error": float(np.max(np.abs(difference)))
                if difference.size
                else 0.0,
                "relative_error_tolerance": float(args.relative_error_tolerance),
                "relative_error_pass": relative_error_pass,
                "block_contract": contract,
                "precision_pass": bool(relative_error_pass and contract["matches"]),
            }
            report["ok"] = bool(
                dense["output"]["finite"]
                and sparse["output"]["finite"]
                and report["parity"]["precision_pass"]
            )
        else:
            raise ValueError(f"?? child_mode?{child_mode}")
    except Exception as exc:
        report = {
            "schema_version": 1,
            "stage": "stage1_sparse_learned_schwarz_memory_validation",
            "mode": str(child_mode),
            "ok": False,
            "failure_stage": "child_execution",
            "reason": repr(exc),
            "traceback": traceback.format_exc(),
        }
    report["child_elapsed_s"] = float(time.perf_counter() - started)
    return report


def _command_for_child(
    args: argparse.Namespace,
    child_mode: str,
    output_json: Path,
    time_log: Path,
) -> List[str]:
    command = [
        "/usr/bin/timeout",
        "-k",
        "30s",
        f"{max(1, int(args.timeout_sec))}s",
        "/usr/bin/time",
        "-v",
        "-o",
        str(time_log),
        sys.executable,
        str(Path(__file__).resolve()),
        "--_child-mode",
        str(child_mode),
        "--child-output-json",
        str(output_json),
        "--trajectory-dir",
        str(args.trajectory_dir),
        "--netlist-dir",
        str(args.netlist_dir),
        "--netlist-path",
        str(args.netlist_path),
        "--circuit-id",
        str(int(args.circuit_id)),
        "--step-offset",
        str(int(args.step_offset)),
        "--learned-schwarz-checkpoint",
        str(args.learned_schwarz_checkpoint),
        "--initial-guess-mode",
        str(args.initial_guess_mode),
        "--max-block-size",
        str(int(args.max_block_size)),
        "--min-block-size",
        str(int(args.min_block_size)),
        "--max-blocks",
        str(int(args.max_blocks)),
        "--probe-source",
        str(args.probe_source),
        "--relative-error-tolerance",
        repr(float(args.relative_error_tolerance)),
        "--memory-saving-target-ratio",
        repr(float(args.memory_saving_target_ratio)),
    ]
    command.append(
        "--apply-gmin-diagonal"
        if bool(args.apply_gmin_diagonal)
        else "--disable-gmin-diagonal"
    )
    if args.positive_gmin_only:
        command.append("--positive-gmin-only")
    return command


def _tail(text: str, limit: int = 2000) -> str:
    return text[-limit:] if len(text) > limit else text


def _run_isolated(
    args: argparse.Namespace,
    child_mode: str,
    run_dir: Path,
) -> Dict[str, Any]:
    child_dir = run_dir / str(child_mode)
    child_dir.mkdir(parents=True, exist_ok=True)
    output_json = child_dir / "result.json"
    time_log = child_dir / "time_v.log"
    stdout_log = child_dir / "stdout.log"
    stderr_log = child_dir / "stderr.log"
    environment = os.environ.copy()
    for key in THREAD_ENV:
        environment[key] = str(int(args.blas_threads))
    command = _command_for_child(args, child_mode, output_json, time_log)
    started = time.perf_counter()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=float(args.timeout_sec) + 45.0,
            check=False,
        )
        stdout_text = completed.stdout or ""
        stderr_text = completed.stderr or ""
        return_code = int(completed.returncode)
        timed_out = return_code == 124
    except subprocess.TimeoutExpired as exc:
        stdout_text = exc.stdout or ""
        stderr_text = exc.stderr or ""
        return_code = None
        timed_out = True
    stdout_log.write_text(str(stdout_text), encoding="utf-8")
    stderr_log.write_text(str(stderr_text), encoding="utf-8")
    elapsed = float(time.perf_counter() - started)
    time_info = _read_time_v(time_log)
    isolated = {
        "return_code": return_code,
        "timed_out": bool(timed_out),
        "wall_elapsed_s": elapsed,
        "max_rss_kb": time_info.get("isolated_max_rss_kb"),
        "time_v": time_info,
        "stdout_log": _relative_path(stdout_log),
        "stderr_log": _relative_path(stderr_log),
        "time_log": _relative_path(time_log),
    }
    if output_json.is_file():
        try:
            result = json.loads(output_json.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            result = {
                "schema_version": 1,
                "stage": "stage1_sparse_learned_schwarz_memory_validation",
                "mode": str(child_mode),
                "ok": False,
                "failure_stage": "child_result_parse",
                "reason": repr(exc),
            }
    else:
        result = {
            "schema_version": 1,
            "stage": "stage1_sparse_learned_schwarz_memory_validation",
            "mode": str(child_mode),
            "ok": False,
            "failure_stage": "child_output_missing",
            "reason": "child process did not write a JSON result",
            "stdout_tail": _tail(str(stdout_text)),
            "stderr_tail": _tail(str(stderr_text)),
        }
    result["isolated_process"] = isolated
    return result


def _rss_kb(result: Dict[str, Any]) -> Any:
    value = result.get("isolated_process", {}).get("max_rss_kb")
    if isinstance(value, (int, float)) and float(value) >= 0.0:
        return int(value)
    return None


def _actual_rss_summary(
    baseline: Dict[str, Any],
    dense: Dict[str, Any],
    sparse: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_rss = _rss_kb(baseline)
    dense_rss = _rss_kb(dense)
    sparse_rss = _rss_kb(sparse)
    summary: Dict[str, Any] = {
        "baseline_max_rss_kb": baseline_rss,
        "dense_max_rss_kb": dense_rss,
        "sparse_max_rss_kb": sparse_rss,
        "comparable": bool(
            isinstance(dense_rss, int)
            and isinstance(sparse_rss, int)
            and dense_rss > 0
        ),
    }
    if summary["comparable"]:
        saved = int(dense_rss - sparse_rss)
        summary.update(
            {
                "saved_rss_kb": saved,
                "saving_ratio": float(saved / dense_rss),
            }
        )
    return summary


def _preconditioner_incremental_rss(
    baseline: Dict[str, Any],
    dense: Dict[str, Any],
    sparse: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_rss = _rss_kb(baseline)
    dense_rss = _rss_kb(dense)
    sparse_rss = _rss_kb(sparse)
    summary: Dict[str, Any] = {
        "baseline_mode": "shared_case_and_model_only",
        "baseline_max_rss_kb": baseline_rss,
        "dense_max_rss_kb": dense_rss,
        "sparse_max_rss_kb": sparse_rss,
        "dense_increment_rss_kb": None,
        "sparse_increment_rss_kb": None,
        "sparse_increment_used_for_ratio_kb": None,
        "saving_ratio": None,
        "comparable": False,
        "formula": "(dense_increment_rss_kb - max(sparse_increment_rss_kb, 0)) / dense_increment_rss_kb",
    }
    if not all(isinstance(value, int) for value in (baseline_rss, dense_rss, sparse_rss)):
        summary["reason"] = "missing_isolated_max_rss"
        return summary

    dense_increment = int(dense_rss - baseline_rss)
    sparse_increment = int(sparse_rss - baseline_rss)
    sparse_for_ratio = int(max(sparse_increment, 0))
    summary.update(
        {
            "dense_increment_rss_kb": dense_increment,
            "sparse_increment_rss_kb": sparse_increment,
            "sparse_increment_used_for_ratio_kb": sparse_for_ratio,
            "negative_sparse_increment_clamped": bool(sparse_increment < 0),
        }
    )
    if dense_increment <= 0:
        summary["reason"] = "dense_increment_not_positive"
        return summary

    summary.update(
        {
            "comparable": True,
            "saving_ratio": float(
                (dense_increment - sparse_for_ratio) / dense_increment
            ),
        }
    )
    return summary


def _run_parent(args: argparse.Namespace) -> Dict[str, Any]:
    work_dir = _repo_path(args.work_dir, "work_dir")
    work_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(
        tempfile.mkdtemp(
            prefix="verify_sparse_learned_schwarz_",
            dir=str(work_dir),
        )
    )
    if args.mode in {"baseline", "dense", "sparse"}:
        result = _run_isolated(args, args.mode, run_dir)
        result["requested_mode"] = str(args.mode)
        result["run_dir"] = _relative_path(run_dir)
        result["passed"] = bool(result.get("ok"))
        return result

    baseline = _run_isolated(args, "baseline", run_dir)
    dense = _run_isolated(args, "dense", run_dir)
    sparse = _run_isolated(args, "sparse", run_dir)
    parity = _run_isolated(args, "parity", run_dir)
    parity_data = dict(parity.get("parity") or {})
    sparse_memory = dict((sparse.get("sparse") or {}).get("memory") or {})
    precision_pass = bool(parity_data.get("precision_pass"))
    memory_ratio = sparse_memory.get("estimated_peak_memory_saving_ratio")
    memory_pass = bool(
        isinstance(memory_ratio, (int, float))
        and float(memory_ratio) > float(args.memory_saving_target_ratio)
    )
    incremental_rss = _preconditioner_incremental_rss(baseline, dense, sparse)
    incremental_rss_ratio = incremental_rss.get("saving_ratio")
    incremental_memory_pass = bool(
        incremental_rss.get("comparable")
        and isinstance(incremental_rss_ratio, (int, float))
        and float(incremental_rss_ratio) > float(args.memory_saving_target_ratio)
    )
    incremental_rss["strict_target"] = f"> {float(args.memory_saving_target_ratio):.6g}"
    incremental_rss["passed"] = incremental_memory_pass
    all_children_ok = bool(
        baseline.get("ok") and dense.get("ok") and sparse.get("ok") and parity.get("ok")
    )
    return {
        "schema_version": 1,
        "stage": "stage1_sparse_learned_schwarz_memory_validation",
        "mode": "parity",
        "requested_mode": "parity",
        "run_dir": _relative_path(run_dir),
        "matrix": parity.get("matrix") or sparse.get("matrix") or dense.get("matrix") or baseline.get("matrix"),
        "step": parity.get("step") or sparse.get("step") or dense.get("step") or baseline.get("step"),
        "checkpoint": parity.get("checkpoint")
        or sparse.get("checkpoint")
        or dense.get("checkpoint")
        or baseline.get("checkpoint"),
        "configuration": parity.get("configuration")
        or sparse.get("configuration")
        or dense.get("configuration")
        or baseline.get("configuration"),
        "preconditioner_incremental_rss": incremental_rss,
        "children": {
            "baseline": baseline,
            "dense": dense,
            "sparse": sparse,
            "parity": parity,
        },
        "acceptance": {
            "all_children_ok": all_children_ok,
            "precision": {
                "relative_l2_error": parity_data.get("relative_l2_error"),
                "relative_error_tolerance": float(args.relative_error_tolerance),
                "block_contract_matches": (parity_data.get("block_contract") or {}).get(
                    "matches"
                ),
                "passed": precision_pass,
            },
            "conservative_memory": {
                "dense_v1_global_matrix_bytes": sparse_memory.get(
                    "dense_v1_global_matrix_bytes"
                ),
                "sparse_retained_estimated_bytes": sparse_memory.get(
                    "sparse_retained_estimated_bytes"
                ),
                "sparse_peak_estimated_bytes": sparse_memory.get(
                    "sparse_peak_estimated_bytes"
                ),
                "estimated_peak_memory_saving_ratio": memory_ratio,
                "strict_target": f"> {float(args.memory_saving_target_ratio):.6g}",
                "passed": memory_pass,
            },
            "isolated_peak_rss": _actual_rss_summary(baseline, dense, sparse),
            "preconditioner_incremental_rss": incremental_rss,
            "actual_incremental_memory": {
                "saving_ratio": incremental_rss_ratio,
                "strict_target": f"> {float(args.memory_saving_target_ratio):.6g}",
                "comparable": bool(incremental_rss.get("comparable")),
                "passed": incremental_memory_pass,
            },
        },
        "passed": bool(
            all_children_ok
            and precision_pass
            and memory_pass
            and incremental_memory_pass
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "\u771f\u5b9e 10x10 \u7535\u8def\u9996\u4e2a Newton \u6b65\u7684 "
            "learned Schwarz V1 \u7a20\u5bc6/\u7a00\u758f\u4e00\u81f4\u6027\u4e0e\u5185\u5b58\u9a8c\u8bc1\u3002"
        )
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "dense", "sparse", "parity"),
        default="parity",
    )
    parser.add_argument("--trajectory-dir", default=DEFAULT_TRAJECTORY_DIR)
    parser.add_argument("--netlist-dir", default=DEFAULT_NETLIST_DIR)
    parser.add_argument("--netlist-path", default="")
    parser.add_argument("--circuit-id", type=int, default=0)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--positive-gmin-only", action="store_true")
    parser.add_argument("--learned-schwarz-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--initial-guess-mode", choices=sorted(INITIAL_GUESS_MODES), default=INITIAL_GUESS_MODE_RHSOLD)
    parser.add_argument("--max-block-size", type=int, default=32)
    parser.add_argument("--min-block-size", type=int, default=2)
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument(
        "--probe-source",
        choices=("rhsnew", "initial_residual", "ones"),
        default="rhsnew",
    )
    parser.add_argument("--relative-error-tolerance", type=float, default=1e-12)
    parser.add_argument("--memory-saving-target-ratio", type=float, default=0.50)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--timeout-sec", type=float, default=600.0)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument("--apply-gmin-diagonal", action="store_true", default=True)
    parser.add_argument(
        "--disable-gmin-diagonal",
        dest="apply_gmin_diagonal",
        action="store_false",
    )
    parser.add_argument(
        "--_child-mode",
        choices=("baseline", "dense", "sparse", "parity"),
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--child-output-json", default="", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not bool(args.apply_gmin_diagonal):
        raise ValueError("schema v4 verification requires --apply-gmin-diagonal")
    if int(args.max_block_size) < 1 or int(args.min_block_size) < 1:
        raise ValueError("block sizes must be positive")
    if int(args.min_block_size) > int(args.max_block_size):
        raise ValueError("min_block_size cannot exceed max_block_size")
    if float(args.timeout_sec) <= 0.0:
        raise ValueError("timeout_sec must be positive")
    if float(args.relative_error_tolerance) < 0.0:
        raise ValueError("relative_error_tolerance cannot be negative")
    if not 0.0 <= float(args.memory_saving_target_ratio) < 1.0:
        raise ValueError("memory_saving_target_ratio must lie in [0, 1)")

    if args._child_mode:
        if not args.child_output_json:
            raise ValueError("child_output_json is required for internal child mode")
        output_path = _repo_path(args.child_output_json, "child_output_json")
        _write_json(output_path, _run_child(args, args._child_mode))
        return

    result = _run_parent(args)
    if args.output_json:
        _write_json(_repo_path(args.output_json, "output_json"), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    if args.strict and not bool(result.get("passed")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
