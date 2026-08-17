import argparse
import hashlib
import json
import math
import os
import resource
import re
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres, spilu

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pypath.aggregator.preconditioner_targets import infer_row_kind
from pypath.baseline import DEFAULT_AGGREGATOR_ROOT
from pypath.preconditioner.block_schwarz import BlockPlanConfig, BlockSchwarzPlan, build_block_schwarz_plan
from pypath.preconditioner.linear_system_contract import (
    compute_initial_residual,
    resolve_initial_guess,
    validate_learned_schwarz_checkpoint_contract,
)
from pypath.preconditioner.learned_schwarz import (
    BoundaryCorrectionPreconditioner,
    LearnedSchwarzPreconditioner,
    build_learned_schwarz_sample,
)
from pypath.preconditioner.schur_interface import (
    ExplicitSchurInterfacePreconditioner,
    HybridSparseSchurLowRankPreconditioner,
    LearnedLocalSparseSchurPreconditioner,
    LearningAugmentedSparseSchurPreconditioner,
    LocalSchurAdditiveInversePreconditioner,
    LearnedSchurDiagonalPreconditioner,
    LearnedSparseSchurPreconditioner,
    LocalSparseSchurPreconditioner,
    PowerSchurPreconditioner,
    PowerSparseSchurPreconditioner,
    PowerSparseSchurArnoldiPreconditioner,
    SparseSchurPreconditioner,
    SelectedLocalSparseSchurPreconditioner,
    load_learned_local_sparse_schur_model,
    load_learned_schur_diagonal_model,
    load_learned_sparse_schur_model,
)
from pypath.utils.ngspice_utils import read_J, read_continuation_step, run_ngspice_linear_system_corpus


DEFAULT_OUTPUT_ROOT = os.path.join(DEFAULT_AGGREGATOR_ROOT, "value", "external_gmres")
DEFAULT_NETLIST_DIR = os.path.join(DEFAULT_AGGREGATOR_ROOT, "generated_netlists")
DEFAULT_RTOL = 1e-8
DEFAULT_ATOL = 1e-10
DEFAULT_RESTART = 30
DEFAULT_MAX_ITERS = 120
DEFAULT_SCALE_CLIP = 12.0
DEFAULT_RESIDUAL_WORSEN_RATIO = 1.01
DEFAULT_RESIDUAL_WORSEN_ABS = 1e-12
DEFAULT_RESIDUAL_SUCCESS_RTOL = 1e-8
DEFAULT_RESIDUAL_SUCCESS_ATOL = 1e-14
POWER_SPARSE_SCHUR_ARNOLDI_MODES = {f"power_sparse_schur_m{m}_r{r}" for m in (0, 1, 2, 3) for r in (2, 4, 8, 16, 32)}


def _residual_ratio(chosen_raw_residual: Any, initial_raw_residual: Any) -> Optional[float]:
    if chosen_raw_residual is None or initial_raw_residual is None:
        return None
    try:
        return float(chosen_raw_residual) / max(float(initial_raw_residual), 1e-30)
    except (TypeError, ValueError, OverflowError):
        return None


def _true_rel_residual(chosen_raw_residual: Any, rhs_norm: Any) -> Optional[float]:
    if chosen_raw_residual is None or rhs_norm is None:
        return None
    try:
        return float(chosen_raw_residual) / max(float(rhs_norm), 1e-30)
    except (TypeError, ValueError, OverflowError):
        return None


def _success_by_true_residual(chosen_raw_residual: Any, rhs_norm: Any) -> Optional[bool]:
    value = _true_rel_residual(chosen_raw_residual, rhs_norm)
    if value is None:
        return None
    return bool(value < DEFAULT_RESIDUAL_SUCCESS_RTOL)


def _success_by_residual_ratio(chosen_raw_residual: Any, initial_raw_residual: Any) -> Optional[bool]:
    ratio = _residual_ratio(chosen_raw_residual, initial_raw_residual)
    if ratio is None:
        return None
    return bool(ratio < DEFAULT_RESIDUAL_SUCCESS_RTOL)


def _chosen_success(chosen_raw_residual: Any, rhs_norm: Any, initial_raw_residual: Any) -> Optional[bool]:
    true_success = _success_by_true_residual(chosen_raw_residual, rhs_norm)
    ratio_success = _success_by_residual_ratio(chosen_raw_residual, initial_raw_residual)
    observed = [value for value in (true_success, ratio_success) if value is not None]
    if not observed:
        return None
    return bool(any(observed))


def _residual_success(chosen_raw_residual: Any, initial_raw_residual: Any) -> Optional[bool]:
    # Backward-compatible alias for older consumers. New summaries should use
    # success_by_true_residual, success_by_residual_ratio, and chosen_success.
    return _success_by_residual_ratio(chosen_raw_residual, initial_raw_residual)

DEFAULT_SIDECAR_SCHEMA_VERSION = 1
DEFAULT_SIDECAR_SCALE_MODE = "log10_scale"
DEFAULT_BENCHMARK_MODES = [
    "identity",
    "jacobi",
    "row_sum",
    "ilu0",
    "ilut",
    "ilu_drop_tol",
    "ilu_fill_factor",
    "generic_block_jacobi",
    "branch_incident_block",
    "cell_block_jacobi",
    "explicit_schur_interface",
    "exact_schur_p_equals_s",
    "learned_diagonal_synthetic",
    "learned_schwarz_v1",
    "learned_boundary_correction_v1",
]
RANGE_TOKEN_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")


def _supported_sidecar_scale_modes() -> Tuple[str, ...]:
    return ("log10_scale", "log_scale_dense", "log10_dense_scale")


@dataclass
class PrototypeConfig:
    requested_mode: str
    fallback_mode: str
    restart: int
    max_iters: int
    rtol: float
    atol: float
    use_rhsold_as_x0: bool
    apply_gmin_diagonal: bool
    residual_worsen_ratio: float
    residual_worsen_abs: float
    sidecar_path: str
    synthetic_source_mode: str = "row_sum"
    block_mode: str = "branch_incident"
    max_block_size: int = 3
    min_block_size: int = 2
    max_blocks: int = 0
    max_total_block_nnz: int = 0
    uncovered_row_policy: str = "row_sum"
    learned_schwarz_checkpoint: str = ""
    learned_schur_checkpoint: str = ""
    learned_sparse_schur_checkpoint: str = ''
    sparse_schur_edge_budget: int = 64
    sparse_schur_candidate_edge_limit: int = 512
    sparse_schur_max_nnz: int = 0
    sparse_schur_max_degree: int = 0
    sparse_schur_max_exact_entries: int = 0
    sparse_schur_diagonal_shift: float = 1e-8
    sparse_schur_row_topk: int = 2
    sparse_schur_relative_threshold: float = 0.05
    local_schur_budget_multiplier: float = 2.0
    local_schur_additive_include_diagonal: bool = False
    local_schur_additive_diagonal_weight: float = 0.0
    local_schur_additive_local_weight: float = 1.0
    local_schur_additive_cluster_weight: float = 0.0
    local_schur_additive_cluster_hops: int = 0
    local_schur_additive_max_cluster_size: int = 16
    local_schur_additive_patch_svd_rcond: float = 0.0
    local_schur_additive_inner_iterations: int = 1
    learned_local_sparse_schur_checkpoint: str = ''
    learned_local_sparse_schur_selection_policy: str = 'learned_union_slack'
    learned_local_sparse_schur_slack_fraction: float = 0.1
    learned_local_sparse_schur_slack_min: int = 1
    learned_local_sparse_schur_probe_iterations: int = 1
    schur_low_rank_rank: int = 4
    schur_low_rank_mode: str = 'slow_eig'
    schur_low_rank_strength: float = 1.0


def _parse_circuit_ids(raw_value: str) -> List[int]:
    circuit_ids: List[int] = []
    seen = set()
    for token in str(raw_value).split(","):
        token = token.strip()
        if not token:
            continue
        match = RANGE_TOKEN_RE.match(token)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            step = 1 if start <= end else -1
            for circuit_id in range(start, end + step, step):
                if circuit_id not in seen:
                    seen.add(circuit_id)
                    circuit_ids.append(circuit_id)
            continue
        circuit_id = int(token)
        if circuit_id not in seen:
            seen.add(circuit_id)
            circuit_ids.append(circuit_id)
    if not circuit_ids:
        raise ValueError("No circuit ids were parsed")
    return circuit_ids


def _build_ilu_operator(
    matrix: np.ndarray,
    *,
    drop_tol: float,
    fill_factor: float,
    mode_name: str,
) -> Tuple[Any, Dict[str, Any], float]:
    load_start = time.perf_counter()
    sparse = csr_matrix(matrix).tocsc()
    try:
        factor = spilu(
            sparse,
            drop_tol=float(drop_tol),
            fill_factor=float(fill_factor),
            permc_spec="NATURAL",
            diag_pivot_thresh=0.0,
            relax=0,
            panel_size=1,
        )
        load_time = time.perf_counter() - load_start
        info = {
            "preconditioner_mode": mode_name,
            "fallback_reason": None,
            f"{mode_name}_loaded": True,
            "drop_tol": float(drop_tol),
            "fill_factor": float(fill_factor),
        }
        return factor.solve, info, load_time
    except Exception as exc:
        load_time = time.perf_counter() - load_start
        info = {
            "preconditioner_mode": mode_name,
            "fallback_reason": f"{mode_name}_failed:{exc}",
            f"{mode_name}_loaded": False,
            "drop_tol": float(drop_tol),
            "fill_factor": float(fill_factor),
        }
        return None, info, load_time


def _build_ilu0_operator(matrix: np.ndarray) -> Tuple[Any, Dict[str, Any], float]:
    return _build_ilu_operator(
        matrix,
        drop_tol=0.0,
        fill_factor=1.0,
        mode_name="ilu0",
    )


def _build_block_jacobi_plan(
    *,
    matrix: np.ndarray,
    node_map: Dict[str, int],
    config: PrototypeConfig,
    netlist_path: str,
) -> BlockSchwarzPlan:
    return build_block_schwarz_plan(
        matrix=matrix,
        node_map=node_map,
        netlist_path=netlist_path,
        config=BlockPlanConfig(
            block_mode=str(config.block_mode),
            max_block_size=int(config.max_block_size),
            min_block_size=int(config.min_block_size),
            max_blocks=int(config.max_blocks),
            max_total_block_nnz=int(config.max_total_block_nnz),
            uncovered_row_policy=str(config.uncovered_row_policy),
        ),
    )


def _prepare_netlist_dir(
    source_netlist_dir: str,
    circuit_id: int,
    mode: str,
    task_root: str,
) -> Tuple[str, str]:
    if mode == "as_is":
        return source_netlist_dir, os.path.join(source_netlist_dir, f"{circuit_id}.sp")

    task_netlist_dir = os.path.join(task_root, f"prototype_netlists_{mode}")
    os.makedirs(task_netlist_dir, exist_ok=True)
    target_path = os.path.join(task_netlist_dir, f"{circuit_id}.sp")

    if mode == "noop":
        with open(target_path, "w", encoding="utf-8") as handle:
            handle.write("* no newton step prototype\n")
            handle.write(".TITLE external_gmres_noop\n")
            handle.write(".CONTROL\n")
            handle.write("quit\n")
            handle.write(".ENDC\n")
            handle.write(".END\n")
        return task_netlist_dir, target_path

    source_path = os.path.join(source_netlist_dir, f"{circuit_id}.sp")
    with open(source_path, "r", encoding="utf-8") as handle:
        netlist_text = handle.read()

    option_line = ".OPTIONS sparse\n" if mode == "force_sparse" else ".OPTIONS klu\n"
    marker = ".OPTIONS gmin=1e-12 Reltol=1e-3\n"
    if marker in netlist_text:
        netlist_text = netlist_text.replace(marker, marker + option_line, 1)
    else:
        netlist_text = option_line + netlist_text

    with open(target_path, "w", encoding="utf-8") as handle:
        handle.write(netlist_text)
    return task_netlist_dir, target_path


def _compute_node_map_hash(node_map: Dict[str, int]) -> str:
    items = sorted((int(idx), str(name).lower()) for name, idx in node_map.items())
    payload = "".join(f"{idx}:{name}\n" for idx, name in items)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_system_matrix(step: Dict[str, Any], *, apply_gmin_diagonal: bool) -> np.ndarray:
    jacobian = np.asarray(step.get("jacobian", np.zeros((0, 0))))
    if jacobian.ndim != 2 or jacobian.shape[0] == 0 or jacobian.shape[0] != jacobian.shape[1]:
        raise ValueError("step does not contain a usable square Jacobian matrix")

    if np.iscomplexobj(jacobian):
        imag_max = float(np.max(np.abs(np.imag(jacobian)))) if jacobian.size else 0.0
        if imag_max > 1e-12:
            raise ValueError(f"jacobian has non-negligible imaginary part: {imag_max}")
        jacobian = np.real(jacobian)

    matrix = np.array(jacobian, dtype=np.float64, copy=True)
    gmin_val = float(step.get("gmin_val", step.get("meta", {}).get("gmin", 0.0)) or 0.0)
    if apply_gmin_diagonal and gmin_val != 0.0:
        diag_idx = np.diag_indices_from(matrix)
        matrix[diag_idx] += gmin_val
    return matrix


def _build_analytic_scales(mode: str, matrix: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    if mode == "identity":
        return np.ones(matrix.shape[0], dtype=np.float64)

    row_abs_sum = np.abs(matrix).sum(axis=1)
    diag_abs = np.abs(np.diag(matrix))

    if mode == "jacobi":
        return 1.0 / np.maximum(diag_abs, eps)
    if mode == "row_sum":
        return 1.0 / np.maximum(row_abs_sum, eps)
    raise ValueError(f"Unsupported analytic preconditioner mode: {mode}")


def _build_synthetic_learned_scales(
    *,
    matrix: np.ndarray,
    node_map: Dict[str, int],
    source_mode: str,
    scale_clip: float = DEFAULT_SCALE_CLIP,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    analytic_scales = _build_analytic_scales(source_mode, matrix)
    log_scale_dense = np.log10(np.maximum(analytic_scales, 1e-30))
    clipped_logs = np.clip(log_scale_dense, -scale_clip, scale_clip)
    scales = np.power(10.0, clipped_logs)

    load_info = {
        "sidecar_path": None,
        "sidecar_loaded": True,
        "invalid_row_count": 0,
        "branch_row_count": 0,
        "fallback_reason": None,
        "scale_clip": float(scale_clip),
        "default_scale": 1.0,
        "synthetic_source_mode": source_mode,
    }
    idx_to_name = {int(idx): str(name).lower() for name, idx in node_map.items()}
    for zero_idx in range(matrix.shape[0]):
        row_name = idx_to_name.get(zero_idx + 1, "")
        if infer_row_kind(row_name) == "branch":
            scales[zero_idx] = 1.0
            load_info["branch_row_count"] += 1
    return scales.astype(np.float64), load_info


def build_synthetic_sidecar_payload(
    step: Dict[str, Any],
    *,
    source_mode: str = "row_sum",
    scale_clip: float = DEFAULT_SCALE_CLIP,
    default_scale: float = 1.0,
) -> Dict[str, Any]:
    matrix = _make_system_matrix(step, apply_gmin_diagonal=True)
    node_map = step.get("node_map", {})
    analytic_scales = _build_analytic_scales(source_mode, matrix)
    log_scale_dense = np.log10(np.maximum(analytic_scales, 1e-30))
    valid_mask_dense = np.ones(matrix.shape[0], dtype=bool)
    idx_to_name = {int(idx): str(name).lower() for name, idx in node_map.items()}

    for zero_idx in range(matrix.shape[0]):
        row_name = idx_to_name.get(zero_idx + 1, "")
        if infer_row_kind(row_name) == "branch":
            log_scale_dense[zero_idx] = 0.0
            valid_mask_dense[zero_idx] = False

    return {
        "schema_version": DEFAULT_SIDECAR_SCHEMA_VERSION,
        "circuit_id": step.get("meta", {}).get("circuit_id"),
        "time": float(step.get("time", step.get("meta", {}).get("time", 0.0))),
        "gmin_val": float(step.get("gmin_val", step.get("meta", {}).get("gmin", 0.0))),
        "iteration": int(step.get("iteration", step.get("meta", {}).get("iteration", -1))),
        "matrix_size": int(matrix.shape[0]),
        "node_map_hash": _compute_node_map_hash(node_map),
        "scale_mode": DEFAULT_SIDECAR_SCALE_MODE,
        "scale_clip": float(scale_clip),
        "default_scale": float(default_scale),
        "log_scale_dense": np.asarray(log_scale_dense, dtype=np.float64).tolist(),
        "valid_mask_dense": np.asarray(valid_mask_dense, dtype=bool).tolist(),
        "confidence_dense": np.ones(matrix.shape[0], dtype=np.float64).tolist(),
        "rows_debug": {
            "synthetic_source_mode": str(source_mode),
        },
    }


def write_sidecar_payload(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _load_learned_scales(
    sidecar_path: str,
    *,
    node_map: Dict[str, int],
    matrix_size: int,
    scale_clip_default: float = DEFAULT_SCALE_CLIP,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    load_info = {
        "sidecar_path": sidecar_path,
        "sidecar_loaded": False,
        "invalid_row_count": 0,
        "branch_row_count": 0,
        "fallback_reason": None,
    }
    with open(sidecar_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    schema_version = int(payload.get("schema_version", -1))
    if schema_version != int(DEFAULT_SIDECAR_SCHEMA_VERSION):
        load_info["fallback_reason"] = "sidecar_invalid_schema_version"
        raise ValueError(load_info["fallback_reason"])

    scale_mode = str(payload.get("scale_mode", ""))
    if scale_mode not in _supported_sidecar_scale_modes():
        load_info["fallback_reason"] = "sidecar_invalid_scale_mode"
        raise ValueError(load_info["fallback_reason"])

    payload_matrix_size = int(payload.get("matrix_size", -1))
    if payload_matrix_size != int(matrix_size):
        load_info["fallback_reason"] = "sidecar_invalid_matrix_size"
        raise ValueError(load_info["fallback_reason"])

    expected_hash = payload.get("node_map_hash")
    if expected_hash:
        actual_hash = _compute_node_map_hash(node_map)
        if str(expected_hash) != actual_hash:
            load_info["fallback_reason"] = "sidecar_invalid_node_map_hash"
            raise ValueError(load_info["fallback_reason"])

    log_scale_dense = np.asarray(payload.get("log_scale_dense", []), dtype=np.float64)
    if log_scale_dense.shape[0] != int(matrix_size):
        load_info["fallback_reason"] = "sidecar_invalid_scale_length"
        raise ValueError(load_info["fallback_reason"])
    if not np.all(np.isfinite(log_scale_dense)):
        load_info["fallback_reason"] = "sidecar_invalid_nan_inf"
        raise ValueError(load_info["fallback_reason"])

    if "valid_mask_dense" in payload:
        valid_mask = np.asarray(payload.get("valid_mask_dense", []), dtype=bool)
        if valid_mask.shape[0] != int(matrix_size):
            load_info["fallback_reason"] = "sidecar_invalid_valid_mask_length"
            raise ValueError(load_info["fallback_reason"])
    else:
        valid_mask = np.ones(matrix_size, dtype=bool)

    default_scale = float(payload.get("default_scale", 1.0))
    if not np.isfinite(default_scale) or default_scale <= 0.0:
        load_info["fallback_reason"] = "sidecar_invalid_default_scale"
        raise ValueError(load_info["fallback_reason"])

    scale_clip = float(payload.get("scale_clip", scale_clip_default))
    if not np.isfinite(scale_clip) or scale_clip <= 0.0:
        load_info["fallback_reason"] = "sidecar_invalid_scale_clip"
        raise ValueError(load_info["fallback_reason"])
    clipped_logs = np.clip(log_scale_dense, -scale_clip, scale_clip)
    scales = np.power(10.0, clipped_logs)

    idx_to_name = {int(idx): str(name).lower() for name, idx in node_map.items()}
    for zero_idx in range(int(matrix_size)):
        row_name = idx_to_name.get(zero_idx + 1, "")
        if infer_row_kind(row_name) == "branch":
            scales[zero_idx] = 1.0
            load_info["branch_row_count"] += 1
            continue
        if not valid_mask[zero_idx] or (not np.isfinite(scales[zero_idx])) or scales[zero_idx] <= 0.0:
            scales[zero_idx] = default_scale
            load_info["invalid_row_count"] += 1

    load_info["sidecar_loaded"] = True
    load_info["scale_clip"] = scale_clip
    load_info["default_scale"] = default_scale
    return scales.astype(np.float64), load_info


def _build_preconditioner_scales(
    requested_mode: str,
    fallback_mode: str,
    matrix: np.ndarray,
    step: Dict[str, Any],
    sidecar_path: str,
    synthetic_source_mode: str,
) -> Tuple[np.ndarray, str, Dict[str, Any], float]:
    load_start = time.perf_counter()
    matrix_size = int(matrix.shape[0])
    if requested_mode not in {"learned_diagonal", "learned_diagonal_synthetic"}:
        return (
            _build_analytic_scales(requested_mode, matrix),
            requested_mode,
            {"fallback_reason": None},
            time.perf_counter() - load_start,
        )

    if requested_mode == "learned_diagonal_synthetic":
        scales, load_info = _build_synthetic_learned_scales(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            source_mode=str(synthetic_source_mode),
        )
        return scales, requested_mode, load_info, time.perf_counter() - load_start

    if not sidecar_path:
        scales = _build_analytic_scales(fallback_mode, matrix)
        return scales, fallback_mode, {"fallback_reason": "sidecar_missing"}, time.perf_counter() - load_start

    try:
        scales, load_info = _load_learned_scales(
            sidecar_path,
            node_map=step.get("node_map", {}),
            matrix_size=matrix_size,
        )
        return scales, requested_mode, load_info, time.perf_counter() - load_start
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        scales = _build_analytic_scales(fallback_mode, matrix)
        reason = str(exc) if str(exc) else "sidecar_load_failed"
        return scales, fallback_mode, {"fallback_reason": reason}, time.perf_counter() - load_start



_LEARNED_SCHWARZ_MODEL_CACHE: Dict[str, torch.nn.Module] = {}


def _load_learned_schwarz_model(
    checkpoint_path: str,
    *,
    dtype: torch.dtype,
    initial_guess_mode: str,
    expected_model_kind: str,
) -> torch.nn.Module:
    if not checkpoint_path:
        raise ValueError("learned_schwarz_checkpoint_missing")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    validate_learned_schwarz_checkpoint_contract(
        checkpoint,
        expected_initial_guess_mode=initial_guess_mode,
    )
    model_kind = str(checkpoint.get("model_kind", ""))
    if model_kind != expected_model_kind:
        raise ValueError(
            f"learned Schwarz checkpoint model_kind mismatch: expected {expected_model_kind}, got {model_kind}"
        )
    cache_key = (
        f"{os.path.abspath(checkpoint_path)}::{dtype}::{initial_guess_mode}::{model_kind}"
    )
    cached = _LEARNED_SCHWARZ_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    model_args = dict(checkpoint.get("model_args") or {})
    if model_kind == "boundary_correction_v1":
        if "correction_feature_dim" not in model_args:
            model_args["correction_feature_dim"] = 0
        model_args.setdefault("projection_mode", "none")
        model_args.setdefault("projection_max_scale", 1.0)
        model_args.setdefault("local_solution_gain_limit", 0.0)
        model_args.setdefault("block_contribution_budget_ratio", 0.0)
        model_args.setdefault("block_contribution_absolute_cap", 0.0)
        model = BoundaryCorrectionPreconditioner(**model_args)
    else:
        model = LearnedSchwarzPreconditioner(**model_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(dtype=dtype)
    model.eval()
    _LEARNED_SCHWARZ_MODEL_CACHE[cache_key] = model
    return model


def _build_block_metadata(plan: BlockSchwarzPlan) -> Dict[str, Any]:
    if hasattr(plan, "metadata"):
        return plan.metadata()
    memory_estimate = int(plan.total_block_nnz * 8 + sum(int(rows.shape[0]) ** 2 * 8 for rows in plan.blocks))
    return {
        "preconditioner_mode": "block_jacobi",
        "block_mode": plan.block_mode,
        "semantic_block_mode": plan.block_mode,
        "num_blocks": int(len(plan.blocks)),
        "covered_rows": int(np.count_nonzero(plan.covered_mask)),
        "coverage_ratio": float(plan.coverage_ratio),
        "uncovered_rows": int(plan.covered_mask.shape[0] - np.count_nonzero(plan.covered_mask)),
        "max_block_size": int(plan.max_block_size),
        "total_block_nnz": int(plan.total_block_nnz),
        "memory_estimate": memory_estimate,
        "candidate_block_count": int(plan.candidate_block_count),
        "skipped_block_count": int(plan.skipped_block_count),
        "factor_modes": list(plan.factor_modes),
        "blocks": [
            {
                "block_id": int(block_id),
                "rows": [int(row) + 1 for row in rows.tolist()],
                "size": int(rows.shape[0]),
            }
            for block_id, rows in enumerate(plan.blocks)
        ],
    }


def _attach_block_candidates(
    step_summary: Dict[str, Any],
    block_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    linear_system_key = (
        f"c{int(step_summary.get('circuit_id', -1))}"
        f"_step{int(step_summary.get('selected_step_index', -1))}"
        f"_t{float(step_summary.get('time', 0.0)):.17e}"
        f"_g{float(step_summary.get('gmin_val', 0.0)):.17e}"
        f"_it{int(step_summary.get('corpus_iteration', -1))}"
    )
    attached: List[Dict[str, Any]] = []
    for candidate in block_candidates:
        item = dict(candidate)
        item.update(
            {
                "linear_system_key": linear_system_key,
                "circuit_id": int(step_summary.get("circuit_id", -1)),
                "selected_step_index": int(step_summary.get("selected_step_index", -1)),
                "corpus_iteration": int(step_summary.get("corpus_iteration", -1)),
                "time": float(step_summary.get("time", 0.0)),
                "gmin_val": float(step_summary.get("gmin_val", 0.0)),
                "matrix_size": int(step_summary.get("matrix_size", -1)),
                "node_map_hash": step_summary.get("node_map_hash"),
            }
        )
        attached.append(item)
    return attached


def _workpoint_key(
    *,
    circuit_id: int,
    time_value: float,
    gmin_value: float,
    iteration: int,
) -> Tuple[int, str, str, int]:
    return (
        int(circuit_id),
        f"{float(time_value):.17e}",
        f"{float(gmin_value):.17e}",
        int(iteration),
    )


def _load_workpoint_manifest(manifest_path: str) -> Dict[int, List[Dict[str, Any]]]:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError("workpoint_manifest_schema_version_must_be_1")
    raw_workpoints = payload.get("workpoints")
    if not isinstance(raw_workpoints, list) or not raw_workpoints:
        raise ValueError("workpoint_manifest_requires_nonempty_workpoints")

    by_circuit: Dict[int, List[Dict[str, Any]]] = {}
    seen = set()
    for manifest_index, item in enumerate(raw_workpoints):
        if not isinstance(item, dict):
            raise ValueError("workpoint_manifest_item_must_be_object")
        try:
            circuit_id = int(item["circuit_id"])
            time_value = float(item["time"])
            gmin_value = float(item["gmin_val"])
            iteration = int(item["iteration"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("workpoint_manifest_item_has_invalid_key_fields") from exc
        if (
            isinstance(item.get("circuit_id"), bool)
            or isinstance(item.get("iteration"), bool)
            or circuit_id < 0
            or iteration < 0
            or not np.isfinite(time_value)
            or not np.isfinite(gmin_value)
        ):
            raise ValueError("workpoint_manifest_item_has_nonfinite_or_negative_key_fields")
        key = _workpoint_key(
            circuit_id=circuit_id,
            time_value=time_value,
            gmin_value=gmin_value,
            iteration=iteration,
        )
        if key in seen:
            raise ValueError(f"workpoint_manifest_duplicate:{key}")
        seen.add(key)
        by_circuit.setdefault(circuit_id, []).append(
            {"key": key, "manifest_index": int(manifest_index)}
        )
    return by_circuit


def _load_trajectory_linear_system_steps(
    *,
    trajectory_dir: str,
    circuit_id: int,
    netlist_path: str,
    requested_workpoints: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    step_pattern = re.compile(
        rf"(?:continuation_)?circuit_{int(circuit_id)}_time_([0-9.eE+-]+)_gmin_(?:\d+_)?([0-9.eE+-]+)_iter_(\d+)\.txt$"
    )
    requested_by_key: Dict[Tuple[int, str, str, int], Dict[str, Any]] = {}
    if requested_workpoints is not None:
        for entry in requested_workpoints:
            key = tuple(entry.get("key", ()))
            if len(key) != 4 or int(key[0]) != int(circuit_id):
                raise ValueError("workpoint_manifest_circuit_mismatch")
            if key in requested_by_key:
                raise ValueError(f"workpoint_manifest_duplicate:{key}")
            requested_by_key[key] = entry

    steps: List[Dict[str, Any]] = []
    found_keys = set()
    if not trajectory_dir or not os.path.isdir(trajectory_dir):
        return {
            "success": False,
            "reason": f"trajectory_dir_not_found:{trajectory_dir}",
            "steps": [],
            "task_dir": trajectory_dir,
            "stderr": "",
            "stdout": "",
        }

    for filename in os.listdir(trajectory_dir):
        match = step_pattern.match(filename)
        if not match:
            continue
        time_val_str, gmin_val_str, iter_id_str = match.groups()
        time_value = float(time_val_str)
        gmin_value = float(gmin_val_str)
        iteration = int(iter_id_str)
        key = _workpoint_key(
            circuit_id=int(circuit_id),
            time_value=time_value,
            gmin_value=gmin_value,
            iteration=iteration,
        )
        manifest_entry = requested_by_key.get(key)
        if requested_workpoints is not None and manifest_entry is None:
            continue
        if key in found_keys:
            raise ValueError(f"trajectory_duplicate_workpoint:{key}")
        filepath = os.path.join(trajectory_dir, filename)
        jac_filepath = filepath[:-4] + "_jac.txt"
        if not os.path.exists(jac_filepath):
            if requested_workpoints is not None:
                raise ValueError(f"workpoint_manifest_jacobian_missing:{key}")
            continue
        payload = read_continuation_step(filepath)
        rhs = np.asarray(payload.get("rhsnew", []), dtype=np.float64)
        rhsold = np.asarray(payload.get("rhsold", []), dtype=np.float64)
        raw_residual = np.asarray(payload.get("residual", []), dtype=np.float64)
        step = {
            "iteration": iteration,
            "time": time_value,
            "gmin_val": gmin_value,
            "rhsold": rhsold,
            "rhs": rhs,
            "state0": np.asarray(payload.get("state0_in", []), dtype=np.float64),
            "raw_residual": raw_residual,
            "raw_residual_norm": float(np.linalg.norm(raw_residual)) if raw_residual.size else None,
            "node_map": payload.get("node_map", {}),
            "meta": {
                "circuit_id": int(circuit_id),
                "time": time_value,
                "gmin": gmin_value,
                "iteration": iteration,
                "matrix_size": int(rhs.shape[0]),
            },
            "jacobian": read_J(jac_filepath),
            "filepath": filepath,
            "jacobian_filepath": jac_filepath,
            "netlist_path": netlist_path,
        }
        if manifest_entry is not None:
            step["_workpoint_manifest_index"] = int(manifest_entry["manifest_index"])
        steps.append(step)
        found_keys.add(key)

    if requested_workpoints is not None:
        missing = [key for key in requested_by_key if key not in found_keys]
        if missing:
            raise ValueError(f"workpoint_manifest_missing:{missing}")
        steps.sort(key=lambda item: int(item["_workpoint_manifest_index"]))
    else:
        steps.sort(key=lambda item: (item["time"], item["gmin_val"], item["iteration"]))
    return {
        "success": bool(steps),
        "reason": None if steps else "trajectory_no_steps",
        "steps": steps,
        "task_dir": trajectory_dir,
        "stderr": "",
        "stdout": "",
    }

def _dense_direct_reference(matrix: np.ndarray, rhs: np.ndarray) -> Dict[str, Any]:
    try:
        solution = np.linalg.solve(matrix, rhs)
        mode = "solve"
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
        mode = "lstsq"

    raw_residual = matrix.dot(solution) - rhs
    return {
        "available": True,
        "solve_mode": mode,
        "solution": solution,
        "raw_residual_norm": float(np.linalg.norm(raw_residual)),
    }


def evaluate_corpus_step(step: Dict[str, Any], config: PrototypeConfig) -> Dict[str, Any]:
    rss_before_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if config.requested_mode in {"learned_schwarz_v1", "learned_boundary_correction_v1"} and not bool(config.apply_gmin_diagonal):
        raise ValueError("schema v3 learned Schwarz modes require the gmin diagonal")
    matrix = _make_system_matrix(step, apply_gmin_diagonal=config.apply_gmin_diagonal)
    rhs = np.asarray(step.get("rhs", []), dtype=np.float64)
    rhsold = np.asarray(step.get("rhsold", []), dtype=np.float64)
    if rhs.shape[0] != matrix.shape[0]:
        raise ValueError("rhs length does not match matrix size")

    initial_guess_mode = "rhsold" if config.use_rhsold_as_x0 else "zero"
    initial_guess = resolve_initial_guess(
        rhsold=rhsold,
        matrix_size=int(matrix.shape[0]),
        initial_guess_mode=initial_guess_mode,
    )
    initial_residual = compute_initial_residual(
        effective_matrix=matrix,
        linear_rhs=rhs,
        initial_guess=initial_guess,
    )
    x0 = initial_guess if initial_guess_mode == "rhsold" else None
    block_plan = None
    block_metadata = None
    learned_schwarz_model = None
    learned_schwarz_sample = None
    learned_boundary_core_sample = None
    learned_boundary_sample = None
    schur_interface_preconditioner = None
    learned_schur_diagonal_preconditioner = None
    learned_sparse_schur_preconditioner = None
    learned_local_sparse_schur_preconditioner = None
    learning_augmented_sparse_schur_preconditioner = None
    hybrid_sparse_schur_low_rank_preconditioner = None
    power_schur_preconditioner = None
    power_sparse_schur_preconditioner = None
    power_sparse_schur_arnoldi_preconditioner = None
    sparse_schur_preconditioner = None
    selected_local_sparse_schur_preconditioner = None
    local_sparse_schur_preconditioner = None
    local_schur_additive_preconditioner = None
    ilu0_solve = None
    ilu0_info: Dict[str, Any] = {"fallback_reason": None}
    if config.requested_mode in {"branch_incident_block", "cell_block_jacobi"}:
        preconditioner_load_start = time.perf_counter()
        block_plan = _build_block_jacobi_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=config,
            netlist_path=str(step.get("netlist_path", "")),
        )
        preconditioner_load_time = time.perf_counter() - preconditioner_load_start
        executed_mode = config.requested_mode
        precond_info = {"fallback_reason": None}
        block_metadata = _build_block_metadata(block_plan)
    elif config.requested_mode == "generic_block_jacobi":
        preconditioner_load_start = time.perf_counter()
        block_plan = _build_block_jacobi_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=PrototypeConfig(
                **{**config.__dict__, "block_mode": "generic"}
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        preconditioner_load_time = time.perf_counter() - preconditioner_load_start
        executed_mode = config.requested_mode
        precond_info = {"fallback_reason": None}
        block_metadata = _build_block_metadata(block_plan)
    elif config.requested_mode == "learned_schwarz_v1":
        preconditioner_load_start = time.perf_counter()
        block_plan = _build_block_jacobi_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=config,
            netlist_path=str(step.get("netlist_path", "")),
        )
        learned_schwarz_model = _load_learned_schwarz_model(
            config.learned_schwarz_checkpoint,
            dtype=torch.float64,
            initial_guess_mode=initial_guess_mode,
            expected_model_kind="learned_schwarz_v1",
        )
        learned_schwarz_sample = build_learned_schwarz_sample(
            matrix=matrix,
            plan=block_plan,
            linear_rhs=rhs,
            initial_residual=initial_residual,
            gmin=float(step.get("gmin_val", step.get("meta", {}).get("gmin", 0.0)) or 0.0),
            dtype=torch.float64,
        )
        preconditioner_load_time = time.perf_counter() - preconditioner_load_start
        executed_mode = config.requested_mode
        precond_info = {"fallback_reason": None, "learned_schwarz_checkpoint": config.learned_schwarz_checkpoint}
        block_metadata = _build_block_metadata(block_plan)
    elif config.requested_mode == "learned_boundary_correction_v1":
        preconditioner_load_start = time.perf_counter()
        core_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        boundary_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode=config.block_mode,
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        block_plan = core_plan
        learned_schwarz_model = _load_learned_schwarz_model(
            config.learned_schwarz_checkpoint,
            dtype=torch.float64,
            initial_guess_mode=initial_guess_mode,
            expected_model_kind="boundary_correction_v1",
        )
        learned_boundary_core_sample = build_learned_schwarz_sample(
            matrix=matrix,
            plan=core_plan,
            linear_rhs=rhs,
            initial_residual=initial_residual,
            gmin=float(step.get("gmin_val", step.get("meta", {}).get("gmin", 0.0)) or 0.0),
            dtype=torch.float64,
        )
        learned_boundary_sample = build_learned_schwarz_sample(
            matrix=matrix,
            plan=boundary_plan,
            linear_rhs=rhs,
            initial_residual=initial_residual,
            gmin=float(step.get("gmin_val", step.get("meta", {}).get("gmin", 0.0)) or 0.0),
            dtype=torch.float64,
        )
        preconditioner_load_time = time.perf_counter() - preconditioner_load_start
        executed_mode = config.requested_mode
        precond_info = {
            "fallback_reason": None,
            "learned_schwarz_checkpoint": config.learned_schwarz_checkpoint,
            "core_block_mode": "cell_core",
            "boundary_block_mode": config.block_mode,
        }
        block_metadata = {
            "core": _build_block_metadata(core_plan),
            "boundary": _build_block_metadata(boundary_plan),
        }
    elif config.requested_mode in {"explicit_schur_interface", "exact_schur_p_equals_s"}:
        preconditioner_load_start = time.perf_counter()
        core_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        boundary_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core_plus_onehop_boundary",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        schur_interface_preconditioner = ExplicitSchurInterfacePreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            uncovered_row_policy=config.uncovered_row_policy,
        )
        preconditioner_load_time = time.perf_counter() - preconditioner_load_start
        executed_mode = config.requested_mode
        precond_info = {"fallback_reason": None}
        block_metadata = schur_interface_preconditioner.metadata()
    elif config.requested_mode == "learned_schur_diagonal":
        preconditioner_load_start = time.perf_counter()
        core_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        boundary_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core_plus_onehop_boundary",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        checkpoint_path = config.learned_schur_checkpoint or config.learned_schwarz_checkpoint
        learned_schur_model = load_learned_schur_diagonal_model(checkpoint_path)
        learned_schur_diagonal_preconditioner = LearnedSchurDiagonalPreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            model=learned_schur_model,
            uncovered_row_policy=config.uncovered_row_policy,
        )
        preconditioner_load_time = time.perf_counter() - preconditioner_load_start
        executed_mode = config.requested_mode
        precond_info = {"fallback_reason": None, "learned_schur_checkpoint": checkpoint_path}
        block_metadata = learned_schur_diagonal_preconditioner.metadata()
    elif config.requested_mode == "learned_sparse_schur":
        preconditioner_load_start = time.perf_counter()
        core_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        boundary_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core_plus_onehop_boundary",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        checkpoint_path = config.learned_sparse_schur_checkpoint or config.learned_schur_checkpoint or config.learned_schwarz_checkpoint
        sparse_schur_model = load_learned_sparse_schur_model(checkpoint_path)
        learned_sparse_schur_preconditioner = LearnedSparseSchurPreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            model=sparse_schur_model,
            edge_budget=int(config.sparse_schur_edge_budget),
            candidate_edge_limit=int(config.sparse_schur_candidate_edge_limit),
            diagonal_shift=float(config.sparse_schur_diagonal_shift),
            uncovered_row_policy=config.uncovered_row_policy,
        )
        preconditioner_load_time = time.perf_counter() - preconditioner_load_start
        executed_mode = config.requested_mode
        precond_info = {"fallback_reason": None, "learned_sparse_schur_checkpoint": checkpoint_path}
        block_metadata = learned_sparse_schur_preconditioner.metadata()
    elif config.requested_mode in POWER_SPARSE_SCHUR_ARNOLDI_MODES:
        preconditioner_load_start = time.perf_counter()
        mode_tail = str(config.requested_mode).replace("power_sparse_schur_m", "")
        power_raw, rank_raw = mode_tail.split("_r", 1)
        power_terms = int(power_raw)
        arnoldi_rank = int(rank_raw)
        core_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        boundary_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core_plus_onehop_boundary",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        power_sparse_schur_arnoldi_preconditioner = PowerSparseSchurArnoldiPreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            strategy='topk_abs',
            edge_budget=int(config.sparse_schur_edge_budget),
            budget_multiplier=float(config.local_schur_budget_multiplier),
            candidate_edge_limit=int(config.sparse_schur_candidate_edge_limit),
            power_terms=power_terms,
            arnoldi_rank=arnoldi_rank,
            diagonal_shift=float(config.sparse_schur_diagonal_shift),
            uncovered_row_policy=config.uncovered_row_policy,
        )
        preconditioner_load_time = time.perf_counter() - preconditioner_load_start
        executed_mode = config.requested_mode
        precond_info = {"fallback_reason": None}
        block_metadata = power_sparse_schur_arnoldi_preconditioner.metadata()
    elif config.requested_mode in {"power_sparse_schur_m0", "power_sparse_schur_m1", "power_sparse_schur_m2", "power_sparse_schur_m3"}:
        preconditioner_load_start = time.perf_counter()
        power_terms = int(config.requested_mode.rsplit("m", 1)[1])
        core_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        boundary_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core_plus_onehop_boundary",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        power_sparse_schur_preconditioner = PowerSparseSchurPreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            strategy='topk_abs',
            edge_budget=int(config.sparse_schur_edge_budget),
            budget_multiplier=float(config.local_schur_budget_multiplier),
            candidate_edge_limit=int(config.sparse_schur_candidate_edge_limit),
            power_terms=power_terms,
            diagonal_shift=float(config.sparse_schur_diagonal_shift),
            uncovered_row_policy=config.uncovered_row_policy,
        )
        preconditioner_load_time = time.perf_counter() - preconditioner_load_start
        executed_mode = config.requested_mode
        precond_info = {"fallback_reason": None}
        block_metadata = power_sparse_schur_preconditioner.metadata()
    elif config.requested_mode in {"power_schur_m0", "power_schur_m1", "power_schur_m2", "power_schur_m3"}:
        preconditioner_load_start = time.perf_counter()
        power_terms = int(config.requested_mode.rsplit("m", 1)[1])
        core_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        boundary_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core_plus_onehop_boundary",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        power_schur_preconditioner = PowerSchurPreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            power_terms=power_terms,
            diagonal_shift=float(config.sparse_schur_diagonal_shift),
            uncovered_row_policy=config.uncovered_row_policy,
        )
        preconditioner_load_time = time.perf_counter() - preconditioner_load_start
        executed_mode = config.requested_mode
        precond_info = {"fallback_reason": None}
        block_metadata = power_schur_preconditioner.metadata()
    elif config.requested_mode in {"local_sparse_schur_topk_abs", "local_sparse_schur_per_instance_topk", "hard_sparse_schur_topk_abs"}:
        preconditioner_load_start = time.perf_counter()
        strategy = "topk_abs" if config.requested_mode == "hard_sparse_schur_topk_abs" else config.requested_mode.replace("local_sparse_schur_", "")
        core_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        boundary_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core_plus_onehop_boundary",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        local_sparse_schur_preconditioner = LocalSparseSchurPreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            strategy=strategy,
            edge_budget=int(config.sparse_schur_edge_budget),
            budget_multiplier=float(config.local_schur_budget_multiplier),
            candidate_edge_limit=int(config.sparse_schur_candidate_edge_limit),
            diagonal_shift=float(config.sparse_schur_diagonal_shift),
            uncovered_row_policy=config.uncovered_row_policy,
        )
        preconditioner_load_time = time.perf_counter() - preconditioner_load_start
        executed_mode = config.requested_mode
        precond_info = {"fallback_reason": None}
        block_metadata = local_sparse_schur_preconditioner.metadata()
    elif config.requested_mode == "local_schur_additive":
        preconditioner_load_start = time.perf_counter()
        core_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        boundary_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core_plus_onehop_boundary",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        local_schur_additive_preconditioner = LocalSchurAdditiveInversePreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            diagonal_shift=float(config.sparse_schur_diagonal_shift),
            uncovered_row_policy=config.uncovered_row_policy,
            include_diagonal_fallback=bool(config.local_schur_additive_include_diagonal),
            diagonal_weight=float(config.local_schur_additive_diagonal_weight),
            local_weight=float(config.local_schur_additive_local_weight),
            cluster_weight=float(config.local_schur_additive_cluster_weight),
            cluster_hops=int(config.local_schur_additive_cluster_hops),
            max_cluster_size=int(config.local_schur_additive_max_cluster_size),
            patch_svd_rcond=float(config.local_schur_additive_patch_svd_rcond),
            inner_iterations=int(config.local_schur_additive_inner_iterations),
        )
        preconditioner_load_time = time.perf_counter() - preconditioner_load_start
        executed_mode = config.requested_mode
        precond_info = {"fallback_reason": None}
        block_metadata = local_schur_additive_preconditioner.metadata()
    elif config.requested_mode in {"learning_augmented", "learned_sparse_schur_safe_add", "learned_sparse_schur_safe_add_probe"}:
        preconditioner_load_start = time.perf_counter()
        core_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        boundary_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core_plus_onehop_boundary",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        selection_policy = "safe_add_probe" if str(config.requested_mode) == "learned_sparse_schur_safe_add_probe" else "safe_add"
        learning_augmented_sparse_schur_preconditioner = LearningAugmentedSparseSchurPreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            edge_budget=int(config.sparse_schur_edge_budget),
            budget_multiplier=float(config.local_schur_budget_multiplier),
            candidate_edge_limit=int(config.sparse_schur_candidate_edge_limit),
            diagonal_shift=float(config.sparse_schur_diagonal_shift),
            uncovered_row_policy=config.uncovered_row_policy,
            probe_rhs=rhs,
            probe_x0=x0,
            probe_iterations=int(config.learned_local_sparse_schur_probe_iterations),
            selection_policy=selection_policy,
            add_fraction=float(config.learned_local_sparse_schur_slack_fraction),
            add_min=int(config.learned_local_sparse_schur_slack_min),
            node_map=step.get('node_map', {}),
            max_schur_nnz=int(config.sparse_schur_max_nnz),
            max_degree=int(config.sparse_schur_max_degree),
            max_exact_entries=int(config.sparse_schur_max_exact_entries),
        )
        preconditioner_load_time = time.perf_counter() - preconditioner_load_start
        executed_mode = config.requested_mode
        precond_info = {"fallback_reason": None, "learning_augmented_selection_policy": selection_policy}
        block_metadata = learning_augmented_sparse_schur_preconditioner.metadata()
    elif config.requested_mode == "learned_local_sparse_schur":
        preconditioner_load_start = time.perf_counter()
        core_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        boundary_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core_plus_onehop_boundary",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        checkpoint_path = config.learned_local_sparse_schur_checkpoint or config.learned_sparse_schur_checkpoint or config.learned_schur_checkpoint or config.learned_schwarz_checkpoint
        local_model = load_learned_local_sparse_schur_model(checkpoint_path)
        learned_local_sparse_schur_preconditioner = LearnedLocalSparseSchurPreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            model=local_model,
            edge_budget=int(config.sparse_schur_edge_budget),
            budget_multiplier=float(config.local_schur_budget_multiplier),
            candidate_edge_limit=int(config.sparse_schur_candidate_edge_limit),
            diagonal_shift=float(config.sparse_schur_diagonal_shift),
            uncovered_row_policy=config.uncovered_row_policy,
            probe_rhs=rhs,
            probe_x0=x0,
            probe_iterations=int(config.learned_local_sparse_schur_probe_iterations),
            selection_policy=str(config.learned_local_sparse_schur_selection_policy),
            slack_fraction=float(config.learned_local_sparse_schur_slack_fraction),
            slack_min=int(config.learned_local_sparse_schur_slack_min),
        )
        preconditioner_load_time = time.perf_counter() - preconditioner_load_start
        executed_mode = config.requested_mode
        precond_info = {"fallback_reason": None, "learned_local_sparse_schur_checkpoint": checkpoint_path}
        block_metadata = learned_local_sparse_schur_preconditioner.metadata()
    elif config.requested_mode in {"hybrid_sparse_schur_low_rank_topk_abs", "hybrid_sparse_schur_low_rank_relative_threshold", "hybrid_sparse_schur_low_rank_row_topk"}:
        preconditioner_load_start = time.perf_counter()
        strategy = config.requested_mode.replace("hybrid_sparse_schur_low_rank_", "")
        core_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        boundary_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core_plus_onehop_boundary",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        hybrid_sparse_schur_low_rank_preconditioner = HybridSparseSchurLowRankPreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            strategy=strategy,
            edge_budget=int(config.sparse_schur_edge_budget),
            row_topk=int(config.sparse_schur_row_topk),
            relative_threshold=float(config.sparse_schur_relative_threshold),
            diagonal_shift=float(config.sparse_schur_diagonal_shift),
            low_rank_rank=int(config.schur_low_rank_rank),
            low_rank_mode=str(config.schur_low_rank_mode),
            low_rank_strength=float(config.schur_low_rank_strength),
            uncovered_row_policy=config.uncovered_row_policy,
        )
        preconditioner_load_time = time.perf_counter() - preconditioner_load_start
        executed_mode = config.requested_mode
        precond_info = {"fallback_reason": None}
        block_metadata = hybrid_sparse_schur_low_rank_preconditioner.metadata()
    elif config.requested_mode == 'sparse_schur_row_topk_selected_local':
        preconditioner_load_start = time.perf_counter()
        core_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get('node_map', {}),
            config=BlockPlanConfig(
                block_mode='cell_core',
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get('netlist_path', '')),
        )
        boundary_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get('node_map', {}),
            config=BlockPlanConfig(
                block_mode='cell_core_plus_onehop_boundary',
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get('netlist_path', '')),
        )
        selected_local_sparse_schur_preconditioner = SelectedLocalSparseSchurPreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            edge_budget=int(config.sparse_schur_edge_budget),
            row_topk=int(config.sparse_schur_row_topk),
            diagonal_shift=float(config.sparse_schur_diagonal_shift),
            uncovered_row_policy=config.uncovered_row_policy,
        )
        preconditioner_load_time = time.perf_counter() - preconditioner_load_start
        executed_mode = config.requested_mode
        precond_info = {'fallback_reason': None}
        block_metadata = selected_local_sparse_schur_preconditioner.metadata()
    elif config.requested_mode in {"sparse_schur_topk_abs", "sparse_schur_relative_threshold", "sparse_schur_row_topk", "hard_sparse_schur_threshold"}:
        preconditioner_load_start = time.perf_counter()
        strategy = "relative_threshold" if config.requested_mode == "hard_sparse_schur_threshold" else config.requested_mode.replace("sparse_schur_", "")
        core_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        boundary_plan = build_block_schwarz_plan(
            matrix=matrix,
            node_map=step.get("node_map", {}),
            config=BlockPlanConfig(
                block_mode="cell_core_plus_onehop_boundary",
                max_block_size=int(config.max_block_size),
                min_block_size=int(config.min_block_size),
                max_blocks=int(config.max_blocks),
                max_total_block_nnz=int(config.max_total_block_nnz),
                uncovered_row_policy=config.uncovered_row_policy,
            ),
            netlist_path=str(step.get("netlist_path", "")),
        )
        sparse_schur_preconditioner = SparseSchurPreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            strategy=strategy,
            edge_budget=int(config.sparse_schur_edge_budget),
            row_topk=int(config.sparse_schur_row_topk),
            relative_threshold=float(config.sparse_schur_relative_threshold),
            diagonal_shift=float(config.sparse_schur_diagonal_shift),
            uncovered_row_policy=config.uncovered_row_policy,
        )
        preconditioner_load_time = time.perf_counter() - preconditioner_load_start
        executed_mode = config.requested_mode
        precond_info = {"fallback_reason": None}
        block_metadata = sparse_schur_preconditioner.metadata()
    elif config.requested_mode in {"ilu0", "ilut", "ilu_drop_tol", "ilu_fill_factor"}:
        if config.requested_mode == "ilu0":
            ilu0_solve, ilu0_info, preconditioner_load_time = _build_ilu0_operator(matrix)
        elif config.requested_mode == "ilut":
            ilu0_solve, ilu0_info, preconditioner_load_time = _build_ilu_operator(
                matrix,
                drop_tol=1e-3,
                fill_factor=10.0,
                mode_name="ilut",
            )
        elif config.requested_mode == "ilu_drop_tol":
            ilu0_solve, ilu0_info, preconditioner_load_time = _build_ilu_operator(
                matrix,
                drop_tol=1e-4,
                fill_factor=1.0,
                mode_name="ilu_drop_tol",
            )
        else:
            ilu0_solve, ilu0_info, preconditioner_load_time = _build_ilu_operator(
                matrix,
                drop_tol=0.0,
                fill_factor=20.0,
                mode_name="ilu_fill_factor",
            )
        executed_mode = config.requested_mode if ilu0_solve is not None else config.fallback_mode
        if ilu0_solve is None:
            scales = _build_analytic_scales(config.fallback_mode, matrix)
            precond_info = ilu0_info
        else:
            precond_info = ilu0_info
    else:
        scales, executed_mode, precond_info, preconditioner_load_time = _build_preconditioner_scales(
            config.requested_mode,
            config.fallback_mode,
            matrix,
            step,
            config.sidecar_path,
            config.synthetic_source_mode,
        )

    csr = csr_matrix(matrix)
    matvec_count = 0
    psolve_count = 0
    matvec_time = 0.0
    preconditioner_apply_time = 0.0
    preconditioned_residual_history: List[float] = []

    def _matvec(vec: np.ndarray) -> np.ndarray:
        nonlocal matvec_count, matvec_time
        start = time.perf_counter()
        out = csr.dot(vec)
        matvec_time += time.perf_counter() - start
        matvec_count += 1
        return np.asarray(out, dtype=np.float64)

    def _apply_preconditioner(vec: np.ndarray) -> np.ndarray:
        nonlocal psolve_count, preconditioner_apply_time
        start = time.perf_counter()
        if (
            learned_schwarz_model is not None
            and learned_boundary_core_sample is not None
            and learned_boundary_sample is not None
        ):
            with torch.no_grad():
                vec_t = torch.as_tensor(np.asarray(vec, dtype=np.float64), dtype=torch.float64)
                out = learned_schwarz_model.apply(
                    learned_boundary_core_sample,
                    learned_boundary_sample,
                    vec_t,
                ).detach().cpu().numpy()
        elif learned_schwarz_model is not None and learned_schwarz_sample is not None:
            with torch.no_grad():
                vec_t = torch.as_tensor(np.asarray(vec, dtype=np.float64), dtype=torch.float64)
                out = learned_schwarz_model.apply(learned_schwarz_sample, vec_t).detach().cpu().numpy()
        elif learned_schur_diagonal_preconditioner is not None:
            out = learned_schur_diagonal_preconditioner.apply(vec)
        elif learned_sparse_schur_preconditioner is not None:
            out = learned_sparse_schur_preconditioner.apply(vec)
        elif learned_local_sparse_schur_preconditioner is not None:
            out = learned_local_sparse_schur_preconditioner.apply(vec)
        elif learning_augmented_sparse_schur_preconditioner is not None:
            out = learning_augmented_sparse_schur_preconditioner.apply(vec)
        elif hybrid_sparse_schur_low_rank_preconditioner is not None:
            out = hybrid_sparse_schur_low_rank_preconditioner.apply(vec)
        elif power_schur_preconditioner is not None:
            out = power_schur_preconditioner.apply(vec)
        elif power_sparse_schur_preconditioner is not None:
            out = power_sparse_schur_preconditioner.apply(vec)
        elif power_sparse_schur_arnoldi_preconditioner is not None:
            out = power_sparse_schur_arnoldi_preconditioner.apply(vec)
        elif sparse_schur_preconditioner is not None:
            out = sparse_schur_preconditioner.apply(vec)
        elif selected_local_sparse_schur_preconditioner is not None:
            out = selected_local_sparse_schur_preconditioner.apply(vec)
        elif local_sparse_schur_preconditioner is not None:
            out = local_sparse_schur_preconditioner.apply(vec)
        elif local_schur_additive_preconditioner is not None:
            out = local_schur_additive_preconditioner.apply(vec)
        elif schur_interface_preconditioner is not None:
            out = schur_interface_preconditioner.apply(vec)
        elif block_plan is not None:
            out = block_plan.apply(vec)
        elif ilu0_solve is not None:
            out = np.asarray(ilu0_solve(vec), dtype=np.float64)
        else:
            out = scales * vec
        preconditioner_apply_time += time.perf_counter() - start
        psolve_count += 1
        return out

    operator = LinearOperator(shape=matrix.shape, matvec=_matvec, dtype=np.float64)
    preconditioner = LinearOperator(shape=matrix.shape, matvec=_apply_preconditioner, dtype=np.float64)

    initial_raw_residual_vec = initial_residual
    exported_raw_residual_norm = step.get("raw_residual_norm")
    initial_precond_residual_vec = _apply_preconditioner(initial_raw_residual_vec)

    gmres_callback_count = 0

    def _callback(pr_norm: float) -> None:
        nonlocal gmres_callback_count
        gmres_callback_count += 1
        preconditioned_residual_history.append(float(pr_norm))

    start = time.perf_counter()
    max_cycles = max(1, int(math.ceil(float(config.max_iters) / float(config.restart))))
    solution, info = gmres(
        operator,
        rhs,
        x0=x0,
        M=preconditioner,
        restart=config.restart,
        maxiter=max_cycles,
        rtol=config.rtol,
        atol=config.atol,
        callback=_callback,
        callback_type="pr_norm",
    )
    linear_solve_time = time.perf_counter() - start

    final_raw_residual_vec = _matvec(solution) - rhs
    final_precond_residual_vec = _apply_preconditioner(final_raw_residual_vec)
    final_raw_residual = float(np.linalg.norm(final_raw_residual_vec))
    initial_raw_residual = float(np.linalg.norm(initial_raw_residual_vec))
    rhs_norm = float(np.linalg.norm(rhs))
    initial_precond_residual = float(np.linalg.norm(initial_precond_residual_vec))
    final_precond_residual = float(np.linalg.norm(final_precond_residual_vec))

    gmres_converged = bool(info == 0)
    gmres_flag_success = bool(info == 0)
    fallback_to_direct = False
    fallback_reason = precond_info.get("fallback_reason")

    if not gmres_converged:
        fallback_to_direct = True
        fallback_reason = fallback_reason or "gmres_no_convergence"
    elif final_raw_residual > max(
        initial_raw_residual * float(config.residual_worsen_ratio),
        initial_raw_residual + float(config.residual_worsen_abs),
    ):
        fallback_to_direct = True
        fallback_reason = fallback_reason or "residual_worsened"

    direct_reference = _dense_direct_reference(matrix, rhs)
    used_direct_reference = fallback_to_direct
    chosen_solution = direct_reference["solution"] if used_direct_reference else solution
    chosen_raw_residual = direct_reference["raw_residual_norm"] if used_direct_reference else final_raw_residual
    chosen_residual_ratio = _residual_ratio(chosen_raw_residual, initial_raw_residual)
    true_rel_residual = _true_rel_residual(chosen_raw_residual, rhs_norm)
    success_by_true_residual = _success_by_true_residual(chosen_raw_residual, rhs_norm)
    success_by_residual_ratio = _success_by_residual_ratio(chosen_raw_residual, initial_raw_residual)
    chosen_success = _chosen_success(chosen_raw_residual, rhs_norm, initial_raw_residual)
    residual_success_conflict = (
        None
        if success_by_true_residual is None or success_by_residual_ratio is None
        else bool(success_by_true_residual != success_by_residual_ratio)
    )
    residual_based_success = chosen_success
    solution_l2_error_vs_direct = float(np.linalg.norm(solution - direct_reference["solution"]))

    gmres_restart_count = max(0, (gmres_callback_count - 1) // int(config.restart)) if gmres_callback_count else 0
    exported_delta = None
    if exported_raw_residual_norm is not None:
        exported_delta = float(initial_raw_residual - float(exported_raw_residual_norm))
    rss_after_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if learning_augmented_sparse_schur_preconditioner is not None:
        block_metadata = learning_augmented_sparse_schur_preconditioner.metadata()

    return {
        "requested_mode": config.requested_mode,
        "executed_mode": executed_mode,
        "fallback_mode": config.fallback_mode,
        "fallback_to_direct": fallback_to_direct,
        "fallback_reason": fallback_reason,
        "gmres_converged": gmres_converged,
        "gmres_info": int(info),
        "gmres_flag_success": bool(gmres_flag_success),
        "gmres_iterations": int(gmres_callback_count),
        "gmres_restart_count": int(gmres_restart_count),
        "linear_solve_time": float(linear_solve_time),
        "preconditioner_apply_time": float(preconditioner_apply_time),
        "preconditioner_load_time": float(preconditioner_load_time),
        "matvec_time": float(matvec_time),
        "matvec_count": int(matvec_count),
        "psolve_count": int(psolve_count),
        "initial_raw_residual": initial_raw_residual,
        "final_raw_residual": final_raw_residual,
        "rhs_norm": rhs_norm,
        "initial_precond_residual": initial_precond_residual,
        "final_precond_residual": final_precond_residual,
        "chosen_raw_residual": float(chosen_raw_residual),
        "true_rel_residual": None if true_rel_residual is None else float(true_rel_residual),
        "chosen_residual_ratio": None if chosen_residual_ratio is None else float(chosen_residual_ratio),
        "residual_ratio": None if chosen_residual_ratio is None else float(chosen_residual_ratio),
        "success_by_true_residual": None if success_by_true_residual is None else bool(success_by_true_residual),
        "success_by_residual_ratio": None if success_by_residual_ratio is None else bool(success_by_residual_ratio),
        "chosen_success": None if chosen_success is None else bool(chosen_success),
        "residual_success_conflict": None if residual_success_conflict is None else bool(residual_success_conflict),
        "residual_based_success": None if residual_based_success is None else bool(residual_based_success),
        "residual_success_rtol": float(DEFAULT_RESIDUAL_SUCCESS_RTOL),
        "residual_success_atol": float(DEFAULT_RESIDUAL_SUCCESS_ATOL),
        "exported_raw_residual_norm": None if exported_raw_residual_norm is None else float(exported_raw_residual_norm),
        "exported_entry_residual_delta": exported_delta,
        "direct_reference_raw_residual": float(direct_reference["raw_residual_norm"]),
        "direct_reference_mode": direct_reference["solve_mode"],
        "solution_l2_error_vs_direct": solution_l2_error_vs_direct,
        "process_peak_rss_kb_before": int(rss_before_kb),
        "process_peak_rss_kb_after": int(rss_after_kb),
        "process_peak_rss_kb": int(max(rss_before_kb, rss_after_kb)),
        "preconditioned_residual_history": preconditioned_residual_history,
        "preconditioner_info": precond_info,
        "block_metadata": block_metadata,
        "block_candidates": list(block_plan.block_candidates) if block_plan is not None else [],
        "solution_preview": chosen_solution[: min(5, chosen_solution.shape[0])].tolist(),
    }


def _summarize_results(step_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not step_results:
        return {
            "step_count": 0,
            "gmres_success_rate": 0.0,
            "fallback_rate": 0.0,
            "residual_based_success_rate": 0.0,
            "chosen_success_rate": 0.0,
            "chosen_success_count": 0,
            "chosen_success_observed_count": 0,
            "success_by_true_residual_rate": 0.0,
            "success_by_true_residual_count": 0,
            "success_by_residual_ratio_rate": 0.0,
            "success_by_residual_ratio_count": 0,
            "residual_success_conflict_count": 0,
            "residual_success_count": 0,
            "residual_success_observed_count": 0,
            "residual_success_rtol": float(DEFAULT_RESIDUAL_SUCCESS_RTOL),
            "residual_success_atol": float(DEFAULT_RESIDUAL_SUCCESS_ATOL),
            "avg_gmres_iterations": 0.0,
            "avg_chosen_raw_residual": None,
        }

    converged = [item for item in step_results if item.get("gmres_converged")]
    fallbacks = [item for item in step_results if item.get("fallback_to_direct")]
    iteration_values = [float(item["gmres_iterations"]) for item in step_results]
    nonzero_iteration_values = [value for value in iteration_values if value > 0.0]
    chosen_residuals = [float(item["chosen_raw_residual"]) for item in step_results if item.get("chosen_raw_residual") is not None]
    residual_ratios = [
        ratio
        for ratio in (
            item.get("residual_ratio", item.get("chosen_residual_ratio"))
            if item.get("residual_ratio", item.get("chosen_residual_ratio")) is not None
            else _residual_ratio(item.get("chosen_raw_residual"), item.get("initial_raw_residual"))
            for item in step_results
        )
        if ratio is not None
    ]
    true_rel_residuals = [
        value
        for value in (
            item.get("true_rel_residual")
            if item.get("true_rel_residual") is not None
            else _true_rel_residual(item.get("chosen_raw_residual"), item.get("rhs_norm"))
            for item in step_results
        )
        if value is not None
    ]
    chosen_successes = [
        bool(success)
        for success in (
            item.get("chosen_success")
            if item.get("chosen_success") is not None
            else _chosen_success(item.get("chosen_raw_residual"), item.get("rhs_norm"), item.get("initial_raw_residual"))
            for item in step_results
        )
        if success is not None
    ]
    true_residual_successes = [
        bool(success)
        for success in (
            item.get("success_by_true_residual")
            if item.get("success_by_true_residual") is not None
            else _success_by_true_residual(item.get("chosen_raw_residual"), item.get("rhs_norm"))
            for item in step_results
        )
        if success is not None
    ]
    residual_ratio_successes = [
        bool(success)
        for success in (
            item.get("success_by_residual_ratio")
            if item.get("success_by_residual_ratio") is not None
            else _success_by_residual_ratio(item.get("chosen_raw_residual"), item.get("initial_raw_residual"))
            for item in step_results
        )
        if success is not None
    ]
    residual_success_conflicts = [
        bool(item.get("residual_success_conflict"))
        for item in step_results
        if item.get("residual_success_conflict") is not None
    ]
    residual_successes = chosen_successes
    runtimes = [float(item["linear_solve_time"]) for item in step_results if item.get("linear_solve_time") is not None]
    factor_times = [float(item.get("preconditioner_load_time", 0.0)) for item in step_results]
    block_counts = [
        float((item.get("block_metadata") or {}).get("num_blocks", 0))
        for item in step_results
        if item.get("block_metadata") is not None
    ]
    covered_rows = [
        float((item.get("block_metadata") or {}).get("covered_rows", 0))
        for item in step_results
        if item.get("block_metadata") is not None
    ]
    total_block_nnz = [
        float((item.get("block_metadata") or {}).get("total_block_nnz", 0))
        for item in step_results
        if item.get("block_metadata") is not None
    ]
    max_block_sizes = [
        float((item.get("block_metadata") or {}).get("max_block_size", 0))
        for item in step_results
        if item.get("block_metadata") is not None
    ]
    memory_estimates = [
        float((item.get("block_metadata") or {}).get("memory_estimate", 0))
        for item in step_results
        if item.get("block_metadata") is not None
    ]
    peak_rss_kb_values = [
        float(item.get("process_peak_rss_kb", 0.0))
        for item in step_results
        if item.get("process_peak_rss_kb") is not None
    ]
    return {
        "step_count": len(step_results),
        "convergence_rate": float(len(converged) / len(step_results)),
        "gmres_success_rate": float(len(converged) / len(step_results)),
        "fallback_rate": float(len(fallbacks) / len(step_results)),
        "residual_based_success_rate": float(np.mean(chosen_successes)) if chosen_successes else 0.0,
        "chosen_success_rate": float(np.mean(chosen_successes)) if chosen_successes else 0.0,
        "chosen_success_count": int(sum(chosen_successes)),
        "chosen_success_observed_count": int(len(chosen_successes)),
        "success_by_true_residual_rate": float(np.mean(true_residual_successes)) if true_residual_successes else 0.0,
        "success_by_true_residual_count": int(sum(true_residual_successes)),
        "success_by_true_residual_observed_count": int(len(true_residual_successes)),
        "success_by_residual_ratio_rate": float(np.mean(residual_ratio_successes)) if residual_ratio_successes else 0.0,
        "success_by_residual_ratio_count": int(sum(residual_ratio_successes)),
        "success_by_residual_ratio_observed_count": int(len(residual_ratio_successes)),
        "residual_success_conflict_count": int(sum(residual_success_conflicts)),
        "residual_success_conflict_observed_count": int(len(residual_success_conflicts)),
        "residual_success_count": int(sum(chosen_successes)),
        "residual_success_observed_count": int(len(chosen_successes)),
        "residual_success_rtol": float(DEFAULT_RESIDUAL_SUCCESS_RTOL),
        "residual_success_atol": float(DEFAULT_RESIDUAL_SUCCESS_ATOL),
        "avg_gmres_iterations": float(np.mean(iteration_values)),
        "median_iterations": float(np.median(iteration_values)),
        "p90_iterations": float(np.percentile(iteration_values, 90)),
        "nonzero_iteration_step_count": int(len(nonzero_iteration_values)),
        "median_iterations_nonzero": float(np.median(nonzero_iteration_values)) if nonzero_iteration_values else 0.0,
        "avg_chosen_raw_residual": float(np.mean(chosen_residuals)) if chosen_residuals else None,
        "median_chosen_raw_residual": float(np.median(chosen_residuals)) if chosen_residuals else None,
        "avg_true_rel_residual": float(np.mean(true_rel_residuals)) if true_rel_residuals else None,
        "median_true_rel_residual": float(np.median(true_rel_residuals)) if true_rel_residuals else None,
        "avg_residual_ratio": float(np.mean(residual_ratios)) if residual_ratios else None,
        "median_residual_ratio": float(np.median(residual_ratios)) if residual_ratios else None,
        "avg_linear_solve_time": float(np.mean(runtimes)),
        "median_runtime": float(np.median(runtimes)),
        "total_runtime": float(np.sum(runtimes)),
        "avg_block_factor_time": float(np.mean(factor_times)),
        "median_block_factor_time": float(np.median(factor_times)),
        "avg_num_blocks": float(np.mean(block_counts)) if block_counts else 0.0,
        "avg_covered_rows": float(np.mean(covered_rows)) if covered_rows else 0.0,
        "avg_total_block_nnz": float(np.mean(total_block_nnz)) if total_block_nnz else 0.0,
        "avg_max_block_size": float(np.mean(max_block_sizes)) if max_block_sizes else 0.0,
        "avg_memory_estimate": float(np.mean(memory_estimates)) if memory_estimates else 0.0,
        "avg_peak_rss_kb": float(np.mean(peak_rss_kb_values)) if peak_rss_kb_values else 0.0,
        "max_peak_rss_kb": float(np.max(peak_rss_kb_values)) if peak_rss_kb_values else 0.0,
        "requested_modes": sorted({str(item["requested_mode"]) for item in step_results}),
        "executed_modes": sorted({str(item["executed_mode"]) for item in step_results}),
        "fallback_reasons": sorted({str(item["fallback_reason"]) for item in step_results if item.get("fallback_reason")}),
    }


def _bucket_value(field: str, item: Dict[str, Any]) -> str:
    if field == "gmin_val":
        return f"{float(item.get(field, 0.0)):.17e}"
    if field in {"corpus_iteration", "matrix_size"}:
        return str(int(item.get(field, -1)))
    return str(item.get(field))


def _build_bucketed_summary(step_results: List[Dict[str, Any]], fields: List[str]) -> Dict[str, Any]:
    buckets: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for field in fields:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in step_results:
            key = _bucket_value(field, item)
            grouped.setdefault(key, []).append(item)
        buckets[field] = {bucket_key: _summarize_results(items) for bucket_key, items in sorted(grouped.items())}
    return buckets


def _mode_step_jsonl_record(step_summary: Dict[str, Any]) -> Dict[str, Any]:
    record = dict(step_summary)
    if "preconditioned_residual_history" in record:
        record["preconditioned_residual_history_len"] = len(record["preconditioned_residual_history"])
        del record["preconditioned_residual_history"]
    if "solution_preview" in record:
        del record["solution_preview"]
    return record


def _compact_preconditioner_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in {"blocks", "factor_modes"}:
                continue
            out[key] = _compact_preconditioner_metadata(item)
        return out
    if isinstance(value, list):
        if len(value) > 16:
            return {"count": len(value)}
        return [_compact_preconditioner_metadata(item) for item in value]
    return value


def _solver_outcome_record(step_summary: Dict[str, Any]) -> Dict[str, Any]:
    block_metadata = step_summary.get("block_metadata") or {}
    record = {
        "schema_version": 1,
        "solver_family": "external_gmres_prototype",
        "linear_system_key": (
            f"c{int(step_summary.get('circuit_id', -1))}"
            f"_step{int(step_summary.get('selected_step_index', -1))}"
            f"_t{float(step_summary.get('time', 0.0)):.17e}"
            f"_g{float(step_summary.get('gmin_val', 0.0)):.17e}"
            f"_it{int(step_summary.get('corpus_iteration', -1))}"
        ),
        "circuit_id": int(step_summary.get("circuit_id", -1)),
        "selected_step_index": int(step_summary.get("selected_step_index", -1)),
        "corpus_iteration": int(step_summary.get("corpus_iteration", -1)),
        "time": float(step_summary.get("time", 0.0)),
        "gmin_val": float(step_summary.get("gmin_val", 0.0)),
        "matrix_size": int(step_summary.get("matrix_size", -1)),
        "node_map_hash": step_summary.get("node_map_hash"),
        "candidate_mode": str(step_summary.get("requested_mode")),
        "executed_mode": str(step_summary.get("executed_mode")),
        "preconditioner_kind": "block" if step_summary.get("requested_mode") in {"branch_incident_block", "cell_block_jacobi", "explicit_schur_interface", "exact_schur_p_equals_s", "learning_augmented", "learned_sparse_schur_safe_add", "learned_sparse_schur_safe_add_probe", "hard_sparse_schur_topk_abs", "hard_sparse_schur_threshold", "learned_schwarz_v1", "learned_boundary_correction_v1", "learned_schur_diagonal", "learned_sparse_schur", "sparse_schur_topk_abs", "sparse_schur_relative_threshold", "sparse_schur_row_topk", "sparse_schur_row_topk_selected_local", "hard_sparse_schur_topk_abs", "hard_sparse_schur_threshold", "local_sparse_schur_topk_abs", "local_sparse_schur_per_instance_topk", "local_schur_additive", "learning_augmented", "learned_sparse_schur_safe_add", "learned_sparse_schur_safe_add_probe", "learned_local_sparse_schur", "hybrid_sparse_schur_low_rank_topk_abs", "hybrid_sparse_schur_low_rank_relative_threshold", "hybrid_sparse_schur_low_rank_row_topk", "power_schur_m0", "power_schur_m1", "power_schur_m2", "power_schur_m3", "power_sparse_schur_m0", "power_sparse_schur_m1", "power_sparse_schur_m2", "power_sparse_schur_m3"} or str(step_summary.get("requested_mode", "")) in POWER_SPARSE_SCHUR_ARNOLDI_MODES else "diagonal",
        "converged": bool(step_summary.get("gmres_converged", False)),
        "gmres_info": int(step_summary.get("gmres_info", 0)),
        "gmres_flag_success": bool(step_summary.get("gmres_flag_success", step_summary.get("gmres_converged", False))),
        "iterations": int(step_summary.get("gmres_iterations", 0)),
        "restart_count": int(step_summary.get("gmres_restart_count", 0)),
        "solve_time": float(step_summary.get("linear_solve_time", 0.0)),
        "block_factor_time": float(step_summary.get("preconditioner_load_time", 0.0)),
        "preconditioner_apply_time": float(step_summary.get("preconditioner_apply_time", 0.0)),
        "matvec_time": float(step_summary.get("matvec_time", 0.0)),
        "residual": step_summary.get("chosen_raw_residual"),
        "initial_residual": step_summary.get("initial_raw_residual"),
        "rhs_norm": step_summary.get("rhs_norm"),
        "true_rel_residual": step_summary.get("true_rel_residual"),
        "residual_ratio": step_summary.get("residual_ratio", step_summary.get("chosen_residual_ratio")),
        "success_by_true_residual": step_summary.get("success_by_true_residual"),
        "success_by_residual_ratio": step_summary.get("success_by_residual_ratio"),
        "chosen_success": step_summary.get("chosen_success"),
        "residual_success_conflict": step_summary.get("residual_success_conflict"),
        "residual_based_success": step_summary.get("residual_based_success", step_summary.get("chosen_success")),
        "residual_success_rtol": step_summary.get("residual_success_rtol", DEFAULT_RESIDUAL_SUCCESS_RTOL),
        "residual_success_atol": step_summary.get("residual_success_atol", DEFAULT_RESIDUAL_SUCCESS_ATOL),
        "final_raw_residual": step_summary.get("final_raw_residual"),
        "fallback_to_direct": bool(step_summary.get("fallback_to_direct", False)),
        "fallback_reason": step_summary.get("fallback_reason"),
        "process_peak_rss_kb": int(step_summary.get("process_peak_rss_kb", 0)),
        "block_stats": {
            "num_blocks": int(block_metadata.get("num_blocks", 0)),
            "covered_rows": int(block_metadata.get("covered_rows", 0)),
            "max_block_size": int(block_metadata.get("max_block_size", 0)),
            "total_block_nnz": int(block_metadata.get("total_block_nnz", 0)),
            "memory_estimate": int(block_metadata.get("memory_estimate", 0)),
        },
        "preconditioner_metadata": _compact_preconditioner_metadata(block_metadata),
    }
    block_candidates = step_summary.get("block_candidates") or []
    if block_candidates:
        record["block_candidates"] = _attach_block_candidates(record, block_candidates)
    return record


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=_json_default))
            handle.write("\n")


def _compute_mode_improvement_vs_row_sum(step_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[Tuple[int, int], Dict[str, Dict[str, Any]]] = {}
    for item in step_results:
        key = (int(item.get("circuit_id", -1)), int(item.get("selected_step_index", -1)))
        grouped.setdefault(key, {})[str(item.get("requested_mode"))] = item

    improvement: Dict[str, Dict[str, List[float]]] = {}
    for by_mode in grouped.values():
        row_sum = by_mode.get("row_sum")
        if row_sum is None:
            continue
        row_sum_iters = float(row_sum.get("gmres_iterations", 0.0))
        row_sum_time = float(row_sum.get("linear_solve_time", 0.0))
        for mode_name, item in by_mode.items():
            if mode_name == "row_sum":
                continue
            slot = improvement.setdefault(
                mode_name,
                {
                    "iteration_delta": [],
                    "solve_time_delta": [],
                    "better_iteration_count": [],
                    "better_time_count": [],
                },
            )
            slot["iteration_delta"].append(float(item.get("gmres_iterations", 0.0)) - row_sum_iters)
            slot["solve_time_delta"].append(float(item.get("linear_solve_time", 0.0)) - row_sum_time)
            slot["better_iteration_count"].append(float(item.get("gmres_iterations", 0.0) < row_sum_iters))
            slot["better_time_count"].append(float(item.get("linear_solve_time", 0.0) < row_sum_time))

    output: Dict[str, Any] = {}
    for mode_name, values in improvement.items():
        iteration_delta = values["iteration_delta"]
        solve_time_delta = values["solve_time_delta"]
        output[mode_name] = {
            "mean_iteration_delta_vs_row_sum": float(np.mean(iteration_delta)) if iteration_delta else 0.0,
            "median_iteration_delta_vs_row_sum": float(np.median(iteration_delta)) if iteration_delta else 0.0,
            "p90_iteration_delta_vs_row_sum": float(np.percentile(iteration_delta, 90)) if iteration_delta else 0.0,
            "mean_solve_time_delta_vs_row_sum": float(np.mean(solve_time_delta)) if solve_time_delta else 0.0,
            "better_iteration_rate_vs_row_sum": float(np.mean(values["better_iteration_count"])) if values["better_iteration_count"] else 0.0,
            "better_time_rate_vs_row_sum": float(np.mean(values["better_time_count"])) if values["better_time_count"] else 0.0,
            "comparison_count": int(len(iteration_delta)),
        }
    return output


def _evaluate_selected_steps(
    *,
    circuit_id: int,
    selected_steps: List[Dict[str, Any]],
    mode_configs: List[PrototypeConfig],
    step_offset: int,
) -> List[Dict[str, Any]]:
    step_rows: List[Dict[str, Any]] = []
    for local_idx, step in enumerate(selected_steps):
        base_summary = {
            "circuit_id": int(circuit_id),
            "netlist_path": step.get("netlist_path"),
            "selected_step_index": int(
                step.get("_workpoint_manifest_index", int(step_offset) + local_idx)
            ),
            "workpoint_manifest_index": step.get("_workpoint_manifest_index"),
            "corpus_iteration": int(step.get("iteration", -1)),
            "time": float(step.get("time", 0.0)),
            "gmin_val": float(step.get("gmin_val", 0.0)),
            "matrix_size": int(step.get("meta", {}).get("matrix_size", len(step.get("rhs", [])))),
            "node_map_hash": _compute_node_map_hash(step.get("node_map", {})),
            "synthetic_source_mode": "row_sum",
        }
        for config in mode_configs:
            step_summary = dict(base_summary)
            try:
                step_summary.update(evaluate_corpus_step(step, config))
            except Exception as exc:
                step_summary.update(
                    {
                        "requested_mode": config.requested_mode,
                        "executed_mode": config.fallback_mode if config.requested_mode in {"learned_diagonal", "learned_diagonal_synthetic"} else config.requested_mode,
                        "fallback_mode": config.fallback_mode,
                        "fallback_to_direct": False,
                        "fallback_reason": f"prototype_step_failed:{exc}",
                        "gmres_converged": False,
                        "gmres_iterations": 0,
                        "gmres_restart_count": 0,
                        "chosen_raw_residual": None,
                    }
                )
            step_rows.append(step_summary)
    return step_rows


def _json_default(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the external GMRES prototype on exported NIiter corpus steps.")
    parser.add_argument("--netlist-dir", default=DEFAULT_NETLIST_DIR)
    parser.add_argument("--circuit-id", type=int, default=0)
    parser.add_argument("--circuit-ids", default="", help="Comma-separated ids or ranges like 0-4. Overrides --circuit-id when set.")
    parser.add_argument("--netlist-mode", choices=["as_is", "force_sparse", "force_klu", "noop"], default="force_sparse")
    parser.add_argument("--requested-mode", choices=["identity", "jacobi", "row_sum", "ilu0", "ilut", "ilu_drop_tol", "ilu_fill_factor", "generic_block_jacobi", "branch_incident_block", "cell_block_jacobi", "explicit_schur_interface", "exact_schur_p_equals_s", "learned_schur_diagonal", "learned_sparse_schur", "learning_augmented", "learned_sparse_schur_safe_add", "learned_sparse_schur_safe_add_probe", "hard_sparse_schur_topk_abs", "hard_sparse_schur_threshold", "hybrid_sparse_schur_low_rank_topk_abs", "hybrid_sparse_schur_low_rank_relative_threshold", "hybrid_sparse_schur_low_rank_row_topk", "power_schur_m0", "power_schur_m1", "power_schur_m2", "power_schur_m3", "power_sparse_schur_m0", "power_sparse_schur_m1", "power_sparse_schur_m2", "power_sparse_schur_m3", *sorted(POWER_SPARSE_SCHUR_ARNOLDI_MODES), "sparse_schur_topk_abs", "sparse_schur_relative_threshold", "sparse_schur_row_topk", "sparse_schur_row_topk_selected_local", "hard_sparse_schur_topk_abs", "hard_sparse_schur_threshold", "local_sparse_schur_topk_abs", "local_sparse_schur_per_instance_topk", "local_schur_additive", "learning_augmented", "learned_sparse_schur_safe_add", "learned_sparse_schur_safe_add_probe", "learned_local_sparse_schur", "learned_diagonal", "learned_diagonal_synthetic", "learned_schwarz_v1", "learned_boundary_correction_v1"], default="identity")
    parser.add_argument("--fallback-mode", choices=["identity", "jacobi", "row_sum"], default="identity")
    parser.add_argument("--sidecar-path", default="")
    parser.add_argument("--learned-schwarz-checkpoint", default="")
    parser.add_argument("--learned-schur-checkpoint", default="")
    parser.add_argument("--learned-sparse-schur-checkpoint", default="")
    parser.add_argument("--learned-local-sparse-schur-checkpoint", default="")
    parser.add_argument("--learned-local-sparse-schur-selection-policy", choices=["hard_topk", "learned_replace", "learned_union_slack", "probe_union_slack", "probe_greedy_slack", "probe_best_of"], default="learned_union_slack")
    parser.add_argument("--learned-local-sparse-schur-slack-fraction", type=float, default=0.1)
    parser.add_argument("--learned-local-sparse-schur-slack-min", type=int, default=1)
    parser.add_argument("--learned-local-sparse-schur-probe-iterations", type=int, default=1)
    parser.add_argument("--sparse-schur-edge-budget", type=int, default=64)
    parser.add_argument("--sparse-schur-candidate-edge-limit", type=int, default=512)
    parser.add_argument("--sparse-schur-max-nnz", type=int, default=0)
    parser.add_argument("--sparse-schur-max-degree", type=int, default=0)
    parser.add_argument("--sparse-schur-max-exact-entries", type=int, default=0)
    parser.add_argument("--sparse-schur-diagonal-shift", type=float, default=1e-8)
    parser.add_argument('--sparse-schur-row-topk', type=int, default=2)
    parser.add_argument('--sparse-schur-relative-threshold', type=float, default=0.05)
    parser.add_argument('--local-schur-budget-multiplier', type=float, default=2.0)
    parser.add_argument('--local-schur-additive-include-diagonal', action='store_true')
    parser.add_argument('--local-schur-additive-diagonal-weight', type=float, default=0.0)
    parser.add_argument('--local-schur-additive-local-weight', type=float, default=1.0)
    parser.add_argument('--local-schur-additive-cluster-weight', type=float, default=0.0)
    parser.add_argument('--local-schur-additive-cluster-hops', type=int, default=0)
    parser.add_argument('--local-schur-additive-max-cluster-size', type=int, default=16)
    parser.add_argument('--local-schur-additive-patch-svd-rcond', type=float, default=0.0)
    parser.add_argument('--local-schur-additive-inner-iterations', type=int, default=1)
    parser.add_argument('--schur-low-rank-rank', type=int, default=4)
    parser.add_argument('--schur-low-rank-mode', choices=['constant', 'contiguous', 'slow_eig', 'correction_svd', 'precond_error_svd', 'inverse_error_svd'], default='slow_eig')
    parser.add_argument('--schur-low-rank-strength', type=float, default=1.0)
    parser.add_argument(
        "--benchmark-modes",
        default="",
        help="Comma-separated mode list. Example: identity,jacobi,row_sum,learned_diagonal_synthetic",
    )
    parser.add_argument(
        "--phase-1b5-benchmark",
        action="store_true",
        help="Run the standard Phase 1B.5 benchmark mode set.",
    )
    parser.add_argument("--restart", type=int, default=DEFAULT_RESTART)
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--trajectory-dir", default="", help="Use pre-generated data_coupler trajectory files instead of re-running ngspice linear-system export.")
    parser.add_argument(
        "--workpoint-manifest",
        default="",
        help="Schema-1 JSON workpoint list. When set, it controls exact trajectory selection.",
    )
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--case", default="")
    parser.add_argument("--keep-temp-netlists", action="store_true")
    parser.add_argument("--use-zero-x0", action="store_true")
    parser.add_argument("--disable-gmin-diagonal", action="store_true")
    parser.add_argument("--block-mode", choices=["branch_incident", "cell_instance", "cell_core", "cell_full", "cell_core_plus_onehop_boundary", "generic"], default="branch_incident")
    parser.add_argument("--max-block-size", type=int, default=32)
    parser.add_argument("--min-block-size", type=int, default=2)
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument("--max-total-block-nnz", type=int, default=0)
    parser.add_argument("--uncovered-row-policy", choices=["identity", "jacobi", "row_sum"], default="row_sum")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    workpoint_manifest_path = os.path.abspath(str(args.workpoint_manifest).strip()) if str(args.workpoint_manifest).strip() else ""
    workpoint_manifest_by_circuit = (
        _load_workpoint_manifest(workpoint_manifest_path)
        if workpoint_manifest_path
        else {}
    )
    circuit_ids = (
        sorted(workpoint_manifest_by_circuit)
        if workpoint_manifest_by_circuit
        else (_parse_circuit_ids(args.circuit_ids) if str(args.circuit_ids).strip() else [int(args.circuit_id)])
    )
    run_tag = args.run_tag or (
        f"c{circuit_ids[0]}_{args.netlist_mode}_{args.requested_mode}_{ts}"
        if len(circuit_ids) == 1
        else f"multi_{len(circuit_ids)}_{args.netlist_mode}_{args.requested_mode}_{ts}"
    )
    output_dir = os.path.join(args.output_root, run_tag)
    os.makedirs(output_dir, exist_ok=True)

    payload: Dict[str, Any] = {
        "run_tag": run_tag,
        "timestamp": ts,
        "prepared_netlist_path": None,
        "prepared_netlist_dir": None,
        "circuit_ids": [int(item) for item in circuit_ids],
        "corpus_results": [],
        "prototype_config": {
            "requested_mode": args.requested_mode,
            "fallback_mode": args.fallback_mode,
            "benchmark_modes": [],
            "restart": args.restart,
            "max_iters": args.max_iters,
            "rtol": args.rtol,
            "atol": args.atol,
            "use_rhsold_as_x0": not args.use_zero_x0,
            "apply_gmin_diagonal": not args.disable_gmin_diagonal,
            "sidecar_path": args.sidecar_path,
            "block_mode": args.block_mode,
            "max_block_size": args.max_block_size,
            "min_block_size": args.min_block_size,
            "max_blocks": args.max_blocks,
            "max_total_block_nnz": args.max_total_block_nnz,
            "uncovered_row_policy": args.uncovered_row_policy,
            "learned_schwarz_checkpoint": args.learned_schwarz_checkpoint,
            "learned_schur_checkpoint": args.learned_schur_checkpoint,
            "learned_sparse_schur_checkpoint": args.learned_sparse_schur_checkpoint,
            "learned_local_sparse_schur_checkpoint": args.learned_local_sparse_schur_checkpoint,
            "learned_local_sparse_schur_selection_policy": args.learned_local_sparse_schur_selection_policy,
            "learned_local_sparse_schur_slack_fraction": args.learned_local_sparse_schur_slack_fraction,
            "learned_local_sparse_schur_slack_min": args.learned_local_sparse_schur_slack_min,
            "learned_local_sparse_schur_probe_iterations": args.learned_local_sparse_schur_probe_iterations,
            "sparse_schur_edge_budget": args.sparse_schur_edge_budget,
            "sparse_schur_candidate_edge_limit": args.sparse_schur_candidate_edge_limit,
            "sparse_schur_max_nnz": args.sparse_schur_max_nnz,
            "sparse_schur_max_degree": args.sparse_schur_max_degree,
            "sparse_schur_max_exact_entries": args.sparse_schur_max_exact_entries,
            "sparse_schur_diagonal_shift": args.sparse_schur_diagonal_shift,
            "schur_low_rank_rank": args.schur_low_rank_rank,
            "schur_low_rank_mode": args.schur_low_rank_mode,
            "schur_low_rank_strength": args.schur_low_rank_strength,
            "local_schur_budget_multiplier": args.local_schur_budget_multiplier,
            "workpoint_manifest": (
                os.path.relpath(workpoint_manifest_path, REPO_ROOT)
                if workpoint_manifest_path
                else None
            ),
        },
        "steps": [],
        "aggregate": {},
        "aggregate_by_mode": {},
        "buckets_by_mode": {},
    }

    mode_names: List[str]
    if args.phase_1b5_benchmark:
        mode_names = list(DEFAULT_BENCHMARK_MODES)
    elif args.benchmark_modes:
        mode_names = [item.strip() for item in str(args.benchmark_modes).split(",") if item.strip()]
    else:
        mode_names = [str(args.requested_mode)]
    payload["prototype_config"]["benchmark_modes"] = mode_names

    mode_configs = [
        PrototypeConfig(
            requested_mode=mode_name,
            fallback_mode=args.fallback_mode,
            restart=int(args.restart),
            max_iters=int(args.max_iters),
            rtol=float(args.rtol),
            atol=float(args.atol),
            use_rhsold_as_x0=not args.use_zero_x0,
            apply_gmin_diagonal=not args.disable_gmin_diagonal,
            residual_worsen_ratio=DEFAULT_RESIDUAL_WORSEN_RATIO,
            residual_worsen_abs=DEFAULT_RESIDUAL_WORSEN_ABS,
            sidecar_path=str(args.sidecar_path),
            synthetic_source_mode="row_sum",
            block_mode=(
                str(args.block_mode)
                if str(mode_name) == "cell_block_jacobi"
                else str(args.block_mode)
            ),
            max_block_size=int(args.max_block_size),
            min_block_size=int(args.min_block_size),
            max_blocks=int(args.max_blocks),
            max_total_block_nnz=int(args.max_total_block_nnz),
            uncovered_row_policy=str(args.uncovered_row_policy),
            learned_schwarz_checkpoint=str(args.learned_schwarz_checkpoint),
            learned_schur_checkpoint=str(args.learned_schur_checkpoint),
            learned_sparse_schur_checkpoint=str(args.learned_sparse_schur_checkpoint),
            learned_local_sparse_schur_checkpoint=str(args.learned_local_sparse_schur_checkpoint),
            learned_local_sparse_schur_selection_policy=str(args.learned_local_sparse_schur_selection_policy),
            learned_local_sparse_schur_slack_fraction=float(args.learned_local_sparse_schur_slack_fraction),
            learned_local_sparse_schur_slack_min=int(args.learned_local_sparse_schur_slack_min),
            learned_local_sparse_schur_probe_iterations=int(args.learned_local_sparse_schur_probe_iterations),
            sparse_schur_edge_budget=int(args.sparse_schur_edge_budget),
            sparse_schur_candidate_edge_limit=int(args.sparse_schur_candidate_edge_limit),
            sparse_schur_max_nnz=int(args.sparse_schur_max_nnz),
            sparse_schur_max_degree=int(args.sparse_schur_max_degree),
            sparse_schur_max_exact_entries=int(args.sparse_schur_max_exact_entries),
            sparse_schur_diagonal_shift=float(args.sparse_schur_diagonal_shift),
            sparse_schur_row_topk=int(args.sparse_schur_row_topk),
            sparse_schur_relative_threshold=float(args.sparse_schur_relative_threshold),
            local_schur_budget_multiplier=float(args.local_schur_budget_multiplier),
            local_schur_additive_include_diagonal=bool(args.local_schur_additive_include_diagonal),
            local_schur_additive_diagonal_weight=float(args.local_schur_additive_diagonal_weight),
            local_schur_additive_local_weight=float(args.local_schur_additive_local_weight),
            local_schur_additive_cluster_weight=float(args.local_schur_additive_cluster_weight),
            local_schur_additive_cluster_hops=int(args.local_schur_additive_cluster_hops),
            local_schur_additive_max_cluster_size=int(args.local_schur_additive_max_cluster_size),
            local_schur_additive_patch_svd_rcond=float(args.local_schur_additive_patch_svd_rcond),
            local_schur_additive_inner_iterations=int(args.local_schur_additive_inner_iterations),
            schur_low_rank_rank=int(args.schur_low_rank_rank),
            schur_low_rank_mode=str(args.schur_low_rank_mode),
            schur_low_rank_strength=float(args.schur_low_rank_strength),
        )
        for mode_name in mode_names
    ]
    all_step_rows: List[Dict[str, Any]] = []
    prepared_netlist_dirs = set()
    for circuit_idx, circuit_id in enumerate(circuit_ids):
        prepared_netlist_dir, prepared_netlist_path = _prepare_netlist_dir(
            args.netlist_dir,
            int(circuit_id),
            args.netlist_mode,
            output_dir,
        )
        prepared_netlist_dirs.add(prepared_netlist_dir)
        if circuit_idx == 0:
            payload["prepared_netlist_path"] = prepared_netlist_path
            payload["prepared_netlist_dir"] = prepared_netlist_dir

        case = f"{args.case}_c{circuit_id}" if args.case else f"external_gmres_{run_tag}_c{circuit_id}"
        if str(args.trajectory_dir).strip():
            corpus_result = _load_trajectory_linear_system_steps(
                trajectory_dir=os.path.abspath(str(args.trajectory_dir).strip()),
                circuit_id=int(circuit_id),
                netlist_path=prepared_netlist_path,
                requested_workpoints=(
                    workpoint_manifest_by_circuit.get(int(circuit_id))
                    if workpoint_manifest_by_circuit
                    else None
                ),
            )
        else:
            corpus_result = run_ngspice_linear_system_corpus(
                val_dir=output_dir,
                netlist_dir=prepared_netlist_dir,
                real_ckt_id=int(circuit_id),
                case=case,
                timeout=args.timeout,
            )
        payload["corpus_results"].append(
            {
                "circuit_id": int(circuit_id),
                "success": bool(corpus_result.get("success")),
                "reason": corpus_result.get("reason"),
                "task_dir": corpus_result.get("task_dir"),
                "stderr": corpus_result.get("stderr", ""),
                "stdout": corpus_result.get("stdout", ""),
                "step_count": len(corpus_result.get("steps", [])),
            }
        )

        selected_steps = corpus_result.get("steps", [])
        if not workpoint_manifest_by_circuit:
            selected_steps = selected_steps[int(args.step_offset):]
            if int(args.max_steps) > 0:
                selected_steps = selected_steps[: int(args.max_steps)]
        for step in selected_steps:
            step["netlist_path"] = prepared_netlist_path
        all_step_rows.extend(
            _evaluate_selected_steps(
                circuit_id=int(circuit_id),
                selected_steps=selected_steps,
                mode_configs=mode_configs,
                step_offset=int(args.step_offset),
            )
        )

    if workpoint_manifest_by_circuit:
        all_step_rows.sort(
            key=lambda item: int(item.get("workpoint_manifest_index", 1 << 30))
        )
    payload["steps"] = all_step_rows
    payload["corpus_result"] = {
        "success": all(bool(item.get("success")) for item in payload["corpus_results"]),
        "reason": None,
        "task_dir": None,
        "stderr": "",
        "stdout": "",
        "step_count": int(sum(int(item.get("step_count", 0)) for item in payload["corpus_results"])),
        "circuit_count": int(len(circuit_ids)),
    }

    payload["aggregate"] = _summarize_results(payload["steps"])
    requested_modes = sorted({str(item["requested_mode"]) for item in payload["steps"]})
    payload["aggregate_by_mode"] = {
        mode_name: _summarize_results([item for item in payload["steps"] if str(item["requested_mode"]) == mode_name])
        for mode_name in requested_modes
    }
    payload["buckets_by_mode"] = {
        mode_name: _build_bucketed_summary(
            [item for item in payload["steps"] if str(item["requested_mode"]) == mode_name],
            ["circuit_id", "gmin_val", "corpus_iteration", "matrix_size"],
        )
        for mode_name in requested_modes
    }
    payload["improvement_vs_row_sum"] = _compute_mode_improvement_vs_row_sum(payload["steps"])

    summary_json = os.path.join(output_dir, "summary.json")
    per_step_jsonl = os.path.join(output_dir, "per_step.jsonl")
    solver_outcome_jsonl = os.path.join(output_dir, "solver_outcome.jsonl")
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    _write_jsonl(per_step_jsonl, [_mode_step_jsonl_record(item) for item in payload["steps"]])
    _write_jsonl(solver_outcome_jsonl, [_solver_outcome_record(item) for item in payload["steps"]])

    print(f"run_tag={run_tag}")
    print(f"summary_json={summary_json}")
    print(f"per_step_jsonl={per_step_jsonl}")
    print(f"solver_outcome_jsonl={solver_outcome_jsonl}")
    print(f"corpus_success={payload['corpus_result']['success']}")
    print(f"corpus_reason={payload['corpus_result']['reason']}")
    print(f"evaluated_steps={len(payload['steps'])}")
    print(f"gmres_success_rate={payload['aggregate'].get('gmres_success_rate')}")
    print(f"fallback_rate={payload['aggregate'].get('fallback_rate')}")
    print(f"executed_modes={payload['aggregate'].get('executed_modes')}")
    print(f"requested_modes={requested_modes}")

    if args.netlist_mode != "as_is" and not args.keep_temp_netlists:
        for prepared_netlist_dir in sorted(prepared_netlist_dirs):
            shutil.rmtree(prepared_netlist_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
