import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pypath.preconditioner.block_schwarz import BlockPlanConfig, build_block_schwarz_plan
from pypath.preconditioner.linear_system_contract import (
    CHECKPOINT_SCHEMA_VERSION,
    EFFECTIVE_MATRIX_CONTRACT,
    FEATURE_CONTRACT,
    INITIAL_GUESS_MODE_RHSOLD,
    INITIAL_GUESS_MODES,
    INITIAL_RESIDUAL_FORMULA,
    LOCAL_SHIFT_CONTRACT,
    LOCAL_SHIFT_FLOOR_RELATIVE,
    require_local_shift_floor_relative,
    compute_initial_residual,
    resolve_initial_guess,
)
from pypath.preconditioner.learned_schwarz import (
    BLOCK_FEATURE_DIM,
    CORRECTION_FEATURE_DIM,
    ROW_FEATURE_DIM,
    BoundaryCorrectionPreconditioner,
    LearnedSchwarzPreconditioner,
    build_learned_schwarz_sample,
    make_probe_matrix,
)
from pypath.utils.external_gmres_prototype import (
    _load_trajectory_linear_system_steps,
    _make_system_matrix,
    _parse_circuit_ids,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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


def _load_training_workpoint_manifest(
    manifest_path: str,
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[str, Any]]:
    with open(manifest_path, "rb") as handle:
        manifest_bytes = handle.read()
    try:
        payload = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("workpoint_manifest_is_not_valid_utf8_json") from exc
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
        entry: Dict[str, Any] = {
            "key": key,
            "manifest_index": int(manifest_index),
        }
        for field_name in ("step_sha256", "jacobian_sha256", "netlist_sha256"):
            expected = item.get(field_name)
            if expected is None:
                continue
            if (
                not isinstance(expected, str)
                or len(expected) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in expected)
            ):
                raise ValueError(f"workpoint_manifest_invalid_{field_name}")
            entry[field_name] = expected.lower()
        by_circuit.setdefault(circuit_id, []).append(entry)
    source_hashes_required = bool(raw_workpoints) and all(
        all(field_name in entry for field_name in ("step_sha256", "jacobian_sha256", "netlist_sha256"))
        for entries in by_circuit.values()
        for entry in entries
    )
    return by_circuit, {
        "path": os.path.relpath(os.path.abspath(manifest_path), REPO_ROOT),
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "workpoint_count": int(len(raw_workpoints)),
        "source_hashes_required": bool(source_hashes_required),
        "source_hashes_verified": False,
    }


def _verify_manifest_file_hash(
    *,
    entry: Dict[str, Any],
    field_name: str,
    path: str,
) -> None:
    expected = entry.get(field_name)
    if expected is None:
        return
    if not os.path.isfile(path):
        raise ValueError(f"workpoint_manifest_{field_name}_missing:{entry['key']}")
    actual = _file_sha256(path)
    if actual != expected:
        raise ValueError(f"workpoint_manifest_{field_name}_mismatch:{entry['key']}")


def _numeric_summary(values: np.ndarray) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
        }
    if not np.all(np.isfinite(array)):
        raise ValueError("local shift statistics contain non-finite values")
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
    }


def _summarize_learned_shifts(
    *,
    model: LearnedSchwarzPreconditioner,
    samples: List[Any],
    local_shift_floor_relative: float,
) -> Dict[str, Any]:
    lambda_pred_chunks: List[np.ndarray] = []
    lambda_floor_chunks: List[np.ndarray] = []
    lambda_eff_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for sample in samples:
            parameters = model.predict_parameters(sample)
            lambda_pred_chunks.append(
                parameters["lambda_pred"].detach().cpu().numpy()
            )
            lambda_floor_chunks.append(
                parameters["lambda_floor"].detach().cpu().numpy()
            )
            lambda_eff_chunks.append(
                parameters["lambdas"].detach().cpu().numpy()
            )
    lambda_pred = np.concatenate(lambda_pred_chunks, axis=0)
    lambda_floor = np.concatenate(lambda_floor_chunks, axis=0)
    lambda_eff = np.concatenate(lambda_eff_chunks, axis=0)
    floor_active = lambda_pred <= lambda_floor
    return {
        "block_scale": _numeric_summary(
            lambda_floor / float(local_shift_floor_relative)
        ),
        "lambda_pred": _numeric_summary(lambda_pred),
        "lambda_floor": _numeric_summary(lambda_floor),
        "lambda_eff": _numeric_summary(lambda_eff),
        "floor_active_block_count": int(np.count_nonzero(floor_active)),
        "floor_active_block_ratio": float(
            np.count_nonzero(floor_active) / max(lambda_pred.size, 1)
        ),
    }


def _load_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    manifest_path = str(getattr(args, "workpoint_manifest", "") or "").strip()
    manifest_by_circuit: Optional[Dict[int, List[Dict[str, Any]]]] = None
    manifest_by_key: Dict[Tuple[int, str, str, int], Dict[str, Any]] = {}
    if manifest_path:
        manifest_by_circuit, provenance = _load_training_workpoint_manifest(manifest_path)
        manifest_count = sum(len(entries) for entries in manifest_by_circuit.values())
        if int(args.max_samples) > 0 and int(args.max_samples) != manifest_count:
            raise ValueError("workpoint_manifest_requires_max_samples_0_or_count")
        setattr(args, "_workpoint_manifest_provenance", provenance)
        circuit_ids = list(manifest_by_circuit)
        manifest_by_key = {
            tuple(entry["key"]): entry
            for entries in manifest_by_circuit.values()
            for entry in entries
        }
    else:
        circuit_ids = _parse_circuit_ids(args.circuit_ids)
    for circuit_id in circuit_ids:
        netlist_path = os.path.join(args.netlist_dir, f"{int(circuit_id)}.sp")
        corpus = _load_trajectory_linear_system_steps(
            trajectory_dir=args.trajectory_dir,
            circuit_id=int(circuit_id),
            netlist_path=netlist_path,
            requested_workpoints=(
                manifest_by_circuit.get(int(circuit_id))
                if manifest_by_circuit is not None
                else None
            ),
        )
        steps = list(corpus.get("steps", []))
        if manifest_by_circuit is None and args.positive_gmin_only:
            steps = [step for step in steps if float(step.get("gmin_val", 0.0)) > 0.0]
        if manifest_by_circuit is None:
            steps = steps[int(args.step_offset):]
        if manifest_by_circuit is None and int(args.max_steps_per_circuit) > 0:
            steps = steps[: int(args.max_steps_per_circuit)]
        for step in steps:
            manifest_entry = None
            if manifest_by_circuit is not None:
                key = _workpoint_key(
                    circuit_id=int(circuit_id),
                    time_value=float(step.get("time", 0.0)),
                    gmin_value=float(step.get("gmin_val", 0.0)),
                    iteration=int(step.get("iteration", -1)),
                )
                manifest_entry = manifest_by_key.get(key)
                if manifest_entry is None:
                    raise ValueError(f"workpoint_manifest_unexpected_step:{key}")
                _verify_manifest_file_hash(
                    entry=manifest_entry,
                    field_name="step_sha256",
                    path=str(step.get("filepath", "")),
                )
                _verify_manifest_file_hash(
                    entry=manifest_entry,
                    field_name="jacobian_sha256",
                    path=str(step.get("jacobian_filepath", "")),
                )
                _verify_manifest_file_hash(
                    entry=manifest_entry,
                    field_name="netlist_sha256",
                    path=netlist_path,
                )
            matrix = _make_system_matrix(
                step,
                apply_gmin_diagonal=not args.disable_gmin_diagonal,
            )
            linear_rhs = np.asarray(step.get("rhs", []), dtype=np.float64)
            initial_guess = resolve_initial_guess(
                rhsold=step.get("rhsold", []),
                matrix_size=int(matrix.shape[0]),
                initial_guess_mode=args.initial_guess_mode,
            )
            initial_residual = compute_initial_residual(
                effective_matrix=matrix,
                linear_rhs=linear_rhs,
                initial_guess=initial_guess,
            )
            plan = build_block_schwarz_plan(
                matrix=matrix,
                node_map=step.get("node_map", {}),
                netlist_path=netlist_path,
                config=BlockPlanConfig(
                    block_mode=args.block_mode,
                    max_block_size=int(args.max_block_size),
                    min_block_size=int(args.min_block_size),
                    max_blocks=int(args.max_blocks),
                    max_total_block_nnz=int(args.max_total_block_nnz),
                    uncovered_row_policy=args.uncovered_row_policy,
                ),
            )
            core_plan = None
            boundary_plan = None
            if args.model_kind == "boundary_correction_v1":
                core_plan = build_block_schwarz_plan(
                    matrix=matrix,
                    node_map=step.get("node_map", {}),
                    netlist_path=netlist_path,
                    config=BlockPlanConfig(
                        block_mode=args.core_block_mode,
                        max_block_size=int(args.max_block_size),
                        min_block_size=int(args.min_block_size),
                        max_blocks=int(args.max_blocks),
                        max_total_block_nnz=int(args.max_total_block_nnz),
                        uncovered_row_policy=args.uncovered_row_policy,
                    ),
                )
                boundary_plan = build_block_schwarz_plan(
                    matrix=matrix,
                    node_map=step.get("node_map", {}),
                    netlist_path=netlist_path,
                    config=BlockPlanConfig(
                        block_mode=args.boundary_block_mode,
                        max_block_size=int(args.max_block_size),
                        min_block_size=int(args.min_block_size),
                        max_blocks=int(args.max_blocks),
                        max_total_block_nnz=int(args.max_total_block_nnz),
                        uncovered_row_policy=args.uncovered_row_policy,
                    ),
                )
            if not plan.blocks:
                if manifest_entry is not None:
                    raise ValueError(
                        f"workpoint_manifest_empty_block_plan:{manifest_entry['key']}"
                    )
                continue
            if args.model_kind == "boundary_correction_v1" and (
                core_plan is None or boundary_plan is None or not core_plan.blocks or not boundary_plan.blocks
            ):
                if manifest_entry is not None:
                    raise ValueError(
                        f"workpoint_manifest_empty_block_plan:{manifest_entry['key']}"
                    )
                continue
            samples.append(
                {
                    "circuit_id": int(circuit_id),
                    "time": float(step.get("time", 0.0)),
                    "iteration": int(step.get("iteration", -1)),
                    "matrix": matrix,
                    "linear_rhs": linear_rhs,
                    "initial_guess": initial_guess,
                    "initial_residual": initial_residual,
                    "initial_guess_mode": str(args.initial_guess_mode),
                    "gmin": float(step.get("gmin_val", 0.0)),
                    "plan": plan,
                    "core_plan": core_plan,
                    "boundary_plan": boundary_plan,
                    "workpoint_manifest_index": (
                        int(manifest_entry["manifest_index"])
                        if manifest_entry is not None
                        else None
                    ),
                    "source_hashes_verified": bool(
                        manifest_entry is not None
                        and all(
                            field_name in manifest_entry
                            for field_name in (
                                "step_sha256",
                                "jacobian_sha256",
                                "netlist_sha256",
                            )
                        )
                    ),
                }
            )
            if (
                manifest_by_circuit is None
                and int(args.max_samples) > 0
                and len(samples) >= int(args.max_samples)
            ):
                return samples
    if manifest_by_circuit is not None:
        expected_count = sum(len(entries) for entries in manifest_by_circuit.values())
        if len(samples) != expected_count:
            raise ValueError(
                f"workpoint_manifest_sample_count_mismatch:{len(samples)}!={expected_count}"
            )
        samples.sort(key=lambda item: int(item["workpoint_manifest_index"]))
        provenance = getattr(args, "_workpoint_manifest_provenance")
        provenance["source_hashes_verified"] = bool(
            provenance["source_hashes_required"]
            and all(bool(item["source_hashes_verified"]) for item in samples)
        )
    return samples


def _make_core_arnoldi_probe_matrix(
    *,
    matrix: np.ndarray,
    linear_rhs: np.ndarray,
    initial_residual: np.ndarray,
    core_plan: Any,
    probe_count: int,
    seed: int,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    matrix_np = np.asarray(matrix, dtype=np.float64)
    matrix_size = int(matrix_np.shape[0])
    rng = np.random.default_rng(int(seed))
    starts: List[np.ndarray] = []
    for candidate in (initial_residual, linear_rhs):
        candidate_np = np.asarray(candidate, dtype=np.float64)
        if candidate_np.shape[0] == matrix_size and np.linalg.norm(candidate_np) > 0.0:
            starts.append(candidate_np)
    if not starts:
        starts.append(rng.standard_normal(matrix_size).astype(np.float64))

    q = starts[0].astype(np.float64, copy=True)
    q_norm = float(np.linalg.norm(q))
    if q_norm <= 0.0:
        q = rng.standard_normal(matrix_size).astype(np.float64)
        q_norm = float(np.linalg.norm(q))
    q = q / max(q_norm, 1e-30)
    basis: List[np.ndarray] = [q]
    probes: List[np.ndarray] = []

    for _ in range(int(max(probe_count, 0))):
        preconditioner_input = matrix_np.dot(q)
        probes.append(preconditioner_input.astype(np.float64, copy=True))
        w = np.asarray(core_plan.apply(preconditioner_input), dtype=np.float64)
        for previous in basis:
            w = w - float(np.dot(previous, w)) * previous
        w_norm = float(np.linalg.norm(w))
        if w_norm <= 1e-20:
            w = rng.standard_normal(matrix_size).astype(np.float64)
            for previous in basis:
                w = w - float(np.dot(previous, w)) * previous
            w_norm = float(np.linalg.norm(w))
        if w_norm <= 1e-20:
            break
        q = w / w_norm
        basis.append(q)

    if not probes:
        return torch.empty((0, matrix_size), dtype=dtype)
    return torch.as_tensor(np.stack(probes, axis=0), dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the first learned Schwarz preconditioner with probe loss.")
    parser.add_argument("--netlist-dir", required=True)
    parser.add_argument("--trajectory-dir", required=True)
    parser.add_argument("--circuit-ids", default="0")
    parser.add_argument(
        "--workpoint-manifest",
        default="",
        help="schema-1 exact training workpoint manifest; requires --max-samples 0 or its exact count",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--model-kind", choices=["learned_schwarz_v1", "boundary_correction_v1"], default="learned_schwarz_v1")
    parser.add_argument("--block-mode", default="cell_core_plus_onehop_boundary")
    parser.add_argument("--core-block-mode", default="cell_core")
    parser.add_argument("--boundary-block-mode", default="cell_core_plus_onehop_boundary")
    parser.add_argument("--max-block-size", type=int, default=32)
    parser.add_argument("--min-block-size", type=int, default=2)
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument("--max-total-block-nnz", type=int, default=0)
    parser.add_argument("--uncovered-row-policy", choices=["identity", "jacobi", "row_sum"], default="row_sum")
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--max-steps-per-circuit", type=int, default=1)
    parser.add_argument("--positive-gmin-only", action="store_true")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lambda-scale", type=float, default=1e-6)
    parser.add_argument("--lambda-min", type=float, default=0.0)
    parser.add_argument("--lambda-initial-relative", type=float, default=1e-5)
    parser.add_argument(
        "--local-shift-floor-relative",
        type=float,
        default=LOCAL_SHIFT_FLOOR_RELATIVE,
    )
    parser.add_argument("--correction-scale", type=float, default=1.0)
    parser.add_argument("--max-correction-ratio", type=float, default=0.0)
    parser.add_argument("--projection-mode", choices=["none", "residual_scalar"], default="none")
    parser.add_argument("--projection-max-scale", type=float, default=1.0)
    parser.add_argument("--local-solution-gain-limit", type=float, default=0.0)
    parser.add_argument("--block-contribution-budget-ratio", type=float, default=0.0)
    parser.add_argument("--block-contribution-absolute-cap", type=float, default=0.0)
    parser.add_argument("--correction-feature-dim", type=int, default=CORRECTION_FEATURE_DIM)
    parser.add_argument("--boundary-residual-floor", type=float, default=1e-6)
    parser.add_argument("--boundary-correction-weight", type=float, default=1e-4)
    parser.add_argument("--boundary-do-no-harm-weight", type=float, default=0.0)
    parser.add_argument("--boundary-do-no-harm-margin", type=float, default=0.0)
    parser.add_argument("--boundary-alignment-weight", type=float, default=0.0)
    parser.add_argument("--boundary-alignment-min-cos", type=float, default=0.0)
    parser.add_argument("--arnoldi-probes", type=int, default=0)
    parser.add_argument("--gaussian-probes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--initial-guess-mode",
        choices=sorted(INITIAL_GUESS_MODES),
        default=INITIAL_GUESS_MODE_RHSOLD,
        help="linear solver initial guess used to construct training features",
    )
    parser.add_argument("--disable-gmin-diagonal", action="store_true")
    args = parser.parse_args()
    if args.disable_gmin_diagonal:
        raise ValueError(
            "schema v4 learned Schwarz training requires the effective "
            "matrix J + gmin I; do not disable the gmin diagonal"
        )

    local_shift_floor_relative = None
    if args.model_kind == "learned_schwarz_v1":
        local_shift_floor_relative = require_local_shift_floor_relative(
            args.local_shift_floor_relative
        )
        if (
            not np.isfinite(float(args.lambda_initial_relative))
            or float(args.lambda_initial_relative) <= 0.0
        ):
            raise ValueError("lambda_initial_relative must be finite and positive")
        if not np.isfinite(float(args.lambda_scale)) or float(args.lambda_scale) <= 0.0:
            raise ValueError("lambda_scale must be finite and positive")
        if not np.isfinite(float(args.lambda_min)) or float(args.lambda_min) < 0.0:
            raise ValueError("lambda_min must be finite and non-negative")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.set_default_dtype(torch.float64)

    run_tag = args.run_tag or time.strftime("learned_schwarz_%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, run_tag)
    os.makedirs(output_dir, exist_ok=True)

    sample_payloads = _load_samples(args)
    if not sample_payloads:
        raise RuntimeError("no learned Schwarz training samples were loaded")

    if args.model_kind == "boundary_correction_v1":
        model_args = {
            "block_feature_dim": BLOCK_FEATURE_DIM,
            "row_feature_dim": ROW_FEATURE_DIM,
            "correction_feature_dim": int(args.correction_feature_dim),
            "hidden_dim": int(args.hidden_dim),
            "correction_scale": float(args.correction_scale),
            "max_correction_ratio": float(args.max_correction_ratio),
            "projection_mode": str(args.projection_mode),
            "projection_max_scale": float(args.projection_max_scale),
            "local_solution_gain_limit": float(args.local_solution_gain_limit),
            "block_contribution_budget_ratio": float(args.block_contribution_budget_ratio),
            "block_contribution_absolute_cap": float(args.block_contribution_absolute_cap),
        }
        model = BoundaryCorrectionPreconditioner(**model_args).to(dtype=torch.float64)
        stat_samples = [
            build_learned_schwarz_sample(
                matrix=payload["matrix"],
                plan=payload["boundary_plan"],
                linear_rhs=payload["linear_rhs"],
                initial_residual=payload["initial_residual"],
                gmin=payload["gmin"],
                dtype=torch.float64,
            )
            for payload in sample_payloads
        ]
    else:
        model_args = {
            "block_feature_dim": BLOCK_FEATURE_DIM,
            "row_feature_dim": ROW_FEATURE_DIM,
            "hidden_dim": int(args.hidden_dim),
            "lambda_scale": float(args.lambda_scale),
            "lambda_min": float(args.lambda_min),
        }
        model = LearnedSchwarzPreconditioner(**model_args).to(dtype=torch.float64)
        stat_samples = [
            build_learned_schwarz_sample(
                matrix=payload["matrix"],
                plan=payload["plan"],
                linear_rhs=payload["linear_rhs"],
                initial_residual=payload["initial_residual"],
                gmin=payload["gmin"],
                dtype=torch.float64,
            )
            for payload in sample_payloads
        ]
    block_feature_rows = torch.cat([sample.block_features for sample in stat_samples], dim=0)
    row_feature_rows = torch.cat([row_features for sample in stat_samples for row_features in sample.row_features], dim=0)
    model.set_feature_stats(
        block_mean=block_feature_rows.mean(dim=0),
        block_std=block_feature_rows.std(dim=0, unbiased=False).clamp_min(1e-12),
        row_mean=row_feature_rows.mean(dim=0),
        row_std=row_feature_rows.std(dim=0, unbiased=False).clamp_min(1e-12),
    )
    local_shift_initialization: Dict[str, Any] = {}
    if args.model_kind == "learned_schwarz_v1":
        if local_shift_floor_relative is None:
            raise AssertionError("missing learned Schwarz floor relative")
        lambda_floors = torch.cat(
            [sample.lambda_floors.reshape(-1) for sample in stat_samples],
            dim=0,
        ).detach().cpu().numpy()
        if lambda_floors.size == 0:
            raise RuntimeError("no learned Schwarz local shift floors were built")
        block_scales = lambda_floors / float(local_shift_floor_relative)
        initial_block_scale_reference = float(np.median(block_scales))
        initial_lambda_target = float(
            float(args.lambda_initial_relative)
            * initial_block_scale_reference
        )
        model.initialize_lambda_prediction(initial_lambda_target)
        local_shift_initialization = {
            "local_shift_contract": LOCAL_SHIFT_CONTRACT,
            "local_shift_floor_relative": float(local_shift_floor_relative),
            "lambda_initial_relative": float(args.lambda_initial_relative),
            "initial_block_scale_statistic": "median",
            "initial_block_scale_reference": initial_block_scale_reference,
            "initial_lambda_target": initial_lambda_target,
        }

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    history: List[Dict[str, Any]] = []

    for epoch in range(int(args.epochs)):
        total_loss = 0.0
        for sample_idx, payload in enumerate(sample_payloads):
            sample = build_learned_schwarz_sample(
                matrix=payload["matrix"],
                plan=payload["plan"],
                linear_rhs=payload["linear_rhs"],
                initial_residual=payload["initial_residual"],
                gmin=payload["gmin"],
                dtype=torch.float64,
            )
            probes = make_probe_matrix(
                matrix_size=int(payload["matrix"].shape[0]),
                linear_rhs=payload["linear_rhs"],
                initial_residual=payload["initial_residual"],
                gaussian_count=int(args.gaussian_probes),
                seed=int(args.seed) + epoch * 1009 + sample_idx,
                dtype=torch.float64,
            )
            if args.model_kind == "boundary_correction_v1" and int(args.arnoldi_probes) > 0:
                arnoldi_probes = _make_core_arnoldi_probe_matrix(
                    matrix=payload["matrix"],
                    linear_rhs=payload["linear_rhs"],
                    initial_residual=payload["initial_residual"],
                    core_plan=payload["core_plan"],
                    probe_count=int(args.arnoldi_probes),
                    seed=int(args.seed) + epoch * 1009 + sample_idx + 1000003,
                    dtype=torch.float64,
                )
                if arnoldi_probes.numel() > 0:
                    probes = torch.cat([probes, arnoldi_probes], dim=0)
            optimizer.zero_grad(set_to_none=True)
            if args.model_kind == "boundary_correction_v1":
                core_sample = build_learned_schwarz_sample(
                    matrix=payload["matrix"],
                    plan=payload["core_plan"],
                    linear_rhs=payload["linear_rhs"],
                    initial_residual=payload["initial_residual"],
                    gmin=payload["gmin"],
                    dtype=torch.float64,
                )
                boundary_sample = build_learned_schwarz_sample(
                    matrix=payload["matrix"],
                    plan=payload["boundary_plan"],
                    linear_rhs=payload["linear_rhs"],
                    initial_residual=payload["initial_residual"],
                    gmin=payload["gmin"],
                    dtype=torch.float64,
                )
                loss = model.probe_loss(
                    core_sample,
                    boundary_sample,
                    probes,
                    residual_floor=float(args.boundary_residual_floor),
                    correction_weight=float(args.boundary_correction_weight),
                    do_no_harm_weight=float(args.boundary_do_no_harm_weight),
                    do_no_harm_margin=float(args.boundary_do_no_harm_margin),
                    alignment_weight=float(args.boundary_alignment_weight),
                    alignment_min_cos=float(args.boundary_alignment_min_cos),
                )
            else:
                loss = model.probe_loss(sample, probes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        row = {"epoch": epoch + 1, "mean_probe_loss": total_loss / max(len(sample_payloads), 1)}
        history.append(row)
        print(json.dumps(row), flush=True)

    local_shift_summary: Dict[str, Any] = {}
    if args.model_kind == "learned_schwarz_v1":
        if local_shift_floor_relative is None:
            raise AssertionError("missing learned Schwarz floor relative")
        local_shift_summary = {
            **local_shift_initialization,
            **_summarize_learned_shifts(
                model=model,
                samples=stat_samples,
                local_shift_floor_relative=float(local_shift_floor_relative),
            ),
        }

    checkpoint_name = "boundary_correction_v1.pt" if args.model_kind == "boundary_correction_v1" else "learned_schwarz_v1.pt"
    checkpoint_path = os.path.join(output_dir, checkpoint_name)
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "feature_contract": FEATURE_CONTRACT,
            "effective_matrix_contract": EFFECTIVE_MATRIX_CONTRACT,
            "initial_residual_formula": INITIAL_RESIDUAL_FORMULA,
            "initial_guess_mode": str(args.initial_guess_mode),
            **(
                {
                    "local_shift_contract": LOCAL_SHIFT_CONTRACT,
                    "local_shift_floor_relative": float(
                        local_shift_floor_relative
                    ),
                    "local_shift_initialization": local_shift_initialization,
                    "local_shift_final": local_shift_summary,
                }
                if args.model_kind == "learned_schwarz_v1"
                else {}
            ),
            "model_kind": args.model_kind,
            "model_args": model_args,
            "model_state_dict": model.state_dict(),
            "train_args": vars(args),
            "workpoint_manifest": getattr(args, "_workpoint_manifest_provenance", None),
            "history": history,
            "sample_count": len(sample_payloads),
        },
        checkpoint_path,
    )
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_tag": run_tag,
                "checkpoint_path": checkpoint_path,
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "feature_contract": FEATURE_CONTRACT,
                "effective_matrix_contract": EFFECTIVE_MATRIX_CONTRACT,
                "initial_residual_formula": INITIAL_RESIDUAL_FORMULA,
                "initial_guess_mode": str(args.initial_guess_mode),
                "workpoint_manifest": getattr(
                    args, "_workpoint_manifest_provenance", None
                ),
                **(
                    {"local_shift": local_shift_summary}
                    if args.model_kind == "learned_schwarz_v1"
                    else {}
                ),
                "sample_count": len(sample_payloads),
                "history": history,
                "samples": [
                    {
                        "circuit_id": item["circuit_id"],
                        "time": float(item["time"]),
                        "gmin": float(item["gmin"]),
                        "iteration": item["iteration"],
                        "workpoint_manifest_index": item["workpoint_manifest_index"],
                        "source_hashes_verified": item["source_hashes_verified"],
                        "matrix_size": int(item["matrix"].shape[0]),
                        "initial_guess_mode": item["initial_guess_mode"],
                        "initial_residual_norm": float(
                            np.linalg.norm(item["initial_residual"])
                        ),
                        "num_blocks": int(len(item["plan"].blocks)),
                        "covered_rows": int(np.count_nonzero(item["plan"].covered_mask)),
                        "coverage_ratio": float(item["plan"].coverage_ratio),
                    }
                    for item in sample_payloads
                ],
            },
            handle,
            indent=2,
            default=_json_default,
        )
    print(f"checkpoint={checkpoint_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
