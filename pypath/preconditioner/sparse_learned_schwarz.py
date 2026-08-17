"""Sparse deployment for the learned Schwarz V1 preconditioner.

The training implementation keeps a dense matrix because it differentiates
through full probe residuals. Deployment does not need that matrix: it only
needs local block matrices and a sparse row-sum fallback. This module keeps
the global matrix in CSR form and retains only double-precision local LU
factors, so the deployed operator follows the same V1 feature and numerical
contract without global dense materialization.
"""

import re
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator

from pypath.preconditioner.linear_system_contract import (
    FEATURE_CONTRACT,
    LOCAL_SHIFT_CONTRACT,
    LOCAL_SHIFT_FLOOR_RELATIVE,
    compute_effective_local_shift_floor,
    require_initial_guess_mode,
    require_real_finite_vector,
    validate_learned_schwarz_checkpoint_contract,
)
from pypath.preconditioner.learned_schwarz import (
    BLOCK_FEATURE_DIM,
    DEFAULT_LEARNED_SCHWARZ_PARAMETER_MODE,
    ROW_FEATURE_DIM,
    LearnedSchwarzPreconditioner,
    LearnedSchwarzSample,
    _build_neighbor_block_features,
    _role_flags,
    _safe_log,
    require_learned_schwarz_parameter_mode,
)


PG_INSTANCE_RE = re.compile(
    r"^(?P<inst>X(?:cell_)?\d+_\d+)\s+(?P<body>.+)$",
    re.IGNORECASE,
)
_MODEL_CACHE: Dict[str, LearnedSchwarzPreconditioner] = {}


def load_learned_schwarz_v1_model(
    checkpoint_path: str,
    *,
    initial_guess_mode: Optional[str] = None,
) -> LearnedSchwarzPreconditioner:
    checkpoint = str(checkpoint_path or "").strip()
    if not checkpoint:
        raise ValueError(
            "learned_schwarz_v1_sparse requires --learned-schwarz-checkpoint"
        )
    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("learned Schwarz checkpoint must contain a mapping")
    checkpoint_initial_guess_mode = validate_learned_schwarz_checkpoint_contract(
        payload,
        expected_initial_guess_mode=initial_guess_mode,
    )
    if str(payload.get("model_kind", "learned_schwarz_v1")) != "learned_schwarz_v1":
        raise ValueError("unsupported learned Schwarz checkpoint kind")
    cache_key = (
        f"{Path(checkpoint).resolve()}::{checkpoint_initial_guess_mode}"
    )
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    model_args = dict(payload.get("model_args") or {})
    block_dim = int(model_args.get("block_feature_dim", BLOCK_FEATURE_DIM))
    row_dim = int(model_args.get("row_feature_dim", ROW_FEATURE_DIM))
    if block_dim != BLOCK_FEATURE_DIM or row_dim != ROW_FEATURE_DIM:
        raise ValueError(
            "learned Schwarz checkpoint feature dimensions do not match V1: "
            f"block={block_dim}, row={row_dim}"
        )

    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("learned Schwarz checkpoint is missing model_state_dict")
    model = LearnedSchwarzPreconditioner(**model_args)
    model.load_state_dict(state_dict)
    model = model.to(dtype=torch.float64).eval()
    _MODEL_CACHE[cache_key] = model
    return model


def _node_map_row(raw_index: Any, matrix_size: int) -> Optional[int]:
    try:
        index = int(raw_index)
    except (TypeError, ValueError):
        return None
    if 1 <= index <= int(matrix_size):
        return index - 1
    if 0 <= index < int(matrix_size):
        return index
    return None


def _rows_sorted_unique(values: List[int]) -> np.ndarray:
    if not values:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(sorted({int(value) for value in values}), dtype=np.int64)


def _aggregator_cell_pool() -> Dict[str, Dict[str, Any]]:
    try:
        from pypath.aggregator.cell_pool_utils import get_aggregator_cell_pool
    except Exception:
        return {}
    return get_aggregator_cell_pool()


def _extract_pg_blocks(
    *,
    node_map: Dict[str, int],
    netlist_path: str,
    max_block_size: int,
    min_block_size: int,
    max_blocks: int,
    matrix_size: int,
) -> Optional[Tuple[List[np.ndarray], List[Dict[str, Any]], Dict[str, Any]]]:
    path = Path(netlist_path)
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            header = handle.read(256)
    except OSError:
        return None
    is_pg_v4 = "Auto-generated PG v4 netlist" in header
    is_ngspice_pg = "Auto-generated ngspice netlist for a 100x100 power grid" in header
    if not (is_pg_v4 or is_ngspice_pg):
        return None

    instance_connections: Dict[str, List[str]] = {}
    instance_types: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            match = PG_INSTANCE_RE.match(raw.strip())
            if not match:
                continue
            body = match.group("body").split()
            if len(body) < 2:
                continue
            instance = str(match.group("inst")).lower()
            instance_connections[instance] = [item.lower() for item in body[:-1]]
            instance_types[instance] = str(body[-1]).upper()

    internal_by_instance: Dict[str, List[Tuple[int, str]]] = {
        instance: [] for instance in instance_connections
    }
    for name, raw_index in node_map.items():
        lower = str(name).lower()
        if lower.startswith("x") and "." in lower:
            instance = lower.split(".", 1)[0]
        elif lower.startswith("m.x"):
            instance = lower[2:].split(".", 1)[0]
        else:
            instance = ""
        if instance not in internal_by_instance:
            continue
        row = _node_map_row(raw_index, matrix_size)
        if row is not None:
            internal_by_instance[instance].append((int(row), str(name)))

    cell_pool = _aggregator_cell_pool()
    shared_count: Dict[str, int] = {}
    for connections in instance_connections.values():
        for connection in connections:
            if connection not in {"vdd", "vss", "gnd", "0"}:
                shared_count[connection] = shared_count.get(connection, 0) + 1

    blocks: List[np.ndarray] = []
    candidates: List[Dict[str, Any]] = []
    skipped = 0
    for instance in sorted(instance_connections):
        row_metadata: Dict[int, Tuple[str, str, str]] = {
            int(row): (str(local_name), str(local_name).lower(), "internal_node")
            for row, local_name in internal_by_instance.get(instance, [])
        }
        pins = list(cell_pool.get(instance_types.get(instance, ""), {}).get("pins", ()))
        for pin_index, connection in enumerate(instance_connections.get(instance, [])):
            if connection in {"vdd", "vss", "gnd", "0"}:
                continue
            row = _node_map_row(node_map.get(connection), matrix_size)
            if row is None:
                continue
            local_name = str(pins[pin_index]) if pin_index < len(pins) else f"pin_{pin_index}"
            role = "external_pin" if shared_count.get(connection, 0) > 1 else "unknown"
            row_metadata[int(row)] = (local_name, connection, role)

        rows = _rows_sorted_unique(list(row_metadata))
        if rows.shape[0] < int(min_block_size) or rows.shape[0] > int(max_block_size):
            skipped += 1
            continue
        blocks.append(rows)
        candidates.append(
            {
                "cell_type": instance_types.get(instance, ""),
                "instance_name": instance,
                "source_instance_id": instance,
                "row_indices": rows.tolist(),
                "row_role_by_index": {
                    str(int(row) + 1): row_metadata[int(row)][2]
                    for row in rows.tolist()
                },
            }
        )
        if max_blocks > 0 and len(blocks) >= int(max_blocks):
            break

    return blocks, candidates, {
        "semantic_mode": "cell_core_plus_onehop_boundary",
        "semantic_extractor": "pg_gen_v4_fast" if is_pg_v4 else "ngspice_pg_fast",
        "candidate_instance_count": int(len(instance_connections)),
        "block_count": int(len(blocks)),
        "skipped_block_count": int(skipped),
        "overlap_skip_count": 0,
    }


def _extract_generic_blocks(
    *,
    node_map: Dict[str, int],
    netlist_path: str,
    max_block_size: int,
    min_block_size: int,
    max_blocks: int,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
    from pypath.aggregator.preprocess_coupler import (
        build_instance_feature_cache,
        parse_netlist_for_instances,
    )
    from pypath.preconditioner.block_schwarz import _infer_row_role

    instances = parse_netlist_for_instances(netlist_path)
    cache = build_instance_feature_cache(node_map, instances)
    shared_global_counter: Dict[str, int] = {}
    for cached in cache.values():
        for global_name in (
            cached.get("external_pin_to_global_node", {}) or {}
        ).values():
            key = str(global_name).lower()
            shared_global_counter[key] = shared_global_counter.get(key, 0) + 1

    blocks: List[np.ndarray] = []
    candidates: List[Dict[str, Any]] = []
    skipped = 0
    for instance_name, cached in sorted(cache.items()):
        indices = [int(index) for index in cached.get("uut_indices", []).tolist()]
        local_names = [str(name) for name in cached.get("uut_local_node_names", ())]
        external = {
            str(local_name): str(global_name).lower()
            for local_name, global_name in (
                cached.get("external_pin_to_global_node", {}) or {}
            ).items()
        }
        selected_rows: List[int] = []
        selected_local_names: List[str] = []
        selected_global_names: List[str] = []
        shared_global_nodes: List[str] = []
        for row, local_name in zip(indices, local_names):
            global_name = str(external.get(local_name, local_name)).lower()
            is_external_pin = local_name in external
            if global_name in {"vdd", "vss", "gnd", "0"}:
                continue
            selected_rows.append(int(row))
            selected_local_names.append(local_name)
            selected_global_names.append(global_name)
            if is_external_pin and shared_global_counter.get(global_name, 0) > 1:
                shared_global_nodes.append(global_name)

        rows = _rows_sorted_unique(selected_rows)
        if rows.shape[0] < int(min_block_size) or rows.shape[0] > int(max_block_size):
            skipped += 1
            continue
        shared_global_set = set(shared_global_nodes)
        row_role_by_index = {
            str(int(row) + 1): _infer_row_role(
                local_name=local_name,
                global_name=global_name,
                shared_nodes=shared_global_set,
            )
            for row, local_name, global_name in zip(
                selected_rows,
                selected_local_names,
                selected_global_names,
            )
        }
        blocks.append(rows)
        candidates.append(
            {
                "cell_type": str(cached.get("cell_type", "")),
                "instance_name": str(instance_name),
                "source_instance_id": str(instance_name),
                "row_indices": rows.tolist(),
                "row_role_by_index": row_role_by_index,
            }
        )
        if max_blocks > 0 and len(blocks) >= int(max_blocks):
            break

    return blocks, candidates, {
        "semantic_mode": "cell_core_plus_onehop_boundary",
        "candidate_instance_count": int(len(cache)),
        "block_count": int(len(blocks)),
        "skipped_block_count": int(skipped),
        "overlap_skip_count": 0,
    }


def extract_learned_schwarz_blocks(
    *,
    node_map: Dict[str, int],
    netlist_path: str,
    max_block_size: int,
    min_block_size: int,
    max_blocks: int,
    matrix_size: int,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
    fast = _extract_pg_blocks(
        node_map=node_map,
        netlist_path=netlist_path,
        max_block_size=max_block_size,
        min_block_size=min_block_size,
        max_blocks=max_blocks,
        matrix_size=matrix_size,
    )
    if fast is not None:
        return fast
    return _extract_generic_blocks(
        node_map=node_map,
        netlist_path=netlist_path,
        max_block_size=max_block_size,
        min_block_size=min_block_size,
        max_blocks=max_blocks,
    )


def _feature_vector_or_zeros(
    values: Optional[np.ndarray],
    matrix_size: int,
    label: str,
) -> np.ndarray:
    if values is None:
        return np.zeros(int(matrix_size), dtype=np.float64)
    return require_real_finite_vector(
        values,
        matrix_size=int(matrix_size),
        label=label,
    )


def build_sparse_learned_schwarz_sample(
    *,
    matrix: sp.spmatrix,
    blocks: List[np.ndarray],
    block_candidates: Optional[List[Dict[str, Any]]] = None,
    linear_rhs: Optional[np.ndarray] = None,
    initial_residual: Optional[np.ndarray] = None,
    gmin: float = 0.0,
    retain_local_matrices: bool = False,
) -> Tuple[LearnedSchwarzSample, List[np.ndarray]]:
    """Build V1 features while keeping the full matrix sparse."""
    csr = matrix.tocsr().real.astype(np.float64, copy=False)
    matrix_size = int(csr.shape[0])
    linear_rhs_values = _feature_vector_or_zeros(
        linear_rhs, matrix_size, "linear_rhs"
    )
    initial_residual_values = _feature_vector_or_zeros(
        initial_residual, matrix_size, "initial_residual"
    )
    normalized_blocks = [np.asarray(rows, dtype=np.int64) for rows in blocks]
    coverage = np.zeros(matrix_size, dtype=np.float64)
    for rows in normalized_blocks:
        coverage[rows] += 1.0

    abs_matrix = abs(csr)
    row_abs_sum = np.asarray(abs_matrix.sum(axis=1)).reshape(-1)
    col_abs_sum = np.asarray(abs_matrix.sum(axis=0)).reshape(-1)
    diag_abs = np.abs(csr.diagonal())
    candidates = list(block_candidates or [])
    block_features: List[List[float]] = []
    row_features: List[torch.Tensor] = []
    block_tensors: List[torch.Tensor] = []
    local_matrices: List[np.ndarray] = []
    lambda_floors: List[float] = []

    for block_id, rows in enumerate(normalized_blocks):
        # matrix is already Aeff = J + gmin I, so the local floor must use it.
        local = csr[rows, :][:, rows].toarray()
        lambda_floors.append(float(compute_effective_local_shift_floor(local)))
        size = int(rows.shape[0])
        nonzero_count = int(np.count_nonzero(local))
        diagonal = np.diag(local)
        diagonal_sum = float(np.abs(diagonal).sum())
        offdiagonal_sum = float(max(np.abs(local).sum() - diagonal_sum, 0.0))
        block_features.append(
            [
                float(size),
                float(nonzero_count),
                float(nonzero_count / max(size * size, 1)),
                _safe_log(float(np.linalg.norm(local, ord="fro"))),
                _safe_log(float(np.linalg.norm(diagonal))),
                _safe_log(float(diagonal_sum / max(offdiagonal_sum, 1e-30))),
                float(gmin),
                float(np.linalg.norm(linear_rhs_values[rows])),
                float(np.linalg.norm(initial_residual_values[rows])),
            ]
        )

        candidate = candidates[block_id] if block_id < len(candidates) else {}
        role_by_index = candidate.get("row_role_by_index") or {}
        per_row: List[List[float]] = []
        for local_row, row in enumerate(rows.tolist()):
            role = str(role_by_index.get(str(int(row) + 1), "unknown"))
            row_to_block = float(np.linalg.norm(local[local_row, :]))
            block_to_row = float(np.linalg.norm(local[:, local_row]))
            per_row.append(
                [
                    *_role_flags(role),
                    _safe_log(float(diag_abs[row])),
                    _safe_log(float(row_abs_sum[row])),
                    _safe_log(float(col_abs_sum[row])),
                    _safe_log(row_to_block),
                    _safe_log(block_to_row),
                    float(coverage[row]),
                    float(abs(linear_rhs_values[row])),
                    float(abs(initial_residual_values[row])),
                    float(gmin),
                ]
            )
        if retain_local_matrices:
            local_matrices.append(local)
        block_tensors.append(torch.as_tensor(rows, dtype=torch.long))
        row_features.append(torch.as_tensor(per_row, dtype=torch.float64))

    if block_features:
        block_feature_array = np.asarray(block_features, dtype=np.float64)
    else:
        block_feature_array = np.zeros((0, BLOCK_FEATURE_DIM), dtype=np.float64)
    lambda_floor_array = np.asarray(lambda_floors, dtype=np.float64)
    neighbor_feature_array = _build_neighbor_block_features(
        block_tensors,
        block_feature_array,
        matrix_size,
    )
    fallback_scales = 1.0 / np.maximum(row_abs_sum, 1e-30)
    return (
        LearnedSchwarzSample(
            matrix=torch.empty((matrix_size, 0), dtype=torch.float64),
            blocks=block_tensors,
            block_features=torch.as_tensor(block_feature_array, dtype=torch.float64),
            neighbor_block_features=torch.as_tensor(
                neighbor_feature_array,
                dtype=torch.float64,
            ),
            row_features=row_features,
            row_coverage_count=torch.as_tensor(coverage, dtype=torch.float64),
            fallback_scales=torch.as_tensor(fallback_scales, dtype=torch.float64),
            lambda_floors=torch.as_tensor(lambda_floor_array, dtype=torch.float64),
            gmin=float(gmin),
        ),
        local_matrices,
    )


def _predict_parameters(
    model: LearnedSchwarzPreconditioner,
    sample: LearnedSchwarzSample,
    *,
    parameter_mode: object = DEFAULT_LEARNED_SCHWARZ_PARAMETER_MODE,
) -> Dict[str, Any]:
    """Reuse the shared parameter transformation for dense/sparse parity."""
    return model.predict_parameters(sample, parameter_mode=parameter_mode)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


class SparseLearnedSchwarzV1Preconditioner:
    """Sparse, fixed-parameter deployment of the learned Schwarz V1 operator."""

    def __init__(
        self,
        *,
        matrix: sp.spmatrix,
        blocks: List[np.ndarray],
        block_candidates: Optional[List[Dict[str, Any]]],
        model: LearnedSchwarzPreconditioner,
        linear_rhs: np.ndarray,
        initial_residual: np.ndarray,
        gmin: float = 0.0,
        parameter_mode: object = DEFAULT_LEARNED_SCHWARZ_PARAMETER_MODE,
    ) -> None:
        started = time.perf_counter()
        self.parameter_mode = require_learned_schwarz_parameter_mode(parameter_mode)
        self.matrix = matrix.tocsr().real.astype(np.float64, copy=False)
        self.model = model.to(dtype=torch.float64).eval()
        original_blocks = [np.asarray(rows, dtype=np.int64) for rows in blocks]
        provided_candidates = list(block_candidates or [])
        original_candidates = [
            provided_candidates[index] if index < len(provided_candidates) else {}
            for index in range(len(original_blocks))
        ]
        self.original_block_count = int(len(original_blocks))
        self.sample, _ = build_sparse_learned_schwarz_sample(
            matrix=self.matrix,
            blocks=original_blocks,
            block_candidates=original_candidates,
            linear_rhs=linear_rhs,
            initial_residual=initial_residual,
            gmin=gmin,
            retain_local_matrices=False,
        )
        self.lambda_floors = (
            self.sample.lambda_floors.detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=True)
        )
        self.local_shift_scales = (
            self.lambda_floors / float(LOCAL_SHIFT_FLOOR_RELATIVE)
        )
        if self.lambda_floors.shape[0] != self.original_block_count:
            raise ValueError(
                "learned Schwarz local shift floor count does not match blocks"
            )
        if not np.all(np.isfinite(self.lambda_floors)) or np.any(
            self.lambda_floors <= 0.0
        ):
            raise ValueError(
                "learned Schwarz local shift floors must be finite and positive"
            )

        if self.original_block_count:
            with torch.no_grad():
                parameters = _predict_parameters(
                    self.model,
                    self.sample,
                    parameter_mode=self.parameter_mode,
                )
            try:
                self.lambda_pred = (
                    parameters["lambda_pred"]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=True)
                )
                parameter_floors = (
                    parameters["lambda_floor"]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=True)
                )
                self.learned_lambdas = (
                    parameters.get("learned_lambdas", parameters["lambdas"])
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=True)
                )
                self.lambdas = (
                    parameters["lambdas"]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=True)
                )
                learned_weight_tensors = list(
                    parameters.get("learned_weights", parameters["weights"])
                )
                self.learned_weights = [
                    weight.detach().cpu().numpy().astype(np.float64, copy=True)
                    for weight in learned_weight_tensors
                ]
                self.weights = [
                    weight.detach().cpu().numpy().astype(np.float64, copy=True)
                    for weight in parameters["weights"]
                ]
                self.shift_source = str(
                    parameters.get("shift_source", "learned_effective")
                )
                self.weight_source = str(
                    parameters.get("weight_source", "learned_overlap")
                )
            except (AttributeError, KeyError, TypeError) as exc:
                raise ValueError(
                    "learned Schwarz model must return lambda_pred, "
                    "lambda_floor, learned_lambdas, lambdas, and weights"
                ) from exc
        else:
            self.lambda_pred = np.zeros(0, dtype=np.float64)
            parameter_floors = np.zeros(0, dtype=np.float64)
            self.learned_lambdas = np.zeros(0, dtype=np.float64)
            self.lambdas = np.zeros(0, dtype=np.float64)
            self.learned_weights = []
            self.weights = []
            self.shift_source = (
                "lambda_floor"
                if self.parameter_mode
                in {
                    "fixed_same_blocks",
                    "learned_overlap_weights_only",
                }
                else "learned_effective"
            )
            self.weight_source = (
                "uniform_row_coverage"
                if self.parameter_mode
                in {"fixed_same_blocks", "learned_shift_only"}
                else "learned_overlap"
            )

        expected_count = self.original_block_count
        if (
            self.lambda_pred.shape[0] != expected_count
            or parameter_floors.shape[0] != expected_count
            or self.learned_lambdas.shape[0] != expected_count
            or self.lambdas.shape[0] != expected_count
            or len(self.learned_weights) != expected_count
            or len(self.weights) != expected_count
        ):
            raise ValueError(
                "learned Schwarz predicted parameter count does not match blocks"
            )
        if not (
            np.all(np.isfinite(self.lambda_pred))
            and np.all(np.isfinite(parameter_floors))
            and np.all(np.isfinite(self.learned_lambdas))
            and np.all(np.isfinite(self.lambdas))
        ):
            raise ValueError("learned Schwarz predicted non-finite block shifts")
        if not np.array_equal(parameter_floors, self.lambda_floors):
            raise ValueError(
                "learned Schwarz model returned local shift floors that do not "
                "match the sparse sample"
            )
        if np.any(self.learned_lambdas < self.lambda_floors):
            raise ValueError(
                "learned Schwarz effective shifts must not be below local floors"
            )
        if np.any(self.lambdas < self.lambda_floors):
            raise ValueError(
                "selected learned Schwarz shifts must not be below local floors"
            )
        if any(
            np.asarray(weights, dtype=np.float64).shape != rows.shape
            for rows, weights in zip(original_blocks, self.learned_weights)
        ) or any(
            np.asarray(weights, dtype=np.float64).shape != rows.shape
            for rows, weights in zip(original_blocks, self.weights)
        ):
            raise ValueError("learned Schwarz weight lengths do not match blocks")

        self.blocks = original_blocks
        self.factors: List[Tuple[np.ndarray, np.ndarray]] = []
        total_block_nnz = 0
        max_local_bytes = 0
        for block_id, (rows, shift) in enumerate(
            zip(self.blocks, self.lambdas.tolist())
        ):
            local = self.matrix[rows, :][:, rows].toarray()
            max_local_bytes = max(max_local_bytes, int(local.nbytes))
            shifted = local + float(shift) * np.eye(
                local.shape[0],
                dtype=local.dtype,
            )
            try:
                if not np.all(np.isfinite(shifted)):
                    raise ValueError("non_finite_shifted_block")
                with warnings.catch_warnings():
                    warnings.simplefilter("error", LinAlgWarning)
                    factor = lu_factor(shifted, check_finite=False)
                if np.any(np.abs(np.diag(factor[0])) <= 1e-30):
                    raise np.linalg.LinAlgError("singular_shifted_block")
            except (ValueError, np.linalg.LinAlgError, LinAlgWarning) as exc:
                reason = str(exc) or type(exc).__name__
                raise ValueError(
                    "learned Schwarz requires every candidate block to "
                    f"factorize; block_id={block_id}, reason={reason}"
                ) from exc
            self.factors.append(factor)
            total_block_nnz += int(np.count_nonzero(local))

        if len(self.factors) != self.original_block_count:
            raise RuntimeError(
                "learned Schwarz factor count does not match candidate blocks"
            )
        self.all_candidate_blocks_factorized = True
        self.covered_mask = np.zeros(int(self.matrix.shape[0]), dtype=bool)
        for rows in self.blocks:
            self.covered_mask[rows] = True
        self.uncovered_mask = ~self.covered_mask
        self.uncovered_scales = (
            self.sample.fallback_scales.detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=True)
        )
        self.skipped_block_count = 0
        self.skipped_reasons: Dict[str, int] = {}
        self.total_block_nnz = int(total_block_nnz)
        self.max_block_size = int(
            max((rows.shape[0] for rows in self.blocks), default=0)
        )
        self.max_local_dense_bytes = int(max_local_bytes)
        self.apply_count = 0
        self.apply_time_total = 0.0
        self.setup_time = float(time.perf_counter() - started)

    def apply(self, vec: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        values = np.asarray(vec, dtype=np.float64)
        out = np.zeros_like(values)
        for rows, factor, weights in zip(self.blocks, self.factors, self.weights):
            local_solution = lu_solve(factor, values[rows], check_finite=False)
            out[rows] += weights * local_solution
        out[self.uncovered_mask] = (
            self.uncovered_scales[self.uncovered_mask]
            * values[self.uncovered_mask]
        )
        self.apply_count += 1
        self.apply_time_total += float(time.perf_counter() - started)
        return out

    def export_parameters(self) -> Dict[str, Any]:
        """Return the fixed deployment parameters needed by a native apply."""
        return {
            "blocks": [rows.astype(np.int64, copy=True) for rows in self.blocks],
            "parameter_mode": self.parameter_mode,
            "shift_source": self.shift_source,
            "weight_source": self.weight_source,
            "lambda_pred": self.lambda_pred.astype(np.float64, copy=True),
            "lambda_floors": self.lambda_floors.astype(np.float64, copy=True),
            "learned_lambdas": self.learned_lambdas.astype(
                np.float64,
                copy=True,
            ),
            "local_shift_scales": self.local_shift_scales.astype(
                np.float64,
                copy=True,
            ),
            "lambdas": self.lambdas.astype(np.float64, copy=True),
            "learned_weights": [
                weight.astype(np.float64, copy=True)
                for weight in self.learned_weights
            ],
            "weights": [
                weight.astype(np.float64, copy=True)
                for weight in self.weights
            ],
            "uncovered_scales": self.uncovered_scales.astype(np.float64, copy=True),
            "metadata": self.metadata(),
        }

    def _retained_bytes(self) -> int:
        matrix_bytes = int(
            self.matrix.data.nbytes
            + self.matrix.indices.nbytes
            + self.matrix.indptr.nbytes
        )
        factor_bytes = int(
            sum(factor[0].nbytes + factor[1].nbytes for factor in self.factors)
        )
        feature_bytes = (
            _tensor_bytes(self.sample.block_features)
            + _tensor_bytes(self.sample.neighbor_block_features)
            + _tensor_bytes(self.sample.row_coverage_count)
            + _tensor_bytes(self.sample.fallback_scales)
            + _tensor_bytes(self.sample.lambda_floors)
            + sum(_tensor_bytes(rows) for rows in self.sample.blocks)
            + sum(_tensor_bytes(features) for features in self.sample.row_features)
        )
        array_bytes = int(
            self.lambda_pred.nbytes
            + self.lambda_floors.nbytes
            + self.local_shift_scales.nbytes
            + self.learned_lambdas.nbytes
            + self.lambdas.nbytes
            + self.uncovered_scales.nbytes
            + self.covered_mask.nbytes
            + self.uncovered_mask.nbytes
            + sum(rows.nbytes for rows in self.blocks)
            + sum(weights.nbytes for weights in self.learned_weights)
            + sum(weights.nbytes for weights in self.weights)
        )
        return int(matrix_bytes + factor_bytes + feature_bytes + array_bytes)

    def metadata(self) -> Dict[str, Any]:
        all_weights = (
            np.concatenate(self.weights) if self.weights else np.zeros(0, dtype=np.float64)
        )
        all_learned_weights = (
            np.concatenate(self.learned_weights)
            if self.learned_weights
            else np.zeros(0, dtype=np.float64)
        )
        matrix_size = int(self.matrix.shape[0])
        dense_global_bytes = int(matrix_size * matrix_size * np.dtype(np.float64).itemsize)
        retained_bytes = self._retained_bytes()
        peak_estimate_bytes = int(retained_bytes + 2 * self.max_local_dense_bytes)
        saved_bytes = int(max(dense_global_bytes - peak_estimate_bytes, 0))
        saving_ratio = (
            float(saved_bytes / dense_global_bytes) if dense_global_bytes else 0.0
        )
        floor_active_count = int(
            np.count_nonzero(self.lambdas == self.lambda_floors)
        )
        shift_metadata_bytes = int(
            self.lambda_pred.nbytes
            + self.lambda_floors.nbytes
            + self.local_shift_scales.nbytes
            + self.learned_lambdas.nbytes
            + self.lambdas.nbytes
            + sum(weights.nbytes for weights in self.learned_weights)
            + sum(weights.nbytes for weights in self.weights)
        )
        return {
            "preconditioner_mode": "learned_schwarz_v1_sparse",
            "implementation": "csr_global_dense_local_lu",
            "parameter_mode": self.parameter_mode,
            "shift_source": self.shift_source,
            "weight_source": self.weight_source,
            "no_global_dense_materialization": True,
            "local_shift_contract": LOCAL_SHIFT_CONTRACT,
            "local_shift_floor_relative": float(LOCAL_SHIFT_FLOOR_RELATIVE),
            "all_candidate_blocks_factorized": bool(
                self.all_candidate_blocks_factorized
            ),
            "block_mode": "cell_core_plus_onehop_boundary",
            "candidate_block_count": int(self.original_block_count),
            "block_count": int(len(self.blocks)),
            "skipped_block_count": int(self.skipped_block_count),
            "skipped_block_reasons": dict(self.skipped_reasons),
            "covered_rows": int(np.count_nonzero(self.covered_mask)),
            "uncovered_rows": int(np.count_nonzero(self.uncovered_mask)),
            "coverage_ratio": float(
                np.count_nonzero(self.covered_mask) / max(matrix_size, 1)
            ),
            "max_block_size": int(self.max_block_size),
            "total_block_nnz": int(self.total_block_nnz),
            "factor_modes": {"dense_lu": int(len(self.factors))},
            "block_local_shift_scales": self.local_shift_scales.tolist(),
            "block_lambda_pred": self.lambda_pred.tolist(),
            "block_lambda_floor": self.lambda_floors.tolist(),
            "block_lambda_learned_effective": self.learned_lambdas.tolist(),
            "block_lambda_effective": self.lambdas.tolist(),
            "lambda_pred_min": float(np.min(self.lambda_pred))
            if self.lambda_pred.size
            else None,
            "lambda_pred_max": float(np.max(self.lambda_pred))
            if self.lambda_pred.size
            else None,
            "lambda_pred_mean": float(np.mean(self.lambda_pred))
            if self.lambda_pred.size
            else None,
            "lambda_floor_min": float(np.min(self.lambda_floors))
            if self.lambda_floors.size
            else None,
            "lambda_floor_max": float(np.max(self.lambda_floors))
            if self.lambda_floors.size
            else None,
            "lambda_floor_mean": float(np.mean(self.lambda_floors))
            if self.lambda_floors.size
            else None,
            "lambda_floor_active_block_count": floor_active_count,
            "lambda_floor_active_ratio": float(
                floor_active_count / max(self.original_block_count, 1)
            ),
            "lambda_eff_min": float(np.min(self.lambdas))
            if self.lambdas.size
            else None,
            "lambda_eff_max": float(np.max(self.lambdas))
            if self.lambdas.size
            else None,
            "lambda_eff_mean": float(np.mean(self.lambdas))
            if self.lambdas.size
            else None,
            "lambda_learned_eff_min": float(np.min(self.learned_lambdas))
            if self.learned_lambdas.size
            else None,
            "lambda_learned_eff_max": float(np.max(self.learned_lambdas))
            if self.learned_lambdas.size
            else None,
            "lambda_learned_eff_mean": float(np.mean(self.learned_lambdas))
            if self.learned_lambdas.size
            else None,
            "lambda_min": float(np.min(self.lambdas)) if self.lambdas.size else None,
            "lambda_max": float(np.max(self.lambdas)) if self.lambdas.size else None,
            "lambda_mean": float(np.mean(self.lambdas)) if self.lambdas.size else None,
            "learned_weight_min": float(np.min(all_learned_weights))
            if all_learned_weights.size
            else None,
            "learned_weight_max": float(np.max(all_learned_weights))
            if all_learned_weights.size
            else None,
            "learned_weight_mean": float(np.mean(all_learned_weights))
            if all_learned_weights.size
            else None,
            "weight_min": float(np.min(all_weights)) if all_weights.size else None,
            "weight_max": float(np.max(all_weights)) if all_weights.size else None,
            "weight_mean": float(np.mean(all_weights)) if all_weights.size else None,
            "setup_time": float(self.setup_time),
            "apply_count": int(self.apply_count),
            "apply_time_total": float(self.apply_time_total),
            "apply_time_per_call": float(
                self.apply_time_total / max(self.apply_count, 1)
            ),
            "dense_v1_global_matrix_bytes": dense_global_bytes,
            "local_shift_metadata_estimated_bytes": shift_metadata_bytes,
            "sparse_retained_estimated_bytes": retained_bytes,
            "sparse_peak_estimated_bytes": peak_estimate_bytes,
            "estimated_memory_saved_bytes": saved_bytes,
            "estimated_peak_memory_saving_ratio": saving_ratio,
            "memory_saving_target_over_50pct": bool(saving_ratio > 0.50),
        }


def build_sparse_learned_schwarz_v1_preconditioner(
    *,
    matrix: sp.spmatrix,
    node_map: Dict[str, int],
    netlist_path: str,
    checkpoint_path: str,
    linear_rhs: np.ndarray,
    initial_residual: np.ndarray,
    initial_guess_mode: str,
    gmin: float,
    max_block_size: int,
    min_block_size: int,
    max_blocks: int,
    parameter_mode: object = DEFAULT_LEARNED_SCHWARZ_PARAMETER_MODE,
) -> Tuple[LinearOperator, Dict[str, Any]]:
    blocks, candidates, block_debug = extract_learned_schwarz_blocks(
        node_map=node_map,
        netlist_path=netlist_path,
        max_block_size=max_block_size,
        min_block_size=min_block_size,
        max_blocks=max_blocks,
        matrix_size=int(matrix.shape[0]),
    )
    resolved_initial_guess_mode = require_initial_guess_mode(
        initial_guess_mode
    )
    model = load_learned_schwarz_v1_model(
        checkpoint_path,
        initial_guess_mode=resolved_initial_guess_mode,
    )
    preconditioner = SparseLearnedSchwarzV1Preconditioner(
        matrix=matrix,
        blocks=blocks,
        block_candidates=candidates,
        model=model,
        linear_rhs=linear_rhs,
        initial_residual=initial_residual,
        gmin=gmin,
        parameter_mode=parameter_mode,
    )
    info: Dict[str, Any] = {
        "mode": "learned_schwarz_v1_sparse",
        "fallback_reason": None,
        "netlist_path": str(netlist_path),
        "learned_schwarz_checkpoint": str(checkpoint_path),
        "feature_contract": FEATURE_CONTRACT,
        "initial_guess_mode": resolved_initial_guess_mode,
        "learned_schwarz_parameter_mode": preconditioner.parameter_mode,
        "block_debug": block_debug,
        "core": preconditioner.metadata(),
    }
    operator = LinearOperator(matrix.shape, matvec=preconditioner.apply, dtype=matrix.dtype)
    setattr(
        operator,
        "pals_preconditioner_info",
        lambda: {**info, "core": preconditioner.metadata()},
    )
    return operator, info


__all__ = [
    "SparseLearnedSchwarzV1Preconditioner",
    "build_sparse_learned_schwarz_sample",
    "build_sparse_learned_schwarz_v1_preconditioner",
    "extract_learned_schwarz_blocks",
    "load_learned_schwarz_v1_model",
]
