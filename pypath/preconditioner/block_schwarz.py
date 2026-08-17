from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pypath.aggregator.preconditioner_targets import infer_row_kind


@dataclass(frozen=True)
class BlockPlanConfig:
    block_mode: str = "branch_incident"
    max_block_size: int = 32
    min_block_size: int = 2
    max_blocks: int = 0
    max_total_block_nnz: int = 0
    uncovered_row_policy: str = "row_sum"


@dataclass
class BlockSchwarzPlan:
    block_mode: str
    blocks: List[np.ndarray]
    block_candidates: List[Dict[str, Any]]
    covered_mask: np.ndarray
    uncovered_scales: np.ndarray
    factor_solvers: List[Any]
    factor_modes: List[str]
    total_block_nnz: int
    max_block_size: int
    candidate_block_count: int
    skipped_block_count: int
    coverage_ratio: float

    def apply(self, vec: np.ndarray) -> np.ndarray:
        out = np.asarray(vec, dtype=np.float64).copy()
        out[~self.covered_mask] = self.uncovered_scales[~self.covered_mask] * out[~self.covered_mask]
        for rows, factor in zip(self.blocks, self.factor_solvers):
            out[rows] = factor.dot(vec[rows])
        return out

    def metadata(self) -> Dict[str, Any]:
        memory_estimate = int(self.total_block_nnz * 8 + sum(int(rows.shape[0]) ** 2 * 8 for rows in self.blocks))
        return {
            "preconditioner_mode": "block_jacobi",
            "block_mode": self.block_mode,
            "semantic_block_mode": self.block_mode,
            "num_blocks": int(len(self.blocks)),
            "covered_rows": int(np.count_nonzero(self.covered_mask)),
            "coverage_ratio": float(self.coverage_ratio),
            "uncovered_rows": int(self.covered_mask.shape[0] - np.count_nonzero(self.covered_mask)),
            "max_block_size": int(self.max_block_size),
            "total_block_nnz": int(self.total_block_nnz),
            "memory_estimate": memory_estimate,
            "candidate_block_count": int(self.candidate_block_count),
            "skipped_block_count": int(self.skipped_block_count),
            "factor_modes": list(self.factor_modes),
            "blocks": [
                {
                    "block_id": int(block_id),
                    "rows": [int(row) + 1 for row in rows.tolist()],
                    "size": int(rows.shape[0]),
                }
                for block_id, rows in enumerate(self.blocks)
            ],
        }


def build_block_schwarz_plan(
    *,
    matrix: np.ndarray,
    node_map: Dict[str, int],
    config: BlockPlanConfig,
    netlist_path: str,
) -> BlockSchwarzPlan:
    if config.block_mode == "branch_incident":
        blocks, block_candidates, debug = _build_branch_incident_blocks(
            matrix=matrix,
            node_map=node_map,
            max_block_size=int(config.max_block_size),
            min_block_size=int(config.min_block_size),
            max_blocks=int(config.max_blocks),
            max_total_block_nnz=int(config.max_total_block_nnz),
        )
    elif config.block_mode in {"cell_instance", "cell_core", "cell_full", "cell_core_plus_onehop_boundary"}:
        blocks, block_candidates, debug = _build_cell_instance_blocks(
            matrix=matrix,
            node_map=node_map,
            netlist_path=netlist_path,
            max_block_size=int(config.max_block_size),
            min_block_size=int(config.min_block_size),
            max_blocks=int(config.max_blocks),
            max_total_block_nnz=int(config.max_total_block_nnz),
            semantic_mode=("cell_core" if config.block_mode == "cell_instance" else str(config.block_mode)),
        )
    elif config.block_mode == "generic":
        blocks, block_candidates, debug = _build_generic_blocks(
            matrix=matrix,
            max_block_size=int(config.max_block_size),
            min_block_size=int(config.min_block_size),
            max_blocks=int(config.max_blocks),
            max_total_block_nnz=int(config.max_total_block_nnz),
        )
    else:
        raise ValueError(f"Unsupported block_mode: {config.block_mode}")

    covered_mask = np.zeros(int(matrix.shape[0]), dtype=bool)
    factor_solvers: List[Any] = []
    factor_modes: List[str] = []
    max_block_size_seen = 0
    total_block_nnz = int(debug["total_block_nnz"])
    for rows in blocks:
        covered_mask[rows] = True
        block_matrix = np.asarray(matrix[np.ix_(rows, rows)], dtype=np.float64)
        factor, factor_mode = _factorize_block(block_matrix)
        factor_solvers.append(factor)
        factor_modes.append(factor_mode)
        max_block_size_seen = max(max_block_size_seen, int(rows.shape[0]))

    coverage_ratio = float(np.count_nonzero(covered_mask) / max(int(matrix.shape[0]), 1))
    return BlockSchwarzPlan(
        block_mode=str(config.block_mode),
        blocks=blocks,
        block_candidates=block_candidates,
        covered_mask=covered_mask,
        uncovered_scales=np.asarray(build_analytic_scales(config.uncovered_row_policy, matrix), dtype=np.float64),
        factor_solvers=factor_solvers,
        factor_modes=factor_modes,
        total_block_nnz=total_block_nnz,
        max_block_size=int(max_block_size_seen),
        candidate_block_count=len(blocks),
        skipped_block_count=int(debug["skipped_block_count"]),
        coverage_ratio=coverage_ratio,
    )


def build_analytic_scales(mode: str, matrix: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    if mode == "identity":
        return np.ones(matrix.shape[0], dtype=np.float64)

    row_abs_sum = np.abs(matrix).sum(axis=1)
    diag_abs = np.abs(np.diag(matrix))
    if mode == "jacobi":
        return 1.0 / np.maximum(diag_abs, eps)
    if mode == "row_sum":
        return 1.0 / np.maximum(row_abs_sum, eps)
    raise ValueError(f"Unsupported analytic preconditioner mode: {mode}")


def _idx_to_name_map(node_map: Dict[str, int]) -> Dict[int, str]:
    return {int(idx): str(name).lower() for name, idx in node_map.items()}


def _rows_sorted_unique(rows: List[int]) -> np.ndarray:
    return np.asarray(sorted({int(row) for row in rows if int(row) >= 0}), dtype=np.int64)


def _candidate_mode_to_source(block_mode: str) -> str:
    if block_mode == "branch_incident":
        return "branch_seed_incident_rows"
    if block_mode in {"cell_instance", "cell_core", "cell_full", "cell_core_plus_onehop_boundary"}:
        return "instance_cell_rows"
    if block_mode == "generic":
        return "generic_row_chunks"
    return "unknown"


def _make_block_candidate_record(
    *,
    candidate_mode: str,
    candidate_source: str,
    rows: np.ndarray,
    matrix: np.ndarray,
    candidate_index: int,
    seed_row: Optional[int] = None,
    seed_row_kind: Optional[str] = None,
    source_instance_id: Optional[str] = None,
    cell_type: Optional[str] = None,
    semantic_block_type: Optional[str] = None,
    source_object_type: Optional[str] = None,
    instance_name: Optional[str] = None,
    local_node_names: Optional[List[str]] = None,
    global_node_names: Optional[List[str]] = None,
    boundary_pin_names: Optional[List[str]] = None,
    internal_node_names: Optional[List[str]] = None,
    shared_global_nodes: Optional[List[str]] = None,
    row_role_by_index: Optional[Dict[str, str]] = None,
    semantic_validity: str = "valid",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row_list = [int(row) for row in rows.tolist()]
    return {
        "schema_version": 1,
        "candidate_id": f"{candidate_mode}:block_{int(candidate_index):04d}",
        "candidate_mode": str(candidate_mode),
        "candidate_source": str(candidate_source),
        "row_indices": row_list,
        "row_index_base": 0,
        "row_count": int(len(row_list)),
        "block_nnz": int(np.count_nonzero(matrix[np.ix_(rows, rows)])),
        "seed_row": None if seed_row is None else int(seed_row),
        "seed_row_kind": seed_row_kind,
        "source_instance_id": source_instance_id,
        "cell_type": cell_type,
        "semantic_block_type": semantic_block_type,
        "source_object_type": source_object_type,
        "instance_name": instance_name or source_instance_id,
        "local_node_names": list(local_node_names or []),
        "global_node_names": list(global_node_names or []),
        "boundary_pin_names": list(boundary_pin_names or []),
        "internal_node_names": list(internal_node_names or []),
        "shared_global_nodes": list(shared_global_nodes or []),
        "row_role_by_index": dict(row_role_by_index or {}),
        "semantic_validity": str(semantic_validity),
        "metadata": dict(metadata or {}),
    }


def _infer_row_role(*, local_name: str, global_name: str, shared_nodes: set) -> str:
    local_name_str = str(local_name)
    global_name_str = str(global_name).lower()
    lowered_local = local_name_str.lower()
    if infer_row_kind(global_name_str) == "branch":
        return "branch_current"
    if global_name_str in {"vdd", "vss", "gnd", "0"}:
        return "supply"
    if lowered_local == global_name_str or "." in lowered_local or "#" in lowered_local:
        return "internal_node"
    if global_name_str in shared_nodes:
        return "external_pin"
    return "unknown"


def _build_branch_incident_blocks(
    *,
    matrix: np.ndarray,
    node_map: Dict[str, int],
    max_block_size: int,
    min_block_size: int,
    max_blocks: int,
    max_total_block_nnz: int,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
    idx_to_name = _idx_to_name_map(node_map)
    matrix_size = int(matrix.shape[0])
    branch_zero_rows = [
        zero_idx
        for zero_idx in range(matrix_size)
        if infer_row_kind(idx_to_name.get(zero_idx + 1, "")) == "branch"
    ]
    used_rows = set()
    blocks: List[np.ndarray] = []
    block_candidates: List[Dict[str, Any]] = []
    total_block_nnz = 0
    skipped_block_count = 0
    for branch_zero_idx in branch_zero_rows:
        if branch_zero_idx in used_rows:
            continue
        row_abs = np.abs(matrix[branch_zero_idx])
        candidate_neighbors = [
            int(idx)
            for idx in np.argsort(-row_abs)
            if int(idx) != branch_zero_idx and row_abs[int(idx)] > 0.0
        ]
        chosen_rows = [int(branch_zero_idx)]
        for neighbor_zero_idx in candidate_neighbors:
            neighbor_name = idx_to_name.get(neighbor_zero_idx + 1, "")
            if infer_row_kind(neighbor_name) == "branch" or neighbor_zero_idx in used_rows:
                continue
            chosen_rows.append(int(neighbor_zero_idx))
            if len(chosen_rows) >= int(max_block_size):
                break
        rows = _rows_sorted_unique(chosen_rows)
        if rows.shape[0] < int(min_block_size):
            skipped_block_count += 1
            continue
        block_nnz = int(np.count_nonzero(matrix[np.ix_(rows, rows)]))
        if max_total_block_nnz > 0 and total_block_nnz + block_nnz > int(max_total_block_nnz):
            skipped_block_count += 1
            continue
        blocks.append(rows)
        block_candidates.append(
            _make_block_candidate_record(
                candidate_mode="branch_incident_block",
                candidate_source=_candidate_mode_to_source("branch_incident"),
                rows=rows,
                matrix=matrix,
                candidate_index=len(block_candidates),
                seed_row=int(branch_zero_idx),
                seed_row_kind="branch",
                metadata={"block_mode": "branch_incident", "neighbor_row_count": int(max(rows.shape[0] - 1, 0))},
            )
        )
        total_block_nnz += block_nnz
        used_rows.update(int(row) for row in rows.tolist())
        if max_blocks > 0 and len(blocks) >= int(max_blocks):
            break
    return blocks, block_candidates, {
        "candidate_branch_rows": len(branch_zero_rows),
        "total_block_nnz": int(total_block_nnz),
        "skipped_block_count": int(skipped_block_count),
    }


def _build_generic_blocks(
    *,
    matrix: np.ndarray,
    max_block_size: int,
    min_block_size: int,
    max_blocks: int,
    max_total_block_nnz: int,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
    matrix_size = int(matrix.shape[0])
    blocks: List[np.ndarray] = []
    block_candidates: List[Dict[str, Any]] = []
    total_block_nnz = 0
    skipped_block_count = 0
    cursor = 0
    while cursor < matrix_size:
        end = min(cursor + int(max_block_size), matrix_size)
        rows = np.arange(cursor, end, dtype=np.int64)
        cursor = end
        if rows.shape[0] < int(min_block_size):
            skipped_block_count += 1
            continue
        block_nnz = int(np.count_nonzero(matrix[np.ix_(rows, rows)]))
        if max_total_block_nnz > 0 and total_block_nnz + block_nnz > int(max_total_block_nnz):
            skipped_block_count += 1
            continue
        blocks.append(rows)
        block_candidates.append(
            _make_block_candidate_record(
                candidate_mode="generic_block_jacobi",
                candidate_source=_candidate_mode_to_source("generic"),
                rows=rows,
                matrix=matrix,
                candidate_index=len(block_candidates),
                semantic_block_type="generic_row_chunk",
                source_object_type="matrix_rows",
                metadata={"block_mode": "generic", "row_start": int(rows[0]), "row_end": int(rows[-1])},
            )
        )
        total_block_nnz += block_nnz
        if max_blocks > 0 and len(blocks) >= int(max_blocks):
            break
    return blocks, block_candidates, {
        "candidate_row_count": matrix_size,
        "total_block_nnz": int(total_block_nnz),
        "skipped_block_count": int(skipped_block_count),
        "semantic_mode": "generic",
    }


def _build_cell_instance_blocks(
    *,
    matrix: np.ndarray,
    node_map: Dict[str, int],
    netlist_path: str,
    max_block_size: int,
    min_block_size: int,
    max_blocks: int,
    max_total_block_nnz: int,
    semantic_mode: str = "cell_core",
) -> Tuple[List[np.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
    from pypath.aggregator.preprocess_coupler import build_instance_feature_cache, parse_netlist_for_instances

    instances = parse_netlist_for_instances(netlist_path)
    cache = build_instance_feature_cache(node_map, instances)
    used_rows = set()
    blocks: List[np.ndarray] = []
    block_candidates: List[Dict[str, Any]] = []
    total_block_nnz = 0
    skipped_block_count = 0
    overlap_skip_count = 0
    shared_global_counter: Dict[str, int] = {}
    for cached in cache.values():
        for global_name in (cached.get("external_pin_to_global_node", {}) or {}).values():
            key = str(global_name).lower()
            shared_global_counter[key] = shared_global_counter.get(key, 0) + 1

    for instance_name, cached in sorted(cache.items()):
        uut_indices = [int(idx) for idx in cached.get("uut_indices", []).tolist()]
        local_names_all = [str(name) for name in cached.get("uut_local_node_names", ())]
        external_map = {
            str(local_name): str(global_name).lower()
            for local_name, global_name in (cached.get("external_pin_to_global_node", {}) or {}).items()
        }
        selected_rows: List[int] = []
        selected_local_names: List[str] = []
        selected_global_names: List[str] = []
        boundary_pin_names: List[str] = []
        internal_node_names: List[str] = []
        shared_global_nodes: List[str] = []
        row_role_by_index: Dict[str, str] = {}
        for row_idx, local_name in zip(uut_indices, local_names_all):
            global_name = str(external_map.get(local_name, local_name)).lower()
            is_external_pin = local_name in external_map
            is_supply = global_name in {"vdd", "vss", "gnd", "0"}
            if semantic_mode == "cell_core" and is_external_pin:
                continue
            if semantic_mode == "cell_core_plus_onehop_boundary" and is_supply:
                continue
            selected_rows.append(int(row_idx))
            selected_local_names.append(str(local_name))
            selected_global_names.append(global_name)
            if is_external_pin:
                boundary_pin_names.append(str(local_name))
                if shared_global_counter.get(global_name, 0) > 1:
                    shared_global_nodes.append(global_name)
            else:
                internal_node_names.append(str(local_name))
        rows = _rows_sorted_unique(selected_rows)
        if rows.shape[0] < int(min_block_size) or rows.shape[0] > int(max_block_size):
            skipped_block_count += 1
            continue
        allow_overlap = semantic_mode == "cell_core_plus_onehop_boundary"
        if (not allow_overlap) and any(int(row) in used_rows for row in rows.tolist()):
            overlap_skip_count += 1
            continue
        shared_global_set = set(shared_global_nodes)
        for row_idx, local_name, global_name in zip(selected_rows, selected_local_names, selected_global_names):
            row_role_by_index[str(int(row_idx) + 1)] = _infer_row_role(
                local_name=local_name,
                global_name=global_name,
                shared_nodes=shared_global_set,
            )
        block_nnz = int(np.count_nonzero(matrix[np.ix_(rows, rows)]))
        if max_total_block_nnz > 0 and total_block_nnz + block_nnz > int(max_total_block_nnz):
            skipped_block_count += 1
            continue
        blocks.append(rows)
        block_candidates.append(
            _make_block_candidate_record(
                candidate_mode="cell_block_jacobi",
                candidate_source=_candidate_mode_to_source(semantic_mode),
                rows=rows,
                matrix=matrix,
                candidate_index=len(block_candidates),
                source_instance_id=str(instance_name),
                cell_type=str(cached.get("cell_type", "")) or None,
                semantic_block_type=(
                    "cell_instance_stamp"
                    if semantic_mode == "cell_full"
                    else "cell_core_plus_onehop_boundary"
                    if semantic_mode == "cell_core_plus_onehop_boundary"
                    else "cell_core_block"
                ),
                source_object_type="standard_cell_instance",
                instance_name=str(instance_name),
                local_node_names=selected_local_names,
                global_node_names=selected_global_names,
                boundary_pin_names=boundary_pin_names,
                internal_node_names=internal_node_names,
                shared_global_nodes=sorted(set(shared_global_nodes)),
                row_role_by_index=row_role_by_index,
                metadata={
                    "block_mode": str(semantic_mode),
                    "instance_name": str(instance_name),
                    "valid_local_names": selected_local_names,
                    "global_nodes": selected_global_names,
                    "original_local_node_count": len(local_names_all),
                    "selected_local_node_count": len(selected_local_names),
                },
            )
        )
        total_block_nnz += block_nnz
        used_rows.update(int(row) for row in rows.tolist())
        if max_blocks > 0 and len(blocks) >= int(max_blocks):
            break
    return blocks, block_candidates, {
        "candidate_instance_count": len(cache),
        "total_block_nnz": int(total_block_nnz),
        "skipped_block_count": int(skipped_block_count),
        "overlap_skip_count": int(overlap_skip_count),
        "semantic_mode": str(semantic_mode),
    }


def _factorize_block(block_matrix: np.ndarray) -> Tuple[Any, str]:
    try:
        return np.linalg.inv(block_matrix), "dense_inv"
    except np.linalg.LinAlgError:
        return np.linalg.pinv(block_matrix), "dense_pinv"
