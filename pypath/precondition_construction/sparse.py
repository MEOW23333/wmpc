"""Sparse preconditioner construction for PALS linear-system benchmarks.

This module owns preconditioner construction. Solver runners should call the
small factory functions here instead of embedding block discovery, Schur
assembly, or direct-factor setup in experiment scripts.
"""

import argparse
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, spilu, splu
from pypath.preconditioner.sparse_learned_schwarz import (
    build_sparse_learned_schwarz_v1_preconditioner,
)

SPARSE_SEMANTIC_MODES = {
    'learned_schwarz_v1_sparse',
    'semantic_cell_core_sparse',
    'semantic_coarse_r1_sparse',
    'semantic_coarse_r2_sparse',
    'local_sparse_schur_sparse',
    'learned_local_sparse_schur_sparse',
    'learned_sparse_schur_safe_add_sparse',
    'learned_sparse_schur_safe_add_probe_sparse',
}

def semantic_netlist_path(args: argparse.Namespace) -> str:
    raw = str(getattr(args, 'netlist_path', '') or '').strip()
    if raw:
        return raw
    trajectory_dir = Path(str(getattr(args, 'trajectory_dir', '') or ''))
    root = trajectory_dir.parent if trajectory_dir.name == 'trajectory' else trajectory_dir
    candidates = [
        root / 'generated_netlists' / f"{int(getattr(args, 'circuit_id', 0))}.sp",
        root / 'generated_pairs' / 'ngspice' / f"power_grid_100x100_{int(getattr(args, 'circuit_id', 0)):02d}.sp",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError('semantic sparse modes require --netlist-path or a recognized generated_netlists path')


def _rows_sorted_unique(values: List[int]) -> np.ndarray:
    if not values:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(sorted(set(int(v) for v in values)), dtype=np.int64)


PG_INSTANCE_RE = re.compile(r'^(?P<inst>X(?:cell_)?\d+_\d+)\s+(?P<body>.+)$', re.IGNORECASE)


def _node_map_row(raw_idx: Any, matrix_size: int) -> Optional[int]:
    try:
        idx = int(raw_idx)
    except (TypeError, ValueError):
        return None
    if 1 <= idx <= int(matrix_size):
        return idx - 1
    if 0 <= idx < int(matrix_size):
        return idx
    return None


def _extract_pg_gen_v4_blocks_fast(
    *,
    node_map: Dict[str, int],
    netlist_path: str,
    semantic_mode: str,
    max_block_size: int,
    min_block_size: int,
    max_blocks: int,
    matrix_size: int,
) -> Optional[Tuple[List[np.ndarray], Dict[str, Any]]]:
    path = Path(netlist_path)
    try:
        head = path.read_text(encoding='utf-8', errors='ignore')[:256]
    except OSError:
        return None
    is_pg_v4 = 'Auto-generated PG v4 netlist' in head
    is_ngspice_pg = 'Auto-generated ngspice netlist for a 100x100 power grid' in head
    if not (is_pg_v4 or is_ngspice_pg):
        return None

    instance_connections: Dict[str, List[str]] = {}
    with path.open('r', encoding='utf-8', errors='ignore') as handle:
        for raw in handle:
            match = PG_INSTANCE_RE.match(raw.strip())
            if not match:
                continue
            body = match.group('body').split()
            if len(body) < 2:
                continue
            inst = str(match.group('inst')).lower()
            instance_connections[inst] = [item.lower() for item in body[:-1]]

    core_by_inst: Dict[str, List[int]] = {inst: [] for inst in instance_connections}
    for name, raw_idx in node_map.items():
        lower = str(name).lower()
        inst = ''
        if lower.startswith('x') and '.' in lower:
            inst = lower.split('.', 1)[0]
        elif lower.startswith('m.x'):
            rest = lower[2:]
            inst = rest.split('.', 1)[0]
        if inst in core_by_inst:
            row = _node_map_row(raw_idx, matrix_size)
            if row is not None:
                core_by_inst[inst].append(int(row))

    blocks: List[np.ndarray] = []
    skipped_block_count = 0
    for inst in sorted(instance_connections):
        rows = list(core_by_inst.get(inst, []))
        if semantic_mode == 'cell_core_plus_onehop_boundary':
            for conn in instance_connections.get(inst, []):
                if conn in {'vdd', 'vss', 'gnd', '0'}:
                    continue
                idx = node_map.get(conn)
                row = _node_map_row(idx, matrix_size)
                if row is not None:
                    rows.append(int(row))
        rows_np = _rows_sorted_unique(rows)
        if rows_np.shape[0] < int(min_block_size) or rows_np.shape[0] > int(max_block_size):
            skipped_block_count += 1
            continue
        blocks.append(rows_np)
        if max_blocks > 0 and len(blocks) >= int(max_blocks):
            break
    return blocks, {
        'semantic_mode': semantic_mode,
        'semantic_extractor': 'pg_gen_v4_fast' if is_pg_v4 else 'ngspice_pg_fast',
        'candidate_instance_count': len(instance_connections),
        'block_count': len(blocks),
        'skipped_block_count': int(skipped_block_count),
        'overlap_skip_count': 0,
    }


def _extract_semantic_blocks_sparse(
    *,
    matrix: sp.spmatrix,
    node_map: Dict[str, int],
    netlist_path: str,
    semantic_mode: str,
    max_block_size: int,
    min_block_size: int,
    max_blocks: int,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    fast = _extract_pg_gen_v4_blocks_fast(
        node_map=node_map,
        netlist_path=netlist_path,
        semantic_mode=semantic_mode,
        max_block_size=max_block_size,
        min_block_size=min_block_size,
        max_blocks=max_blocks,
        matrix_size=int(matrix.shape[0]),
    )
    if fast is not None:
        return fast

    from pypath.aggregator.preprocess_coupler import build_instance_feature_cache, parse_netlist_for_instances

    instances = parse_netlist_for_instances(netlist_path)
    cache = build_instance_feature_cache(node_map, instances)
    blocks: List[np.ndarray] = []
    used_rows = set()
    skipped_block_count = 0
    overlap_skip_count = 0
    shared_global_counter: Dict[str, int] = {}
    for cached in cache.values():
        for global_name in (cached.get('external_pin_to_global_node', {}) or {}).values():
            key = str(global_name).lower()
            shared_global_counter[key] = shared_global_counter.get(key, 0) + 1

    for _, cached in sorted(cache.items()):
        uut_indices = [int(idx) for idx in cached.get('uut_indices', []).tolist()]
        local_names_all = [str(name) for name in cached.get('uut_local_node_names', ())]
        external_map = {
            str(local_name): str(global_name).lower()
            for local_name, global_name in (cached.get('external_pin_to_global_node', {}) or {}).items()
        }
        selected_rows: List[int] = []
        for row_idx, local_name in zip(uut_indices, local_names_all):
            global_name = str(external_map.get(local_name, local_name)).lower()
            is_external_pin = local_name in external_map
            is_supply = global_name in {'vdd', 'vss', 'gnd', '0'}
            if semantic_mode == 'cell_core' and is_external_pin:
                continue
            if semantic_mode == 'cell_core_plus_onehop_boundary' and is_supply:
                continue
            selected_rows.append(int(row_idx))
        rows = _rows_sorted_unique(selected_rows)
        if rows.shape[0] < int(min_block_size) or rows.shape[0] > int(max_block_size):
            skipped_block_count += 1
            continue
        allow_overlap = semantic_mode == 'cell_core_plus_onehop_boundary'
        if (not allow_overlap) and any(int(row) in used_rows for row in rows.tolist()):
            overlap_skip_count += 1
            continue
        blocks.append(rows)
        if not allow_overlap:
            used_rows.update(int(row) for row in rows.tolist())
        if max_blocks > 0 and len(blocks) >= int(max_blocks):
            break
    return blocks, {
        'semantic_mode': semantic_mode,
        'candidate_instance_count': len(cache),
        'block_count': len(blocks),
        'skipped_block_count': int(skipped_block_count),
        'overlap_skip_count': int(overlap_skip_count),
    }


def _factor_dense_block(block: np.ndarray) -> Tuple[np.ndarray, str]:
    try:
        return np.linalg.inv(block), 'dense_inv'
    except np.linalg.LinAlgError:
        return np.linalg.pinv(block), 'dense_pinv'


class SparseSemanticBlockJacobi:
    def __init__(self, matrix: sp.spmatrix, blocks: List[np.ndarray], uncovered_policy: str = 'row_sum'):
        self.matrix = matrix.tocsr()
        self.blocks = [np.asarray(rows, dtype=np.int64) for rows in blocks]
        self.factor_solvers: List[np.ndarray] = []
        self.factor_modes: Dict[str, int] = {}
        self.covered_mask = np.zeros(int(matrix.shape[0]), dtype=bool)
        total_block_nnz = 0
        for rows in self.blocks:
            local = self.matrix[rows[:, None], rows].toarray().astype(np.complex128)
            total_block_nnz += int(np.count_nonzero(local))
            factor, mode = _factor_dense_block(local)
            self.factor_solvers.append(factor)
            self.factor_modes[mode] = self.factor_modes.get(mode, 0) + 1
            self.covered_mask[rows] = True
        self.uncovered_mask = ~self.covered_mask
        if uncovered_policy == 'jacobi_diagonal':
            diag = self.matrix.diagonal()
            self.uncovered_scales = 1.0 / np.maximum(np.abs(diag), 1e-30)
        elif uncovered_policy == 'identity':
            self.uncovered_scales = np.ones(int(matrix.shape[0]), dtype=np.complex128)
        else:
            row_sum = np.asarray(abs(self.matrix).sum(axis=1)).reshape(-1)
            self.uncovered_scales = 1.0 / np.maximum(row_sum, 1e-30)
        self.total_block_nnz = int(total_block_nnz)

    def apply(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.complex128)
        out = np.zeros_like(vec)
        for rows, factor in zip(self.blocks, self.factor_solvers):
            out[rows] = factor.dot(vec[rows])
        out[self.uncovered_mask] = self.uncovered_scales[self.uncovered_mask] * vec[self.uncovered_mask]
        return out

    def apply_core_only(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.complex128)
        out = np.zeros_like(vec)
        for rows, factor in zip(self.blocks, self.factor_solvers):
            out[rows] = factor.dot(vec[rows])
        return out

    def metadata(self) -> Dict[str, Any]:
        return {
            'block_count': int(len(self.blocks)),
            'covered_rows': int(np.count_nonzero(self.covered_mask)),
            'uncovered_rows': int(np.count_nonzero(self.uncovered_mask)),
            'coverage_ratio': float(np.count_nonzero(self.covered_mask) / max(int(self.matrix.shape[0]), 1)),
            'total_block_nnz': int(self.total_block_nnz),
            'factor_modes': dict(self.factor_modes),
        }


def _interface_rows_from_plans(core: SparseSemanticBlockJacobi, boundary_blocks: List[np.ndarray], n: int) -> Tuple[np.ndarray, np.ndarray]:
    boundary_mask = np.zeros(int(n), dtype=bool)
    for rows in boundary_blocks:
        boundary_mask[np.asarray(rows, dtype=np.int64)] = True
    interface_mask = np.logical_and(boundary_mask, ~core.covered_mask)
    return np.flatnonzero(interface_mask).astype(np.int64), interface_mask


class SparseLocalSchurPreconditioner:
    """Sparse-native strict exact-on-demand local Schur preconditioner.

    This mirrors the strict Route-B logic in schur_interface.py while keeping
    100x10 artifacts sparse: build cheap proxy edge groups, select safe base
    plus optional learned additions, compute exact local Schur values only for
    selected groups, assemble sparse P_theta, and solve P_theta in apply.
    """

    def __init__(
        self,
        *,
        matrix: sp.spmatrix,
        core: SparseSemanticBlockJacobi,
        boundary_blocks: List[np.ndarray],
        learned_model: Any = None,
        strategy: str = 'topk_abs',
        edge_budget: int = 0,
        budget_multiplier: float = 2.0,
        candidate_edge_limit: int = 512,
        diagonal_shift: float = 1e-8,
        factor_drop_tol: float = 1e-4,
        factor_fill_factor: float = 10.0,
        interface_solve_mode: str = 'spilu',
        eps: float = 1e-30,
        probe_rhs: Optional[np.ndarray] = None,
        probe_x0: Optional[np.ndarray] = None,
        probe_restart: int = 8,
        probe_iterations: int = 1,
        add_fraction: float = 0.1,
        add_min: int = 1,
        node_map: Optional[Dict[str, int]] = None,
        max_schur_nnz: int = 0,
        max_degree: int = 0,
        max_exact_entries: int = 0,
    ):
        self.matrix = matrix.tocsr().astype(np.complex128)
        self.matrix_csc = self.matrix.tocsc()
        self.core = core
        self.interface_rows, self.interface_mask = _interface_rows_from_plans(core, boundary_blocks, int(matrix.shape[0]))
        self.interface_count = int(self.interface_rows.shape[0])
        self.core_rows = np.flatnonzero(self.core.covered_mask).astype(np.int64)
        self.uncovered_mask = ~np.logical_or(self.core.covered_mask, self.interface_mask)
        self.strategy = str(strategy)
        self.edge_budget = int(edge_budget)
        self.budget_multiplier = float(budget_multiplier)
        self.candidate_edge_limit = int(candidate_edge_limit)
        self.max_schur_nnz = int(max_schur_nnz)
        self.max_degree = int(max_degree)
        self.max_exact_entries = int(max_exact_entries)
        self.diagonal_shift = float(diagonal_shift)
        self.factor_drop_tol = float(factor_drop_tol)
        self.factor_fill_factor = float(factor_fill_factor)
        self.interface_solve_mode = str(interface_solve_mode)
        self.eps = float(eps)
        self.probe_rhs = None if probe_rhs is None else np.asarray(probe_rhs, dtype=np.complex128)
        self.probe_x0 = None if probe_x0 is None else np.asarray(probe_x0, dtype=np.complex128)
        self.probe_restart = int(probe_restart)
        self.probe_iterations = int(probe_iterations)
        self.add_fraction = float(add_fraction)
        self.add_min = int(add_min)
        self.node_map = dict(node_map or {})
        self.index_to_name = {int(idx): str(name).lower() for name, idx in self.node_map.items() if str(idx).lstrip('-').isdigit()}
        self.interface_pos = {int(row): idx for idx, row in enumerate(self.interface_rows.tolist())}
        self.learned_model = learned_model
        self.abb = self.matrix[self.interface_rows, :][:, self.interface_rows].tocsc() if self.interface_count else sp.csc_matrix((0, 0), dtype=np.complex128)
        self.groups, self.edge_proxy, self.edge_source_counts, self.total_possible_local_schur_entries = self._build_proxy_groups()
        self.edges = np.asarray(list(self.groups.keys()), dtype=np.int64) if self.groups else np.zeros((0, 2), dtype=np.int64)
        self.proxy_values = np.asarray([self.edge_proxy[tuple(edge)] for edge in self.edges.tolist()], dtype=np.float64) if self.edges.shape[0] else np.zeros(0, dtype=np.float64)
        self.source_counts = np.asarray([self.edge_source_counts.get(tuple(edge), 1) for edge in self.edges.tolist()], dtype=np.int64) if self.edges.shape[0] else np.zeros(0, dtype=np.int64)
        self.features = self._build_proxy_features()
        self.learning_scores = self._cheap_learning_scores()
        self.safe_edge_indices, self.learned_add_indices, self.selected_edge_indices = self._select_edges()
        self.schur_matrix, self.exact_schur_entry_count, self.exact_schur_entry_compute_time = self._assemble_selected_sparse_schur(self.selected_edge_indices)
        self.factorization_success = True
        self.factorization_failure_reason = ''
        start_time = time.perf_counter()
        try:
            self.schur_factor, self.schur_factor_mode = self._factor_schur(self.schur_matrix)
        except Exception as exc:
            self.factorization_success = False
            self.schur_factor_mode = 'factorization_failed_jacobi_fallback'
            self.factorization_failure_reason = repr(exc)
            diag = self.schur_matrix.diagonal().astype(np.complex128) if self.interface_count else np.zeros(0, dtype=np.complex128)
            self.schur_factor = np.where(np.abs(diag) > self.eps, 1.0 / diag, 0.0 + 0.0j)
        self.factorization_time = float(time.perf_counter() - start_time)
        self.apply_count = 0
        self.apply_time_total = 0.0

    def _active_interface_for_core_rows(self, rows: np.ndarray) -> np.ndarray:
        active = set()
        csr = self.matrix
        csc = self.matrix_csc
        for row in rows.tolist():
            start, end = int(csr.indptr[row]), int(csr.indptr[row + 1])
            for col in csr.indices[start:end].tolist():
                pos = self.interface_pos.get(int(col))
                if pos is not None:
                    active.add(pos)
            start, end = int(csc.indptr[row]), int(csc.indptr[row + 1])
            for src in csc.indices[start:end].tolist():
                pos = self.interface_pos.get(int(src))
                if pos is not None:
                    active.add(pos)
        return np.asarray(sorted(active), dtype=np.int64)

    def _build_proxy_groups(self) -> Tuple[Dict[Tuple[int, int], List[Tuple[int, int, int]]], Dict[Tuple[int, int], float], Dict[Tuple[int, int], int], int]:
        groups: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
        proxy: Dict[Tuple[int, int], float] = {}
        counts: Dict[Tuple[int, int], int] = {}
        total = 0
        for block_id, rows in enumerate(self.core.blocks):
            rows = np.asarray(rows, dtype=np.int64)
            active = self._active_interface_for_core_rows(rows)
            if active.shape[0] == 0 or rows.shape[0] == 0:
                continue
            active_rows = self.interface_rows[active]
            total += int(active_rows.shape[0] * active_rows.shape[0])
            diag = np.abs(self.matrix[rows[:, None], rows].diagonal())
            diag_scale = 1.0 / max(float(np.max(diag)) if diag.size else 0.0, self.eps)
            a_bi = self.matrix[active_rows[:, None], rows]
            a_ib = self.matrix[rows[:, None], active_rows]
            row_norm = np.sqrt(np.asarray(abs(a_bi).power(2).sum(axis=1)).reshape(-1))
            col_norm = np.sqrt(np.asarray(abs(a_ib).power(2).sum(axis=0)).reshape(-1))
            for ai, bi in enumerate(active_rows.tolist()):
                if float(row_norm[ai]) <= self.eps:
                    continue
                for aj, bj in enumerate(active_rows.tolist()):
                    if float(col_norm[aj]) <= self.eps:
                        continue
                    key = (int(active[ai]), int(active[aj]))
                    value = float(row_norm[ai]) * diag_scale * float(col_norm[aj])
                    if value <= self.eps:
                        continue
                    groups.setdefault(key, []).append((int(block_id), int(bi), int(bj)))
                    proxy[key] = float(proxy.get(key, 0.0) + value)
                    counts[key] = int(counts.get(key, 0) + 1)
        if self.candidate_edge_limit > 0 and len(groups) > self.candidate_edge_limit:
            keep = set(key for key, _ in sorted(proxy.items(), key=lambda kv: kv[1], reverse=True)[: self.candidate_edge_limit])
            groups = {key: value for key, value in groups.items() if key in keep}
            proxy = {key: value for key, value in proxy.items() if key in keep}
            counts = {key: value for key, value in counts.items() if key in keep}
        return groups, proxy, counts, int(total)

    def _build_proxy_features(self) -> np.ndarray:
        if self.edges.shape[0] == 0:
            return np.zeros((0, 14), dtype=np.float64)
        features = np.zeros((int(self.edges.shape[0]), 14), dtype=np.float64)
        max_proxy = max(float(np.max(self.proxy_values)), self.eps)
        max_count = max(float(np.max(self.source_counts)) if self.source_counts.size else 1.0, 1.0)
        diag = self.abb.diagonal().astype(np.complex128) if self.interface_count else np.zeros(0, dtype=np.complex128)
        for idx, (ri, cj) in enumerate(self.edges.tolist()):
            proxy_value = float(self.proxy_values[idx])
            reverse = float(self.edge_proxy.get((int(cj), int(ri)), 0.0))
            dnorm = np.sqrt(max((abs(diag[int(ri)]) if diag.size else 0.0) * (abs(diag[int(cj)]) if diag.size else 0.0), self.eps))
            features[idx, 0] = np.log(max(proxy_value, self.eps))
            features[idx, 1] = np.log(max(proxy_value / max(dnorm, self.eps), self.eps))
            features[idx, 2] = np.log(max(proxy_value / max(max_proxy, self.eps), self.eps))
            features[idx, 3] = np.log(max(reverse, self.eps))
            features[idx, 4] = 1.0
            features[idx, 5] = np.log(max(abs(diag[int(ri)]) if diag.size else 0.0, self.eps))
            features[idx, 6] = np.log(max(abs(diag[int(cj)]) if diag.size else 0.0, self.eps))
            features[idx, 11] = float(len(self.groups.get((int(ri), int(cj)), []))) / max(float(self.interface_count), 1.0)
            features[idx, 12] = min(float(self.source_counts[idx]) / max_count, 1.0)
            features[idx, 13] = float(int(ri) - int(cj)) / float(max(self.interface_count - 1, 1))
        return features

    def _budget(self) -> int:
        if self.budget_multiplier > 0.0:
            return max(0, int(round(self.budget_multiplier * float(max(self.interface_count, 0)))))
        return max(0, int(self.edge_budget))

    def _cheap_learning_scores(self) -> np.ndarray:
        if self.proxy_values.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        counts = self.source_counts.astype(np.float64) if self.source_counts.shape[0] else np.ones_like(self.proxy_values)
        source_bonus = np.log1p(counts) / max(float(np.log1p(np.max(counts))) if counts.size else 1.0, 1.0)
        diag_bonus = np.asarray([1.0 if int(edge[0]) == int(edge[1]) else 0.0 for edge in self.edges], dtype=np.float64)
        reverse = np.asarray([self.edge_proxy.get((int(edge[1]), int(edge[0])), 0.0) for edge in self.edges], dtype=np.float64)
        symmetry_bonus = np.minimum(reverse / np.maximum(self.proxy_values, self.eps), 1.0)
        return self.proxy_values * (1.0 + 0.25 * source_bonus + 0.10 * diag_bonus + 0.05 * symmetry_bonus)

    def _edge_exact_cost(self, idx: int) -> int:
        key = tuple(int(x) for x in self.edges[int(idx)].tolist())
        return int(len(self.groups.get(key, [])))

    def _select_edges(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.learned_edge_proposed_count = 0
        self.learned_edge_accepted_count = 0
        self.learned_edge_rejected_count = 0
        self.probe_eval_count = 0
        self.probe_accept_mean_eta = None
        self.probe_reject_reason_counts: Dict[str, int] = {}
        self.safe_rule_counts = {'diagonal': 0, 'abb_nonzero': 0, 'per_row_proxy': 0, 'global_proxy': 0, 'semantic_conservative': 0}
        self.semantic_safe_rule_counts = {'supply_like': 0, 'ground_like': 0, 'high_degree': 0}
        self.semantic_protected_interface_positions: List[int] = []
        self.learned_add_per_row_limit = max(1, int(self.add_min), 2)
        if self.edges.shape[0] == 0:
            empty = np.zeros(0, dtype=np.int64)
            self.selection_mode = 'empty'
            return empty, empty, empty
        budget = min(self._budget(), int(self.edges.shape[0]))
        if budget <= 0:
            empty = np.zeros(0, dtype=np.int64)
            self.selection_mode = 'zero_budget'
            return empty, empty, empty
        if self.strategy == 'learned' and self.learned_model is not None:
            import torch
            with torch.no_grad():
                scores = self.learned_model(torch.as_tensor(self.features, dtype=torch.float64)).detach().cpu().numpy().reshape(-1)
            order = np.argsort(-scores).astype(np.int64)[:budget]
            self.learned_score_min = float(np.min(scores)) if scores.size else None
            self.learned_score_max = float(np.max(scores)) if scores.size else None
            self.learned_score_mean = float(np.mean(scores)) if scores.size else None
            self.selection_mode = 'learned_replace_proxy_exact_on_demand'
            return order, np.zeros(0, dtype=np.int64), order
        if self.strategy == 'topk_abs':
            order = np.argsort(-self.proxy_values).astype(np.int64)[:budget]
            self.selection_mode = 'hard_topk_proxy_exact_on_demand'
            return order, np.zeros(0, dtype=np.int64), order

        edge_list = [tuple(int(x) for x in edge.tolist()) for edge in self.edges]
        by_row: Dict[int, List[int]] = {}
        for idx, (ri, cj) in enumerate(edge_list):
            by_row.setdefault(int(ri), []).append(int(idx))
        degree_values = np.asarray([len(v) for v in by_row.values()], dtype=np.float64) if by_row else np.zeros(0, dtype=np.float64)
        high_degree_threshold = max(8, int(np.percentile(degree_values, 90)) if degree_values.size else 8)
        protected_rows = set()
        for ri, idxs in by_row.items():
            global_row = int(self.interface_rows[int(ri)])
            name = str(self.index_to_name.get(global_row, self.index_to_name.get(global_row + 1, ''))).lower()
            is_ground = name in {'0', 'gnd', 'ground'} or 'gnd' in name or 'vss' in name
            is_supply = ('vdd' in name) or ('vcc' in name) or ('vss' in name) or ('vsub' in name) or ('supply' in name) or ('power' in name)
            is_high_degree = len(idxs) >= high_degree_threshold and len(idxs) > 2
            if is_supply or is_ground or is_high_degree:
                protected_rows.add(int(ri))
                self.semantic_protected_interface_positions.append(int(ri))
                if is_supply:
                    self.semantic_safe_rule_counts['supply_like'] = int(self.semantic_safe_rule_counts.get('supply_like', 0) + 1)
                if is_ground:
                    self.semantic_safe_rule_counts['ground_like'] = int(self.semantic_safe_rule_counts.get('ground_like', 0) + 1)
                if is_high_degree:
                    self.semantic_safe_rule_counts['high_degree'] = int(self.semantic_safe_rule_counts.get('high_degree', 0) + 1)

        safe: List[int] = []
        seen = set()

        def selected_exact_cost(extra_idx: Optional[int] = None) -> int:
            total_cost = sum(self._edge_exact_cost(i) for i in seen)
            return total_cost if extra_idx is None else total_cost + self._edge_exact_cost(int(extra_idx))

        def selected_row_degree(ri: int, extra_idx: Optional[int] = None) -> int:
            deg = sum(1 for i in seen if int(edge_list[int(i)][0]) == int(ri))
            if extra_idx is not None and int(edge_list[int(extra_idx)][0]) == int(ri):
                deg += 1
            return int(deg)

        def optional_allowed(idx: int) -> bool:
            idx = int(idx)
            ri = int(edge_list[idx][0])
            if self.max_schur_nnz > 0 and len(seen) + 1 > int(self.max_schur_nnz):
                return False
            if self.max_exact_entries > 0 and selected_exact_cost(idx) > int(self.max_exact_entries):
                return False
            if self.max_degree > 0 and selected_row_degree(ri, idx) > int(self.max_degree):
                return False
            return True

        def add_safe(idx: int, rule: str, mandatory: bool = True) -> bool:
            idx = int(idx)
            if idx in seen:
                return False
            if not mandatory and not optional_allowed(idx):
                return False
            safe.append(idx)
            seen.add(idx)
            self.safe_rule_counts[rule] = int(self.safe_rule_counts.get(rule, 0) + 1)
            return True

        diag_idx = [idx for idx, (ri, cj) in enumerate(edge_list) if int(ri) == int(cj)]
        for idx in diag_idx:
            add_safe(idx, 'diagonal')
        if self.abb.shape[0]:
            for idx, (ri, cj) in enumerate(edge_list):
                try:
                    if abs(complex(self.abb[int(ri), int(cj)])) > self.eps:
                        add_safe(idx, 'abb_nonzero')
                except Exception:
                    pass
        for ri, idxs in by_row.items():
            ordered = sorted(idxs, key=lambda ix: float(self.proxy_values[int(ix)]), reverse=True)
            offdiag = [ix for ix in ordered if int(edge_list[int(ix)][1]) != int(ri)]
            target = offdiag[:1] if offdiag else ordered[:1]
            for idx in target:
                add_safe(idx, 'per_row_proxy')
            if int(ri) in protected_rows:
                for idx in ordered[: min(2, len(ordered))]:
                    add_safe(idx, 'semantic_conservative')
        for idx in np.argsort(-self.proxy_values).astype(np.int64).tolist():
            if len(safe) >= max(budget, len(diag_idx)):
                break
            add_safe(int(idx), 'global_proxy', mandatory=False)

        safe_arr = np.asarray(safe, dtype=np.int64)
        safe_set = set(int(x) for x in safe_arr.tolist())
        add_budget = min(int(self.edges.shape[0]) - len(safe_set), max(int(self.add_min), int(round(self.add_fraction * max(budget, 1)))))
        add: List[int] = []
        row_add_counts: Dict[int, int] = {}
        for idx in np.argsort(-self.learning_scores).astype(np.int64).tolist():
            idx = int(idx)
            if idx in safe_set or idx in seen:
                continue
            ri = int(edge_list[idx][0])
            if ri in protected_rows:
                continue
            if int(row_add_counts.get(ri, 0)) >= int(self.learned_add_per_row_limit):
                continue
            if not optional_allowed(idx):
                continue
            seen.add(idx)
            add.append(idx)
            row_add_counts[ri] = int(row_add_counts.get(ri, 0) + 1)
            if len(add) >= max(0, add_budget):
                break
        add_arr = np.asarray(add, dtype=np.int64)
        self.learned_edge_proposed_count = int(add_arr.shape[0])
        union = np.concatenate([safe_arr, add_arr]).astype(np.int64) if add_arr.shape[0] else safe_arr
        selected = self._select_with_probe(safe_arr, union) if self.strategy == 'safe_add_probe' else union
        if self.strategy != 'safe_add_probe':
            self.selection_mode = 'safe_add' if add_arr.shape[0] else 'hard_safe_base'
        selected_set = set(int(x) for x in selected.tolist())
        self.learned_edge_accepted_count = int(sum(1 for idx in add_arr.tolist() if int(idx) in selected_set))
        self.learned_edge_rejected_count = int(add_arr.shape[0] - self.learned_edge_accepted_count)
        return safe_arr, add_arr, selected.astype(np.int64)

    def _exact_value_for_source(self, block_id: int, bi: int, bj: int) -> complex:
        rows = self.core.blocks[int(block_id)]
        factor = self.core.factor_solvers[int(block_id)]
        rhs_raw = self.matrix[rows, int(bj)]
        rhs = (rhs_raw.toarray() if hasattr(rhs_raw, 'toarray') else np.asarray(rhs_raw)).reshape(-1).astype(np.complex128)
        solved = factor.dot(rhs)
        left_raw = self.matrix[int(bi), rows]
        left = (left_raw.toarray() if hasattr(left_raw, 'toarray') else np.asarray(left_raw)).reshape(-1).astype(np.complex128)
        return -complex(left.dot(solved))

    def _assemble_selected_sparse_schur(self, selected_indices: np.ndarray) -> Tuple[sp.csc_matrix, int, float]:
        start_time = time.perf_counter()
        row_idx: List[int] = []
        col_idx: List[int] = []
        vals: List[complex] = []
        exact_count = 0
        for idx in np.asarray(selected_indices, dtype=np.int64).tolist():
            key = tuple(int(x) for x in self.edges[int(idx)].tolist())
            value = 0.0 + 0.0j
            for block_id, bi, bj in self.groups.get(key, []):
                value += self._exact_value_for_source(block_id, bi, bj)
                exact_count += 1
            if abs(value) > self.eps:
                row_idx.append(int(key[0]))
                col_idx.append(int(key[1]))
                vals.append(value)
        if row_idx:
            exact_mat = sp.coo_matrix((np.asarray(vals, dtype=np.complex128), (row_idx, col_idx)), shape=(self.interface_count, self.interface_count), dtype=np.complex128)
            schur = (self.abb + exact_mat).tocsc()
        else:
            schur = self.abb.copy().tocsc()
        if self.diagonal_shift > 0.0 and self.interface_count > 0:
            abs_s = abs(schur).tocsr()
            offdiag_abs = np.asarray(abs_s.sum(axis=1)).reshape(-1) - np.abs(schur.diagonal())
            base_diag = schur.diagonal()
            signs = np.where(np.real(base_diag) >= 0.0, 1.0, -1.0)
            shifted = base_diag + signs * self.diagonal_shift * np.maximum(offdiag_abs, self.eps)
            schur = schur + sp.diags(shifted - base_diag, offsets=0, shape=schur.shape, dtype=np.complex128)
        return schur.tocsc(), int(exact_count), float(time.perf_counter() - start_time)

    def _factor_schur(self, schur_matrix: Optional[sp.csc_matrix] = None) -> Tuple[Any, str]:
        schur = self.schur_matrix if schur_matrix is None else schur_matrix
        if self.interface_count <= 0:
            return None, 'empty'
        if self.interface_solve_mode in {'jacobi', 'jacobi_neumann1'}:
            diag = schur.diagonal().astype(np.complex128)
            inv_diag = np.where(np.abs(diag) > self.eps, 1.0 / diag, 0.0 + 0.0j)
            return inv_diag, self.interface_solve_mode
        try:
            return spilu(schur, drop_tol=self.factor_drop_tol, fill_factor=self.factor_fill_factor), 'spilu'
        except Exception:
            return splu(schur), 'splu'

    def _solve_interface(self, rhs: np.ndarray, factor: Any = None, factor_mode: Optional[str] = None, schur_matrix: Optional[sp.csc_matrix] = None) -> np.ndarray:
        factor = self.schur_factor if factor is None else factor
        factor_mode = self.schur_factor_mode if factor_mode is None else factor_mode
        schur = self.schur_matrix if schur_matrix is None else schur_matrix
        if factor is None:
            return np.zeros_like(rhs, dtype=np.complex128)
        if isinstance(factor, np.ndarray):
            y = factor * rhs
            if factor_mode == 'jacobi_neumann1':
                y = y + factor * (rhs - schur.dot(y))
            return np.asarray(y, dtype=np.complex128)
        return factor.solve(rhs)

    def _apply_with_factor(self, vec: np.ndarray, factor: Any, factor_mode: str, schur_matrix: sp.csc_matrix) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.complex128)
        out = np.zeros_like(vec)
        core_solution = self.core.apply_core_only(vec)
        out[self.core.covered_mask] = core_solution[self.core.covered_mask]
        if self.interface_count > 0:
            interface_rhs = vec[self.interface_rows] - self.matrix[self.interface_rows, :].dot(core_solution)
            interface_solution = self._solve_interface(interface_rhs, factor=factor, factor_mode=factor_mode, schur_matrix=schur_matrix)
            out[self.interface_rows] = interface_solution
            for rows, core_factor in zip(self.core.blocks, self.core.factor_solvers):
                correction_rhs = self.matrix[rows, :][:, self.interface_rows].dot(interface_solution)
                if np.any(np.abs(correction_rhs) > self.eps):
                    out[rows] = out[rows] - core_factor.dot(correction_rhs)
        out[self.uncovered_mask] = self.core.uncovered_scales[self.uncovered_mask] * vec[self.uncovered_mask]
        return out

    def _probe_score(self, selected_indices: np.ndarray) -> float:
        self.probe_eval_count = int(getattr(self, 'probe_eval_count', 0)) + 1
        if self.probe_rhs is None or self.probe_rhs.shape[0] != self.matrix.shape[0]:
            return float('inf')
        try:
            schur, _, _ = self._assemble_selected_sparse_schur(selected_indices)
            factor, factor_mode = self._factor_schur(schur)
            preconditioner = LinearOperator(self.matrix.shape, matvec=lambda x: self._apply_with_factor(x, factor, factor_mode, schur), dtype=np.complex128)
            history: List[float] = []
            solution, _ = gmres(
                self.matrix,
                self.probe_rhs,
                x0=self.probe_x0,
                M=preconditioner,
                restart=max(1, self.probe_restart),
                maxiter=max(1, self.probe_iterations),
                rtol=0.0,
                atol=0.0,
                callback=lambda value: history.append(float(value)),
                callback_type='pr_norm',
            )
            raw = self.matrix.dot(solution) - self.probe_rhs
            score = float(np.linalg.norm(raw))
            return min(score, float(history[-1])) if history else score
        except Exception:
            return float('inf')

    def _select_with_probe(self, safe: np.ndarray, union: np.ndarray) -> np.ndarray:
        self.probe_eval_count = 0
        self.probe_accept_mean_eta = None
        if self.probe_rhs is None or self.probe_rhs.shape[0] != self.matrix.shape[0] or np.array_equal(safe, union):
            self.selection_mode = 'safe_add' if not np.array_equal(safe, union) else 'hard_safe_base'
            if not np.array_equal(safe, union):
                self.probe_reject_reason_counts = {'probe_unavailable_accepted_without_probe': 1}
            return union
        safe_score = self._probe_score(safe)
        union_score = self._probe_score(union)
        if np.isfinite(safe_score) and np.isfinite(union_score):
            self.probe_accept_mean_eta = float(union_score / max(safe_score, self.eps))
        if union_score < safe_score:
            self.selection_mode = 'safe_add_probe:accepted'
            self.probe_reject_reason_counts = {}
            return union
        self.selection_mode = 'safe_add_probe:rejected'
        reason = 'probe_failure_or_factorization_failure' if not np.isfinite(union_score) else 'probe_no_improvement'
        self.probe_reject_reason_counts = {reason: 1}
        return safe

    def apply(self, vec: np.ndarray) -> np.ndarray:
        start_time = time.perf_counter()
        out = self._apply_with_factor(vec, self.schur_factor, self.schur_factor_mode, self.schur_matrix)
        self.apply_count += 1
        self.apply_time_total += float(time.perf_counter() - start_time)
        return out

    def metadata(self) -> Dict[str, Any]:
        selected = int(self.selected_edge_indices.shape[0])
        row_degrees = np.diff(self.schur_matrix.tocsr().indptr) if self.schur_matrix.shape[0] else np.zeros(0, dtype=np.int64)
        solve_per_apply = float(self.apply_time_total / max(int(self.apply_count), 1))
        factor_fill = None
        if self.schur_factor is not None and hasattr(self.schur_factor, 'L'):
            factor_fill = int(self.schur_factor.L.nnz + self.schur_factor.U.nnz)
        elif isinstance(self.schur_factor, np.ndarray):
            factor_fill = int(np.count_nonzero(self.schur_factor))
        budget_constraint_violations = {
            'max_schur_nnz': bool(self.max_schur_nnz > 0 and selected > int(self.max_schur_nnz)),
            'max_degree': bool(self.max_degree > 0 and row_degrees.size and int(np.max(row_degrees)) > int(self.max_degree)),
            'max_exact_entries': bool(self.max_exact_entries > 0 and int(self.exact_schur_entry_count) > int(self.max_exact_entries)),
        }
        return {
            'preconditioner_mode': 'sparse_' + self.strategy + '_local_schur',
            'route': 'route_b_sparse_schur_construction',
            'implementation': 'strict_exact_on_demand_sparse_native',
            'full_schur_constructed': False,
            'selected_local_constructed': True,
            'core_inverse_interface_constructed': False,
            'interface_rows': int(self.interface_count),
            'core_rows': int(self.core_rows.shape[0]),
            'uncovered_rows': int(np.count_nonzero(self.uncovered_mask)),
            'local_candidate_edges': int(self.total_possible_local_schur_entries),
            'global_candidate_edges': int(self.edges.shape[0]),
            'total_possible_local_schur_entries': int(self.total_possible_local_schur_entries),
            'total_candidate_count': int(self.edges.shape[0]),
            'cheap_candidate_count': int(self.edges.shape[0]),
            'approximate_schur_score_count': int(self.edges.shape[0]),
            'selected_edges': selected,
            'selected_edge_count': selected,
            'selected_edge_ratio': float(selected / max(int(self.edges.shape[0]), 1)),
            'exact_schur_entry_count': int(self.exact_schur_entry_count),
            'exact_schur_entry_compute_time': float(self.exact_schur_entry_compute_time),
            'exact_compute_ratio': float(self.exact_schur_entry_count / max(int(self.total_possible_local_schur_entries), 1)),
            'skipped_patch_count': 0,
            'skipped_entry_count': 0,
            'P_theta_shape': [int(self.schur_matrix.shape[0]), int(self.schur_matrix.shape[1])],
            'P_theta_nnz': int(self.schur_matrix.nnz),
            'P_theta_density': float(self.schur_matrix.nnz / max(self.interface_count * self.interface_count, 1)),
            'P_shape': [int(self.schur_matrix.shape[0]), int(self.schur_matrix.shape[1])],
            'P_nnz': int(self.schur_matrix.nnz),
            'P_density': float(self.schur_matrix.nnz / max(self.interface_count * self.interface_count, 1)),
            'P_diag_nnz': int(np.count_nonzero(self.schur_matrix.diagonal())) if self.interface_count else 0,
            'P_offdiag_nnz': int(self.schur_matrix.nnz - (int(np.count_nonzero(self.schur_matrix.diagonal())) if self.interface_count else 0)),
            'P_theta_factorization_success': bool(self.factorization_success),
            'P_theta_factorization_time': float(self.factorization_time),
            'P_theta_factor_fill_nnz': None if factor_fill is None else int(factor_fill),
            'P_theta_apply_count': int(self.apply_count),
            'P_theta_apply_time_total': float(self.apply_time_total),
            'P_theta_solve_time_total': float(self.apply_time_total),
            'P_theta_solve_time_per_apply': solve_per_apply,
            'P_theta_assembly_time': float(self.exact_schur_entry_compute_time),
            'P_theta_failure_reason': str(self.factorization_failure_reason),
            'fallback_used': not bool(self.factorization_success),
            'sparse_schur_nnz': int(self.schur_matrix.nnz),
            'schur_factor_mode': str(self.schur_factor_mode),
            'schur_factor_nnz': None if factor_fill is None else int(factor_fill),
            'per_row_degree_stats': {
                'min': int(np.min(row_degrees)) if row_degrees.size else 0,
                'max': int(np.max(row_degrees)) if row_degrees.size else 0,
                'mean': float(np.mean(row_degrees)) if row_degrees.size else 0.0,
                'p95': float(np.percentile(row_degrees, 95)) if row_degrees.size else 0.0,
            },
            'safe_edge_count': int(self.safe_edge_indices.shape[0]),
            'safe_rule_counts': dict(getattr(self, 'safe_rule_counts', {})),
            'semantic_safe_rule_counts': dict(getattr(self, 'semantic_safe_rule_counts', {})),
            'semantic_protected_interface_count': int(len(getattr(self, 'semantic_protected_interface_positions', []))),
            'learned_add_per_row_limit': int(getattr(self, 'learned_add_per_row_limit', 0)),
            'learned_edge_proposed_count': int(getattr(self, 'learned_edge_proposed_count', 0)),
            'learned_edge_accepted_count': int(getattr(self, 'learned_edge_accepted_count', 0)),
            'learned_edge_rejected_count': int(getattr(self, 'learned_edge_rejected_count', 0)),
            'probe_eval_count': int(getattr(self, 'probe_eval_count', 0)),
            'probe_accept_mean_eta': getattr(self, 'probe_accept_mean_eta', None),
            'probe_reject_reason_counts': dict(getattr(self, 'probe_reject_reason_counts', {})),
            'selection_mode': str(getattr(self, 'selection_mode', self.strategy)),
            'edge_budget': int(self.edge_budget),
            'budget_multiplier': float(self.budget_multiplier),
            'candidate_edge_limit': int(self.candidate_edge_limit),
            'max_candidates': int(self.candidate_edge_limit),
            'max_schur_nnz': int(self.max_schur_nnz),
            'max_degree': int(self.max_degree),
            'max_exact_entries': int(self.max_exact_entries),
            'budget_constraint_violations': budget_constraint_violations,
            'diagonal_shift': float(self.diagonal_shift),
            'memory_estimate': int(self.matrix.data.nbytes + self.matrix.indices.nbytes + self.matrix.indptr.nbytes + self.schur_matrix.data.nbytes + self.schur_matrix.indices.nbytes + self.schur_matrix.indptr.nbytes),
            'core': self.core.metadata(),
        }

def build_sparse_semantic_preconditioner(
    matrix: sp.spmatrix,
    mode: str,
    step: Dict[str, Any],
    payload: Dict[str, Any],
    args: argparse.Namespace,
    *,
    linear_rhs: Optional[np.ndarray] = None,
    initial_residual: Optional[np.ndarray] = None,
    initial_guess_mode: Optional[str] = None,
) -> Tuple[LinearOperator, Dict[str, Any]]:
    if args.backend != 'scipy_sparse':
        raise ValueError('sparse semantic Schur modes are currently implemented for backend=scipy_sparse')
    if not isinstance(payload.get('node_map'), dict) or not payload.get('node_map'):
        raise ValueError('sparse semantic modes require node_map in the continuation step payload')
    netlist_path = semantic_netlist_path(args)
    node_map = payload.get('node_map') or {}
    if mode == 'learned_schwarz_v1_sparse':
        if (
            linear_rhs is None
            or initial_residual is None
            or initial_guess_mode is None
        ):
            raise ValueError(
                'learned_schwarz_v1_sparse requires explicit linear_rhs, '
                'initial_residual, and initial_guess_mode'
            )
        return build_sparse_learned_schwarz_v1_preconditioner(
            matrix=matrix,
            node_map=node_map,
            netlist_path=netlist_path,
            checkpoint_path=str(getattr(args, 'learned_schwarz_checkpoint', '') or ''),
            linear_rhs=np.asarray(linear_rhs, dtype=np.float64),
            initial_residual=np.asarray(initial_residual, dtype=np.float64),
            initial_guess_mode=str(initial_guess_mode),
            gmin=float(step.get('gmin_val', 0.0)),
            max_block_size=int(args.semantic_boundary_max_block_size),
            min_block_size=int(args.semantic_min_block_size),
            max_blocks=int(args.semantic_max_blocks),
            parameter_mode=getattr(
                args,
                "learned_schwarz_parameter_mode",
                "learned_full",
            ),
        )
    core_blocks, core_debug = _extract_semantic_blocks_sparse(
        matrix=matrix,
        node_map=node_map,
        netlist_path=netlist_path,
        semantic_mode='cell_core',
        max_block_size=int(args.semantic_max_block_size),
        min_block_size=int(args.semantic_min_block_size),
        max_blocks=int(args.semantic_max_blocks),
    )
    core = SparseSemanticBlockJacobi(matrix, core_blocks, uncovered_policy=str(args.semantic_uncovered_policy))
    info: Dict[str, Any] = {
        'mode': mode,
        'fallback_reason': None,
        'netlist_path': netlist_path,
        'core_debug': core_debug,
        'core': core.metadata(),
    }
    if mode == 'semantic_cell_core_sparse':
        return LinearOperator(matrix.shape, matvec=core.apply, dtype=matrix.dtype), info
    if mode in {'semantic_coarse_r1_sparse', 'semantic_coarse_r2_sparse'}:
        boundary_blocks, boundary_debug = _extract_semantic_blocks_sparse(
            matrix=matrix,
            node_map=node_map,
            netlist_path=netlist_path,
            semantic_mode='cell_core_plus_onehop_boundary',
            max_block_size=int(getattr(args, 'semantic_boundary_max_block_size', 128)),
            min_block_size=int(getattr(args, 'semantic_min_block_size', 2)),
            max_blocks=int(getattr(args, 'semantic_max_blocks', 0)),
        )
        from pypath.preconditioner.semantic_coarse_space import (
            build_semantic_coarse_operator,
        )

        mode_count = 1 if mode.endswith('r1_sparse') else 2
        operator, coarse_info = build_semantic_coarse_operator(
            matrix=matrix,
            local_apply=core.apply,
            coarse_blocks=boundary_blocks,
            mode_count=mode_count,
            max_condition=float(getattr(args, 'semantic_coarse_max_condition', 1.0e12)),
            rank_tol=float(getattr(args, 'semantic_coarse_rank_tol', 1.0e-10)),
            semantic_mode=mode,
        )
        info['boundary_debug'] = boundary_debug
        coarse_public = {
            key: value
            for key, value in coarse_info.items()
            if key != 'operator_state'
        }
        state = coarse_info.get('operator_state')
        if state is not None:
            state.info = coarse_public
            state.metadata()
        info['coarse'] = coarse_public
        info['fallback_reason'] = coarse_info.get('fallback_reason')
        return operator, info
    boundary_blocks, boundary_debug = _extract_semantic_blocks_sparse(
        matrix=matrix,
        node_map=node_map,
        netlist_path=netlist_path,
        semantic_mode='cell_core_plus_onehop_boundary',
        max_block_size=int(args.semantic_boundary_max_block_size),
        min_block_size=int(args.semantic_min_block_size),
        max_blocks=int(args.semantic_max_blocks),
    )
    learned_model = None
    strategy = 'topk_abs'
    if mode == 'learned_local_sparse_schur_sparse':
        checkpoint = str(args.learned_local_sparse_schur_checkpoint or '').strip()
        if not checkpoint:
            raise ValueError('learned_local_sparse_schur_sparse requires --learned-local-sparse-schur-checkpoint')
        from pypath.preconditioner.schur_interface import load_learned_local_sparse_schur_model
        learned_model = load_learned_local_sparse_schur_model(checkpoint)
        strategy = 'learned'
        info['learned_local_sparse_schur_checkpoint'] = checkpoint
    elif mode == 'learned_sparse_schur_safe_add_sparse':
        strategy = 'safe_add'
    elif mode == 'learned_sparse_schur_safe_add_probe_sparse':
        strategy = 'safe_add_probe'
    schur = SparseLocalSchurPreconditioner(
        matrix=matrix,
        core=core,
        boundary_blocks=boundary_blocks,
        learned_model=learned_model,
        strategy=strategy,
        edge_budget=int(args.sparse_schur_edge_budget),
        budget_multiplier=float(args.local_schur_budget_multiplier),
        candidate_edge_limit=int(args.sparse_schur_candidate_edge_limit),
        diagonal_shift=float(args.sparse_schur_diagonal_shift),
        factor_drop_tol=float(args.sparse_schur_factor_drop_tol),
        factor_fill_factor=float(args.sparse_schur_factor_fill_factor),
        interface_solve_mode=str(args.sparse_schur_interface_solve),
        probe_rhs=np.asarray(payload.get('rhsnew', []), dtype=np.complex128),
        probe_x0=(np.asarray(payload.get('rhsold', []), dtype=np.complex128) if bool(args.use_rhsold_as_x0) else None),
        probe_restart=int(args.learning_sparse_schur_probe_restart),
        probe_iterations=int(args.learning_sparse_schur_probe_iterations),
        add_fraction=float(args.learning_sparse_schur_add_fraction),
        add_min=int(args.learning_sparse_schur_add_min),
        node_map=node_map,
        max_schur_nnz=int(args.sparse_schur_max_nnz),
        max_degree=int(args.sparse_schur_max_degree),
        max_exact_entries=int(args.sparse_schur_max_exact_entries),
    )
    info['boundary_debug'] = boundary_debug
    info['schur'] = schur.metadata()
    return LinearOperator(matrix.shape, matvec=schur.apply, dtype=matrix.dtype), info

def build_preconditioner(matrix: sp.spmatrix, mode: str) -> tuple[Optional[LinearOperator], Dict[str, Any]]:
    mode = str(mode)
    n = int(matrix.shape[0])
    info: Dict[str, Any] = {'mode': mode, 'fallback_reason': None}
    if mode == 'identity':
        return None, info
    if mode == 'row_sum':
        row_sum = np.asarray(abs(matrix).sum(axis=1)).reshape(-1)
        inv = 1.0 / np.maximum(row_sum, 1e-30)
        return LinearOperator((n, n), matvec=lambda vec: inv * vec, dtype=matrix.dtype), info
    if mode in {'jacobi', 'jacobi_diagonal'}:
        diag = matrix.diagonal()
        info['diag_zero_count'] = int(np.count_nonzero(np.abs(diag) < 1e-30))
        inv = 1.0 / np.maximum(np.abs(diag), 1e-30)
        return LinearOperator((n, n), matvec=lambda vec: inv * vec, dtype=matrix.dtype), info
    csc = matrix.tocsc()
    if mode == 'ilu0':
        factor = spilu(csc, drop_tol=0.0, fill_factor=1.0, permc_spec='NATURAL')
    elif mode == 'ilut':
        factor = spilu(csc, drop_tol=1e-4, fill_factor=10.0)
    elif mode == 'splu':
        factor = splu(csc)
    else:
        raise ValueError(f'Unsupported sparse mode: {mode}')
    info['L_nnz'] = int(factor.L.nnz)
    info['U_nnz'] = int(factor.U.nnz)
    info['factor_nnz'] = int(factor.L.nnz + factor.U.nnz)
    return LinearOperator((n, n), matvec=factor.solve, dtype=matrix.dtype), info




__all__ = [
    'SPARSE_SEMANTIC_MODES',
    'SparseSemanticBlockJacobi',
    'SparseLocalSchurPreconditioner',
    'build_preconditioner',
    'build_sparse_semantic_preconditioner',
    'semantic_netlist_path',
]
