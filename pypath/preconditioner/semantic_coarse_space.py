"""语义块驱动的全阶低秩粗空间预条件器。

该模块不替换原始线性方程，只在现有局部 Schwarz 校正后增加一个
低秩全局校正。局部基由语义块矩阵的近零右奇异模态构成，用于捕捉
跨单元的慢误差方向；所有数值守卫失败时回退到传入的局部校正。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator


DEFAULT_COARSE_MAX_CONDITION = 1.0e12
DEFAULT_COARSE_RANK_TOL = 1.0e-10


def _normalise_block(rows: Sequence[int], matrix_size: int) -> np.ndarray:
    values = sorted({int(value) for value in rows})
    if not values:
        return np.zeros(0, dtype=np.int64)
    array = np.asarray(values, dtype=np.int64)
    if int(array[0]) < 0 or int(array[-1]) >= int(matrix_size):
        raise ValueError("semantic coarse block row is outside matrix")
    return array


def _build_local_mode_vectors(
    matrix: sp.csr_matrix,
    blocks: Sequence[Sequence[int]],
    mode_count: int,
    rank_tol: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n = int(matrix.shape[0])
    normalised = [_normalise_block(rows, n) for rows in blocks]
    normalised = [rows for rows in normalised if rows.size]
    coverage = np.zeros(n, dtype=np.float64)
    for rows in normalised:
        coverage[rows] += 1.0
    inverse_coverage = np.zeros(n, dtype=np.float64)
    covered = coverage > 0.0
    inverse_coverage[covered] = 1.0 / coverage[covered]

    raw_vectors: List[np.ndarray] = []
    local_mode_counts: List[int] = []
    for rows in normalised:
        local = np.asarray(matrix[rows, :][:, rows].toarray(), dtype=np.float64)
        if local.size == 0 or not np.all(np.isfinite(local)):
            raise ValueError("semantic coarse local matrix is non-finite")
        _, singular_values, vh = np.linalg.svd(
            local,
            full_matrices=False,
            compute_uv=True,
        )
        order = np.argsort(np.asarray(singular_values, dtype=np.float64))
        take = min(max(int(mode_count), 1), int(vh.shape[0]))
        selected = 0
        for mode_index in order[:take].tolist():
            local_vector = np.asarray(vh[int(mode_index), :], dtype=np.float64)
            local_norm = float(np.linalg.norm(local_vector))
            if not np.isfinite(local_norm) or local_norm <= 0.0:
                continue
            local_vector = local_vector / local_norm
            global_vector = np.zeros(n, dtype=np.float64)
            global_vector[rows] = inverse_coverage[rows] * local_vector
            global_norm = float(np.linalg.norm(global_vector))
            if not np.isfinite(global_norm) or global_norm <= 0.0:
                continue
            raw_vectors.append(global_vector / global_norm)
            selected += 1
        local_mode_counts.append(int(selected))

    if not raw_vectors:
        return np.zeros((n, 0), dtype=np.float64), {
            "candidate_block_count": int(len(normalised)),
            "candidate_mode_count": 0,
            "covered_rows": int(np.count_nonzero(covered)),
            "coverage_ratio": float(np.mean(covered)) if n else 0.0,
            "local_mode_counts": local_mode_counts,
        }

    raw = np.column_stack(raw_vectors)
    q, r = np.linalg.qr(raw, mode="reduced")
    diagonal = np.abs(np.diag(r)) if r.size else np.zeros(0, dtype=np.float64)
    scale = max(float(np.max(diagonal)) if diagonal.size else 0.0, 1.0)
    keep = diagonal > float(rank_tol) * scale
    basis = np.asarray(q[:, keep], dtype=np.float64)
    if basis.ndim != 2:
        basis = basis.reshape(n, -1)
    if basis.size and not np.all(np.isfinite(basis)):
        raise ValueError("semantic coarse basis is non-finite")
    return basis, {
        "candidate_block_count": int(len(normalised)),
        "candidate_mode_count": int(len(raw_vectors)),
        "covered_rows": int(np.count_nonzero(covered)),
        "coverage_ratio": float(np.mean(covered)) if n else 0.0,
        "local_mode_counts": local_mode_counts,
        "raw_basis_rank": int(raw.shape[1]),
    }


class SemanticCoarseOperator:
    """一层局部校正加低秩粗校正的全阶线性算子。"""

    def __init__(
        self,
        *,
        matrix: sp.spmatrix,
        local_apply: Callable[[np.ndarray], np.ndarray],
        basis: np.ndarray,
        coarse_matrix: np.ndarray,
        info: Dict[str, Any],
    ) -> None:
        self.matrix = matrix.tocsr().astype(np.float64, copy=False)
        self.local_apply = local_apply
        self.basis = np.asarray(basis, dtype=np.float64)
        self.coarse_matrix = np.asarray(coarse_matrix, dtype=np.float64)
        self.info = info
        self.enabled = True
        self.runtime_fallback_count = 0

    def _fallback(self, local: np.ndarray, reason: str) -> np.ndarray:
        self.enabled = False
        self.runtime_fallback_count += 1
        self.info["runtime_fallback_count"] = int(self.runtime_fallback_count)
        self.info["runtime_fallback_reason"] = str(reason)
        self.info["coarse_enabled_after_guard"] = False
        return local

    def apply(self, vector: np.ndarray) -> np.ndarray:
        vec = np.asarray(vector, dtype=np.float64)
        local_raw = np.asarray(self.local_apply(vec))
        if np.iscomplexobj(local_raw):
            imag_max = float(np.max(np.abs(local_raw.imag))) if local_raw.size else 0.0
            real_max = float(np.max(np.abs(local_raw.real))) if local_raw.size else 0.0
            if imag_max > 1.0e-12 * max(real_max, 1.0):
                return self._fallback(vec, "local_complex_output")
            local_raw = local_raw.real
        local = np.asarray(local_raw, dtype=np.float64)
        if local.shape != vec.shape or not np.all(np.isfinite(local)):
            return self._fallback(vec, "local_invalid_output")
        if not self.enabled:
            return local
        try:
            defect = vec - np.asarray(self.matrix.dot(local), dtype=np.float64)
            coarse_rhs = self.basis.T.dot(defect)
            correction = np.linalg.solve(self.coarse_matrix, coarse_rhs)
            output = local + self.basis.dot(correction)
            if not np.all(np.isfinite(output)):
                return self._fallback(local, "runtime_nonfinite_output")
            return output
        except Exception as exc:
            return self._fallback(local, f"runtime_coarse_solve:{exc!r}")

    def as_linear_operator(self) -> LinearOperator:
        return LinearOperator(
            self.matrix.shape,
            matvec=self.apply,
            dtype=np.float64,
        )

    def metadata(self) -> Dict[str, Any]:
        self.info["runtime_fallback_count"] = int(self.runtime_fallback_count)
        self.info["coarse_enabled_after_guard"] = bool(self.enabled)
        self.info["basis_storage_bytes"] = int(self.basis.nbytes)
        self.info["coarse_matrix_storage_bytes"] = int(self.coarse_matrix.nbytes)
        self.info["retained_coarse_bytes"] = int(
            self.basis.nbytes + self.coarse_matrix.nbytes
        )
        return self.info


def build_semantic_coarse_operator(
    *,
    matrix: sp.spmatrix,
    local_apply: Callable[[np.ndarray], np.ndarray],
    coarse_blocks: Sequence[Sequence[int]],
    mode_count: int,
    max_condition: float = DEFAULT_COARSE_MAX_CONDITION,
    rank_tol: float = DEFAULT_COARSE_RANK_TOL,
    semantic_mode: str = "semantic_coarse_sparse",
) -> Tuple[LinearOperator, Dict[str, Any]]:
    """构造粗空间算子；守卫失败时返回原局部算子。"""
    csr = matrix.tocsr().astype(np.float64, copy=False)
    n = int(csr.shape[0])
    info: Dict[str, Any] = {
        "mode": str(semantic_mode),
        "basis_method": "local_smallest_right_singular_vectors",
        "mode_count_per_block": int(mode_count),
        "max_condition": float(max_condition),
        "rank_tolerance": float(rank_tol),
        "fallback_reason": None,
        "coarse_enabled_after_guard": False,
        "runtime_fallback_count": 0,
    }
    if n == 0 or csr.shape[0] != csr.shape[1]:
        info["fallback_reason"] = "matrix_not_nonempty_square"
        return LinearOperator(csr.shape, matvec=local_apply, dtype=np.float64), info
    if not np.isfinite(float(max_condition)) or float(max_condition) <= 1.0:
        info["fallback_reason"] = "invalid_max_condition"
        return LinearOperator(csr.shape, matvec=local_apply, dtype=np.float64), info
    try:
        basis, basis_info = _build_local_mode_vectors(
            csr, coarse_blocks, mode_count, rank_tol
        )
        info.update(basis_info)
        if basis.shape[1] == 0:
            raise ValueError("empty_coarse_basis")
        if basis.shape[0] != n or not np.all(np.isfinite(basis)):
            raise ValueError("invalid_coarse_basis")
        gram = basis.T.dot(basis)
        gram_condition = float(np.linalg.cond(gram))
        info["basis_gram_condition"] = gram_condition
        if not np.isfinite(gram_condition) or gram_condition > float(max_condition):
            raise ValueError(f"coarse_basis_condition:{gram_condition}")
        coarse_matrix = np.asarray(basis.T.dot(csr.dot(basis)), dtype=np.float64)
        if coarse_matrix.ndim != 2 or coarse_matrix.shape[0] != coarse_matrix.shape[1]:
            raise ValueError("coarse_matrix_not_square")
        if not np.all(np.isfinite(coarse_matrix)):
            raise ValueError("coarse_matrix_nonfinite")
        coarse_condition = float(np.linalg.cond(coarse_matrix))
        info["coarse_rank"] = int(basis.shape[1])
        info["coarse_condition"] = coarse_condition
        if not np.isfinite(coarse_condition) or coarse_condition > float(max_condition):
            raise ValueError(f"coarse_matrix_condition:{coarse_condition}")
        np.linalg.solve(coarse_matrix, np.zeros(basis.shape[1], dtype=np.float64))
        state = SemanticCoarseOperator(
            matrix=csr,
            local_apply=local_apply,
            basis=basis,
            coarse_matrix=coarse_matrix,
            info=info,
        )
        info["coarse_enabled_after_guard"] = True
        info["fallback_reason"] = None
        info["basis_shape"] = [int(value) for value in basis.shape]
        info["coarse_matrix_shape"] = [int(value) for value in coarse_matrix.shape]
        info["operator_state"] = state
        return state.as_linear_operator(), info
    except Exception as exc:
        info["fallback_reason"] = repr(exc)
        info["coarse_rank"] = 0
        info["coarse_condition"] = None
        return LinearOperator(csr.shape, matvec=local_apply, dtype=np.float64), info


__all__ = [
    "DEFAULT_COARSE_MAX_CONDITION",
    "DEFAULT_COARSE_RANK_TOL",
    "SemanticCoarseOperator",
    "build_semantic_coarse_operator",
]
