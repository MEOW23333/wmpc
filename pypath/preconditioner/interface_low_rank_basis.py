"""接口低秩基与伍德伯里校正的统一实现。

该模块只处理接口自由度上的小规模基构造和低秩校正，不改变全阶线性
方程。所有基都经过有限性、正交性、秩和条件数守卫；守卫失败时可回退
到常数基。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


DEFAULT_RANK_TOL = 1.0e-10
DEFAULT_MAX_CONDITION = 1.0e12


def _dense_real(value: Any, *, name: str) -> np.ndarray:
    if sp.issparse(value):
        array = value.toarray()
    else:
        array = np.asarray(value)
    if np.iscomplexobj(array):
        imag = float(np.max(np.abs(np.imag(array)))) if array.size else 0.0
        real = float(np.max(np.abs(np.real(array)))) if array.size else 0.0
        if imag > 1.0e-12 * max(real, 1.0):
            raise ValueError(f"{name}_has_non_negligible_imaginary_part")
        array = np.real(array)
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name}_must_be_square")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}_contains_nonfinite_values")
    return array


def _dimension(
    interface_matrix: Optional[Any],
    graph_weights: Optional[Any],
    snapshots: Optional[Any],
) -> int:
    for value in (interface_matrix, graph_weights):
        if value is not None:
            array = np.asarray(value.toarray() if sp.issparse(value) else value)
            if array.ndim != 2 or array.shape[0] != array.shape[1]:
                raise ValueError("interface_input_must_be_square")
            return int(array.shape[0])
    if snapshots is not None:
        array = np.asarray(snapshots)
        if array.ndim != 2:
            raise ValueError("snapshots_must_be_two_dimensional")
        return int(array.shape[0])
    raise ValueError("basis_dimension_is_unknown")


def _orthonormalize(
    raw: np.ndarray,
    *,
    requested_rank: int,
    rank_tol: float,
) -> Tuple[np.ndarray, int]:
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError("raw_basis_must_be_two_dimensional")
    n = int(raw.shape[0])
    if raw.shape[1] == 0:
        return np.zeros((n, 0), dtype=np.float64), 0
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw_basis_contains_nonfinite_values")
    q, r = np.linalg.qr(raw, mode="reduced")
    diagonal = np.abs(np.diag(r)) if r.size else np.zeros(0, dtype=np.float64)
    scale = max(float(np.max(diagonal)) if diagonal.size else 0.0, 1.0)
    keep = diagonal > float(rank_tol) * scale
    basis = np.asarray(q[:, keep], dtype=np.float64)
    rank = min(max(int(requested_rank), 0), int(basis.shape[1]))
    return basis[:, :rank], int(rank)


def _normalise_weights(
    weights: Optional[Any],
    n: int,
    *,
    eps: float,
) -> np.ndarray:
    if weights is None:
        return np.zeros((n, n), dtype=np.float64)
    array = _dense_real(weights, name="graph_weights")
    if array.shape != (n, n):
        raise ValueError("graph_weights_dimension_mismatch")
    array = 0.5 * (np.abs(array) + np.abs(array.T))
    np.fill_diagonal(array, 0.0)
    array[array <= float(eps)] = 0.0
    return array


def _component_labels(weights: np.ndarray, eps: float) -> np.ndarray:
    n = int(weights.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    adjacency = sp.csr_matrix(weights > float(eps), dtype=np.int8)
    _, labels = connected_components(adjacency, directed=False, return_labels=True)
    return np.asarray(labels, dtype=np.int64)


def _constant_raw(weights: np.ndarray, eps: float) -> np.ndarray:
    n = int(weights.shape[0])
    labels = _component_labels(weights, eps)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    columns = []
    for component in sorted(set(int(x) for x in labels.tolist())):
        rows = np.flatnonzero(labels == component)
        vector = np.zeros(n, dtype=np.float64)
        vector[rows] = 1.0 / np.sqrt(float(max(rows.size, 1)))
        columns.append(vector)
    return np.column_stack(columns) if columns else np.zeros((n, 0), dtype=np.float64)


def _graph_laplacian_raw(
    weights: np.ndarray,
    requested_rank: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(weights.shape[0])
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64), np.zeros(0, dtype=np.float64)
    degree = np.sum(weights, axis=1)
    laplacian = np.diag(degree) - weights
    values, vectors = np.linalg.eigh(0.5 * (laplacian + laplacian.T))
    order = np.argsort(np.asarray(values, dtype=np.float64))
    take = min(max(int(requested_rank), 1), n)
    selected = order[:take]
    return np.asarray(vectors[:, selected], dtype=np.float64), np.asarray(values[selected], dtype=np.float64)


def _snapshot_raw(
    snapshots: Any,
    requested_rank: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    array = np.asarray(snapshots)
    if array.ndim != 2:
        raise ValueError("snapshots_must_be_two_dimensional")
    if np.iscomplexobj(array):
        imag = float(np.max(np.abs(array.imag))) if array.size else 0.0
        real = float(np.max(np.abs(array.real))) if array.size else 0.0
        if imag > 1.0e-12 * max(real, 1.0):
            raise ValueError("snapshots_have_non_negligible_imaginary_part")
        array = np.real(array)
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("snapshots_contain_nonfinite_values")
    if array.shape[1] == 0 or float(np.linalg.norm(array)) <= 0.0:
        raise ValueError("snapshots_are_empty_or_zero")
    u, singular_values, _ = np.linalg.svd(array, full_matrices=False)
    take = min(max(int(requested_rank), 1), int(u.shape[1]))
    selected = np.asarray(u[:, :take], dtype=np.float64)
    energy = np.asarray(singular_values, dtype=np.float64) ** 2
    retained = float(np.sum(energy[:take]) / max(float(np.sum(energy)), 1.0e-30))
    return selected, np.asarray(singular_values, dtype=np.float64), retained


def _schur_raw(
    interface_matrix: Any,
    requested_rank: int,
) -> Tuple[np.ndarray, np.ndarray]:
    matrix = _dense_real(interface_matrix, name="interface_matrix")
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    order = np.argsort(np.abs(values))
    take = min(max(int(requested_rank), 1), int(vectors.shape[1]))
    selected = order[:take]
    return np.asarray(vectors[:, selected], dtype=np.float64), np.asarray(values[selected], dtype=np.float64)


def build_interface_basis(
    method: str,
    *,
    interface_matrix: Optional[Any] = None,
    graph_weights: Optional[Any] = None,
    snapshots: Optional[Any] = None,
    requested_rank: int = 1,
    rank_tol: float = DEFAULT_RANK_TOL,
    max_condition: float = DEFAULT_MAX_CONDITION,
    fallback_method: str = "constant",
    eps: float = 1.0e-14,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """构造接口低秩基并返回基矩阵和可审计元数据。"""

    method = str(method)
    requested_rank = max(int(requested_rank), 0)
    n = _dimension(interface_matrix, graph_weights, snapshots)
    info: Dict[str, Any] = {
        "method": method,
        "requested_rank": requested_rank,
        "actual_rank": 0,
        "rank_tolerance": float(rank_tol),
        "max_condition": float(max_condition),
        "fallback_method": None,
        "fallback_reason": None,
        "basis_condition": None,
        "basis_storage_bytes": 0,
        "component_count": None,
        "graph_edge_count": None,
        "laplacian_eigenvalues": [],
        "schur_eigenvalues": [],
        "snapshot_count": 0,
        "snapshot_singular_values": [],
        "snapshot_energy_retained": None,
    }
    if n == 0 or requested_rank == 0:
        info["fallback_reason"] = "empty_dimension_or_zero_requested_rank"
        return np.zeros((n, 0), dtype=np.float64), info
    if not np.isfinite(float(max_condition)) or float(max_condition) <= 1.0:
        info["fallback_reason"] = "invalid_condition_limit"
        return np.zeros((n, 0), dtype=np.float64), info

    try:
        weights = _normalise_weights(graph_weights, n, eps=eps)
        info["component_count"] = int(
            len(set(_component_labels(weights, eps).tolist()))
        )
        info["graph_edge_count"] = int(np.count_nonzero(np.triu(weights > eps, 1)))
        if method == "constant":
            raw = _constant_raw(weights, eps)
            raw = raw[:, : min(requested_rank, raw.shape[1])]
        elif method == "graph_laplacian":
            raw, eigenvalues = _graph_laplacian_raw(weights, requested_rank)
            info["laplacian_eigenvalues"] = [float(x) for x in eigenvalues.tolist()]
        elif method in {"schur_slow_eig", "interface_schur"}:
            if interface_matrix is None:
                raise ValueError("schur_basis_requires_interface_matrix")
            raw, eigenvalues = _schur_raw(interface_matrix, requested_rank)
            info["schur_eigenvalues"] = [float(x) for x in eigenvalues.tolist()]
        elif method in {"snapshot_pod", "pod"}:
            if snapshots is None:
                raise ValueError("snapshot_pod_requires_snapshots")
            raw, singular_values, retained = _snapshot_raw(snapshots, requested_rank)
            info["snapshot_count"] = int(np.asarray(snapshots).shape[1])
            info["snapshot_singular_values"] = [
                float(x) for x in singular_values.tolist()
            ]
            info["snapshot_energy_retained"] = float(retained)
        else:
            raise ValueError(f"unsupported_basis_method:{method}")

        basis, actual_rank = _orthonormalize(
            raw, requested_rank=requested_rank, rank_tol=rank_tol
        )
        if actual_rank <= 0:
            raise ValueError("empty_basis_after_rank_guard")
        gram = basis.T.dot(basis)
        condition = float(np.linalg.cond(gram))
        info["basis_condition"] = condition
        if not np.isfinite(condition) or condition > float(max_condition):
            raise ValueError(f"basis_condition_exceeded:{condition}")
        if not np.all(np.isfinite(basis)):
            raise ValueError("basis_contains_nonfinite_values")
        info["actual_rank"] = int(actual_rank)
        info["basis_shape"] = [int(x) for x in basis.shape]
        info["basis_storage_bytes"] = int(basis.nbytes)
        return basis, info
    except Exception as exc:
        info["fallback_reason"] = repr(exc)
        if (
            fallback_method
            and str(fallback_method) != method
            and str(fallback_method) != "none"
        ):
            fallback_basis, fallback_info = build_interface_basis(
                str(fallback_method),
                interface_matrix=interface_matrix,
                graph_weights=graph_weights,
                snapshots=None,
                requested_rank=requested_rank,
                rank_tol=rank_tol,
                max_condition=max_condition,
                fallback_method="none",
                eps=eps,
            )
            fallback_info["method_requested"] = method
            fallback_info["fallback_method"] = str(fallback_method)
            fallback_info["fallback_reason"] = info["fallback_reason"]
            fallback_info["fallback_used"] = True
            return fallback_basis, fallback_info
        return np.zeros((n, 0), dtype=np.float64), info


def build_exact_interface_schur(
    matrix: Any,
    core_indices: np.ndarray,
    interface_indices: np.ndarray,
) -> np.ndarray:
    """在小规模接口上构造全阶精确舒尔矩阵。"""

    dense = _dense_real(matrix, name="matrix")
    core = np.asarray(core_indices, dtype=np.int64)
    interface = np.asarray(interface_indices, dtype=np.int64)
    if core.size and (core.min() < 0 or core.max() >= dense.shape[0]):
        raise ValueError("core_indices_out_of_range")
    if interface.size and (interface.min() < 0 or interface.max() >= dense.shape[0]):
        raise ValueError("interface_indices_out_of_range")
    aii = dense[np.ix_(interface, interface)]
    if core.size == 0 or interface.size == 0:
        return np.asarray(aii, dtype=np.float64)
    acc = dense[np.ix_(core, core)]
    aci = dense[np.ix_(core, interface)]
    aic = dense[np.ix_(interface, core)]
    solved = np.linalg.solve(acc, aci)
    return np.asarray(aii - aic.dot(solved), dtype=np.float64)


class WoodburyLowRankCorrector:
    r"""以基于 \(P\) 的逆作用实现 \(P+ZKZ^T\) 的稳定小矩阵校正。"""

    def __init__(
        self,
        *,
        base_matrix: Any,
        target_matrix: Any,
        basis: np.ndarray,
        base_solve: Callable[[np.ndarray], np.ndarray],
        max_condition: float = DEFAULT_MAX_CONDITION,
        eps: float = 1.0e-14,
    ) -> None:
        self.base_matrix = _dense_real(base_matrix, name="base_matrix")
        self.target_matrix = _dense_real(target_matrix, name="target_matrix")
        self.basis = np.asarray(basis, dtype=np.float64)
        self.base_solve = base_solve
        self.max_condition = float(max_condition)
        self.eps = float(eps)
        self.enabled = False
        self.fallback_reason: Optional[str] = None
        self.runtime_fallback_count = 0
        self.correction_matrix = np.zeros((0, 0), dtype=np.float64)
        self.preconditioned_basis = np.zeros((self.base_matrix.shape[0], 0), dtype=np.float64)
        self.small_matrix = np.zeros((0, 0), dtype=np.float64)
        self.small_condition: Optional[float] = None
        self._build()

    def _build(self) -> None:
        n = int(self.base_matrix.shape[0])
        if self.target_matrix.shape != (n, n):
            self.fallback_reason = "target_dimension_mismatch"
            return
        if self.basis.ndim != 2 or self.basis.shape[0] != n or self.basis.shape[1] == 0:
            self.fallback_reason = "empty_or_invalid_basis"
            return
        try:
            if not np.all(np.isfinite(self.basis)):
                raise ValueError("basis_nonfinite")
            correction = self.target_matrix - self.base_matrix
            self.correction_matrix = self.basis.T.dot(correction).dot(self.basis)
            columns = [
                np.asarray(self.base_solve(self.basis[:, idx]), dtype=np.float64)
                for idx in range(self.basis.shape[1])
            ]
            self.preconditioned_basis = np.column_stack(columns)
            gram = self.basis.T.dot(self.preconditioned_basis)
            self.small_matrix = np.eye(self.basis.shape[1]) + self.correction_matrix.dot(gram)
            self.small_condition = float(np.linalg.cond(self.small_matrix))
            if not np.isfinite(self.small_condition) or self.small_condition > self.max_condition:
                raise ValueError(f"small_matrix_condition:{self.small_condition}")
            np.linalg.solve(
                self.small_matrix,
                np.zeros(self.small_matrix.shape[0], dtype=np.float64),
            )
            self.enabled = True
        except Exception as exc:
            self.fallback_reason = repr(exc)
            self.enabled = False

    def apply(self, vector: np.ndarray) -> np.ndarray:
        raw = np.asarray(vector, dtype=np.float64)
        if not self.enabled:
            return np.asarray(self.base_solve(raw), dtype=np.float64)
        try:
            y = np.asarray(self.base_solve(raw), dtype=np.float64)
            rhs_small = self.correction_matrix.dot(self.basis.T.dot(y))
            coeff = np.linalg.solve(self.small_matrix, rhs_small)
            output = y - self.preconditioned_basis.dot(coeff)
            if not np.all(np.isfinite(output)):
                raise ValueError("runtime_nonfinite_output")
            return output
        except Exception as exc:
            self.runtime_fallback_count += 1
            self.fallback_reason = f"runtime:{exc!r}"
            return np.asarray(self.base_solve(raw), dtype=np.float64)

    def metadata(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "fallback_reason": self.fallback_reason,
            "runtime_fallback_count": int(self.runtime_fallback_count),
            "basis_rank": int(self.basis.shape[1]) if self.basis.ndim == 2 else 0,
            "basis_storage_bytes": int(self.basis.nbytes),
            "correction_storage_bytes": int(self.correction_matrix.nbytes),
            "preconditioned_basis_storage_bytes": int(self.preconditioned_basis.nbytes),
            "small_matrix_storage_bytes": int(self.small_matrix.nbytes),
            "retained_low_rank_bytes": int(
                self.basis.nbytes
                + self.correction_matrix.nbytes
                + self.preconditioned_basis.nbytes
                + self.small_matrix.nbytes
            ),
            "small_matrix_condition": self.small_condition,
        }


class InterfaceLowRankSchurPreconditioner:
    """在现有稀疏局部舒尔预条件子上增加接口低秩校正。"""

    def __init__(
        self,
        *,
        base: Any,
        method: str,
        requested_rank: int,
        snapshots: Optional[Any] = None,
        exact_target: Optional[Any] = None,
        max_condition: float = DEFAULT_MAX_CONDITION,
        rank_tol: float = DEFAULT_RANK_TOL,
    ) -> None:
        self.base = base
        self.method = str(method)
        self.requested_rank = int(requested_rank)
        self.max_condition = float(max_condition)
        self.rank_tol = float(rank_tol)
        self.interface_rows = np.asarray(base.interface_rows, dtype=np.int64)
        self.interface_count = int(self.interface_rows.size)
        self.target_fallback_reason: Optional[str] = None
        p_dense = _dense_real(base.schur_matrix, name="base_schur")
        self.base_schur = p_dense
        target = p_dense
        if exact_target is not None:
            try:
                target = _dense_real(exact_target, name="exact_schur_target")
                if target.shape != p_dense.shape:
                    raise ValueError("exact_target_dimension_mismatch")
            except Exception as exc:
                self.target_fallback_reason = repr(exc)
                target = p_dense
        self.target_schur = target
        graph_weights = np.abs(p_dense)
        if graph_weights.size:
            diagonal = np.maximum(np.abs(np.diag(p_dense)), 1.0e-30)
            scale = np.sqrt(diagonal[:, None] * diagonal[None, :])
            graph_weights = 0.5 * (graph_weights + graph_weights.T) / scale
            np.fill_diagonal(graph_weights, 0.0)
        self.basis, self.basis_info = build_interface_basis(
            self._basis_method(method),
            interface_matrix=target,
            graph_weights=graph_weights,
            snapshots=snapshots,
            requested_rank=self.requested_rank,
            rank_tol=self.rank_tol,
            max_condition=self.max_condition,
            fallback_method="constant",
        )
        self.corrector = WoodburyLowRankCorrector(
            base_matrix=p_dense,
            target_matrix=target,
            basis=self.basis,
            base_solve=self._base_interface_solve,
            max_condition=self.max_condition,
        )

    @staticmethod
    def _basis_method(method: str) -> str:
        aliases = {
            "constant": "constant",
            "graph_laplacian": "graph_laplacian",
            "snapshot_pod": "snapshot_pod",
            "pod": "snapshot_pod",
            "schur_slow_eig": "schur_slow_eig",
            "interface_schur": "schur_slow_eig",
        }
        return aliases.get(str(method), str(method))

    def _base_interface_solve(self, rhs: np.ndarray) -> np.ndarray:
        value = self.base._solve_interface(
            np.asarray(rhs, dtype=np.complex128),
            factor=self.base.schur_factor,
            factor_mode=self.base.schur_factor_mode,
            schur_matrix=self.base.schur_matrix,
        )
        value = np.asarray(value)
        if np.iscomplexobj(value):
            imag = float(np.max(np.abs(value.imag))) if value.size else 0.0
            real = float(np.max(np.abs(value.real))) if value.size else 0.0
            if imag > 1.0e-10 * max(real, 1.0):
                raise ValueError("base_interface_solve_is_complex")
            value = value.real
        return np.asarray(value, dtype=np.float64)

    def apply(self, vec: np.ndarray) -> np.ndarray:
        vector = np.asarray(vec, dtype=np.complex128)
        out = np.zeros_like(vector)
        core_solution = self.base.core.apply_core_only(vector)
        out[self.base.core.covered_mask] = core_solution[self.base.core.covered_mask]
        if self.interface_count:
            rhs = vector[self.interface_rows] - self.base.matrix[self.interface_rows, :].dot(core_solution)
            rhs_real = np.asarray(np.real(rhs), dtype=np.float64)
            interface_solution = self.corrector.apply(rhs_real).astype(np.complex128)
            out[self.interface_rows] = interface_solution
            for rows, core_factor in zip(self.base.core.blocks, self.base.core.factor_solvers):
                correction_rhs = self.base.matrix[rows, :][:, self.interface_rows].dot(interface_solution)
                if np.any(np.abs(correction_rhs) > self.base.eps):
                    out[rows] = out[rows] - core_factor.dot(correction_rhs)
        out[self.base.uncovered_mask] = (
            self.base.core.uncovered_scales[self.base.uncovered_mask]
            * vector[self.base.uncovered_mask]
        )
        return out

    def metadata(self) -> Dict[str, Any]:
        return {
            "preconditioner_mode": "interface_low_rank_" + self.method,
            "basis_method": self.method,
            "basis": dict(self.basis_info),
            "corrector": self.corrector.metadata(),
            "target_schur_source": "exact_full_core_schur"
            if self.target_fallback_reason is None and self.target_schur is not self.base_schur
            else "base_sparse_schur",
            "target_fallback_reason": self.target_fallback_reason,
            "interface_rows": int(self.interface_count),
            "requested_rank": int(self.requested_rank),
            "actual_rank": int(self.basis.shape[1]),
        }


__all__ = [
    "DEFAULT_MAX_CONDITION",
    "DEFAULT_RANK_TOL",
    "InterfaceLowRankSchurPreconditioner",
    "WoodburyLowRankCorrector",
    "build_exact_interface_schur",
    "build_interface_basis",
]
