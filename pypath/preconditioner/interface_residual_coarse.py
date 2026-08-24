"""Residual-driven reduced coarse correction for interface Schur systems.

The original full-order matrix remains unchanged. A reduced trial space is
used only to correct the residual left by the existing sparse interface
preconditioner. All construction and runtime failures fall back to the base
interface solve.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from pypath.preconditioner.interface_low_rank_basis import (
    DEFAULT_MAX_CONDITION,
    DEFAULT_RANK_TOL,
    build_interface_basis,
)


def _as_complex_matrix(value: Any, *, rows: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or int(array.shape[0]) != int(rows):
        raise ValueError(f"{name}_shape_mismatch")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}_contains_nonfinite_values")
    return array


def _orthonormalize(
    value: Any,
    *,
    rows: int,
    rank_tol: float,
    name: str,
) -> np.ndarray:
    matrix = _as_complex_matrix(value, rows=rows, name=name)
    if matrix.shape[1] == 0:
        return np.zeros((rows, 0), dtype=np.complex128)
    q, r = np.linalg.qr(matrix, mode="reduced")
    diagonal = np.abs(np.diag(r)) if r.size else np.zeros(0)
    scale = max(float(np.max(diagonal)) if diagonal.size else 0.0, 1.0)
    keep = diagonal > float(rank_tol) * scale
    return np.asarray(q[:, keep], dtype=np.complex128)


class ResidualCoarseCorrector:
    """Apply a reduced correction to the residual left by a base solve."""

    def __init__(
        self,
        *,
        schur_apply: Callable[[np.ndarray], np.ndarray],
        base_solve: Callable[[np.ndarray], np.ndarray],
        basis: np.ndarray,
        test_space: str = "range_action",
        max_condition: float = DEFAULT_MAX_CONDITION,
        rank_tol: float = DEFAULT_RANK_TOL,
    ) -> None:
        self.schur_apply = schur_apply
        self.base_solve = base_solve
        raw_basis = np.asarray(basis)
        self.interface_count = (
            int(raw_basis.shape[0]) if raw_basis.ndim == 2 else 0
        )
        self.basis = np.asarray(raw_basis, dtype=np.complex128)
        self.test_space_mode = str(test_space)
        self.max_condition = float(max_condition)
        self.rank_tol = float(rank_tol)
        self.schur_basis = np.zeros(
            (self.interface_count, 0), dtype=np.complex128
        )
        self.test_basis = np.zeros(
            (self.interface_count, 0), dtype=np.complex128
        )
        self.reduced_matrix = np.zeros((0, 0), dtype=np.complex128)
        self.basis_condition: Optional[float] = None
        self.test_basis_condition: Optional[float] = None
        self.reduced_condition: Optional[float] = None
        self.enabled = False
        self.fallback_reason: Optional[str] = None
        self.runtime_fallback_count = 0
        self.apply_count = 0
        self._build()

    def _build(self) -> None:
        try:
            if (
                self.basis.ndim != 2
                or self.basis.shape[0] <= 0
                or self.basis.shape[1] <= 0
            ):
                raise ValueError("empty_or_invalid_basis")
            if not np.isfinite(self.max_condition) or self.max_condition <= 1.0:
                raise ValueError("invalid_max_condition")
            orthogonal = _orthonormalize(
                self.basis,
                rows=self.interface_count,
                rank_tol=self.rank_tol,
                name="basis",
            )
            if orthogonal.shape[1] != self.basis.shape[1]:
                raise ValueError("basis_rank_loss")
            self.basis = orthogonal
            basis_gram = self.basis.conj().T.dot(self.basis)
            self.basis_condition = float(np.linalg.cond(basis_gram))
            if (
                not np.isfinite(self.basis_condition)
                or self.basis_condition > self.max_condition
            ):
                raise ValueError(
                    f"basis_condition_exceeded:{self.basis_condition}"
                )

            self.schur_basis = _as_complex_matrix(
                self.schur_apply(self.basis),
                rows=self.interface_count,
                name="schur_basis",
            )
            if self.schur_basis.shape != self.basis.shape:
                raise ValueError("schur_basis_shape_mismatch")

            if self.test_space_mode in {"range_action", "petrov_galerkin"}:
                self.test_basis = _orthonormalize(
                    self.schur_basis,
                    rows=self.interface_count,
                    rank_tol=self.rank_tol,
                    name="schur_basis",
                )
                if self.test_basis.shape[1] != self.basis.shape[1]:
                    raise ValueError("test_basis_rank_loss")
            elif self.test_space_mode == "galerkin":
                self.test_basis = self.basis.copy()
            else:
                raise ValueError(
                    f"unsupported_test_space:{self.test_space_mode}"
                )

            test_gram = self.test_basis.conj().T.dot(self.test_basis)
            self.test_basis_condition = float(np.linalg.cond(test_gram))
            if (
                not np.isfinite(self.test_basis_condition)
                or self.test_basis_condition > self.max_condition
            ):
                raise ValueError(
                    "test_basis_condition_exceeded:"
                    f"{self.test_basis_condition}"
                )

            self.reduced_matrix = self.test_basis.conj().T.dot(
                self.schur_basis
            )
            self.reduced_condition = float(
                np.linalg.cond(self.reduced_matrix)
            )
            if (
                not np.isfinite(self.reduced_condition)
                or self.reduced_condition > self.max_condition
            ):
                raise ValueError(
                    f"reduced_condition_exceeded:{self.reduced_condition}"
                )
            np.linalg.solve(
                self.reduced_matrix,
                np.zeros(self.reduced_matrix.shape[0], dtype=np.complex128),
            )
            self.enabled = True
        except Exception as exc:
            self.enabled = False
            self.fallback_reason = repr(exc)

    def disable(self, reason: str) -> None:
        self.enabled = False
        self.fallback_reason = str(reason)

    def _base(self, rhs: np.ndarray) -> np.ndarray:
        vector = np.asarray(rhs, dtype=np.complex128).reshape(-1)
        value = np.asarray(self.base_solve(vector), dtype=np.complex128)
        if value.shape != vector.shape or not np.all(np.isfinite(value)):
            raise ValueError("base_solve_invalid_output")
        return value

    def apply(self, rhs: np.ndarray) -> np.ndarray:
        vector = np.asarray(rhs, dtype=np.complex128).reshape(-1)
        if vector.shape[0] != self.interface_count:
            raise ValueError("interface_rhs_dimension_mismatch")
        if not self.enabled:
            return self._base(vector)
        try:
            base = self._base(vector)
            defect = vector - np.asarray(
                self.schur_apply(base), dtype=np.complex128
            ).reshape(-1)
            reduced_rhs = self.test_basis.conj().T.dot(defect)
            coefficients = np.linalg.solve(
                self.reduced_matrix, reduced_rhs
            )
            output = base + self.basis.dot(coefficients)
            if output.shape != vector.shape or not np.all(np.isfinite(output)):
                raise ValueError("runtime_nonfinite_or_invalid_output")
            self.apply_count += 1
            return np.asarray(output, dtype=np.complex128)
        except Exception as exc:
            self.runtime_fallback_count += 1
            self.fallback_reason = f"runtime:{exc!r}"
            return self._base(vector)

    def metadata(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "test_space": self.test_space_mode,
            "basis_rank": (
                int(self.basis.shape[1]) if self.basis.ndim == 2 else 0
            ),
            "basis_condition": self.basis_condition,
            "test_basis_condition": self.test_basis_condition,
            "reduced_condition": self.reduced_condition,
            "fallback_reason": self.fallback_reason,
            "runtime_fallback_count": int(self.runtime_fallback_count),
            "apply_count": int(self.apply_count),
            "basis_storage_bytes": int(self.basis.nbytes),
            "schur_basis_storage_bytes": int(self.schur_basis.nbytes),
            "test_basis_storage_bytes": int(self.test_basis.nbytes),
            "reduced_matrix_storage_bytes": int(self.reduced_matrix.nbytes),
            "retained_coarse_bytes": int(
                self.basis.nbytes
                + self.schur_basis.nbytes
                + self.test_basis.nbytes
                + self.reduced_matrix.nbytes
            ),
        }


class InterfaceResidualCoarsePreconditioner:
    """Full-space wrapper around the residual-driven interface correction."""

    def __init__(
        self,
        *,
        base: Any,
        method: str,
        requested_rank: int,
        snapshots: Optional[Any] = None,
        basis_target: Optional[Any] = None,
        test_space: str = "range_action",
        guard_vector: Optional[Any] = None,
        guard_tolerance: float = 1.0e-10,
        max_condition: float = DEFAULT_MAX_CONDITION,
        rank_tol: float = DEFAULT_RANK_TOL,
    ) -> None:
        self.base = base
        self.method = str(method)
        self.requested_rank = int(requested_rank)
        self.test_space = str(test_space)
        self.guard_tolerance = float(guard_tolerance)
        self.interface_rows = np.asarray(base.interface_rows, dtype=np.int64)
        self.interface_count = int(self.interface_rows.size)
        p_dense = np.asarray(base.schur_matrix.toarray())
        target = p_dense if basis_target is None else basis_target
        graph_weights = np.abs(p_dense)
        if graph_weights.size:
            diagonal = np.maximum(np.abs(np.diag(p_dense)), 1.0e-30)
            scale = np.sqrt(diagonal[:, None] * diagonal[None, :])
            graph_weights = (
                0.5 * (graph_weights + graph_weights.T.conj()) / scale
            )
            np.fill_diagonal(graph_weights, 0.0)
        self.basis, self.basis_info = build_interface_basis(
            self._basis_method(self.method),
            interface_matrix=target,
            graph_weights=graph_weights,
            snapshots=snapshots,
            requested_rank=self.requested_rank,
            rank_tol=float(rank_tol),
            max_condition=float(max_condition),
            fallback_method="constant",
        )
        self.corrector = ResidualCoarseCorrector(
            schur_apply=self.base.apply_interface_schur,
            base_solve=self._base_interface_solve,
            basis=self.basis,
            test_space=self.test_space,
            max_condition=float(max_condition),
            rank_tol=float(rank_tol),
        )
        self.guard_info: Dict[str, Any] = {
            "configured": guard_vector is not None,
            "accepted": None,
            "base_residual_norm": None,
            "candidate_residual_norm": None,
            "candidate_to_base_ratio": None,
            "tolerance": self.guard_tolerance,
            "reason": None,
        }
        self._run_guard(guard_vector)

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
        return np.asarray(
            self.base._solve_interface(
                np.asarray(rhs, dtype=np.complex128),
                factor=self.base.schur_factor,
                factor_mode=self.base.schur_factor_mode,
                schur_matrix=self.base.schur_matrix,
            ),
            dtype=np.complex128,
        )

    def _apply_impl(self, vec: np.ndarray) -> np.ndarray:
        vector = np.asarray(vec, dtype=np.complex128).reshape(-1)
        out = np.zeros_like(vector)
        core_solution = self.base.core.apply_core_only(vector)
        out[self.base.core.covered_mask] = core_solution[
            self.base.core.covered_mask
        ]
        if self.interface_count:
            interface_rhs = (
                vector[self.interface_rows]
                - self.base.matrix[self.interface_rows, :].dot(core_solution)
            )
            interface_solution = self.corrector.apply(interface_rhs)
            out[self.interface_rows] = interface_solution
            for rows, core_factor in zip(
                self.base.core.blocks, self.base.core.factor_solvers
            ):
                correction_rhs = self.base.matrix[
                    rows, :
                ][:, self.interface_rows].dot(interface_solution)
                if np.any(np.abs(correction_rhs) > self.base.eps):
                    out[rows] = out[rows] - core_factor.dot(correction_rhs)
        out[self.base.uncovered_mask] = (
            self.base.core.uncovered_scales[self.base.uncovered_mask]
            * vector[self.base.uncovered_mask]
        )
        return out

    def _run_guard(self, guard_vector: Optional[Any]) -> None:
        if guard_vector is None:
            self.guard_info["reason"] = "guard_vector_not_configured"
            return
        try:
            vector = np.asarray(
                guard_vector, dtype=np.complex128
            ).reshape(-1)
            if vector.shape[0] != int(self.base.matrix.shape[0]):
                raise ValueError("guard_vector_dimension_mismatch")
            base_output = self.base._apply_with_factor(
                vector,
                self.base.schur_factor,
                self.base.schur_factor_mode,
                self.base.schur_matrix,
            )
            candidate_output = self._apply_impl(vector)
            base_residual = vector - self.base.matrix.dot(base_output)
            candidate_residual = (
                vector - self.base.matrix.dot(candidate_output)
            )
            base_norm = float(np.linalg.norm(base_residual))
            candidate_norm = float(np.linalg.norm(candidate_residual))
            if not np.isfinite(base_norm) or not np.isfinite(candidate_norm):
                raise ValueError("guard_residual_nonfinite")
            ratio = float(candidate_norm / max(base_norm, 1.0e-30))
            accepted = candidate_norm <= (
                1.0 + self.guard_tolerance
            ) * base_norm
            self.guard_info.update(
