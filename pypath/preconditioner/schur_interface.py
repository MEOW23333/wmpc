from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import LinearOperator, gmres

from pypath.preconditioner.block_schwarz import BlockSchwarzPlan, build_analytic_scales


@dataclass
class ExplicitSchurInterfacePreconditioner:
    matrix: np.ndarray
    core_plan: BlockSchwarzPlan
    boundary_plan: BlockSchwarzPlan
    uncovered_row_policy: str = "row_sum"
    factor_mode: str = "dense_inv"
    factorize_schur: bool = True

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=np.float64)
        self.interface_mask = np.logical_and(self.boundary_plan.covered_mask, ~self.core_plan.covered_mask)
        self.uncovered_mask = ~np.logical_or(self.core_plan.covered_mask, self.interface_mask)
        self.core_rows = np.flatnonzero(self.core_plan.covered_mask).astype(np.int64)
        self.interface_rows = np.flatnonzero(self.interface_mask).astype(np.int64)
        self.uncovered_scales = build_analytic_scales(self.uncovered_row_policy, self.matrix)
        self.core_inverse_interface = self._build_core_inverse_interface()
        self.schur_matrix = self._build_schur_matrix()
        if self.factorize_schur:
            self.schur_factor, self.factor_mode = self._factorize(self.schur_matrix)
        else:
            self.schur_factor = np.zeros_like(self.schur_matrix)
            self.factor_mode = "not_factorized"

    def _apply_core_only(self, vec: np.ndarray) -> np.ndarray:
        out = np.zeros_like(np.asarray(vec, dtype=np.float64))
        for rows, factor in zip(self.core_plan.blocks, self.core_plan.factor_solvers):
            out[rows] = factor.dot(vec[rows])
        return out

    def _build_core_inverse_interface(self) -> np.ndarray:
        matrix_size = int(self.matrix.shape[0])
        interface_count = int(self.interface_rows.shape[0])
        if interface_count == 0:
            return np.zeros((matrix_size, 0), dtype=np.float64)
        out = np.zeros((matrix_size, interface_count), dtype=np.float64)
        for col_id, row in enumerate(self.interface_rows.tolist()):
            rhs = np.zeros(matrix_size, dtype=np.float64)
            rhs[self.core_rows] = self.matrix[np.ix_(self.core_rows, [int(row)])].reshape(-1)
            out[:, col_id] = self._apply_core_only(rhs)
        return out

    def _build_schur_matrix(self) -> np.ndarray:
        if self.interface_rows.shape[0] == 0:
            return np.zeros((0, 0), dtype=np.float64)
        abb = self.matrix[np.ix_(self.interface_rows, self.interface_rows)]
        abi_core_inv_aib = self.matrix[np.ix_(self.interface_rows, self.core_rows)].dot(
            self.core_inverse_interface[self.core_rows, :]
        )
        return np.asarray(abb - abi_core_inv_aib, dtype=np.float64)

    @staticmethod
    def _factorize(matrix: np.ndarray) -> Tuple[np.ndarray, str]:
        if matrix.shape[0] == 0:
            return np.zeros((0, 0), dtype=np.float64), "empty"
        try:
            return np.linalg.inv(matrix), "dense_inv"
        except np.linalg.LinAlgError:
            return np.linalg.pinv(matrix), "dense_pinv"


    def apply(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float64)
        out = np.zeros_like(vec)
        core_solution = self._apply_core_only(vec)
        out[self.core_rows] = core_solution[self.core_rows]
        if self.interface_rows.shape[0] > 0:
            interface_rhs = vec[self.interface_rows] - self.matrix[np.ix_(self.interface_rows, self.core_rows)].dot(
                core_solution[self.core_rows]
            )
            interface_solution = self.schur_factor.dot(interface_rhs)
            out[self.interface_rows] = interface_solution
            out[self.core_rows] = out[self.core_rows] - self.core_inverse_interface[self.core_rows, :].dot(
                interface_solution
            )
        out[self.uncovered_mask] = self.uncovered_scales[self.uncovered_mask] * vec[self.uncovered_mask]
        return out

    def metadata(self) -> Dict[str, Any]:
        interface_count = int(self.interface_rows.shape[0])
        schur_nnz = int(np.count_nonzero(self.schur_matrix))
        memory_estimate = int(
            self.core_plan.metadata().get("memory_estimate", 0)
            + self.boundary_plan.metadata().get("memory_estimate", 0)
            + self.core_inverse_interface.size * 8
            + self.schur_matrix.size * 8
            + self.schur_factor.size * 8
        )
        return {
            "preconditioner_mode": "explicit_schur_interface",
            "core_block_mode": self.core_plan.block_mode,
            "boundary_block_mode": self.boundary_plan.block_mode,
            "interface_rows": interface_count,
            "core_rows": int(self.core_rows.shape[0]),
            "uncovered_rows": int(np.count_nonzero(self.uncovered_mask)),
            "schur_factor_mode": self.factor_mode,
            "schur_nnz": schur_nnz,
            "schur_density": float(schur_nnz / max(interface_count * interface_count, 1)),
            "memory_estimate": memory_estimate,
            "core": self.core_plan.metadata(),
            "boundary": self.boundary_plan.metadata(),
        }


SCHUR_ROW_FEATURE_DIM = 8


def build_schur_row_features(preconditioner: ExplicitSchurInterfacePreconditioner, eps: float = 1e-30) -> np.ndarray:
    matrix = np.asarray(preconditioner.matrix, dtype=np.float64)
    schur = np.asarray(preconditioner.schur_matrix, dtype=np.float64)
    rows = np.asarray(preconditioner.interface_rows, dtype=np.int64)
    if rows.shape[0] == 0:
        return np.zeros((0, SCHUR_ROW_FEATURE_DIM), dtype=np.float64)
    schur_diag = np.diag(schur)
    schur_row_sum = np.abs(schur).sum(axis=1)
    schur_col_sum = np.abs(schur).sum(axis=0)
    matrix_diag = np.diag(matrix)[rows]
    matrix_row_sum = np.abs(matrix[rows, :]).sum(axis=1)
    matrix_col_sum = np.abs(matrix[:, rows]).sum(axis=0)
    core_coupling = np.linalg.norm(matrix[np.ix_(rows, preconditioner.core_rows)], axis=1)
    schur_nnz = np.count_nonzero(np.abs(schur) > eps, axis=1)
    denom = float(max(int(rows.shape[0]), 1))
    return np.stack(
        [
            _safe_log_abs(schur_diag, eps),
            _safe_log_abs(schur_row_sum, eps),
            _safe_log_abs(schur_col_sum, eps),
            _safe_log_abs(matrix_diag, eps),
            _safe_log_abs(matrix_row_sum, eps),
            _safe_log_abs(matrix_col_sum, eps),
            _safe_log_abs(core_coupling, eps),
            np.asarray(schur_nnz, dtype=np.float64) / denom,
        ],
        axis=1,
    )


def _safe_log_abs(values: np.ndarray, eps: float) -> np.ndarray:
    return np.log(np.maximum(np.abs(np.asarray(values, dtype=np.float64)), float(eps)))


class SchurDiagonalScaleNet(nn.Module):
    def __init__(self, feature_dim: int = SCHUR_ROW_FEATURE_DIM, hidden_dim: int = 32, log_scale_clip: float = 4.0):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.log_scale_clip = float(log_scale_clip)
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.register_buffer("feature_mean", torch.zeros(self.feature_dim, dtype=torch.float64))
        self.register_buffer("feature_std", torch.ones(self.feature_dim, dtype=torch.float64))
        self.double()

    def set_feature_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.feature_mean.copy_(torch.as_tensor(mean, dtype=torch.float64))
        self.feature_std.copy_(torch.as_tensor(std, dtype=torch.float64).clamp_min(1e-12))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_std.clamp_min(1e-12)
        log_scale = torch.clamp(self.net(normalized).squeeze(-1), -self.log_scale_clip, self.log_scale_clip)
        return torch.exp(log_scale)


class LearnedSchurDiagonalPreconditioner:
    def __init__(
        self,
        *,
        matrix: np.ndarray,
        core_plan: BlockSchwarzPlan,
        boundary_plan: BlockSchwarzPlan,
        model: SchurDiagonalScaleNet,
        uncovered_row_policy: str = "row_sum",
        eps: float = 1e-30,
    ):
        self.base = ExplicitSchurInterfacePreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            uncovered_row_policy=uncovered_row_policy,
            factorize_schur=False,
        )
        self.model = model.eval()
        self.eps = float(eps)
        self.features = build_schur_row_features(self.base, eps=self.eps)
        diag = np.diag(self.base.schur_matrix) if self.base.schur_matrix.size else np.zeros(0, dtype=np.float64)
        self.inverse_diagonal = np.sign(diag) / np.maximum(np.abs(diag), self.eps)

    def apply(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float64)
        out = np.zeros_like(vec)
        core_solution = self.base._apply_core_only(vec)
        out[self.base.core_rows] = core_solution[self.base.core_rows]
        if self.base.interface_rows.shape[0] > 0:
            interface_rhs = vec[self.base.interface_rows] - self.base.matrix[np.ix_(self.base.interface_rows, self.base.core_rows)].dot(core_solution[self.base.core_rows])
            with torch.no_grad():
                features_t = torch.as_tensor(self.features, dtype=torch.float64)
                scales = self.model(features_t).detach().cpu().numpy()
            interface_solution = scales * self.inverse_diagonal * interface_rhs
            out[self.base.interface_rows] = interface_solution
            out[self.base.core_rows] = out[self.base.core_rows] - self.base.core_inverse_interface[self.base.core_rows, :].dot(interface_solution)
        out[self.base.uncovered_mask] = self.base.uncovered_scales[self.base.uncovered_mask] * vec[self.base.uncovered_mask]
        return out

    def metadata(self) -> Dict[str, Any]:
        metadata = self.base.metadata()
        metadata["preconditioner_mode"] = "learned_schur_diagonal"
        metadata["schur_factor_mode"] = "learned_diagonal"
        metadata["learned_feature_dim"] = SCHUR_ROW_FEATURE_DIM
        return metadata


def load_learned_schur_diagonal_model(checkpoint_path: str) -> SchurDiagonalScaleNet:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(payload.get("model_config", {}))
    model = SchurDiagonalScaleNet(
        feature_dim=int(config.get("feature_dim", SCHUR_ROW_FEATURE_DIM)),
        hidden_dim=int(config.get("hidden_dim", 32)),
        log_scale_clip=float(config.get("log_scale_clip", 4.0)),
    )
    model.load_state_dict(payload["model_state_dict"])
    if "feature_mean" in payload and "feature_std" in payload:
        model.set_feature_stats(np.asarray(payload["feature_mean"]), np.asarray(payload["feature_std"]))
    model.eval()
    return model


SCHUR_EDGE_FEATURE_DIM = 10


def build_schur_edge_features(
    preconditioner: ExplicitSchurInterfacePreconditioner,
    *,
    candidate_edge_limit: int = 0,
    eps: float = 1e-30,
):
    schur = np.asarray(preconditioner.schur_matrix, dtype=np.float64)
    n = int(schur.shape[0])
    if n <= 1:
        return (
            np.zeros((0, SCHUR_EDGE_FEATURE_DIM), dtype=np.float64),
            np.zeros((0, 2), dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )
    diag = np.diag(schur)
    abs_schur = np.abs(schur)
    row_sum = abs_schur.sum(axis=1)
    col_sum = abs_schur.sum(axis=0)
    items = []
    for i in range(n):
        for j in range(i + 1, n):
            sij = float(schur[i, j])
            sji = float(schur[j, i])
            value = 0.5 * (sij + sji)
            strength = max(abs(sij), abs(sji))
            if strength <= eps:
                continue
            denom = np.sqrt(max(abs(diag[i]) * abs(diag[j]), eps))
            rel = strength / max(float(denom), eps)
            features = [
                _safe_log_abs(np.asarray([strength]), eps)[0],
                np.log(max(rel, eps)),
                _safe_log_abs(np.asarray([diag[i]]), eps)[0],
                _safe_log_abs(np.asarray([diag[j]]), eps)[0],
                _safe_log_abs(np.asarray([row_sum[i]]), eps)[0],
                _safe_log_abs(np.asarray([row_sum[j]]), eps)[0],
                _safe_log_abs(np.asarray([col_sum[i]]), eps)[0],
                _safe_log_abs(np.asarray([col_sum[j]]), eps)[0],
                abs(float(i - j)) / float(max(n - 1, 1)),
                1.0 / float(max(n, 1)),
            ]
            items.append((strength, i, j, value, features))
    items.sort(key=lambda item: item[0], reverse=True)
    if candidate_edge_limit and int(candidate_edge_limit) > 0:
        items = items[: int(candidate_edge_limit)]
    if not items:
        return (
            np.zeros((0, SCHUR_EDGE_FEATURE_DIM), dtype=np.float64),
            np.zeros((0, 2), dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )
    edges = np.asarray([[item[1], item[2]] for item in items], dtype=np.int64)
    values = np.asarray([item[3] for item in items], dtype=np.float64)
    features = np.asarray([item[4] for item in items], dtype=np.float64)
    return features, edges, values


class SchurEdgeGateNet(nn.Module):
    def __init__(self, feature_dim: int = SCHUR_EDGE_FEATURE_DIM, hidden_dim: int = 64, logit_clip: float = 8.0):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.logit_clip = float(logit_clip)
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.register_buffer('feature_mean', torch.zeros(self.feature_dim, dtype=torch.float64))
        self.register_buffer('feature_std', torch.ones(self.feature_dim, dtype=torch.float64))
        self.double()

    def set_feature_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.feature_mean.copy_(torch.as_tensor(mean, dtype=torch.float64))
        self.feature_std.copy_(torch.as_tensor(std, dtype=torch.float64).clamp_min(1e-12))

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_std.clamp_min(1e-12)
        return torch.clamp(self.net(normalized).squeeze(-1), -self.logit_clip, self.logit_clip)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(features))


class LearnedSparseSchurPreconditioner:
    def __init__(
        self,
        *,
        matrix: np.ndarray,
        core_plan: BlockSchwarzPlan,
        boundary_plan: BlockSchwarzPlan,
        model: SchurEdgeGateNet,
        edge_budget: int = 0,
        candidate_edge_limit: int = 0,
        diagonal_shift: float = 1e-8,
        uncovered_row_policy: str = 'row_sum',
        eps: float = 1e-30,
    ):
        self.base = ExplicitSchurInterfacePreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            uncovered_row_policy=uncovered_row_policy,
            factorize_schur=False,
        )
        self.model = model.eval()
        self.edge_budget = int(edge_budget)
        self.candidate_edge_limit = int(candidate_edge_limit)
        self.diagonal_shift = float(diagonal_shift)
        self.eps = float(eps)
        self.features, self.edges, _ = build_schur_edge_features(
            self.base,
            candidate_edge_limit=self.candidate_edge_limit,
            eps=self.eps,
        )
        self.sparse_schur, self.edge_gates, self.selected_edge_indices = self._assemble_sparse_schur()
        self.sparse_factor, self.factor_mode = ExplicitSchurInterfacePreconditioner._factorize(self.sparse_schur)

    def _assemble_sparse_schur(self):
        schur = np.asarray(self.base.schur_matrix, dtype=np.float64)
        n = int(schur.shape[0])
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64), np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
        sparse = np.diag(np.diag(schur)).astype(np.float64)
        if self.features.shape[0] == 0:
            gates = np.zeros(0, dtype=np.float64)
            selected = np.zeros(0, dtype=np.int64)
        else:
            with torch.no_grad():
                features_t = torch.as_tensor(self.features, dtype=torch.float64)
                gates = self.model(features_t).detach().cpu().numpy().astype(np.float64)
            strengths = gates * np.maximum(np.abs(schur[self.edges[:, 0], self.edges[:, 1]]), np.abs(schur[self.edges[:, 1], self.edges[:, 0]]))
            budget = int(self.edge_budget) if int(self.edge_budget) > 0 else int(self.edges.shape[0])
            budget = min(budget, int(self.edges.shape[0]))
            selected = np.argsort(-strengths)[:budget].astype(np.int64)
            for edge_idx in selected.tolist():
                i = int(self.edges[edge_idx, 0])
                j = int(self.edges[edge_idx, 1])
                gate = float(gates[edge_idx])
                sparse[i, j] = gate * schur[i, j]
                sparse[j, i] = gate * schur[j, i]
        if self.diagonal_shift > 0.0:
            offdiag_abs = np.abs(sparse).sum(axis=1) - np.abs(np.diag(sparse))
            diag = np.diag(sparse).copy()
            signs = np.where(diag >= 0.0, 1.0, -1.0)
            sparse[np.diag_indices(n)] = diag + signs * self.diagonal_shift * np.maximum(offdiag_abs, self.eps)
        return sparse, gates, selected

    def apply(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float64)
        out = np.zeros_like(vec)
        core_solution = self.base._apply_core_only(vec)
        out[self.base.core_rows] = core_solution[self.base.core_rows]
        if self.base.interface_rows.shape[0] > 0:
            interface_rhs = vec[self.base.interface_rows] - self.base.matrix[np.ix_(self.base.interface_rows, self.base.core_rows)].dot(core_solution[self.base.core_rows])
            interface_solution = self.sparse_factor.dot(interface_rhs)
            out[self.base.interface_rows] = interface_solution
            out[self.base.core_rows] = out[self.base.core_rows] - self.base.core_inverse_interface[self.base.core_rows, :].dot(interface_solution)
        out[self.base.uncovered_mask] = self.base.uncovered_scales[self.base.uncovered_mask] * vec[self.base.uncovered_mask]
        return out

    def metadata(self) -> Dict[str, Any]:
        metadata = self.base.metadata()
        interface_count = int(self.base.interface_rows.shape[0])
        sparse_nnz = int(np.count_nonzero(self.sparse_schur))
        metadata['preconditioner_mode'] = 'learned_sparse_schur'
        metadata['schur_factor_mode'] = self.factor_mode
        metadata['learned_feature_dim'] = SCHUR_EDGE_FEATURE_DIM
        metadata['candidate_edges'] = int(self.edges.shape[0])
        metadata['selected_edges'] = int(self.selected_edge_indices.shape[0])
        metadata['edge_budget'] = int(self.edge_budget)
        metadata['candidate_edge_limit'] = int(self.candidate_edge_limit)
        metadata['sparse_schur_nnz'] = sparse_nnz
        metadata['sparse_schur_density'] = float(sparse_nnz / max(interface_count * interface_count, 1))
        metadata['diagonal_shift'] = float(self.diagonal_shift)
        metadata['memory_estimate'] = int(metadata.get('memory_estimate', 0) + self.sparse_schur.size * 8 + self.sparse_factor.size * 8)
        return metadata


def load_learned_sparse_schur_model(checkpoint_path: str) -> SchurEdgeGateNet:
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = dict(payload.get('model_config', {}))
    model = SchurEdgeGateNet(
        feature_dim=int(config.get('feature_dim', SCHUR_EDGE_FEATURE_DIM)),
        hidden_dim=int(config.get('hidden_dim', 64)),
        logit_clip=float(config.get('logit_clip', 8.0)),
    )
    model.load_state_dict(payload['model_state_dict'])
    if 'feature_mean' in payload and 'feature_std' in payload:
        model.set_feature_stats(np.asarray(payload['feature_mean']), np.asarray(payload['feature_std']))
    model.eval()
    return model


LOCAL_SCHUR_EDGE_FEATURE_DIM = 14


def _local_schur_budget(interface_count: int, budget_multiplier: float, fallback_edge_budget: int) -> int:
    if float(budget_multiplier) > 0.0:
        return max(0, int(round(float(budget_multiplier) * float(max(int(interface_count), 0)))))
    return max(0, int(fallback_edge_budget))


def build_local_schur_edge_features(
    preconditioner: ExplicitSchurInterfacePreconditioner,
    *,
    candidate_edge_limit: int = 0,
    eps: float = 1e-30,
):
    """Build per-core-block local Schur patch edge candidates.

    Each candidate is produced by one semantic core block k:

        delta_k = -A_BI_k A_kk^{-1} A_I_kB

    Edges are directed local interface entries. Duplicate directed entries from
    different instances remain separate proposer candidates; the aggregator
    sums selected local contributions into one sparse Schur approximation.
    """
    matrix = np.asarray(preconditioner.matrix, dtype=np.float64)
    interface_rows = np.asarray(preconditioner.interface_rows, dtype=np.int64)
    interface_count = int(interface_rows.shape[0])
    if interface_count <= 0:
        return (
            np.zeros((0, LOCAL_SCHUR_EDGE_FEATURE_DIM), dtype=np.float64),
            np.zeros((0, 2), dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            np.zeros((interface_count, interface_count), dtype=np.float64),
            np.zeros((interface_count, interface_count), dtype=np.float64),
        )

    abb = np.asarray(matrix[np.ix_(interface_rows, interface_rows)], dtype=np.float64)
    diag_delta = np.zeros((interface_count, interface_count), dtype=np.float64)
    schur_diag_abs = np.abs(np.diag(preconditioner.schur_matrix)) if preconditioner.schur_matrix.size else np.zeros(interface_count, dtype=np.float64)
    abb_diag_abs = np.abs(np.diag(abb))
    items = []
    for block_id, (core_rows, core_factor) in enumerate(zip(preconditioner.core_plan.blocks, preconditioner.core_plan.factor_solvers)):
        core_rows = np.asarray(core_rows, dtype=np.int64)
        if core_rows.shape[0] == 0:
            continue
        a_ib_full = matrix[np.ix_(core_rows, interface_rows)]
        a_bi_full = matrix[np.ix_(interface_rows, core_rows)]
        active_mask = np.logical_or(
            np.any(np.abs(a_ib_full) > eps, axis=0),
            np.any(np.abs(a_bi_full) > eps, axis=1),
        )
        active = np.flatnonzero(active_mask).astype(np.int64)
        if active.shape[0] == 0:
            continue
        a_ib = a_ib_full[:, active]
        a_bi = a_bi_full[active, :]
        local_delta = -np.asarray(a_bi.dot(core_factor).dot(a_ib), dtype=np.float64)
        for local_i, global_i in enumerate(active.tolist()):
            diag_delta[global_i, global_i] += float(local_delta[local_i, local_i])
        local_abs = np.abs(local_delta)
        local_row_abs = local_abs.sum(axis=1)
        local_col_abs = local_abs.sum(axis=0)
        block_size = float(max(int(core_rows.shape[0]), 1))
        active_count = float(max(int(active.shape[0]), 1))
        for ai in range(int(active.shape[0])):
            for aj in range(int(active.shape[0])):
                if ai == aj:
                    continue
                gi = int(active[ai])
                gj = int(active[aj])
                value = float(local_delta[ai, aj])
                reverse_value = float(local_delta[aj, ai])
                strength = abs(value)
                if strength <= eps:
                    continue
                diag_norm = np.sqrt(max(float(schur_diag_abs[gi] * schur_diag_abs[gj]), eps))
                abb_norm = np.sqrt(max(float(abb_diag_abs[gi] * abb_diag_abs[gj]), eps))
                rel_schur = strength / max(diag_norm, eps)
                rel_abb = strength / max(abb_norm, eps)
                features = [
                    _safe_log_abs(np.asarray([strength]), eps)[0],
                    np.log(max(rel_schur, eps)),
                    np.log(max(rel_abb, eps)),
                    _safe_log_abs(np.asarray([reverse_value]), eps)[0],
                    1.0 if value >= 0.0 else -1.0,
                    _safe_log_abs(np.asarray([schur_diag_abs[gi]]), eps)[0],
                    _safe_log_abs(np.asarray([schur_diag_abs[gj]]), eps)[0],
                    _safe_log_abs(np.asarray([local_row_abs[ai]]), eps)[0],
                    _safe_log_abs(np.asarray([local_row_abs[aj]]), eps)[0],
                    _safe_log_abs(np.asarray([local_col_abs[ai]]), eps)[0],
                    _safe_log_abs(np.asarray([local_col_abs[aj]]), eps)[0],
                    block_size / float(max(int(matrix.shape[0]), 1)),
                    active_count / float(max(interface_count, 1)),
                    float(gi - gj) / float(max(interface_count - 1, 1)),
                ]
                items.append((strength, int(block_id), gi, gj, value, features))
    items.sort(key=lambda item: item[0], reverse=True)
    if candidate_edge_limit and int(candidate_edge_limit) > 0:
        items = items[: int(candidate_edge_limit)]
    if not items:
        return (
            np.zeros((0, LOCAL_SCHUR_EDGE_FEATURE_DIM), dtype=np.float64),
            np.zeros((0, 2), dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            abb,
            diag_delta,
        )
    edges = np.asarray([[item[2], item[3]] for item in items], dtype=np.int64)
    values = np.asarray([item[4] for item in items], dtype=np.float64)
    block_ids = np.asarray([item[1] for item in items], dtype=np.int64)
    features = np.asarray([item[5] for item in items], dtype=np.float64)
    return features, edges, values, block_ids, abb, diag_delta


def aggregate_local_schur_edge_features(
    features: np.ndarray,
    edges: np.ndarray,
    values: np.ndarray,
    block_ids: np.ndarray,
    *,
    interface_count: int,
    candidate_edge_limit: int = 0,
    eps: float = 1e-30,
):
    """Aggregate per-instance directed Schur candidates into global edges.

    Local proposer candidates are still produced per standard-cell instance, but
    the sparse Schur budget is spent on unique directed interface entries.  This
    avoids wasting budget on duplicate local contributions to the same (i, j).
    """
    if edges.shape[0] == 0:
        return (
            np.zeros((0, LOCAL_SCHUR_EDGE_FEATURE_DIM), dtype=np.float64),
            np.zeros((0, 2), dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
        )
    grouped: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for idx in range(int(edges.shape[0])):
        i = int(edges[idx, 0])
        j = int(edges[idx, 1])
        value = float(values[idx])
        weight = max(abs(value), eps)
        key = (i, j)
        item = grouped.get(key)
        if item is None:
            item = {'value': 0.0, 'weight': 0.0, 'feature_sum': np.zeros(LOCAL_SCHUR_EDGE_FEATURE_DIM, dtype=np.float64), 'count': 0}
            grouped[key] = item
        item['value'] += value
        item['weight'] += weight
        item['feature_sum'] += weight * np.asarray(features[idx], dtype=np.float64)
        item['count'] += 1
    value_by_key = {key: float(item['value']) for key, item in grouped.items()}
    items = []
    for key, item in grouped.items():
        i, j = key
        total_value = float(item['value'])
        strength = abs(total_value)
        if strength <= eps:
            continue
        feature = item['feature_sum'] / max(float(item['weight']), eps)
        feature = np.asarray(feature, dtype=np.float64).copy()
        feature[0] = _safe_log_abs(np.asarray([total_value]), eps)[0]
        feature[3] = _safe_log_abs(np.asarray([value_by_key.get((j, i), 0.0)]), eps)[0]
        feature[4] = 1.0 if total_value >= 0.0 else -1.0
        feature[12] = min(float(item['count']) / float(max(int(interface_count), 1)), 1.0)
        items.append((strength, i, j, total_value, int(item['count']), feature))
    items.sort(key=lambda item: item[0], reverse=True)
    if candidate_edge_limit and int(candidate_edge_limit) > 0:
        items = items[: int(candidate_edge_limit)]
    if not items:
        return (
            np.zeros((0, LOCAL_SCHUR_EDGE_FEATURE_DIM), dtype=np.float64),
            np.zeros((0, 2), dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
        )
    out_edges = np.asarray([[item[1], item[2]] for item in items], dtype=np.int64)
    out_values = np.asarray([item[3] for item in items], dtype=np.float64)
    source_counts = np.asarray([item[4] for item in items], dtype=np.int64)
    out_features = np.asarray([item[5] for item in items], dtype=np.float64)
    return out_features, out_edges, out_values, source_counts


class LocalSchurEdgeGateNet(nn.Module):
    def __init__(self, feature_dim: int = LOCAL_SCHUR_EDGE_FEATURE_DIM, hidden_dim: int = 96, logit_clip: float = 8.0):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.logit_clip = float(logit_clip)
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.register_buffer('feature_mean', torch.zeros(self.feature_dim, dtype=torch.float64))
        self.register_buffer('feature_std', torch.ones(self.feature_dim, dtype=torch.float64))
        self.double()

    def set_feature_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.feature_mean.copy_(torch.as_tensor(mean, dtype=torch.float64))
        self.feature_std.copy_(torch.as_tensor(std, dtype=torch.float64).clamp_min(1e-12))

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_std.clamp_min(1e-12)
        return torch.clamp(self.net(normalized).squeeze(-1), -self.logit_clip, self.logit_clip)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(features))


def assemble_local_sparse_schur(
    *,
    abb: np.ndarray,
    diag_delta: np.ndarray,
    edges: np.ndarray,
    values: np.ndarray,
    selected_indices: np.ndarray,
    diagonal_shift: float,
    eps: float,
) -> np.ndarray:
    sparse = np.asarray(abb + diag_delta, dtype=np.float64).copy()
    if edges.shape[0] > 0 and selected_indices.shape[0] > 0:
        for edge_idx in selected_indices.tolist():
            i = int(edges[int(edge_idx), 0])
            j = int(edges[int(edge_idx), 1])
            value = float(values[int(edge_idx)])
            sparse[i, j] += value
    if diagonal_shift > 0.0 and sparse.shape[0] > 0:
        offdiag_abs = np.abs(sparse).sum(axis=1) - np.abs(np.diag(sparse))
        diag = np.diag(sparse).copy()
        signs = np.where(diag >= 0.0, 1.0, -1.0)
        sparse[np.diag_indices(int(sparse.shape[0]))] = diag + signs * float(diagonal_shift) * np.maximum(offdiag_abs, eps)
    return sparse


class LocalSparseSchurPreconditioner:
    def __init__(
        self,
        *,
        matrix: np.ndarray,
        core_plan: BlockSchwarzPlan,
        boundary_plan: BlockSchwarzPlan,
        strategy: str = 'topk_abs',
        edge_budget: int = 0,
        budget_multiplier: float = 2.0,
        candidate_edge_limit: int = 0,
        diagonal_shift: float = 1e-8,
        uncovered_row_policy: str = 'row_sum',
        eps: float = 1e-30,
    ):
        self.base = ExplicitSchurInterfacePreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            uncovered_row_policy=uncovered_row_policy,
            factorize_schur=False,
        )
        self.strategy = str(strategy)
        self.edge_budget = int(edge_budget)
        self.budget_multiplier = float(budget_multiplier)
        self.candidate_edge_limit = int(candidate_edge_limit)
        self.diagonal_shift = float(diagonal_shift)
        self.eps = float(eps)
        local_features, local_edges, local_values, local_block_ids, self.abb, self.diag_delta = build_local_schur_edge_features(
            self.base,
            candidate_edge_limit=0 if self.strategy == 'topk_abs' else self.candidate_edge_limit,
            eps=self.eps,
        )
        self.local_candidate_edges = int(local_edges.shape[0])
        if self.strategy == 'topk_abs':
            self.features, self.edges, self.values, self.source_counts = aggregate_local_schur_edge_features(
                local_features,
                local_edges,
                local_values,
                local_block_ids,
                interface_count=int(self.base.interface_rows.shape[0]),
                candidate_edge_limit=self.candidate_edge_limit,
                eps=self.eps,
            )
            self.block_ids = np.zeros(int(self.edges.shape[0]), dtype=np.int64)
        else:
            self.features, self.edges, self.values, self.block_ids = local_features, local_edges, local_values, local_block_ids
            self.source_counts = np.ones(int(self.edges.shape[0]), dtype=np.int64)
        self.selected_edge_indices = self._select_edges()
        self.sparse_schur = assemble_local_sparse_schur(
            abb=self.abb,
            diag_delta=self.diag_delta,
            edges=self.edges,
            values=self.values,
            selected_indices=self.selected_edge_indices,
            diagonal_shift=self.diagonal_shift,
            eps=self.eps,
        )
        self.sparse_factor, self.factor_mode = ExplicitSchurInterfacePreconditioner._factorize(self.sparse_schur)

    def _select_edges(self) -> np.ndarray:
        if self.edges.shape[0] == 0:
            return np.zeros(0, dtype=np.int64)
        interface_count = int(self.base.interface_rows.shape[0])
        budget = _local_schur_budget(interface_count, self.budget_multiplier, self.edge_budget)
        budget = min(budget, int(self.edges.shape[0]))
        if budget <= 0:
            return np.zeros(0, dtype=np.int64)
        if self.strategy == 'topk_abs':
            return np.argsort(-np.abs(self.values))[:budget].astype(np.int64)
        if self.strategy == 'per_instance_topk':
            selected = []
            per_instance = max(1, int(np.ceil(float(budget) / float(max(len(set(self.block_ids.tolist())), 1)))))
            for block_id in sorted(set(self.block_ids.tolist())):
                idx = np.flatnonzero(self.block_ids == int(block_id))
                local = idx[np.argsort(-np.abs(self.values[idx]))[:per_instance]]
                selected.extend(int(item) for item in local.tolist())
            selected = sorted(set(selected), key=lambda item: abs(float(self.values[item])), reverse=True)[:budget]
            return np.asarray(selected, dtype=np.int64)
        raise ValueError(f'Unsupported local sparse Schur strategy: {self.strategy}')

    def apply(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float64)
        out = np.zeros_like(vec)
        core_solution = self.base._apply_core_only(vec)
        out[self.base.core_rows] = core_solution[self.base.core_rows]
        if self.base.interface_rows.shape[0] > 0:
            interface_rhs = vec[self.base.interface_rows] - self.base.matrix[np.ix_(self.base.interface_rows, self.base.core_rows)].dot(core_solution[self.base.core_rows])
            interface_solution = self.sparse_factor.dot(interface_rhs)
            out[self.base.interface_rows] = interface_solution
            out[self.base.core_rows] = out[self.base.core_rows] - self.base.core_inverse_interface[self.base.core_rows, :].dot(interface_solution)
        out[self.base.uncovered_mask] = self.base.uncovered_scales[self.base.uncovered_mask] * vec[self.base.uncovered_mask]
        return out

    def metadata(self) -> Dict[str, Any]:
        metadata = self.base.metadata()
        interface_count = int(self.base.interface_rows.shape[0])
        metadata['preconditioner_mode'] = 'local_sparse_schur_' + self.strategy
        metadata['schur_factor_mode'] = self.factor_mode
        metadata['local_candidate_edges'] = int(getattr(self, 'local_candidate_edges', self.edges.shape[0]))
        metadata['global_candidate_edges'] = int(self.edges.shape[0])
        metadata['selected_edges'] = int(self.selected_edge_indices.shape[0])
        metadata['max_local_sources_per_global_edge'] = int(np.max(self.source_counts)) if getattr(self, 'source_counts', np.zeros(0)).shape[0] else 0
        metadata['edge_budget'] = int(self.edge_budget)
        metadata['budget_multiplier'] = float(self.budget_multiplier)
        metadata['candidate_edge_limit'] = int(self.candidate_edge_limit)
        metadata['sparse_schur_nnz'] = int(np.count_nonzero(self.sparse_schur))
        metadata['sparse_schur_density'] = float(np.count_nonzero(self.sparse_schur) / max(interface_count * interface_count, 1))
        metadata['diagonal_shift'] = float(self.diagonal_shift)
        metadata['memory_estimate'] = int(metadata.get('memory_estimate', 0) + self.sparse_schur.size * 8 + self.sparse_factor.size * 8)
        return metadata



def _spectral_radius(matrix: np.ndarray) -> float | None:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[0] == 0:
        return 0.0
    try:
        values = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        return None
    return float(np.max(np.abs(values))) if values.shape[0] else 0.0


class PowerSchurPreconditioner:
    def __init__(
        self,
        *,
        matrix: np.ndarray,
        core_plan: BlockSchwarzPlan,
        boundary_plan: BlockSchwarzPlan,
        power_terms: int = 1,
        diagonal_shift: float = 1e-8,
        uncovered_row_policy: str = 'row_sum',
        eps: float = 1e-30,
    ):
        self.base = ExplicitSchurInterfacePreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            uncovered_row_policy=uncovered_row_policy,
            factorize_schur=False,
        )
        self.power_terms = max(int(power_terms), 0)
        self.diagonal_shift = float(diagonal_shift)
        self.eps = float(eps)
        self.c0 = self._build_c0()
        self.c0_factor, self.factor_mode = ExplicitSchurInterfacePreconditioner._factorize(self.c0)
        self.es = np.asarray(self.c0 - self.base.schur_matrix, dtype=np.float64)
        self.power_matrix = np.asarray(self.c0_factor.dot(self.es), dtype=np.float64) if self.c0.shape[0] else np.zeros((0, 0), dtype=np.float64)

    def _build_c0(self) -> np.ndarray:
        schur = np.asarray(self.base.schur_matrix, dtype=np.float64)
        n = int(schur.shape[0])
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64)
        c0 = np.diag(np.diag(schur)).astype(np.float64)
        if self.diagonal_shift > 0.0:
            offdiag_abs = np.abs(schur).sum(axis=1) - np.abs(np.diag(schur))
            diag = np.diag(c0).copy()
            signs = np.where(diag >= 0.0, 1.0, -1.0)
            c0[np.diag_indices(n)] = diag + signs * self.diagonal_shift * np.maximum(offdiag_abs, self.eps)
        return c0

    def _apply_interface_power_inverse(self, interface_rhs: np.ndarray) -> np.ndarray:
        current = np.asarray(self.c0_factor.dot(interface_rhs), dtype=np.float64)
        total = current.copy()
        for _ in range(self.power_terms):
            current = np.asarray(self.c0_factor.dot(self.es.dot(current)), dtype=np.float64)
            total = total + current
        return total

    def _apply_power_matrix_to_schur(self) -> np.ndarray:
        schur = np.asarray(self.base.schur_matrix, dtype=np.float64)
        if schur.shape[0] == 0:
            return np.zeros((0, 0), dtype=np.float64)
        current = np.asarray(self.c0_factor.dot(schur), dtype=np.float64)
        total = current.copy()
        for _ in range(self.power_terms):
            current = np.asarray(self.c0_factor.dot(self.es.dot(current)), dtype=np.float64)
            total = total + current
        return total

    def apply(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float64)
        out = np.zeros_like(vec)
        core_solution = self.base._apply_core_only(vec)
        out[self.base.core_rows] = core_solution[self.base.core_rows]
        if self.base.interface_rows.shape[0] > 0:
            interface_rhs = vec[self.base.interface_rows] - self.base.matrix[np.ix_(self.base.interface_rows, self.base.core_rows)].dot(core_solution[self.base.core_rows])
            interface_solution = self._apply_interface_power_inverse(interface_rhs)
            out[self.base.interface_rows] = interface_solution
            out[self.base.core_rows] = out[self.base.core_rows] - self.base.core_inverse_interface[self.base.core_rows, :].dot(interface_solution)
        out[self.base.uncovered_mask] = self.base.uncovered_scales[self.base.uncovered_mask] * vec[self.base.uncovered_mask]
        return out

    def metadata(self) -> Dict[str, Any]:
        metadata = self.base.metadata()
        interface_count = int(self.base.interface_rows.shape[0])
        precond_error = np.eye(interface_count, dtype=np.float64) - self._apply_power_matrix_to_schur()
        metadata['preconditioner_mode'] = 'power_schur'
        metadata['schur_factor_mode'] = 'power_series_' + self.factor_mode
        metadata['power_terms'] = int(self.power_terms)
        metadata['c0_nnz'] = int(np.count_nonzero(self.c0))
        metadata['es_nnz'] = int(np.count_nonzero(self.es))
        metadata['power_matrix_spectral_radius'] = _spectral_radius(self.power_matrix)
        metadata['precond_error_spectral_radius'] = _spectral_radius(precond_error)
        metadata['diagonal_shift'] = float(self.diagonal_shift)
        metadata['memory_estimate'] = int(
            metadata.get('memory_estimate', 0)
            + self.c0.size * 8
            + self.c0_factor.size * 8
            + self.es.size * 8
            + self.power_matrix.size * 8
        )
        return metadata



class PowerSparseSchurPreconditioner:
    def __init__(
        self,
        *,
        matrix: np.ndarray,
        core_plan: BlockSchwarzPlan,
        boundary_plan: BlockSchwarzPlan,
        strategy: str = 'topk_abs',
        edge_budget: int = 0,
        budget_multiplier: float = 2.0,
        candidate_edge_limit: int = 0,
        power_terms: int = 1,
        diagonal_shift: float = 1e-8,
        uncovered_row_policy: str = 'row_sum',
        eps: float = 1e-30,
    ):
        self.sparse = LocalSparseSchurPreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            strategy=strategy,
            edge_budget=edge_budget,
            budget_multiplier=budget_multiplier,
            candidate_edge_limit=candidate_edge_limit,
            diagonal_shift=diagonal_shift,
            uncovered_row_policy=uncovered_row_policy,
            eps=eps,
        )
        self.base = self.sparse.base
        self.power_terms = max(int(power_terms), 0)
        self.eps = float(eps)
        self.c0 = np.asarray(self.sparse.sparse_schur, dtype=np.float64)
        self.c0_factor = np.asarray(self.sparse.sparse_factor, dtype=np.float64)
        self.factor_mode = self.sparse.factor_mode
        self.es = np.asarray(self.c0 - self.base.schur_matrix, dtype=np.float64)
        self.power_matrix = np.asarray(self.c0_factor.dot(self.es), dtype=np.float64) if self.c0.shape[0] else np.zeros((0, 0), dtype=np.float64)

    def _apply_interface_power_inverse(self, interface_rhs: np.ndarray) -> np.ndarray:
        current = np.asarray(self.c0_factor.dot(interface_rhs), dtype=np.float64)
        total = current.copy()
        for _ in range(self.power_terms):
            current = np.asarray(self.c0_factor.dot(self.es.dot(current)), dtype=np.float64)
            total = total + current
        return total

    def _apply_power_matrix_to_schur(self) -> np.ndarray:
        schur = np.asarray(self.base.schur_matrix, dtype=np.float64)
        if schur.shape[0] == 0:
            return np.zeros((0, 0), dtype=np.float64)
        current = np.asarray(self.c0_factor.dot(schur), dtype=np.float64)
        total = current.copy()
        for _ in range(self.power_terms):
            current = np.asarray(self.c0_factor.dot(self.es.dot(current)), dtype=np.float64)
            total = total + current
        return total

    def apply(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float64)
        out = np.zeros_like(vec)
        core_solution = self.base._apply_core_only(vec)
        out[self.base.core_rows] = core_solution[self.base.core_rows]
        if self.base.interface_rows.shape[0] > 0:
            interface_rhs = vec[self.base.interface_rows] - self.base.matrix[np.ix_(self.base.interface_rows, self.base.core_rows)].dot(core_solution[self.base.core_rows])
            interface_solution = self._apply_interface_power_inverse(interface_rhs)
            out[self.base.interface_rows] = interface_solution
            out[self.base.core_rows] = out[self.base.core_rows] - self.base.core_inverse_interface[self.base.core_rows, :].dot(interface_solution)
        out[self.base.uncovered_mask] = self.base.uncovered_scales[self.base.uncovered_mask] * vec[self.base.uncovered_mask]
        return out

    def metadata(self) -> Dict[str, Any]:
        metadata = self.sparse.metadata()
        interface_count = int(self.base.interface_rows.shape[0])
        precond_error = np.eye(interface_count, dtype=np.float64) - self._apply_power_matrix_to_schur()
        metadata['preconditioner_mode'] = 'power_sparse_schur_' + str(self.sparse.strategy)
        metadata['schur_factor_mode'] = 'power_sparse_series_' + self.factor_mode
        metadata['power_terms'] = int(self.power_terms)
        metadata['power_matrix_spectral_radius'] = _spectral_radius(self.power_matrix)
        metadata['precond_error_spectral_radius'] = _spectral_radius(precond_error)
        metadata['memory_estimate'] = int(
            metadata.get('memory_estimate', 0)
            + self.es.size * 8
            + self.power_matrix.size * 8
        )
        return metadata




def _arnoldi_basis(operator: np.ndarray, rank: int, eps: float = 1e-30) -> Tuple[np.ndarray, float]:
    operator = np.asarray(operator, dtype=np.float64)
    n = int(operator.shape[0])
    rank = max(0, min(int(rank), n))
    if n == 0 or rank == 0:
        return np.zeros((n, 0), dtype=np.float64), 0.0
    seed = np.ones(n, dtype=np.float64)
    norm = float(np.linalg.norm(seed))
    if norm <= eps:
        return np.zeros((n, 0), dtype=np.float64), 0.0
    vectors = []
    vectors.append(seed / norm)
    residual_norm = 0.0
    for j in range(rank):
        w = np.asarray(operator.dot(vectors[j]), dtype=np.float64)
        for q in vectors:
            w = w - float(np.dot(q, w)) * q
        # A second pass keeps the basis usable for the small dense interface systems here.
        for q in vectors:
            w = w - float(np.dot(q, w)) * q
        residual_norm = float(np.linalg.norm(w))
        if residual_norm <= eps or len(vectors) >= rank:
            break
        vectors.append(w / residual_norm)
    basis = np.column_stack(vectors[:rank]) if vectors else np.zeros((n, 0), dtype=np.float64)
    return basis, residual_norm


class PowerSparseSchurArnoldiPreconditioner:
    def __init__(
        self,
        *,
        matrix: np.ndarray,
        core_plan: BlockSchwarzPlan,
        boundary_plan: BlockSchwarzPlan,
        strategy: str = 'topk_abs',
        edge_budget: int = 0,
        budget_multiplier: float = 2.0,
        candidate_edge_limit: int = 0,
        power_terms: int = 1,
        arnoldi_rank: int = 4,
        diagonal_shift: float = 1e-8,
        uncovered_row_policy: str = 'row_sum',
        eps: float = 1e-30,
    ):
        self.power = PowerSparseSchurPreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            strategy=strategy,
            edge_budget=edge_budget,
            budget_multiplier=budget_multiplier,
            candidate_edge_limit=candidate_edge_limit,
            power_terms=power_terms,
            diagonal_shift=diagonal_shift,
            uncovered_row_policy=uncovered_row_policy,
            eps=eps,
        )
        self.base = self.power.base
        self.sparse = self.power.sparse
        self.power_terms = int(self.power.power_terms)
        self.arnoldi_rank = max(0, int(arnoldi_rank))
        self.eps = float(eps)
        self.basis, self.arnoldi_last_residual_norm = _arnoldi_basis(self.power.power_matrix, self.arnoldi_rank, self.eps)
        self.correction_left = self._build_low_rank_correction()

    def _apply_interface_power_inverse_matrix(self, rhs_matrix: np.ndarray) -> np.ndarray:
        rhs_matrix = np.asarray(rhs_matrix, dtype=np.float64)
        current = np.asarray(self.power.c0_factor.dot(rhs_matrix), dtype=np.float64)
        total = current.copy()
        for _ in range(self.power_terms):
            current = np.asarray(self.power.c0_factor.dot(self.power.es.dot(current)), dtype=np.float64)
            total = total + current
        return total

    def _build_low_rank_correction(self) -> np.ndarray:
        interface_count = int(self.base.interface_rows.shape[0])
        rank = int(self.basis.shape[1])
        if interface_count == 0 or rank == 0:
            return np.zeros((interface_count, 0), dtype=np.float64)
        schur = np.asarray(self.base.schur_matrix, dtype=np.float64)
        schur_basis = np.asarray(schur.dot(self.basis), dtype=np.float64)
        projected = np.asarray(self.basis.T.dot(schur_basis), dtype=np.float64)
        power_on_schur_basis = self._apply_interface_power_inverse_matrix(schur_basis)
        residual = self.basis - power_on_schur_basis
        try:
            projected_inverse = np.linalg.inv(projected)
        except np.linalg.LinAlgError:
            projected_inverse = np.linalg.pinv(projected)
        return np.asarray(residual.dot(projected_inverse), dtype=np.float64)

    def _apply_interface_inverse(self, interface_rhs: np.ndarray) -> np.ndarray:
        result = self.power._apply_interface_power_inverse(interface_rhs)
        if self.basis.shape[1] > 0:
            result = result + self.correction_left.dot(self.basis.T.dot(interface_rhs))
        return np.asarray(result, dtype=np.float64)

    def _apply_inverse_to_schur(self) -> np.ndarray:
        schur = np.asarray(self.base.schur_matrix, dtype=np.float64)
        if schur.shape[0] == 0:
            return np.zeros((0, 0), dtype=np.float64)
        result = self._apply_interface_power_inverse_matrix(schur)
        if self.basis.shape[1] > 0:
            result = result + self.correction_left.dot(self.basis.T.dot(schur))
        return np.asarray(result, dtype=np.float64)

    def apply(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float64)
        out = np.zeros_like(vec)
        core_solution = self.base._apply_core_only(vec)
        out[self.base.core_rows] = core_solution[self.base.core_rows]
        if self.base.interface_rows.shape[0] > 0:
            interface_rhs = vec[self.base.interface_rows] - self.base.matrix[np.ix_(self.base.interface_rows, self.base.core_rows)].dot(core_solution[self.base.core_rows])
            interface_solution = self._apply_interface_inverse(interface_rhs)
            out[self.base.interface_rows] = interface_solution
            out[self.base.core_rows] = out[self.base.core_rows] - self.base.core_inverse_interface[self.base.core_rows, :].dot(interface_solution)
        out[self.base.uncovered_mask] = self.base.uncovered_scales[self.base.uncovered_mask] * vec[self.base.uncovered_mask]
        return out

    def metadata(self) -> Dict[str, Any]:
        metadata = self.power.metadata()
        interface_count = int(self.base.interface_rows.shape[0])
        precond_error = np.eye(interface_count, dtype=np.float64) - self._apply_inverse_to_schur()
        metadata['preconditioner_mode'] = 'power_sparse_schur_arnoldi_' + str(self.sparse.strategy)
        metadata['schur_factor_mode'] = 'power_sparse_arnoldi_' + self.power.factor_mode
        metadata['power_terms'] = int(self.power_terms)
        metadata['arnoldi_rank_requested'] = int(self.arnoldi_rank)
        metadata['arnoldi_rank_actual'] = int(self.basis.shape[1])
        metadata['arnoldi_last_residual_norm'] = float(self.arnoldi_last_residual_norm)
        metadata['precond_error_spectral_radius'] = _spectral_radius(precond_error)
        metadata['low_rank_correction_nnz'] = int(np.count_nonzero(self.correction_left))
        metadata['memory_estimate'] = int(
            metadata.get('memory_estimate', 0)
            + self.basis.size * 8
            + self.correction_left.size * 8
        )
        return metadata


class LocalSchurAdditiveInversePreconditioner:
    def __init__(
        self,
        *,
        matrix: np.ndarray,
        core_plan: BlockSchwarzPlan,
        boundary_plan: BlockSchwarzPlan,
        diagonal_shift: float = 1e-8,
        uncovered_row_policy: str = 'row_sum',
        eps: float = 1e-30,
        include_diagonal_fallback: bool = False,
        diagonal_weight: float = 0.0,
        local_weight: float = 1.0,
        cluster_weight: float = 0.0,
        cluster_hops: int = 0,
        max_cluster_size: int = 16,
        patch_svd_rcond: float = 0.0,
        inner_iterations: int = 1,
    ):
        self.base = ExplicitSchurInterfacePreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            uncovered_row_policy=uncovered_row_policy,
            factorize_schur=False,
        )
        self.diagonal_shift = float(diagonal_shift)
        self.eps = float(eps)
        self.include_diagonal_fallback = bool(include_diagonal_fallback)
        self.diagonal_weight = float(diagonal_weight)
        self.local_weight = float(local_weight)
        self.cluster_weight = float(cluster_weight)
        self.cluster_hops = int(cluster_hops)
        self.max_cluster_size = int(max_cluster_size)
        self.patch_svd_rcond = float(patch_svd_rcond)
        self.inner_iterations = int(inner_iterations)
        self.local_patches = []
        self.cluster_patches = []
        self.interface_overlap = np.zeros(int(self.base.interface_rows.shape[0]), dtype=np.float64)
        self.diagonal_inverse = self._build_diagonal_inverse()
        self._build_local_inverse_patches()
        self._build_cluster_inverse_patches()

    def _build_diagonal_inverse(self) -> np.ndarray:
        if self.base.schur_matrix.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        diag = np.diag(self.base.schur_matrix).astype(np.float64)
        signs = np.where(diag >= 0.0, 1.0, -1.0)
        return signs / np.maximum(np.abs(diag), self.eps)

    def _build_local_inverse_patches(self) -> None:
        interface_rows = np.asarray(self.base.interface_rows, dtype=np.int64)
        interface_count = int(interface_rows.shape[0])
        if interface_count == 0:
            return
        matrix = np.asarray(self.base.matrix, dtype=np.float64)
        active_by_block = []
        for core_rows in self.base.core_plan.blocks:
            core_rows = np.asarray(core_rows, dtype=np.int64)
            if core_rows.shape[0] == 0:
                active_by_block.append(np.zeros(0, dtype=np.int64))
                continue
            a_ib_full = matrix[np.ix_(core_rows, interface_rows)]
            a_bi_full = matrix[np.ix_(interface_rows, core_rows)]
            active_mask = np.logical_or(
                np.any(np.abs(a_ib_full) > self.eps, axis=0),
                np.any(np.abs(a_bi_full) > self.eps, axis=1),
            )
            active = np.flatnonzero(active_mask).astype(np.int64)
            active_by_block.append(active)
            self.interface_overlap[active] += 1.0
        overlap = np.maximum(self.interface_overlap, 1.0)
        abb_diag = np.diag(matrix[np.ix_(interface_rows, interface_rows)]).astype(np.float64)
        for block_id, (core_rows, core_factor, active) in enumerate(zip(self.base.core_plan.blocks, self.base.core_plan.factor_solvers, active_by_block)):
            core_rows = np.asarray(core_rows, dtype=np.int64)
            active = np.asarray(active, dtype=np.int64)
            if core_rows.shape[0] == 0 or active.shape[0] == 0:
                continue
            a_ib = matrix[np.ix_(core_rows, interface_rows[active])]
            a_bi = matrix[np.ix_(interface_rows[active], core_rows)]
            local_delta = -np.asarray(a_bi.dot(core_factor).dot(a_ib), dtype=np.float64)
            local_matrix = local_delta.copy()
            local_matrix[np.diag_indices(int(active.shape[0]))] += abb_diag[active] / overlap[active]
            if self.diagonal_shift > 0.0 and local_matrix.shape[0] > 0:
                offdiag_abs = np.abs(local_matrix).sum(axis=1) - np.abs(np.diag(local_matrix))
                diag = np.diag(local_matrix).copy()
                signs = np.where(diag >= 0.0, 1.0, -1.0)
                local_matrix[np.diag_indices(int(local_matrix.shape[0]))] = diag + signs * self.diagonal_shift * np.maximum(offdiag_abs, self.eps)
            local_factor, factor_mode = ExplicitSchurInterfacePreconditioner._factorize(local_matrix)
            weights = 1.0 / overlap[active]
            self.local_patches.append(
                {
                    'block_id': int(block_id),
                    'active': active,
                    'factor': local_factor,
                    'factor_mode': factor_mode,
                    'weights': weights.astype(np.float64),
                    'local_size': int(active.shape[0]),
                }
            )


    def _factorize_interface_patch(self, active: np.ndarray, matrix: np.ndarray) -> Tuple[np.ndarray, str]:
        patch_matrix = np.asarray(matrix, dtype=np.float64).copy()
        if self.diagonal_shift > 0.0 and patch_matrix.shape[0] > 0:
            offdiag_abs = np.abs(patch_matrix).sum(axis=1) - np.abs(np.diag(patch_matrix))
            diag = np.diag(patch_matrix).copy()
            signs = np.where(diag >= 0.0, 1.0, -1.0)
            patch_matrix[np.diag_indices(int(patch_matrix.shape[0]))] = diag + signs * self.diagonal_shift * np.maximum(offdiag_abs, self.eps)
        if self.patch_svd_rcond > 0.0 and patch_matrix.shape[0] > 0:
            return np.linalg.pinv(patch_matrix, rcond=self.patch_svd_rcond), 'svd_pinv_rcond'
        return ExplicitSchurInterfacePreconditioner._factorize(patch_matrix)

    def _build_cluster_inverse_patches(self) -> None:
        if self.cluster_hops <= 0 or self.cluster_weight == 0.0 or self.base.interface_rows.shape[0] == 0:
            return
        interface_count = int(self.base.interface_rows.shape[0])
        neighbors = [set() for _ in range(interface_count)]
        seeds = []
        for patch in self.local_patches:
            active = np.asarray(patch['active'], dtype=np.int64)
            if active.shape[0] == 0:
                continue
            seeds.append(active)
            for i in active.tolist():
                neighbors[int(i)].update(int(j) for j in active.tolist() if int(j) != int(i))
        cluster_sets = []
        seen = set()
        for seed in seeds:
            current = set(int(i) for i in seed.tolist())
            frontier = set(current)
            for _ in range(max(0, self.cluster_hops)):
                nxt = set()
                for item in frontier:
                    nxt.update(neighbors[int(item)])
                nxt.difference_update(current)
                if not nxt:
                    break
                current.update(nxt)
                frontier = nxt
            if len(current) > max(1, self.max_cluster_size):
                schur = self.base.schur_matrix
                seed_list = sorted(int(i) for i in seed.tolist())
                candidates = sorted(current, key=lambda idx: max(abs(float(schur[idx, j])) for j in seed_list), reverse=True)
                current = set(candidates[: max(1, self.max_cluster_size)])
            key = tuple(sorted(current))
            if len(key) <= 1 or key in seen:
                continue
            seen.add(key)
            cluster_sets.append(np.asarray(key, dtype=np.int64))
        if not cluster_sets:
            return
        cluster_overlap = np.zeros(interface_count, dtype=np.float64)
        for active in cluster_sets:
            cluster_overlap[active] += 1.0
        cluster_overlap = np.maximum(cluster_overlap, 1.0)
        for cluster_id, active in enumerate(cluster_sets):
            local_matrix = self.base.schur_matrix[np.ix_(active, active)]
            local_factor, factor_mode = self._factorize_interface_patch(active, local_matrix)
            self.cluster_patches.append({
                'cluster_id': int(cluster_id),
                'active': active,
                'factor': local_factor,
                'factor_mode': factor_mode,
                'weights': (1.0 / cluster_overlap[active]).astype(np.float64),
                'local_size': int(active.shape[0]),
            })

    def _apply_patch_once(self, rhs: np.ndarray) -> np.ndarray:
        out = np.zeros_like(np.asarray(rhs, dtype=np.float64))
        if self.include_diagonal_fallback and out.shape[0] > 0:
            out += self.diagonal_weight * self.diagonal_inverse * rhs
        for patch in self.local_patches:
            active = patch['active']
            local = patch['factor'].dot(rhs[active])
            out[active] += self.local_weight * patch['weights'] * local
        for patch in self.cluster_patches:
            active = patch['active']
            local = patch['factor'].dot(rhs[active])
            out[active] += self.cluster_weight * patch['weights'] * local
        return out

    def _apply_interface_inverse(self, interface_rhs: np.ndarray) -> np.ndarray:
        interface_rhs = np.asarray(interface_rhs, dtype=np.float64)
        if self.inner_iterations <= 1:
            return self._apply_patch_once(interface_rhs)
        out = np.zeros_like(interface_rhs)
        residual = interface_rhs.copy()
        for _ in range(max(1, self.inner_iterations)):
            delta = self._apply_patch_once(residual)
            out += delta
            residual = interface_rhs - self.base.schur_matrix.dot(out)
        return out

    def apply(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float64)
        out = np.zeros_like(vec)
        core_solution = self.base._apply_core_only(vec)
        out[self.base.core_rows] = core_solution[self.base.core_rows]
        if self.base.interface_rows.shape[0] > 0:
            interface_rhs = vec[self.base.interface_rows] - self.base.matrix[np.ix_(self.base.interface_rows, self.base.core_rows)].dot(core_solution[self.base.core_rows])
            interface_solution = self._apply_interface_inverse(interface_rhs)
            out[self.base.interface_rows] = interface_solution
            out[self.base.core_rows] = out[self.base.core_rows] - self.base.core_inverse_interface[self.base.core_rows, :].dot(interface_solution)
        out[self.base.uncovered_mask] = self.base.uncovered_scales[self.base.uncovered_mask] * vec[self.base.uncovered_mask]
        return out

    def metadata(self) -> Dict[str, Any]:
        metadata = self.base.metadata()
        patch_sizes = [int(item['local_size']) for item in self.local_patches]
        cluster_sizes = [int(item['local_size']) for item in self.cluster_patches]
        factor_memory = int(sum(int(size) * int(size) * 8 for size in patch_sizes + cluster_sizes))
        metadata['preconditioner_mode'] = 'local_schur_additive'
        metadata['schur_factor_mode'] = 'local_inverse_additive'
        metadata['local_inverse_patches'] = int(len(self.local_patches))
        metadata['cluster_inverse_patches'] = int(len(self.cluster_patches))
        metadata['max_patch_size'] = int(max(patch_sizes) if patch_sizes else 0)
        metadata['mean_patch_size'] = float(np.mean(patch_sizes)) if patch_sizes else 0.0
        metadata['covered_interface_rows'] = int(np.count_nonzero(self.interface_overlap > 0.0))
        metadata['max_interface_overlap'] = int(np.max(self.interface_overlap)) if self.interface_overlap.shape[0] else 0
        metadata['include_diagonal_fallback'] = bool(self.include_diagonal_fallback)
        metadata['diagonal_weight'] = float(self.diagonal_weight)
        metadata['local_weight'] = float(self.local_weight)
        metadata['cluster_weight'] = float(self.cluster_weight)
        metadata['cluster_hops'] = int(self.cluster_hops)
        metadata['max_cluster_size'] = int(self.max_cluster_size)
        metadata['patch_svd_rcond'] = float(self.patch_svd_rcond)
        metadata['inner_iterations'] = int(self.inner_iterations)
        metadata['max_cluster_patch_size'] = int(max(cluster_sizes) if cluster_sizes else 0)
        metadata['mean_cluster_patch_size'] = float(np.mean(cluster_sizes)) if cluster_sizes else 0.0
        metadata['diagonal_shift'] = float(self.diagonal_shift)
        metadata['memory_estimate'] = int(metadata.get('memory_estimate', 0) + factor_memory + self.diagonal_inverse.size * 8)
        return metadata


class LearnedLocalSparseSchurPreconditioner:
    def __init__(
        self,
        *,
        matrix: np.ndarray,
        core_plan: BlockSchwarzPlan,
        boundary_plan: BlockSchwarzPlan,
        model: LocalSchurEdgeGateNet,
        edge_budget: int = 0,
        budget_multiplier: float = 2.0,
        candidate_edge_limit: int = 0,
        diagonal_shift: float = 1e-8,
        uncovered_row_policy: str = 'row_sum',
        eps: float = 1e-30,
        probe_rhs: np.ndarray | None = None,
        probe_x0: np.ndarray | None = None,
        probe_restart: int = 8,
        probe_iterations: int = 1,
        selection_policy: str = 'learned_union_slack',
        slack_fraction: float = 0.1,
        slack_min: int = 1,
    ):
        self.base = ExplicitSchurInterfacePreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            uncovered_row_policy=uncovered_row_policy,
            factorize_schur=False,
        )
        self.model = model.eval()
        self.edge_budget = int(edge_budget)
        self.budget_multiplier = float(budget_multiplier)
        self.candidate_edge_limit = int(candidate_edge_limit)
        self.diagonal_shift = float(diagonal_shift)
        self.eps = float(eps)
        self.probe_rhs = None if probe_rhs is None else np.asarray(probe_rhs, dtype=np.float64)
        self.probe_x0 = None if probe_x0 is None else np.asarray(probe_x0, dtype=np.float64)
        self.probe_restart = int(probe_restart)
        self.probe_iterations = int(probe_iterations)
        self.selection_policy = str(selection_policy)
        self.slack_fraction = float(slack_fraction)
        self.slack_min = int(slack_min)
        self.selection_mode = self.selection_policy
        local_features, local_edges, local_values, local_block_ids, self.abb, self.diag_delta = build_local_schur_edge_features(
            self.base,
            candidate_edge_limit=0,
            eps=self.eps,
        )
        self.local_candidate_edges = int(local_edges.shape[0])
        self.features, self.edges, self.values, self.source_counts = aggregate_local_schur_edge_features(
            local_features,
            local_edges,
            local_values,
            local_block_ids,
            interface_count=int(self.base.interface_rows.shape[0]),
            candidate_edge_limit=self.candidate_edge_limit,
            eps=self.eps,
        )
        self.block_ids = np.zeros(int(self.edges.shape[0]), dtype=np.int64)
        self.edge_gates, self.selected_edge_indices = self._score_and_select()
        self.sparse_schur = assemble_local_sparse_schur(
            abb=self.abb,
            diag_delta=self.diag_delta,
            edges=self.edges,
            values=self.values,
            selected_indices=self.selected_edge_indices,
            diagonal_shift=self.diagonal_shift,
            eps=self.eps,
        )
        self.sparse_factor, self.factor_mode = ExplicitSchurInterfacePreconditioner._factorize(self.sparse_schur)

    def _score_and_select(self):
        if self.features.shape[0] == 0:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
        with torch.no_grad():
            features_t = torch.as_tensor(self.features, dtype=torch.float64)
            gates = self.model(features_t).detach().cpu().numpy().astype(np.float64)
        interface_count = int(self.base.interface_rows.shape[0])
        budget = _local_schur_budget(interface_count, self.budget_multiplier, self.edge_budget)
        budget = min(budget, int(self.edges.shape[0]))
        if budget <= 0:
            return gates, np.zeros(0, dtype=np.int64)
        abs_values = np.abs(self.values)
        hard_selected = np.argsort(-abs_values)[:budget].astype(np.int64)
        scores = abs_values * (0.5 + gates)
        learned_order = np.argsort(-scores).astype(np.int64)
        learned_selected = learned_order[:budget].astype(np.int64)
        policy = str(self.selection_policy)
        if policy == 'hard_topk':
            self.selection_mode = 'hard_topk'
            return gates, hard_selected
        if policy == 'learned_replace':
            self.selection_mode = 'learned_replace'
            return gates, learned_selected
        if policy not in {'learned_union_slack', 'probe_union_slack', 'probe_greedy_slack', 'probe_best_of'}:
            raise ValueError(f'unsupported learned local sparse Schur selection policy: {policy}')

        hard_set = set(int(idx) for idx in hard_selected.tolist())
        slack_by_fraction = int(round(float(self.slack_fraction) * float(max(budget, 0))))
        slack = min(int(self.edges.shape[0]) - budget, max(int(self.slack_min), slack_by_fraction))
        learned_extra = [int(idx) for idx in learned_order.tolist() if int(idx) not in hard_set][:max(0, slack)]
        if learned_extra:
            union_selected = np.concatenate([hard_selected, np.asarray(learned_extra, dtype=np.int64)]).astype(np.int64)
        else:
            union_selected = hard_selected
        if policy == 'probe_union_slack':
            selected = self._select_with_probe(hard_selected, union_selected)
        elif policy == 'probe_greedy_slack':
            selected = self._select_greedy_with_probe(hard_selected, learned_extra)
        elif policy == 'probe_best_of':
            selected = self._select_best_with_probe(hard_selected, learned_selected, union_selected)
        else:
            self.selection_mode = 'learned_union_slack'
            selected = union_selected
        return gates, selected

    def _make_probe_apply(self, selected_indices: np.ndarray):
        sparse = assemble_local_sparse_schur(
            abb=self.abb,
            diag_delta=self.diag_delta,
            edges=self.edges,
            values=self.values,
            selected_indices=selected_indices,
            diagonal_shift=self.diagonal_shift,
            eps=self.eps,
        )
        sparse_factor, _ = ExplicitSchurInterfacePreconditioner._factorize(sparse)

        def apply(vec: np.ndarray) -> np.ndarray:
            vec = np.asarray(vec, dtype=np.float64)
            out = np.zeros_like(vec)
            core_solution = self.base._apply_core_only(vec)
            out[self.base.core_rows] = core_solution[self.base.core_rows]
            if self.base.interface_rows.shape[0] > 0:
                interface_rhs = vec[self.base.interface_rows] - self.base.matrix[np.ix_(self.base.interface_rows, self.base.core_rows)].dot(core_solution[self.base.core_rows])
                interface_solution = sparse_factor.dot(interface_rhs)
                out[self.base.interface_rows] = interface_solution
                out[self.base.core_rows] = out[self.base.core_rows] - self.base.core_inverse_interface[self.base.core_rows, :].dot(interface_solution)
            out[self.base.uncovered_mask] = self.base.uncovered_scales[self.base.uncovered_mask] * vec[self.base.uncovered_mask]
            return out

        return apply

    def _probe_score(self, selected_indices: np.ndarray) -> float:
        if self.probe_rhs is None or self.probe_rhs.shape[0] != self.base.matrix.shape[0]:
            return float('inf')
        operator = LinearOperator(self.base.matrix.shape, matvec=lambda x: self.base.matrix.dot(x), dtype=np.float64)
        apply = self._make_probe_apply(selected_indices)
        preconditioner = LinearOperator(self.base.matrix.shape, matvec=apply, dtype=np.float64)
        history = []
        try:
            solution, _ = gmres(
                operator,
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
            raw = self.base.matrix.dot(solution) - self.probe_rhs
            score = float(np.linalg.norm(raw))
            if history:
                score = min(score, float(history[-1]))
            return score
        except Exception:
            return float('inf')

    def _select_greedy_with_probe(self, hard_selected: np.ndarray, learned_extra: List[int]) -> np.ndarray:
        if self.probe_rhs is None or self.probe_rhs.shape[0] != self.base.matrix.shape[0]:
            self.selection_mode = 'learned_union_slack'
            if learned_extra:
                return np.concatenate([hard_selected, np.asarray(learned_extra, dtype=np.int64)]).astype(np.int64)
            return hard_selected
        selected = [int(idx) for idx in hard_selected.tolist()]
        best_score = self._probe_score(np.asarray(selected, dtype=np.int64))
        accepted = 0
        for edge_idx in learned_extra:
            trial = np.asarray(selected + [int(edge_idx)], dtype=np.int64)
            score = self._probe_score(trial)
            if score < best_score:
                selected.append(int(edge_idx))
                best_score = score
                accepted += 1
        self.selection_mode = 'probe_greedy_slack' if accepted > 0 else 'hard_topk'
        self.greedy_accepted_edges = int(accepted)
        return np.asarray(selected, dtype=np.int64)


    def _select_best_with_probe(self, hard_selected: np.ndarray, learned_selected: np.ndarray, union_selected: np.ndarray) -> np.ndarray:
        if self.probe_rhs is None or self.probe_rhs.shape[0] != self.base.matrix.shape[0]:
            self.selection_mode = 'learned_replace'
            return learned_selected
        candidates = [
            ('hard_topk', hard_selected),
            ('learned_replace', learned_selected),
            ('learned_union_slack', union_selected),
        ]
        scored = [(self._probe_score(selected), label, selected) for label, selected in candidates]
        scored.sort(key=lambda item: item[0])
        self.selection_mode = 'probe_best_of:' + scored[0][1]
        return scored[0][2]

    def _select_with_probe(self, hard_selected: np.ndarray, union_selected: np.ndarray) -> np.ndarray:
        if self.probe_rhs is None or self.probe_rhs.shape[0] != self.base.matrix.shape[0]:
            self.selection_mode = 'learned_union_slack'
            return union_selected
        if np.array_equal(hard_selected, union_selected):
            self.selection_mode = 'hard_topk'
            return hard_selected

        scores = []
        for label, selected_indices in [('hard_topk', hard_selected), ('learned_union_slack', union_selected)]:
            score = self._probe_score(selected_indices)
            scores.append((score, label, selected_indices))
        scores.sort(key=lambda item: item[0])
        self.selection_mode = scores[0][1]
        return scores[0][2]

    def apply(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float64)
        out = np.zeros_like(vec)
        core_solution = self.base._apply_core_only(vec)
        out[self.base.core_rows] = core_solution[self.base.core_rows]
        if self.base.interface_rows.shape[0] > 0:
            interface_rhs = vec[self.base.interface_rows] - self.base.matrix[np.ix_(self.base.interface_rows, self.base.core_rows)].dot(core_solution[self.base.core_rows])
            interface_solution = self.sparse_factor.dot(interface_rhs)
            out[self.base.interface_rows] = interface_solution
            out[self.base.core_rows] = out[self.base.core_rows] - self.base.core_inverse_interface[self.base.core_rows, :].dot(interface_solution)
        out[self.base.uncovered_mask] = self.base.uncovered_scales[self.base.uncovered_mask] * vec[self.base.uncovered_mask]
        return out

    def metadata(self) -> Dict[str, Any]:
        metadata = self.base.metadata()
        interface_count = int(self.base.interface_rows.shape[0])
        metadata['preconditioner_mode'] = 'learned_local_sparse_schur'
        metadata['schur_factor_mode'] = self.factor_mode
        metadata['learned_feature_dim'] = LOCAL_SCHUR_EDGE_FEATURE_DIM
        metadata['local_candidate_edges'] = int(getattr(self, 'local_candidate_edges', self.edges.shape[0]))
        metadata['global_candidate_edges'] = int(self.edges.shape[0])
        metadata['selected_edges'] = int(self.selected_edge_indices.shape[0])
        metadata['selection_mode'] = str(getattr(self, 'selection_mode', 'learned_union_slack'))
        metadata['selection_policy'] = str(getattr(self, 'selection_policy', 'learned_union_slack'))
        metadata['slack_fraction'] = float(getattr(self, 'slack_fraction', 0.1))
        metadata['slack_min'] = int(getattr(self, 'slack_min', 1))
        metadata['probe_iterations'] = int(getattr(self, 'probe_iterations', 1))
        metadata['greedy_accepted_edges'] = int(getattr(self, 'greedy_accepted_edges', 0))
        metadata['max_local_sources_per_global_edge'] = int(np.max(self.source_counts)) if getattr(self, 'source_counts', np.zeros(0)).shape[0] else 0
        metadata['edge_budget'] = int(self.edge_budget)
        metadata['budget_multiplier'] = float(self.budget_multiplier)
        metadata['candidate_edge_limit'] = int(self.candidate_edge_limit)
        metadata['sparse_schur_nnz'] = int(np.count_nonzero(self.sparse_schur))
        metadata['sparse_schur_density'] = float(np.count_nonzero(self.sparse_schur) / max(interface_count * interface_count, 1))
        metadata['diagonal_shift'] = float(self.diagonal_shift)
        metadata['memory_estimate'] = int(metadata.get('memory_estimate', 0) + self.sparse_schur.size * 8 + self.sparse_factor.size * 8)
        return metadata



class LearningAugmentedSparseSchurPreconditioner:
    """Route-B learned-augmented sparse Schur constructor.

    Hard top-k_abs safe base + cheap-score additions, then solve sparse
    P_theta x_B = r_hat_B.  This does not learn or directly apply S^{-1}.
    """

    def __init__(self, *, matrix: np.ndarray, core_plan: BlockSchwarzPlan, boundary_plan: BlockSchwarzPlan, edge_budget: int = 0, budget_multiplier: float = 2.0, candidate_edge_limit: int = 0, diagonal_shift: float = 1e-8, uncovered_row_policy: str = 'row_sum', eps: float = 1e-30, probe_rhs: np.ndarray | None = None, probe_x0: np.ndarray | None = None, probe_restart: int = 8, probe_iterations: int = 1, selection_policy: str = 'safe_add', add_fraction: float = 0.1, add_min: int = 1, node_map: dict | None = None):
        self.base = ExplicitSchurInterfacePreconditioner(matrix=matrix, core_plan=core_plan, boundary_plan=boundary_plan, uncovered_row_policy=uncovered_row_policy, factorize_schur=False)
        self.edge_budget = int(edge_budget); self.budget_multiplier = float(budget_multiplier); self.candidate_edge_limit = int(candidate_edge_limit)
        self.diagonal_shift = float(diagonal_shift); self.eps = float(eps)
        self.probe_rhs = None if probe_rhs is None else np.asarray(probe_rhs, dtype=np.float64)
        self.probe_x0 = None if probe_x0 is None else np.asarray(probe_x0, dtype=np.float64)
        self.probe_restart = int(probe_restart); self.probe_iterations = int(probe_iterations)
        self.selection_policy = str(selection_policy); self.selection_mode = str(selection_policy)
        self.add_fraction = float(add_fraction); self.add_min = int(add_min)
        local_features, local_edges, local_values, local_block_ids, self.abb, self.diag_delta = build_local_schur_edge_features(self.base, candidate_edge_limit=0, eps=self.eps)
        self.local_candidate_edges = int(local_edges.shape[0])
        self.features, self.edges, self.values, self.source_counts = aggregate_local_schur_edge_features(local_features, local_edges, local_values, local_block_ids, interface_count=int(self.base.interface_rows.shape[0]), candidate_edge_limit=self.candidate_edge_limit, eps=self.eps)
        self.learning_scores = self._cheap_learning_scores()
        self.safe_edge_indices, self.learned_add_indices, self.selected_edge_indices = self._select_edges()
        self.sparse_schur = assemble_local_sparse_schur(abb=self.abb, diag_delta=self.diag_delta, edges=self.edges, values=self.values, selected_indices=self.selected_edge_indices, diagonal_shift=self.diagonal_shift, eps=self.eps)
        self.factorization_success = True; self.factorization_failure_reason = ''
        import time as _time
        start = _time.perf_counter()
        try:
            self.sparse_factor, self.factor_mode = ExplicitSchurInterfacePreconditioner._factorize(self.sparse_schur)
        except Exception as exc:
            self.factorization_success = False; self.factor_mode = 'factorization_failed'; self.factorization_failure_reason = str(exc)
            self.sparse_factor = np.diag(build_analytic_scales('row_sum', self.sparse_schur)) if self.sparse_schur.shape[0] else np.zeros((0, 0), dtype=np.float64)
        self.factorization_time = float(_time.perf_counter() - start)
        self.apply_count = 0; self.apply_time_total = 0.0

    def _cheap_learning_scores(self) -> np.ndarray:
        if self.edges.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        abs_values = np.abs(self.values); max_abs = max(float(np.max(abs_values)), self.eps)
        if self.source_counts.shape[0] > 0:
            source_bonus = np.log1p(self.source_counts.astype(np.float64)) / max(float(np.log1p(np.max(self.source_counts))), 1.0)
        else:
            source_bonus = np.zeros_like(abs_values)
        rel_schur = np.tanh(np.asarray(self.features[:, 1], dtype=np.float64)) if self.features.shape[1] > 1 else 0.0
        rel_abb = np.tanh(np.asarray(self.features[:, 2], dtype=np.float64)) if self.features.shape[1] > 2 else 0.0
        reverse_abs = np.exp(np.clip(np.asarray(self.features[:, 3], dtype=np.float64), -60.0, 60.0)) if self.features.shape[1] > 3 else 0.0
        symmetry_bonus = np.minimum(reverse_abs / np.maximum(abs_values, self.eps), 1.0)
        normalized_abs = abs_values / max_abs
        return abs_values * (1.0 + 0.25 * source_bonus + 0.15 * rel_schur + 0.10 * rel_abb + 0.05 * symmetry_bonus + 0.05 * normalized_abs)

    def _select_edges(self):
        self.learned_edge_proposed_count = 0; self.learned_edge_accepted_count = 0; self.learned_edge_rejected_count = 0; self.probe_eval_count = 0; self.probe_accept_mean_eta = None
        if self.edges.shape[0] == 0:
            empty = np.zeros(0, dtype=np.int64); return empty, empty, empty
        interface_count = int(self.base.interface_rows.shape[0])
        budget = min(_local_schur_budget(interface_count, self.budget_multiplier, self.edge_budget), int(self.edges.shape[0]))
        if budget <= 0:
            empty = np.zeros(0, dtype=np.int64); return empty, empty, empty
        abs_values = np.abs(self.values)
        safe = np.argsort(-abs_values)[:budget].astype(np.int64)
        safe_set = {int(idx) for idx in safe.tolist()}
        add_budget = min(int(self.edges.shape[0]) - budget, max(int(self.add_min), int(round(float(self.add_fraction) * float(budget)))))
        learned_order = np.argsort(-self.learning_scores).astype(np.int64)
        learned_extra = [int(idx) for idx in learned_order.tolist() if int(idx) not in safe_set][:max(0, add_budget)]
        add = np.asarray(learned_extra, dtype=np.int64)
        self.learned_edge_proposed_count = int(add.shape[0])
        union = np.concatenate([safe, add]).astype(np.int64) if add.shape[0] else safe
        if self.selection_policy == 'safe_add_probe':
            selected = self._select_with_probe(safe, union)
        else:
            self.selection_mode = 'safe_add' if add.shape[0] else 'hard_topk'; selected = union
        selected_set = {int(idx) for idx in selected.tolist()}
        self.learned_edge_accepted_count = int(sum(1 for idx in add.tolist() if int(idx) in selected_set))
        self.learned_edge_rejected_count = int(add.shape[0] - self.learned_edge_accepted_count)
        return safe, add, selected

    def _make_probe_apply(self, selected_indices: np.ndarray):
        sparse = assemble_local_sparse_schur(abb=self.abb, diag_delta=self.diag_delta, edges=self.edges, values=self.values, selected_indices=selected_indices, diagonal_shift=self.diagonal_shift, eps=self.eps)
        sparse_factor, _ = ExplicitSchurInterfacePreconditioner._factorize(sparse)
        def apply(vec: np.ndarray) -> np.ndarray:
            vec = np.asarray(vec, dtype=np.float64); out = np.zeros_like(vec)
            core_solution = self.base._apply_core_only(vec); out[self.base.core_rows] = core_solution[self.base.core_rows]
            if self.base.interface_rows.shape[0] > 0:
                interface_rhs = vec[self.base.interface_rows] - self.base.matrix[np.ix_(self.base.interface_rows, self.base.core_rows)].dot(core_solution[self.base.core_rows])
                interface_solution = sparse_factor.dot(interface_rhs); out[self.base.interface_rows] = interface_solution
                out[self.base.core_rows] = out[self.base.core_rows] - self.base.core_inverse_interface[self.base.core_rows, :].dot(interface_solution)
            out[self.base.uncovered_mask] = self.base.uncovered_scales[self.base.uncovered_mask] * vec[self.base.uncovered_mask]
            return out
        return apply

    def _probe_score(self, selected_indices: np.ndarray) -> float:
        self.probe_eval_count = int(getattr(self, 'probe_eval_count', 0)) + 1
        if self.probe_rhs is None or self.probe_rhs.shape[0] != self.base.matrix.shape[0]:
            return float('inf')
        operator = LinearOperator(self.base.matrix.shape, matvec=lambda x: self.base.matrix.dot(x), dtype=np.float64)
        preconditioner = LinearOperator(self.base.matrix.shape, matvec=self._make_probe_apply(selected_indices), dtype=np.float64)
        history = []
        try:
            solution, _ = gmres(operator, self.probe_rhs, x0=self.probe_x0, M=preconditioner, restart=max(1, self.probe_restart), maxiter=max(1, self.probe_iterations), rtol=0.0, atol=0.0, callback=lambda value: history.append(float(value)), callback_type='pr_norm')
            raw = self.base.matrix.dot(solution) - self.probe_rhs
            score = float(np.linalg.norm(raw)); return min(score, float(history[-1])) if history else score
        except Exception:
            return float('inf')

    def _select_with_probe(self, safe: np.ndarray, union: np.ndarray) -> np.ndarray:
        self.probe_eval_count = 0; self.probe_accept_mean_eta = None
        if self.probe_rhs is None or self.probe_rhs.shape[0] != self.base.matrix.shape[0] or np.array_equal(safe, union):
            self.selection_mode = 'safe_add' if not np.array_equal(safe, union) else 'hard_topk'; return union
        safe_score = self._probe_score(safe); union_score = self._probe_score(union)
        if np.isfinite(safe_score) and np.isfinite(union_score):
            self.probe_accept_mean_eta = float(union_score / max(safe_score, self.eps))
        if union_score < safe_score:
            self.selection_mode = 'safe_add_probe:accepted'; return union
        self.selection_mode = 'safe_add_probe:rejected'; return safe

    def apply(self, vec: np.ndarray) -> np.ndarray:
        import time as _time
        start = _time.perf_counter(); vec = np.asarray(vec, dtype=np.float64); out = np.zeros_like(vec)
        core_solution = self.base._apply_core_only(vec); out[self.base.core_rows] = core_solution[self.base.core_rows]
        if self.base.interface_rows.shape[0] > 0:
            interface_rhs = vec[self.base.interface_rows] - self.base.matrix[np.ix_(self.base.interface_rows, self.base.core_rows)].dot(core_solution[self.base.core_rows])
            interface_solution = self.sparse_factor.dot(interface_rhs); out[self.base.interface_rows] = interface_solution
            out[self.base.core_rows] = out[self.base.core_rows] - self.base.core_inverse_interface[self.base.core_rows, :].dot(interface_solution)
        out[self.base.uncovered_mask] = self.base.uncovered_scales[self.base.uncovered_mask] * vec[self.base.uncovered_mask]
        self.apply_count += 1; self.apply_time_total += float(_time.perf_counter() - start)
        return out

    def metadata(self) -> Dict[str, Any]:
        metadata = self.base.metadata(); interface_count = int(self.base.interface_rows.shape[0])
        p_nnz = int(np.count_nonzero(self.sparse_schur)); total_possible = int(max(self.local_candidate_edges, int(self.edges.shape[0]))); selected = int(self.selected_edge_indices.shape[0])
        metadata['preconditioner_mode'] = 'learning_augmented'; metadata['route'] = 'route_b_sparse_schur_construction'; metadata['schur_factor_mode'] = self.factor_mode
        metadata['P_theta_shape'] = [int(self.sparse_schur.shape[0]), int(self.sparse_schur.shape[1])]; metadata['P_theta_nnz'] = p_nnz; metadata['P_theta_density'] = float(p_nnz / max(interface_count * interface_count, 1))
        metadata['P_theta_factorization_success'] = bool(self.factorization_success); metadata['P_theta_factorization_time'] = float(self.factorization_time); metadata['P_theta_apply_count'] = int(self.apply_count); metadata['P_theta_apply_time_total'] = float(self.apply_time_total); metadata['P_theta_failure_reason'] = str(self.factorization_failure_reason); metadata['fallback_used'] = not bool(self.factorization_success)
        metadata['total_possible_local_schur_entries'] = total_possible; metadata['cheap_candidate_count'] = int(self.edges.shape[0]); metadata['approximate_schur_score_count'] = int(self.edges.shape[0]); metadata['exact_schur_entry_count'] = selected; metadata['selected_edge_count'] = selected; metadata['selected_edge_ratio'] = float(selected / max(int(self.edges.shape[0]), 1)); metadata['exact_compute_ratio'] = float(selected / max(total_possible, 1))
        metadata['safe_edge_count'] = int(self.safe_edge_indices.shape[0]); metadata['learned_edge_proposed_count'] = int(self.learned_edge_proposed_count); metadata['learned_edge_accepted_count'] = int(self.learned_edge_accepted_count); metadata['learned_edge_rejected_count'] = int(self.learned_edge_rejected_count); metadata['probe_eval_count'] = int(getattr(self, 'probe_eval_count', 0)); metadata['probe_accept_mean_eta'] = self.probe_accept_mean_eta
        metadata['selection_mode'] = str(self.selection_mode); metadata['selection_policy'] = str(self.selection_policy); metadata['edge_budget'] = int(self.edge_budget); metadata['budget_multiplier'] = float(self.budget_multiplier); metadata['candidate_edge_limit'] = int(self.candidate_edge_limit); metadata['diagonal_shift'] = float(self.diagonal_shift)
        metadata['max_local_sources_per_global_edge'] = int(np.max(self.source_counts)) if getattr(self, 'source_counts', np.zeros(0)).shape[0] else 0
        metadata['memory_estimate'] = int(metadata.get('memory_estimate', 0) + self.sparse_schur.size * 8 + self.sparse_factor.size * 8)
        metadata['implementation_note'] = 'safe top-k_abs base plus deterministic cheap-score learned additions; sparse P_theta solve, not S inverse learning'
        return metadata



class LearningAugmentedSparseSchurPreconditionerStrict:
    """Exact-on-demand Route-B sparse Schur constructor.

    This strict variant avoids constructing the full dense Schur matrix.  It
    first scores local interface edges with cheap proxy features, selects a safe
    base plus learned additions, computes exact local Schur values only for the
    selected edge groups, assembles sparse P_theta, and solves P_theta in apply.
    """
    def __init__(self, *, matrix: np.ndarray, core_plan: BlockSchwarzPlan, boundary_plan: BlockSchwarzPlan, edge_budget: int = 0, budget_multiplier: float = 2.0, candidate_edge_limit: int = 0, diagonal_shift: float = 1e-8, uncovered_row_policy: str = 'row_sum', eps: float = 1e-30, probe_rhs: np.ndarray | None = None, probe_x0: np.ndarray | None = None, probe_restart: int = 8, probe_iterations: int = 1, selection_policy: str = 'safe_add', add_fraction: float = 0.1, add_min: int = 1, node_map: dict | None = None, max_schur_nnz: int = 0, max_degree: int = 0, max_exact_entries: int = 0):
        self.matrix=np.asarray(matrix,dtype=np.float64); self.core_plan=core_plan; self.boundary_plan=boundary_plan; self.node_map=dict(node_map or {}); self.index_to_name={int(idx):str(name).lower() for name,idx in self.node_map.items() if str(idx).lstrip('-').isdigit()}; self.node_map=dict(node_map or {}); self.index_to_name={int(idx):str(name).lower() for name,idx in self.node_map.items() if str(idx).lstrip('-').isdigit()}
        self.edge_budget=int(edge_budget); self.budget_multiplier=float(budget_multiplier); self.candidate_edge_limit=int(candidate_edge_limit); self.max_schur_nnz=int(max_schur_nnz); self.max_degree=int(max_degree); self.max_exact_entries=int(max_exact_entries)
        self.diagonal_shift=float(diagonal_shift); self.uncovered_row_policy=str(uncovered_row_policy); self.eps=float(eps)
        self.probe_rhs=None if probe_rhs is None else np.asarray(probe_rhs,dtype=np.float64); self.probe_x0=None if probe_x0 is None else np.asarray(probe_x0,dtype=np.float64)
        self.probe_restart=int(probe_restart); self.probe_iterations=int(probe_iterations); self.selection_policy=str(selection_policy); self.selection_mode=str(selection_policy)
        self.add_fraction=float(add_fraction); self.add_min=int(add_min); self.apply_count=0; self.apply_time_total=0.0
        self.core_rows=np.flatnonzero(self.core_plan.covered_mask).astype(np.int64)
        self.interface_mask=np.logical_and(self.boundary_plan.covered_mask, ~self.core_plan.covered_mask)
        self.interface_rows=np.flatnonzero(self.interface_mask).astype(np.int64)
        self.interface_pos={int(r):i for i,r in enumerate(self.interface_rows.tolist())}
        self.uncovered_mask=~np.logical_or(self.core_plan.covered_mask,self.interface_mask)
        self.uncovered_scales=build_analytic_scales(self.uncovered_row_policy,self.matrix)
        self.abb=self.matrix[np.ix_(self.interface_rows,self.interface_rows)] if self.interface_rows.shape[0] else np.zeros((0,0),dtype=np.float64)
        self.groups,self.edge_proxy,self.edge_source_counts,self.total_possible_local_schur_entries=self._build_proxy_groups()
        self.edges=np.asarray(list(self.groups.keys()),dtype=np.int64) if self.groups else np.zeros((0,2),dtype=np.int64)
        self.proxy_values=np.asarray([self.edge_proxy[tuple(e)] for e in self.edges.tolist()],dtype=np.float64) if self.edges.shape[0] else np.zeros(0,dtype=np.float64)
        self.learning_scores=self._cheap_learning_scores()
        self.safe_edge_indices,self.learned_add_indices,self.selected_edge_indices=self._select_edges()
        self.sparse_schur,self.exact_schur_entry_count,self.exact_schur_entry_compute_time=self._assemble_selected_sparse_schur(self.selected_edge_indices)
        self.factorization_success=True; self.factorization_failure_reason=''
        import time as _time
        t=_time.perf_counter()
        try: self.sparse_factor,self.factor_mode=ExplicitSchurInterfacePreconditioner._factorize(self.sparse_schur)
        except Exception as exc:
            self.factorization_success=False; self.factor_mode='factorization_failed'; self.factorization_failure_reason=str(exc)
            self.sparse_factor=np.diag(build_analytic_scales('row_sum',self.sparse_schur)) if self.sparse_schur.shape[0] else np.zeros((0,0),dtype=np.float64)
        self.factorization_time=float(_time.perf_counter()-t)
        self.factor_fill_nnz=int(np.count_nonzero(self.sparse_factor)) if hasattr(self.sparse_factor,'shape') else None
    def _apply_core_only(self,vec):
        out=np.zeros_like(np.asarray(vec,dtype=np.float64))
        for rows,factor in zip(self.core_plan.blocks,self.core_plan.factor_solvers): out[rows]=factor.dot(vec[rows])
        return out
    def _build_proxy_groups(self):
        groups={}; proxy={}; counts={}; total=0
        for bid,(irows,brows) in enumerate(zip(self.core_plan.blocks,self.boundary_plan.blocks)):
            I=np.asarray(irows,dtype=np.int64); B=[int(r) for r in brows.tolist() if int(r) not in set(I.tolist()) and int(r) in self.interface_pos]
            total += len(B)*len(B)
            if len(I)==0 or len(B)==0: continue
            AII=self.matrix[np.ix_(I,I)]; diag=np.abs(np.diag(AII)); dinv=1.0/max(float(np.max(diag)) if diag.size else 0.0,self.eps)
            ABI=self.matrix[np.ix_(B,I)]; AIB=self.matrix[np.ix_(I,B)]
            row_norm=np.linalg.norm(ABI,axis=1); col_norm=np.linalg.norm(AIB,axis=0)
            for ai,bi in enumerate(B):
                if row_norm[ai] <= self.eps: continue
                for aj,bj in enumerate(B):
                    if col_norm[aj] <= self.eps: continue
                    gp=(self.interface_pos[bi],self.interface_pos[bj]); val=float(row_norm[ai]*dinv*col_norm[aj])
                    if val <= self.eps: continue
                    groups.setdefault(gp,[]).append((bid,bi,bj))
                    proxy[gp]=proxy.get(gp,0.0)+val; counts[gp]=counts.get(gp,0)+1
        if self.candidate_edge_limit>0 and len(groups)>self.candidate_edge_limit:
            keep=set(k for k,_ in sorted(proxy.items(),key=lambda kv:kv[1],reverse=True)[:self.candidate_edge_limit])
            groups={k:v for k,v in groups.items() if k in keep}; proxy={k:v for k,v in proxy.items() if k in keep}; counts={k:v for k,v in counts.items() if k in keep}
        return groups,proxy,counts,int(total)
    def _cheap_learning_scores(self):
        if self.proxy_values.shape[0]==0: return np.zeros(0,dtype=np.float64)
        counts=np.asarray([self.edge_source_counts.get(tuple(e),1) for e in self.edges.tolist()],dtype=np.float64)
        bonus=np.log1p(counts)/max(float(np.log1p(np.max(counts))),1.0)
        diag_bonus=np.asarray([1.0 if int(e[0])==int(e[1]) else 0.0 for e in self.edges],dtype=np.float64)
        return self.proxy_values*(1.0+0.25*bonus+0.10*diag_bonus)
    def _select_edges(self):
        self.learned_edge_proposed_count=0; self.learned_edge_accepted_count=0; self.learned_edge_rejected_count=0; self.probe_eval_count=0; self.probe_accept_mean_eta=None; self.probe_reject_reason_counts={}
        self.safe_rule_counts={'diagonal':0,'abb_nonzero':0,'per_row_proxy':0,'global_proxy':0,'semantic_conservative':0}
        self.semantic_safe_rule_counts={'supply_like':0,'ground_like':0,'high_degree':0}
        self.semantic_protected_interface_positions=[]
        self.learned_add_per_row_limit=max(1,int(self.add_min),2)
        if self.edges.shape[0]==0:
            z=np.zeros(0,dtype=np.int64); return z,z,z
        n=int(self.interface_rows.shape[0]); budget=min(_local_schur_budget(n,self.budget_multiplier,self.edge_budget),int(self.edges.shape[0]))
        edge_list=[tuple(int(x) for x in e.tolist()) for e in self.edges]
        by_row={}
        for idx,(ri,cj) in enumerate(edge_list): by_row.setdefault(int(ri),[]).append(int(idx))
        degree_values=np.asarray([len(v) for v in by_row.values()],dtype=np.float64) if by_row else np.zeros(0,dtype=np.float64)
        high_degree_threshold=max(8,int(np.percentile(degree_values,90)) if degree_values.size else 8)
        protected_rows=set()
        for ri,idxs in by_row.items():
            global_row=int(self.interface_rows[int(ri)])
            name=str(self.index_to_name.get(global_row,self.index_to_name.get(global_row+1,''))).lower()
            is_ground=name in {'0','gnd','ground'} or 'gnd' in name or 'vss' in name
            is_supply=('vdd' in name) or ('vcc' in name) or ('vss' in name) or ('vsub' in name) or ('supply' in name) or ('power' in name)
            is_high_degree=len(idxs)>=high_degree_threshold and len(idxs)>2
            if is_supply or is_ground or is_high_degree:
                protected_rows.add(int(ri)); self.semantic_protected_interface_positions.append(int(ri))
                if is_supply: self.semantic_safe_rule_counts['supply_like']=int(self.semantic_safe_rule_counts.get('supply_like',0))+1
                if is_ground: self.semantic_safe_rule_counts['ground_like']=int(self.semantic_safe_rule_counts.get('ground_like',0))+1
                if is_high_degree: self.semantic_safe_rule_counts['high_degree']=int(self.semantic_safe_rule_counts.get('high_degree',0))+1
        safe=[]; seen=set()
        def edge_exact_cost(idx):
            gp=tuple(int(x) for x in self.edges[int(idx)].tolist()); return int(len(self.groups.get(gp,[])))
        def selected_exact_cost(extra_idx=None):
            total_cost=sum(edge_exact_cost(i) for i in seen)
            return total_cost if extra_idx is None else total_cost+edge_exact_cost(extra_idx)
        def selected_row_degree(ri,extra_idx=None):
            deg=sum(1 for i in seen if int(edge_list[int(i)][0])==int(ri))
            if extra_idx is not None and int(edge_list[int(extra_idx)][0])==int(ri): deg+=1
            return int(deg)
        def optional_allowed(idx):
            idx=int(idx); ri=int(edge_list[idx][0])
            if self.max_schur_nnz>0 and len(seen)+1>int(self.max_schur_nnz): return False
            if self.max_exact_entries>0 and selected_exact_cost(idx)>int(self.max_exact_entries): return False
            if self.max_degree>0 and selected_row_degree(ri,idx)>int(self.max_degree): return False
            return True
        def add_safe(idx,rule,mandatory=True):
            idx=int(idx)
            if idx in seen: return False
            if not mandatory and not optional_allowed(idx): return False
            safe.append(idx); seen.add(idx); self.safe_rule_counts[rule]=int(self.safe_rule_counts.get(rule,0))+1; return True
        diag_idx=[i for i,(ri,cj) in enumerate(edge_list) if ri==cj]
        for idx in diag_idx: add_safe(idx,'diagonal')
        if self.abb.shape[0]:
            for idx,(ri,cj) in enumerate(edge_list):
                if abs(float(self.abb[int(ri),int(cj)]))>self.eps: add_safe(idx,'abb_nonzero')
        for ri,idxs in by_row.items():
            ordered=sorted(idxs,key=lambda ix:float(self.proxy_values[int(ix)]),reverse=True)
            off=[ix for ix in ordered if edge_list[int(ix)][1]!=int(ri)]
            target=off[:1] if off else ordered[:1]
            for idx in target: add_safe(idx,'per_row_proxy')
            if int(ri) in protected_rows:
                for idx in ordered[:min(2,len(ordered))]: add_safe(idx,'semantic_conservative')
        order=np.argsort(-self.proxy_values).astype(np.int64)
        for idx in order.tolist():
            if len(safe)>=max(budget,len(diag_idx)): break
            add_safe(idx,'global_proxy',mandatory=False)
        safe=np.asarray(safe,dtype=np.int64); safe_set=set(safe.tolist())
        add_budget=min(int(self.edges.shape[0])-len(safe_set),max(int(self.add_min),int(round(self.add_fraction*max(budget,1)))))
        learned_order=np.argsort(-self.learning_scores).astype(np.int64)
        add=[]; row_add_counts={}
        for idx in learned_order.tolist():
            idx=int(idx)
            if idx in safe_set: continue
            ri=int(edge_list[idx][0])
            if ri in protected_rows: continue
            if int(row_add_counts.get(ri,0))>=int(self.learned_add_per_row_limit): continue
            if not optional_allowed(idx): continue
            seen.add(idx)
            add.append(idx); row_add_counts[ri]=int(row_add_counts.get(ri,0))+1
            if len(add)>=max(0,add_budget): break
        add=np.asarray(add,dtype=np.int64)
        self.learned_edge_proposed_count=int(add.shape[0]); union=np.concatenate([safe,add]).astype(np.int64) if add.shape[0] else safe
        selected=self._select_with_probe(safe,union) if self.selection_policy=='safe_add_probe' else union
        self.selection_mode='safe_add' if self.selection_policy!='safe_add_probe' and add.shape[0] else getattr(self,'selection_mode','hard_topk')
        sel=set(selected.tolist()); self.learned_edge_accepted_count=sum(1 for i in add.tolist() if i in sel); self.learned_edge_rejected_count=int(add.shape[0]-self.learned_edge_accepted_count)
        return safe,add,selected
    def _exact_value_for_source(self,bid,bi,bj):
        I=self.core_plan.blocks[int(bid)]; factor=self.core_plan.factor_solvers[int(bid)]
        rhs=self.matrix[np.ix_(I,[int(bj)])].reshape(-1); solved=factor.dot(rhs); left=self.matrix[np.ix_([int(bi)],I)].reshape(-1)
        return -float(left.dot(solved))
    def _assemble_selected_sparse_schur(self,selected_indices):
        import time as _time
        t=_time.perf_counter(); sparse=np.asarray(self.abb,dtype=np.float64).copy(); exact_count=0
        for idx in selected_indices.tolist():
            gp=tuple(int(x) for x in self.edges[int(idx)].tolist()); val=0.0
            for bid,bi,bj in self.groups.get(gp,[]): val += self._exact_value_for_source(bid,bi,bj); exact_count += 1
            sparse[gp[0],gp[1]] += val
        if self.diagonal_shift>0.0 and sparse.shape[0]>0:
            off=np.abs(sparse).sum(axis=1)-np.abs(np.diag(sparse)); diag=np.diag(sparse).copy(); signs=np.where(diag>=0.0,1.0,-1.0)
            sparse[np.diag_indices(int(sparse.shape[0]))]=diag+signs*self.diagonal_shift*np.maximum(off,self.eps)
        return sparse,int(exact_count),float(_time.perf_counter()-t)
    def _make_probe_apply(self,selected_indices):
        sparse,_,_=self._assemble_selected_sparse_schur(selected_indices); factor,_=ExplicitSchurInterfacePreconditioner._factorize(sparse)
        def apply(vec): return self._apply_with_factor(vec,factor)
        return apply
    def _probe_score(self,selected_indices):
        self.probe_eval_count=int(getattr(self,'probe_eval_count',0))+1
        if self.probe_rhs is None or self.probe_rhs.shape[0]!=self.matrix.shape[0]: return float('inf')
        op=LinearOperator(self.matrix.shape,matvec=lambda x:self.matrix.dot(x),dtype=np.float64); pc=LinearOperator(self.matrix.shape,matvec=self._make_probe_apply(selected_indices),dtype=np.float64); hist=[]
        try:
            sol,_=gmres(op,self.probe_rhs,x0=self.probe_x0,M=pc,restart=max(1,self.probe_restart),maxiter=max(1,self.probe_iterations),rtol=0.0,atol=0.0,callback=lambda v:hist.append(float(v)),callback_type='pr_norm')
            score=float(np.linalg.norm(self.matrix.dot(sol)-self.probe_rhs)); return min(score,float(hist[-1])) if hist else score
        except Exception: return float('inf')
    def _select_with_probe(self,safe,union):
        self.probe_eval_count=0; self.probe_accept_mean_eta=None
        if self.probe_rhs is None or self.probe_rhs.shape[0]!=self.matrix.shape[0] or np.array_equal(safe,union):
            self.selection_mode='safe_add' if not np.array_equal(safe,union) else 'hard_topk'
            if self.selection_policy=='safe_add_probe' and not np.array_equal(safe,union): self.probe_reject_reason_counts={'probe_unavailable_accepted_without_probe':1}
            return union
        a=self._probe_score(safe); b=self._probe_score(union); self.probe_accept_mean_eta=float(b/max(a,self.eps)) if np.isfinite(a) and np.isfinite(b) else None
        if b<a: self.selection_mode='safe_add_probe:accepted'; self.probe_reject_reason_counts={}; return union
        self.selection_mode='safe_add_probe:rejected'
        reason='probe_failure_or_factorization_failure' if not np.isfinite(b) else 'probe_no_improvement'
        self.probe_reject_reason_counts={reason:1}
        return safe
    def _apply_with_factor(self,vec,factor):
        vec=np.asarray(vec,dtype=np.float64); out=np.zeros_like(vec); core=self._apply_core_only(vec); out[self.core_rows]=core[self.core_rows]
        if self.interface_rows.shape[0]>0:
            rhs=vec[self.interface_rows]-self.matrix[np.ix_(self.interface_rows,self.core_rows)].dot(core[self.core_rows]); xb=factor.dot(rhs); out[self.interface_rows]=xb
            corr_rhs=np.zeros_like(vec); corr_rhs[self.core_rows]=self.matrix[np.ix_(self.core_rows,self.interface_rows)].dot(xb); corr=self._apply_core_only(corr_rhs); out[self.core_rows]=out[self.core_rows]-corr[self.core_rows]
        out[self.uncovered_mask]=self.uncovered_scales[self.uncovered_mask]*vec[self.uncovered_mask]; return out
    def apply(self,vec):
        import time as _time
        t=_time.perf_counter(); out=self._apply_with_factor(vec,self.sparse_factor); self.apply_count+=1; self.apply_time_total+=float(_time.perf_counter()-t); return out
    def metadata(self):
        n=int(self.interface_rows.shape[0]); pnnz=int(np.count_nonzero(self.sparse_schur)); selected=int(self.selected_edge_indices.shape[0]); total=max(int(self.total_possible_local_schur_entries),1)
        row_degrees=np.count_nonzero(self.sparse_schur,axis=1) if self.sparse_schur.shape[0] else np.zeros(0,dtype=np.int64); solve_per_apply=float(self.apply_time_total/max(int(self.apply_count),1)); budget_constraint_violations={'max_schur_nnz':bool(self.max_schur_nnz>0 and selected>int(self.max_schur_nnz)),'max_degree':bool(self.max_degree>0 and row_degrees.size and int(np.max(row_degrees))>int(self.max_degree)),'max_exact_entries':bool(self.max_exact_entries>0 and int(self.exact_schur_entry_count)>int(self.max_exact_entries))}; return {'preconditioner_mode':'learning_augmented','route':'route_b_sparse_schur_construction','implementation':'strict_exact_on_demand','core_block_mode':self.core_plan.block_mode,'boundary_block_mode':self.boundary_plan.block_mode,'interface_rows':n,'core_rows':int(self.core_rows.shape[0]),'uncovered_rows':int(np.count_nonzero(self.uncovered_mask)),'schur_factor_mode':self.factor_mode,'P_theta_shape':[int(self.sparse_schur.shape[0]),int(self.sparse_schur.shape[1])],'P_theta_nnz':pnnz,'P_theta_density':float(pnnz/max(n*n,1)),'P_theta_factorization_success':bool(self.factorization_success),'P_theta_factorization_time':float(self.factorization_time),'P_theta_factor_fill_nnz':None if self.factor_fill_nnz is None else int(self.factor_fill_nnz),'P_theta_apply_count':int(self.apply_count),'P_theta_apply_time_total':float(self.apply_time_total),'P_theta_solve_time_total':float(self.apply_time_total),'P_theta_solve_time_per_apply':solve_per_apply,'P_theta_failure_reason':str(self.factorization_failure_reason),'fallback_used':not bool(self.factorization_success),'total_possible_local_schur_entries':int(self.total_possible_local_schur_entries),'total_candidate_count':int(self.edges.shape[0]),'cheap_candidate_count':int(self.edges.shape[0]),'approximate_schur_score_count':int(self.edges.shape[0]),'exact_schur_entry_count':int(self.exact_schur_entry_count),'exact_schur_entry_compute_time':float(self.exact_schur_entry_compute_time),'P_theta_assembly_time':float(self.exact_schur_entry_compute_time),'selected_edge_count':selected,'selected_edge_ratio':float(selected/max(int(self.edges.shape[0]),1)),'exact_compute_ratio':float(self.exact_schur_entry_count/total),'skipped_patch_count':0,'skipped_entry_count':0,'per_row_degree_stats':{'min':int(np.min(row_degrees)) if row_degrees.size else 0,'max':int(np.max(row_degrees)) if row_degrees.size else 0,'mean':float(np.mean(row_degrees)) if row_degrees.size else 0.0,'p95':float(np.percentile(row_degrees,95)) if row_degrees.size else 0.0},'safe_edge_count':int(self.safe_edge_indices.shape[0]),'safe_rule_counts':dict(getattr(self,'safe_rule_counts',{})),'semantic_safe_rule_counts':dict(getattr(self,'semantic_safe_rule_counts',{})),'semantic_protected_interface_count':int(len(getattr(self,'semantic_protected_interface_positions',[]))),'learned_add_per_row_limit':int(getattr(self,'learned_add_per_row_limit',0)),'learned_edge_proposed_count':int(self.learned_edge_proposed_count),'learned_edge_accepted_count':int(self.learned_edge_accepted_count),'learned_edge_rejected_count':int(self.learned_edge_rejected_count),'probe_eval_count':int(getattr(self,'probe_eval_count',0)),'probe_accept_mean_eta':self.probe_accept_mean_eta,'probe_reject_reason_counts':dict(getattr(self,'probe_reject_reason_counts',{})),'selection_mode':str(self.selection_mode),'selection_policy':str(self.selection_policy),'edge_budget':int(self.edge_budget),'budget_multiplier':float(self.budget_multiplier),'candidate_edge_limit':int(self.candidate_edge_limit),'max_candidates':int(self.candidate_edge_limit),'max_schur_nnz':int(self.max_schur_nnz),'max_degree':int(self.max_degree),'max_exact_entries':int(self.max_exact_entries),'budget_constraint_violations':budget_constraint_violations,'diagonal_shift':float(self.diagonal_shift),'memory_estimate':int(self.core_plan.metadata().get('memory_estimate',0)+self.boundary_plan.metadata().get('memory_estimate',0)+self.sparse_schur.size*8+self.sparse_factor.size*8),'core':self.core_plan.metadata(),'boundary':self.boundary_plan.metadata()}

# Override the earlier compatibility implementation with the strict exact-on-demand version.
LearningAugmentedSparseSchurPreconditioner = LearningAugmentedSparseSchurPreconditionerStrict


def load_learned_local_sparse_schur_model(checkpoint_path: str) -> LocalSchurEdgeGateNet:
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = dict(payload.get('model_config', {}))
    model = LocalSchurEdgeGateNet(
        feature_dim=int(config.get('feature_dim', LOCAL_SCHUR_EDGE_FEATURE_DIM)),
        hidden_dim=int(config.get('hidden_dim', 96)),
        logit_clip=float(config.get('logit_clip', 8.0)),
    )
    model.load_state_dict(payload['model_state_dict'])
    if 'feature_mean' in payload and 'feature_std' in payload:
        model.set_feature_stats(np.asarray(payload['feature_mean']), np.asarray(payload['feature_std']))
    model.eval()
    return model


class SparseSchurPreconditioner:
    def __init__(
        self,
        *,
        matrix: np.ndarray,
        core_plan: BlockSchwarzPlan,
        boundary_plan: BlockSchwarzPlan,
        strategy: str = 'topk_abs',
        edge_budget: int = 64,
        row_topk: int = 2,
        relative_threshold: float = 0.05,
        diagonal_shift: float = 1e-8,
        uncovered_row_policy: str = 'row_sum',
        eps: float = 1e-30,
    ):
        self.base = ExplicitSchurInterfacePreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            uncovered_row_policy=uncovered_row_policy,
            factorize_schur=False,
        )
        self.strategy = str(strategy)
        self.edge_budget = int(edge_budget)
        self.row_topk = int(row_topk)
        self.relative_threshold = float(relative_threshold)
        self.diagonal_shift = float(diagonal_shift)
        self.eps = float(eps)
        self.sparse_schur, self.selected_edges = self._assemble_sparse_schur()
        self.sparse_factor, self.factor_mode = ExplicitSchurInterfacePreconditioner._factorize(self.sparse_schur)

    def _edge_strengths(self):
        schur = np.asarray(self.base.schur_matrix, dtype=np.float64)
        n = int(schur.shape[0])
        diag_abs = np.abs(np.diag(schur))
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                raw = max(abs(float(schur[i, j])), abs(float(schur[j, i])))
                if raw <= self.eps:
                    continue
                rel = raw / max(float(np.sqrt(max(diag_abs[i] * diag_abs[j], self.eps))), self.eps)
                edges.append((i, j, raw, rel))
        return edges

    def _select_edges(self):
        edges = self._edge_strengths()
        if not edges:
            return []
        if self.strategy == 'topk_abs':
            budget = int(self.edge_budget) if int(self.edge_budget) > 0 else len(edges)
            return sorted(edges, key=lambda item: item[2], reverse=True)[: min(budget, len(edges))]
        if self.strategy == 'relative_threshold':
            kept = [item for item in edges if float(item[3]) >= self.relative_threshold]
            if self.edge_budget > 0 and len(kept) > self.edge_budget:
                kept = sorted(kept, key=lambda item: item[3], reverse=True)[: self.edge_budget]
            return kept
        if self.strategy == 'row_topk':
            by_row = {}
            for item in edges:
                i, j = int(item[0]), int(item[1])
                by_row.setdefault(i, []).append(item)
                by_row.setdefault(j, []).append(item)
            seen = {}
            for row_items in by_row.values():
                for item in sorted(row_items, key=lambda x: x[2], reverse=True)[: max(int(self.row_topk), 0)]:
                    key = (int(item[0]), int(item[1]))
                    seen[key] = item
            kept = list(seen.values())
            if self.edge_budget > 0 and len(kept) > self.edge_budget:
                kept = sorted(kept, key=lambda item: item[2], reverse=True)[: self.edge_budget]
            return kept
        raise ValueError(f'Unsupported sparse Schur strategy: {self.strategy}')

    def _assemble_sparse_schur(self):
        schur = np.asarray(self.base.schur_matrix, dtype=np.float64)
        n = int(schur.shape[0])
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64), []
        sparse = np.diag(np.diag(schur)).astype(np.float64)
        selected = self._select_edges()
        for item in selected:
            i = int(item[0])
            j = int(item[1])
            sparse[i, j] = schur[i, j]
            sparse[j, i] = schur[j, i]
        if self.diagonal_shift > 0.0:
            offdiag_abs = np.abs(sparse).sum(axis=1) - np.abs(np.diag(sparse))
            diag = np.diag(sparse).copy()
            signs = np.where(diag >= 0.0, 1.0, -1.0)
            sparse[np.diag_indices(n)] = diag + signs * self.diagonal_shift * np.maximum(offdiag_abs, self.eps)
        return sparse, selected

    def apply(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float64)
        out = np.zeros_like(vec)
        core_solution = self.base._apply_core_only(vec)
        out[self.base.core_rows] = core_solution[self.base.core_rows]
        if self.base.interface_rows.shape[0] > 0:
            interface_rhs = vec[self.base.interface_rows] - self.base.matrix[np.ix_(self.base.interface_rows, self.base.core_rows)].dot(core_solution[self.base.core_rows])
            interface_solution = self.sparse_factor.dot(interface_rhs)
            out[self.base.interface_rows] = interface_solution
            out[self.base.core_rows] = out[self.base.core_rows] - self.base.core_inverse_interface[self.base.core_rows, :].dot(interface_solution)
        out[self.base.uncovered_mask] = self.base.uncovered_scales[self.base.uncovered_mask] * vec[self.base.uncovered_mask]
        return out

    def metadata(self) -> Dict[str, Any]:
        metadata = self.base.metadata()
        interface_count = int(self.base.interface_rows.shape[0])
        sparse_nnz = int(np.count_nonzero(self.sparse_schur))
        metadata['preconditioner_mode'] = 'sparse_schur_' + self.strategy
        metadata['schur_factor_mode'] = self.factor_mode
        metadata['sparse_schur_strategy'] = self.strategy
        metadata['selected_edges'] = int(len(self.selected_edges))
        metadata['edge_budget'] = int(self.edge_budget)
        metadata['row_topk'] = int(self.row_topk)
        metadata['relative_threshold'] = float(self.relative_threshold)
        metadata['sparse_schur_nnz'] = sparse_nnz
        metadata['sparse_schur_density'] = float(sparse_nnz / max(interface_count * interface_count, 1))
        metadata['diagonal_shift'] = float(self.diagonal_shift)
        metadata['memory_estimate'] = int(metadata.get('memory_estimate', 0) + self.sparse_schur.size * 8 + self.sparse_factor.size * 8)
        return metadata


class SelectedLocalSparseSchurPreconditioner:
    def __init__(self, *, matrix: np.ndarray, core_plan: BlockSchwarzPlan, boundary_plan: BlockSchwarzPlan, edge_budget: int = 64, row_topk: int = 2, diagonal_shift: float = 1e-8, uncovered_row_policy: str = 'row_sum', eps: float = 1e-30):
        self.matrix = np.asarray(matrix, dtype=np.float64)
        self.core_plan = core_plan
        self.boundary_plan = boundary_plan
        self.edge_budget = int(edge_budget)
        self.row_topk = int(row_topk)
        self.diagonal_shift = float(diagonal_shift)
        self.uncovered_row_policy = str(uncovered_row_policy)
        self.eps = float(eps)
        self.interface_mask = np.logical_and(self.boundary_plan.covered_mask, ~self.core_plan.covered_mask)
        self.uncovered_mask = ~np.logical_or(self.core_plan.covered_mask, self.interface_mask)
        self.core_rows = np.flatnonzero(self.core_plan.covered_mask).astype(np.int64)
        self.interface_rows = np.flatnonzero(self.interface_mask).astype(np.int64)
        self.uncovered_scales = build_analytic_scales(self.uncovered_row_policy, self.matrix)
        self.block_active_interface_indices: List[np.ndarray] = []
        self.edge_values: Dict[Tuple[int, int], float] = {}
        self.schur_diag = self._assemble_selected_local_entries()
        self.selected_edges = self._select_row_topk_edges()
        self.sparse_schur = self._assemble_sparse_schur()
        self.sparse_factor, self.factor_mode = ExplicitSchurInterfacePreconditioner._factorize(self.sparse_schur)

    def _apply_core_only(self, vec: np.ndarray) -> np.ndarray:
        out = np.zeros_like(np.asarray(vec, dtype=np.float64))
        for rows, factor in zip(self.core_plan.blocks, self.core_plan.factor_solvers):
            out[rows] = factor.dot(vec[rows])
        return out

    def _add_edge(self, i: int, j: int, value: float) -> None:
        if i == j or abs(float(value)) <= self.eps:
            return
        key = (int(i), int(j))
        self.edge_values[key] = float(self.edge_values.get(key, 0.0) + float(value))

    def _assemble_selected_local_entries(self) -> np.ndarray:
        n = int(self.interface_rows.shape[0])
        if n == 0:
            self.direct_interface_candidate_edges = 0
            self.local_candidate_edges = 0
            self.global_candidate_edges = 0
            return np.zeros(0, dtype=np.float64)
        diag = np.asarray(np.diag(self.matrix[np.ix_(self.interface_rows, self.interface_rows)]), dtype=np.float64).copy()
        direct_edges = 0
        for i, row in enumerate(self.interface_rows.tolist()):
            for j, col in enumerate(self.interface_rows.tolist()):
                if i == j:
                    continue
                value = float(self.matrix[int(row), int(col)])
                if abs(value) > self.eps:
                    self._add_edge(i, j, value)
                    direct_edges += 1
        local_edges = 0
        for core_rows, core_factor in zip(self.core_plan.blocks, self.core_plan.factor_solvers):
            core_rows = np.asarray(core_rows, dtype=np.int64)
            if core_rows.shape[0] == 0:
                self.block_active_interface_indices.append(np.zeros(0, dtype=np.int64))
                continue
            a_ib_full = self.matrix[np.ix_(core_rows, self.interface_rows)]
            a_bi_full = self.matrix[np.ix_(self.interface_rows, core_rows)]
            active = np.flatnonzero(np.logical_or(np.any(np.abs(a_ib_full) > self.eps, axis=0), np.any(np.abs(a_bi_full) > self.eps, axis=1))).astype(np.int64)
            self.block_active_interface_indices.append(active)
            if active.shape[0] == 0:
                continue
            local_delta = -np.asarray(a_bi_full[active, :].dot(core_factor).dot(a_ib_full[:, active]), dtype=np.float64)
            for li, gi in enumerate(active.tolist()):
                diag[int(gi)] += float(local_delta[li, li])
            for li, gi in enumerate(active.tolist()):
                for lj, gj in enumerate(active.tolist()):
                    if li == lj:
                        continue
                    value = float(local_delta[li, lj])
                    if abs(value) > self.eps:
                        self._add_edge(int(gi), int(gj), value)
                        local_edges += 1
        self.direct_interface_candidate_edges = int(direct_edges)
        self.local_candidate_edges = int(local_edges)
        self.global_candidate_edges = int(len(self.edge_values))
        return diag

    def _edge_strengths(self) -> List[Tuple[int, int, float]]:
        seen = set()
        out: List[Tuple[int, int, float]] = []
        for i, j in self.edge_values.keys():
            a, b = min(int(i), int(j)), max(int(i), int(j))
            if a == b or (a, b) in seen:
                continue
            seen.add((a, b))
            strength = max(abs(float(self.edge_values.get((a, b), 0.0))), abs(float(self.edge_values.get((b, a), 0.0))))
            if strength > self.eps:
                out.append((a, b, strength))
        return out

    def _select_row_topk_edges(self) -> List[Tuple[int, int, float]]:
        by_row: Dict[int, List[Tuple[int, int, float]]] = {}
        for edge in self._edge_strengths():
            i, j, _ = edge
            by_row.setdefault(int(i), []).append(edge)
            by_row.setdefault(int(j), []).append(edge)
        selected: Dict[Tuple[int, int], Tuple[int, int, float]] = {}
        if self.row_topk > 0:
            for row_edges in by_row.values():
                for edge in sorted(row_edges, key=lambda item: item[2], reverse=True)[: self.row_topk]:
                    selected[(int(edge[0]), int(edge[1]))] = edge
        kept = list(selected.values())
        if self.edge_budget > 0 and len(kept) > self.edge_budget:
            kept = sorted(kept, key=lambda item: item[2], reverse=True)[: self.edge_budget]
        return kept

    def _assemble_sparse_schur(self) -> np.ndarray:
        n = int(self.interface_rows.shape[0])
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64)
        sparse = np.zeros((n, n), dtype=np.float64)
        sparse[np.diag_indices(n)] = self.schur_diag
        for i, j, _ in self.selected_edges:
            sparse[int(i), int(j)] = float(self.edge_values.get((int(i), int(j)), 0.0))
            sparse[int(j), int(i)] = float(self.edge_values.get((int(j), int(i)), 0.0))
        if self.diagonal_shift > 0.0:
            offdiag_abs = np.abs(sparse).sum(axis=1) - np.abs(np.diag(sparse))
            diag = np.diag(sparse).copy()
            signs = np.where(diag >= 0.0, 1.0, -1.0)
            sparse[np.diag_indices(n)] = diag + signs * self.diagonal_shift * np.maximum(offdiag_abs, self.eps)
        return sparse

    def apply(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float64)
        out = np.zeros_like(vec)
        core_solution = self._apply_core_only(vec)
        out[self.core_rows] = core_solution[self.core_rows]
        if self.interface_rows.shape[0] > 0:
            interface_rhs = vec[self.interface_rows] - self.matrix[np.ix_(self.interface_rows, self.core_rows)].dot(core_solution[self.core_rows])
            interface_solution = self.sparse_factor.dot(interface_rhs)
            out[self.interface_rows] = interface_solution
            for block_id, (core_rows, factor) in enumerate(zip(self.core_plan.blocks, self.core_plan.factor_solvers)):
                rows = np.asarray(core_rows, dtype=np.int64)
                rhs = np.asarray(vec[rows], dtype=np.float64).copy()
                active = self.block_active_interface_indices[block_id] if block_id < len(self.block_active_interface_indices) else np.zeros(0, dtype=np.int64)
                if active.shape[0] > 0:
                    rhs -= self.matrix[np.ix_(rows, self.interface_rows[active])].dot(interface_solution[active])
                out[rows] = factor.dot(rhs)
        out[self.uncovered_mask] = self.uncovered_scales[self.uncovered_mask] * vec[self.uncovered_mask]
        return out

    def metadata(self) -> Dict[str, Any]:
        interface_count = int(self.interface_rows.shape[0])
        sparse_nnz = int(np.count_nonzero(self.sparse_schur))
        diag_nnz = int(np.count_nonzero(np.diag(self.sparse_schur))) if interface_count > 0 else 0
        per_row_counts = np.zeros(interface_count, dtype=np.int64)
        for i, j, _ in self.selected_edges:
            per_row_counts[int(i)] += 1
            per_row_counts[int(j)] += 1
        memory_estimate = int(self.core_plan.metadata().get('memory_estimate', 0) + self.boundary_plan.metadata().get('memory_estimate', 0) + self.sparse_schur.size * 8 + self.sparse_factor.size * 8 + len(self.edge_values) * 24)
        return {
            'preconditioner_mode': 'sparse_schur_row_topk_selected_local',
            'core_block_mode': self.core_plan.block_mode,
            'boundary_block_mode': self.boundary_plan.block_mode,
            'schur_factor_mode': self.factor_mode,
            'sparse_schur_strategy': 'row_topk_selected_local',
            'full_schur_constructed': False,
            'selected_local_constructed': True,
            'core_inverse_interface_constructed': False,
            'interface_rows': interface_count,
            'core_rows': int(self.core_rows.shape[0]),
            'uncovered_rows': int(np.count_nonzero(self.uncovered_mask)),
            'selected_edges': int(len(self.selected_edges)),
            'selected_edge_count': int(len(self.selected_edges)),
            'selected_edge_budget': int(self.edge_budget),
            'edge_budget': int(self.edge_budget),
            'row_topk': int(self.row_topk),
            'relative_threshold': None,
            'direct_interface_candidate_edges': int(getattr(self, 'direct_interface_candidate_edges', 0)),
            'local_candidate_edges': int(getattr(self, 'local_candidate_edges', 0)),
            'global_candidate_edges': int(getattr(self, 'global_candidate_edges', 0)),
            'sparse_schur_nnz': sparse_nnz,
            'sparse_schur_density': float(sparse_nnz / max(interface_count * interface_count, 1)),
            'P_shape': [interface_count, interface_count],
            'P_nnz': sparse_nnz,
            'P_density': float(sparse_nnz / max(interface_count * interface_count, 1)),
            'P_diag_nnz': diag_nnz,
            'P_offdiag_nnz': int(sparse_nnz - diag_nnz),
            'selected_edges_per_row_mean': float(np.mean(per_row_counts)) if per_row_counts.shape[0] else 0.0,
            'selected_edges_per_row_max': int(np.max(per_row_counts)) if per_row_counts.shape[0] else 0,
            'diagonal_shift': float(self.diagonal_shift),
            'schur_nnz': None,
            'schur_density': None,
            'memory_estimate': memory_estimate,
            'core': self.core_plan.metadata(),
            'boundary': self.boundary_plan.metadata(),
        }


def _orthonormalize_columns(matrix: np.ndarray, eps: float) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0] if matrix.ndim == 2 else 0, 0), dtype=np.float64)
    q, r = np.linalg.qr(matrix)
    keep = np.abs(np.diag(r)) > float(eps)
    if not np.any(keep):
        return np.zeros((matrix.shape[0], 0), dtype=np.float64)
    return np.asarray(q[:, keep], dtype=np.float64)


def build_fixed_schur_low_rank_basis(
    schur: np.ndarray,
    *,
    rank: int,
    mode: str = 'slow_eig',
    eps: float = 1e-30,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    schur = np.asarray(schur, dtype=np.float64)
    n = int(schur.shape[0])
    rank = max(int(rank), 0)
    if n == 0 or rank == 0:
        return np.zeros((n, 0), dtype=np.float64)
    rank = min(rank, n)
    mode = str(mode)
    if mode == 'constant':
        raw = np.ones((n, 1), dtype=np.float64)
    elif mode == 'contiguous':
        raw = np.zeros((n, rank), dtype=np.float64)
        for row in range(n):
            col = min(int(row * rank / max(n, 1)), rank - 1)
            raw[row, col] = 1.0
    elif mode == 'correction_svd':
        target = schur if reference is None else np.asarray(reference, dtype=np.float64)
        try:
            u, _, _ = np.linalg.svd(target, full_matrices=False)
            raw = np.asarray(u[:, :rank], dtype=np.float64)
        except np.linalg.LinAlgError:
            raw = np.zeros((n, 0), dtype=np.float64)
    elif mode == 'slow_eig':
        try:
            values, vectors = np.linalg.eig(schur)
            order = np.argsort(np.abs(values))[:rank]
            raw = np.real(vectors[:, order]).astype(np.float64)
        except np.linalg.LinAlgError:
            raw = np.zeros((n, 0), dtype=np.float64)
    else:
        raise ValueError(f'Unsupported fixed Schur low-rank mode: {mode}')
    basis = _orthonormalize_columns(raw, eps)
    if basis.shape[1] >= rank:
        return basis[:, :rank]
    if mode != 'contiguous':
        fallback = build_fixed_schur_low_rank_basis(schur, rank=rank, mode='contiguous', eps=eps, reference=reference)
        combined = np.concatenate([basis, fallback], axis=1) if basis.shape[1] else fallback
        return _orthonormalize_columns(combined, eps)[:, :rank]
    return basis


class HybridSparseSchurLowRankPreconditioner:
    def __init__(
        self,
        *,
        matrix: np.ndarray,
        core_plan: BlockSchwarzPlan,
        boundary_plan: BlockSchwarzPlan,
        strategy: str = 'topk_abs',
        edge_budget: int = 64,
        row_topk: int = 2,
        relative_threshold: float = 0.05,
        diagonal_shift: float = 1e-8,
        low_rank_rank: int = 4,
        low_rank_mode: str = 'slow_eig',
        low_rank_strength: float = 1.0,
        uncovered_row_policy: str = 'row_sum',
        eps: float = 1e-30,
    ):
        self.sparse = SparseSchurPreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            strategy=strategy,
            edge_budget=edge_budget,
            row_topk=row_topk,
            relative_threshold=relative_threshold,
            diagonal_shift=diagonal_shift,
            uncovered_row_policy=uncovered_row_policy,
            eps=eps,
        )
        self.base = self.sparse.base
        self.strategy = str(strategy)
        self.low_rank_rank = max(int(low_rank_rank), 0)
        self.low_rank_mode = str(low_rank_mode)
        self.low_rank_strength = float(low_rank_strength)
        self.eps = float(eps)
        self.inverse_low_rank_left = np.zeros((int(self.base.interface_rows.shape[0]), 0), dtype=np.float64)
        self.inverse_low_rank_singular = np.zeros(0, dtype=np.float64)
        self.inverse_low_rank_right = np.zeros((int(self.base.interface_rows.shape[0]), 0), dtype=np.float64)
        self.low_rank_basis = self._build_low_rank_basis()
        self.low_rank_core = self._build_low_rank_core()
        self.woodbury_z, self.woodbury_left, self.woodbury_small_factor, self.factor_mode = self._build_woodbury_state()

    def _build_low_rank_basis(self) -> np.ndarray:
        if self.low_rank_rank <= 0 or self.base.interface_rows.shape[0] == 0:
            return np.zeros((int(self.base.interface_rows.shape[0]), 0), dtype=np.float64)
        correction = np.asarray(self.base.schur_matrix - self.sparse.sparse_schur, dtype=np.float64)
        if self.low_rank_mode == 'inverse_error_svd':
            schur = np.asarray(self.base.schur_matrix, dtype=np.float64)
            exact_factor, _ = ExplicitSchurInterfacePreconditioner._factorize(schur)
            inverse_error = np.asarray(exact_factor - self.sparse.sparse_factor, dtype=np.float64)
            try:
                u, s, vh = np.linalg.svd(inverse_error, full_matrices=False)
                rank = min(self.low_rank_rank, int(u.shape[1]))
                self.inverse_low_rank_left = np.asarray(u[:, :rank], dtype=np.float64)
                self.inverse_low_rank_singular = np.asarray(s[:rank], dtype=np.float64)
                self.inverse_low_rank_right = np.asarray(vh[:rank, :].T, dtype=np.float64)
                return self.inverse_low_rank_left
            except np.linalg.LinAlgError:
                return np.zeros((int(schur.shape[0]), 0), dtype=np.float64)
        if self.low_rank_mode == 'precond_error_svd':
            schur = np.asarray(self.base.schur_matrix, dtype=np.float64)
            n = int(schur.shape[0])
            try:
                precond_error = np.eye(n, dtype=np.float64) - np.asarray(self.sparse.sparse_factor.dot(schur), dtype=np.float64)
                _, _, vh = np.linalg.svd(precond_error, full_matrices=False)
                raw = np.asarray(vh[: min(self.low_rank_rank, n), :].T, dtype=np.float64)
                return _orthonormalize_columns(raw, self.eps)[:, : self.low_rank_rank]
            except np.linalg.LinAlgError:
                return build_fixed_schur_low_rank_basis(
                    schur,
                    rank=self.low_rank_rank,
                    mode='contiguous',
                    eps=self.eps,
                    reference=correction,
                )
        return build_fixed_schur_low_rank_basis(
            self.base.schur_matrix,
            rank=self.low_rank_rank,
            mode=self.low_rank_mode,
            eps=self.eps,
            reference=correction,
        )

    def _build_low_rank_core(self) -> np.ndarray:
        if self.low_rank_mode == 'inverse_error_svd':
            return np.diag(self.inverse_low_rank_singular).astype(np.float64)
        if self.low_rank_basis.shape[1] == 0:
            return np.zeros((0, 0), dtype=np.float64)
        correction_target = np.asarray(self.base.schur_matrix - self.sparse.sparse_schur, dtype=np.float64)
        return self.low_rank_strength * self.low_rank_basis.T.dot(correction_target).dot(self.low_rank_basis)

    def _build_woodbury_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        if self.low_rank_mode == 'inverse_error_svd':
            return (
                np.zeros((int(self.base.interface_rows.shape[0]), 0), dtype=np.float64),
                np.zeros((int(self.base.interface_rows.shape[0]), 0), dtype=np.float64),
                np.zeros((0, 0), dtype=np.float64),
                'inverse_error_svd_additive',
            )
        rank = int(self.low_rank_basis.shape[1])
        if rank == 0:
            return (
                np.zeros((int(self.base.interface_rows.shape[0]), 0), dtype=np.float64),
                np.zeros((int(self.base.interface_rows.shape[0]), 0), dtype=np.float64),
                np.zeros((0, 0), dtype=np.float64),
                'sparse_factor_only',
            )
        # Apply (P + W K W^T)^-1 without forming dense W K W^T:
        # y = P^-1 q
        # x = y - (P^-1 W K) (I + W^T P^-1 W K)^-1 W^T y
        z = np.asarray(self.sparse.sparse_factor.dot(self.low_rank_basis), dtype=np.float64)
        left = np.asarray(z.dot(self.low_rank_core), dtype=np.float64)
        small = np.eye(rank, dtype=np.float64) + self.low_rank_basis.T.dot(left)
        small_factor, small_mode = ExplicitSchurInterfacePreconditioner._factorize(small)
        return z, left, small_factor, 'woodbury_' + str(small_mode)

    def _apply_interface_hybrid_inverse(self, interface_rhs: np.ndarray) -> np.ndarray:
        y = np.asarray(self.sparse.sparse_factor.dot(interface_rhs), dtype=np.float64)
        if self.low_rank_mode == 'inverse_error_svd':
            if self.inverse_low_rank_singular.shape[0] == 0:
                return y
            coeff = self.inverse_low_rank_right.T.dot(interface_rhs)
            return y + self.inverse_low_rank_left.dot(self.inverse_low_rank_singular * coeff)
        if self.low_rank_basis.shape[1] == 0:
            return y
        reduced_rhs = self.low_rank_basis.T.dot(y)
        alpha = self.woodbury_small_factor.dot(reduced_rhs)
        return y - self.woodbury_left.dot(alpha)

    def apply(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float64)
        out = np.zeros_like(vec)
        core_solution = self.base._apply_core_only(vec)
        out[self.base.core_rows] = core_solution[self.base.core_rows]
        if self.base.interface_rows.shape[0] > 0:
            interface_rhs = vec[self.base.interface_rows] - self.base.matrix[np.ix_(self.base.interface_rows, self.base.core_rows)].dot(core_solution[self.base.core_rows])
            interface_solution = self._apply_interface_hybrid_inverse(interface_rhs)
            out[self.base.interface_rows] = interface_solution
            out[self.base.core_rows] = out[self.base.core_rows] - self.base.core_inverse_interface[self.base.core_rows, :].dot(interface_solution)
        out[self.base.uncovered_mask] = self.base.uncovered_scales[self.base.uncovered_mask] * vec[self.base.uncovered_mask]
        return out

    def metadata(self) -> Dict[str, Any]:
        metadata = self.sparse.metadata()
        interface_count = int(self.base.interface_rows.shape[0])
        low_rank_nnz = int(np.count_nonzero(np.abs(self.low_rank_basis) > self.eps))
        metadata['preconditioner_mode'] = 'hybrid_sparse_schur_low_rank_' + self.strategy
        metadata['schur_factor_mode'] = self.factor_mode
        metadata['hybrid_apply_mode'] = 'inverse_error_svd_additive' if self.low_rank_mode == 'inverse_error_svd' else 'implicit_woodbury'
        metadata['low_rank_mode'] = self.low_rank_mode
        metadata['low_rank_requested_rank'] = int(self.low_rank_rank)
        metadata['low_rank_effective_rank'] = int(self.low_rank_basis.shape[1])
        metadata['low_rank_strength'] = float(self.low_rank_strength)
        metadata['low_rank_basis_nnz'] = low_rank_nnz
        metadata['low_rank_core_shape'] = [int(item) for item in self.low_rank_core.shape]
        metadata['woodbury_z_shape'] = [int(item) for item in self.woodbury_z.shape]
        metadata['woodbury_small_shape'] = [int(item) for item in self.woodbury_small_factor.shape]
        metadata['inverse_low_rank_left_shape'] = [int(item) for item in self.inverse_low_rank_left.shape]
        metadata['inverse_low_rank_right_shape'] = [int(item) for item in self.inverse_low_rank_right.shape]
        metadata['memory_estimate'] = int(
            metadata.get('memory_estimate', 0)
            + self.low_rank_basis.size * 8
            + self.low_rank_core.size * 8
            + self.woodbury_z.size * 8
            + self.woodbury_left.size * 8
            + self.woodbury_small_factor.size * 8
            + self.inverse_low_rank_left.size * 8
            + self.inverse_low_rank_right.size * 8
            + self.inverse_low_rank_singular.size * 8
        )
        return metadata

