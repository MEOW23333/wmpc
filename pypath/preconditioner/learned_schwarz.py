from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pypath.preconditioner.block_schwarz import BlockSchwarzPlan

from pypath.preconditioner.linear_system_contract import (
    compute_effective_local_shift_floor,
)


BLOCK_FEATURE_DIM = 9
ROW_FEATURE_DIM = 14
CORRECTION_FEATURE_DIM = 4


@dataclass
class LearnedSchwarzSample:
    matrix: torch.Tensor
    blocks: List[torch.Tensor]
    block_features: torch.Tensor
    neighbor_block_features: torch.Tensor
    row_features: List[torch.Tensor]
    row_coverage_count: torch.Tensor
    fallback_scales: torch.Tensor
    lambda_floors: torch.Tensor
    gmin: float = 0.0


def build_learned_schwarz_sample(
    *,
    matrix: np.ndarray,
    plan: BlockSchwarzPlan,
    linear_rhs: Optional[np.ndarray] = None,
    initial_residual: Optional[np.ndarray] = None,
    gmin: float = 0.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> LearnedSchwarzSample:
    matrix_np = np.asarray(matrix, dtype=np.float64)
    matrix_size = int(matrix_np.shape[0])
    linear_rhs_np = (
        np.zeros(matrix_size, dtype=np.float64)
        if linear_rhs is None
        else np.asarray(linear_rhs, dtype=np.float64)
    )
    initial_residual_np = (
        np.zeros(matrix_size, dtype=np.float64)
        if initial_residual is None
        else np.asarray(initial_residual, dtype=np.float64)
    )
    coverage_count_np = np.zeros(matrix_size, dtype=np.float64)
    for rows in plan.blocks:
        coverage_count_np[np.asarray(rows, dtype=np.int64)] += 1.0

    row_abs_sum = np.abs(matrix_np).sum(axis=1)
    col_abs_sum = np.abs(matrix_np).sum(axis=0)
    diag_abs = np.abs(np.diag(matrix_np))
    block_features: List[List[float]] = []
    row_features: List[torch.Tensor] = []
    blocks: List[torch.Tensor] = []
    lambda_floors: List[float] = []
    candidate_by_index: Dict[int, Dict] = {
        idx: candidate for idx, candidate in enumerate(plan.block_candidates)
    }

    for block_id, rows_raw in enumerate(plan.blocks):
        rows_np = np.asarray(rows_raw, dtype=np.int64)
        block_matrix = matrix_np[np.ix_(rows_np, rows_np)]
        lambda_floors.append(
            compute_effective_local_shift_floor(block_matrix)
        )
        nnz = int(np.count_nonzero(block_matrix))
        size = int(rows_np.shape[0])
        density = float(nnz / max(size * size, 1))
        fro_norm = float(np.linalg.norm(block_matrix, ord="fro"))
        diag_norm = float(np.linalg.norm(np.diag(block_matrix)))
        diag_sum = float(np.abs(np.diag(block_matrix)).sum())
        offdiag_sum = float(max(np.abs(block_matrix).sum() - diag_sum, 0.0))
        diag_dominance = float(diag_sum / max(offdiag_sum, 1e-30))
        block_features.append(
            [
                float(size),
                float(nnz),
                density,
                _safe_log(fro_norm),
                _safe_log(diag_norm),
                _safe_log(diag_dominance),
                float(gmin),
                float(np.linalg.norm(linear_rhs_np[rows_np])),
                float(np.linalg.norm(initial_residual_np[rows_np])),
            ]
        )

        candidate = candidate_by_index.get(block_id, {})
        row_role_by_index = candidate.get("row_role_by_index") or {}
        per_row_features = []
        for row in rows_np.tolist():
            role = str(row_role_by_index.get(str(int(row) + 1), "unknown"))
            role_flags = _role_flags(role)
            row_to_block = float(np.linalg.norm(matrix_np[row, rows_np]))
            block_to_row = float(np.linalg.norm(matrix_np[rows_np, row]))
            per_row_features.append(
                [
                    *role_flags,
                    _safe_log(float(diag_abs[row])),
                    _safe_log(float(row_abs_sum[row])),
                    _safe_log(float(col_abs_sum[row])),
                    _safe_log(row_to_block),
                    _safe_log(block_to_row),
                    float(coverage_count_np[row]),
                    float(abs(linear_rhs_np[row])),
                    float(abs(initial_residual_np[row])),
                    float(gmin),
                ]
            )
        blocks.append(torch.as_tensor(rows_np, dtype=torch.long, device=device))
        row_features.append(torch.as_tensor(per_row_features, dtype=dtype, device=device))

    block_features_np = np.asarray(block_features, dtype=np.float64)
    neighbor_features_np = _build_neighbor_block_features(blocks, block_features_np, matrix_size)

    return LearnedSchwarzSample(
        matrix=torch.as_tensor(matrix_np, dtype=dtype, device=device),
        blocks=blocks,
        block_features=torch.as_tensor(block_features_np, dtype=dtype, device=device),
        neighbor_block_features=torch.as_tensor(neighbor_features_np, dtype=dtype, device=device),
        row_features=row_features,
        row_coverage_count=torch.as_tensor(coverage_count_np, dtype=dtype, device=device),
        fallback_scales=torch.as_tensor(plan.uncovered_scales, dtype=dtype, device=device),
        lambda_floors=torch.as_tensor(
            np.asarray(lambda_floors, dtype=np.float64),
            dtype=dtype,
            device=device,
        ),
        gmin=float(gmin),
    )


LEARNED_SCHWARZ_PARAMETER_MODE_FIXED_SAME_BLOCKS = "fixed_same_blocks"
LEARNED_SCHWARZ_PARAMETER_MODE_LEARNED_SHIFT_ONLY = "learned_shift_only"
LEARNED_SCHWARZ_PARAMETER_MODE_LEARNED_OVERLAP_WEIGHTS_ONLY = (
    "learned_overlap_weights_only"
)
LEARNED_SCHWARZ_PARAMETER_MODE_LEARNED_FULL = "learned_full"
LEARNED_SCHWARZ_PARAMETER_MODES = frozenset(
    {
        LEARNED_SCHWARZ_PARAMETER_MODE_FIXED_SAME_BLOCKS,
        LEARNED_SCHWARZ_PARAMETER_MODE_LEARNED_SHIFT_ONLY,
        LEARNED_SCHWARZ_PARAMETER_MODE_LEARNED_OVERLAP_WEIGHTS_ONLY,
        LEARNED_SCHWARZ_PARAMETER_MODE_LEARNED_FULL,
    }
)
DEFAULT_LEARNED_SCHWARZ_PARAMETER_MODE = (
    LEARNED_SCHWARZ_PARAMETER_MODE_LEARNED_FULL
)


def require_learned_schwarz_parameter_mode(value: object) -> str:
    if not isinstance(value, str) or value not in LEARNED_SCHWARZ_PARAMETER_MODES:
        raise ValueError(
            "learned Schwarz parameter mode must be one of "
            f"{sorted(LEARNED_SCHWARZ_PARAMETER_MODES)}"
        )
    return value


def _uniform_overlap_weights(
    *,
    sample: LearnedSchwarzSample,
    dtype: torch.dtype,
    device: torch.device,
) -> List[torch.Tensor]:
    if not sample.blocks:
        return []
    coverage = sample.row_coverage_count.to(dtype=dtype, device=device)
    if coverage.ndim != 1 or coverage.shape[0] != int(sample.matrix.shape[0]):
        raise ValueError("learned Schwarz row coverage has invalid shape")
    weights: List[torch.Tensor] = []
    for rows in sample.blocks:
        block_coverage = coverage.index_select(0, rows).clamp_min(0.0)
        if not bool(torch.all(torch.isfinite(block_coverage)).item()):
            raise ValueError("learned Schwarz row coverage is non-finite")
        if bool(torch.any(block_coverage <= 0.0).item()):
            raise ValueError("learned Schwarz block row has zero coverage")
        weights.append(torch.reciprocal(block_coverage))
    return weights


def apply_learned_schwarz_parameter_mode(
    *,
    sample: LearnedSchwarzSample,
    lambda_pred: torch.Tensor,
    lambda_floor: torch.Tensor,
    learned_lambdas: torch.Tensor,
    learned_weights: List[torch.Tensor],
    parameter_mode: object = DEFAULT_LEARNED_SCHWARZ_PARAMETER_MODE,
) -> Dict[str, object]:
    """Select auditable shift and overlap-weight ablations for one sample."""
    mode = require_learned_schwarz_parameter_mode(parameter_mode)
    expected_count = len(sample.blocks)
    if (
        lambda_pred.ndim != 1
        or lambda_floor.ndim != 1
        or learned_lambdas.ndim != 1
        or lambda_pred.shape[0] != expected_count
        or lambda_floor.shape != lambda_pred.shape
        or learned_lambdas.shape != lambda_pred.shape
        or len(learned_weights) != expected_count
    ):
        raise ValueError("learned Schwarz parameter tensors do not match blocks")
    if not (
        bool(torch.all(torch.isfinite(lambda_pred)).item())
        and bool(torch.all(torch.isfinite(lambda_floor)).item())
        and bool(torch.all(torch.isfinite(learned_lambdas)).item())
    ):
        raise ValueError("learned Schwarz parameter tensors are non-finite")
    if bool(torch.any(learned_lambdas < lambda_floor).item()):
        raise ValueError("learned Schwarz effective shifts are below local floors")

    use_uniform_weights = mode in {
        LEARNED_SCHWARZ_PARAMETER_MODE_FIXED_SAME_BLOCKS,
        LEARNED_SCHWARZ_PARAMETER_MODE_LEARNED_SHIFT_ONLY,
    }
    use_floor_shifts = mode in {
        LEARNED_SCHWARZ_PARAMETER_MODE_FIXED_SAME_BLOCKS,
        LEARNED_SCHWARZ_PARAMETER_MODE_LEARNED_OVERLAP_WEIGHTS_ONLY,
    }
    weights = (
        _uniform_overlap_weights(
            sample=sample,
            dtype=lambda_floor.dtype,
            device=lambda_floor.device,
        )
        if use_uniform_weights
        else learned_weights
    )
    lambdas = lambda_floor if use_floor_shifts else learned_lambdas
    return {
        "parameter_mode": mode,
        "lambda_pred": lambda_pred,
        "lambda_floor": lambda_floor,
        "learned_lambdas": learned_lambdas,
        "lambdas": lambdas,
        "learned_weights": learned_weights,
        "weights": weights,
        "shift_source": "lambda_floor" if use_floor_shifts else "learned_effective",
        "weight_source": "uniform_row_coverage" if use_uniform_weights else "learned_overlap",
    }


class LearnedSchwarzPreconditioner(nn.Module):
    def __init__(
        self,
        *,
        block_feature_dim: int = BLOCK_FEATURE_DIM,
        row_feature_dim: int = ROW_FEATURE_DIM,
        hidden_dim: int = 64,
        lambda_min: float = 0.0,
        lambda_scale: float = 1e-6,
    ) -> None:
        super().__init__()
        self.lambda_min = float(lambda_min)
        self.lambda_scale = float(lambda_scale)
        self.lambda_mlp = nn.Sequential(
            nn.Linear(block_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.weight_mlp = nn.Sequential(
            nn.Linear(block_feature_dim * 2 + row_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("block_feature_mean", torch.zeros(block_feature_dim))
        self.register_buffer("block_feature_std", torch.ones(block_feature_dim))
        self.register_buffer("row_feature_mean", torch.zeros(row_feature_dim))
        self.register_buffer("row_feature_std", torch.ones(row_feature_dim))
        self._initialize_conservative_parameters()

    def _initialize_conservative_parameters(self) -> None:
        last_lambda = self.lambda_mlp[-1]
        last_weight = self.weight_mlp[-1]
        nn.init.zeros_(last_lambda.weight)
        nn.init.constant_(last_lambda.bias, -50.0)
        nn.init.zeros_(last_weight.weight)
        nn.init.zeros_(last_weight.bias)

    def initialize_lambda_prediction(self, target_lambda: float) -> None:
        """Set a uniform initial absolute block shift by inverse softplus."""
        target = float(target_lambda)
        if not np.isfinite(target) or target <= float(self.lambda_min):
            raise ValueError(
                "target_lambda must be finite and exceed lambda_min"
            )
        if not np.isfinite(self.lambda_scale) or float(self.lambda_scale) <= 0.0:
            raise ValueError("lambda_scale must be finite and positive")
        softplus_target = (target - float(self.lambda_min)) / float(self.lambda_scale)
        if not np.isfinite(softplus_target) or softplus_target <= 0.0:
            raise ValueError("target_lambda cannot be represented by lambda_scale")
        if softplus_target > 20.0:
            inverse = softplus_target + np.log1p(-np.exp(-softplus_target))
        else:
            inverse = np.log(np.expm1(softplus_target))
        if not np.isfinite(inverse):
            raise ValueError("inverse softplus initialization is non-finite")
        last_lambda = self.lambda_mlp[-1]
        with torch.no_grad():
            last_lambda.weight.zero_()
            last_lambda.bias.fill_(float(inverse))

    def set_feature_stats(
        self,
        *,
        block_mean: torch.Tensor,
        block_std: torch.Tensor,
        row_mean: torch.Tensor,
        row_std: torch.Tensor,
    ) -> None:
        self.block_feature_mean.copy_(block_mean.to(dtype=self.block_feature_mean.dtype, device=self.block_feature_mean.device))
        self.block_feature_std.copy_(block_std.to(dtype=self.block_feature_std.dtype, device=self.block_feature_std.device).clamp_min(1e-12))
        self.row_feature_mean.copy_(row_mean.to(dtype=self.row_feature_mean.dtype, device=self.row_feature_mean.device))
        self.row_feature_std.copy_(row_std.to(dtype=self.row_feature_std.dtype, device=self.row_feature_std.device).clamp_min(1e-12))

    def _normalize_block_features(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.block_feature_mean.to(dtype=features.dtype, device=features.device)) / self.block_feature_std.to(dtype=features.dtype, device=features.device)

    def _normalize_row_features(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.row_feature_mean.to(dtype=features.dtype, device=features.device)) / self.row_feature_std.to(dtype=features.dtype, device=features.device)

    def predict_parameters(
        self,
        sample: LearnedSchwarzSample,
        *,
        parameter_mode: object = DEFAULT_LEARNED_SCHWARZ_PARAMETER_MODE,
    ) -> Dict[str, object]:
        normalized_block_features = self._normalize_block_features(sample.block_features)
        normalized_neighbor_features = self._normalize_block_features(
            sample.neighbor_block_features
        )
        lambda_logits = self.lambda_mlp(normalized_block_features).squeeze(-1)
        lambda_pred = self.lambda_scale * F.softplus(lambda_logits) + self.lambda_min
        lambda_floor = sample.lambda_floors.to(
            dtype=lambda_pred.dtype,
            device=lambda_pred.device,
        )
        if lambda_floor.ndim != 1 or lambda_floor.shape != lambda_pred.shape:
            raise ValueError("learned Schwarz block shift shapes do not match")
        if not bool(torch.all(torch.isfinite(lambda_pred)).item()):
            raise ValueError("learned Schwarz predicted non-finite block shifts")
        if not bool(torch.all(torch.isfinite(lambda_floor)).item()):
            raise ValueError("learned Schwarz block shift floors are non-finite")
        learned_lambdas = torch.maximum(lambda_pred, lambda_floor)
        raw_scores: List[torch.Tensor] = []
        for block_id, rows in enumerate(sample.blocks):
            block_feat = normalized_block_features[block_id].expand(rows.numel(), -1)
            neighbor_feat = normalized_neighbor_features[block_id].expand(
                rows.numel(),
                -1,
            )
            row_feat = self._normalize_row_features(sample.row_features[block_id])
            features = torch.cat([block_feat, neighbor_feat, row_feat], dim=-1)
            raw_scores.append(self.weight_mlp(features).squeeze(-1))
        learned_weights = _normalize_block_row_scores(
            rows_by_block=sample.blocks,
            scores_by_block=raw_scores,
            matrix_size=int(sample.matrix.shape[0]),
        )
        selected = apply_learned_schwarz_parameter_mode(
            sample=sample,
            lambda_pred=lambda_pred,
            lambda_floor=lambda_floor,
            learned_lambdas=learned_lambdas,
            learned_weights=learned_weights,
            parameter_mode=parameter_mode,
        )
        return {
            **selected,
            "raw_scores": raw_scores,
        }

    def apply(
        self,
        sample: LearnedSchwarzSample,
        vec: torch.Tensor,
        *,
        parameter_mode: object = DEFAULT_LEARNED_SCHWARZ_PARAMETER_MODE,
    ) -> torch.Tensor:
        params = self.predict_parameters(sample, parameter_mode=parameter_mode)
        lambdas = params["lambdas"]
        weights_by_block = params["weights"]
        matrix_size = int(sample.matrix.shape[0])
        out = torch.zeros(
            matrix_size,
            dtype=sample.matrix.dtype,
            device=sample.matrix.device,
        )
        covered = torch.zeros(
            matrix_size,
            dtype=torch.bool,
            device=sample.matrix.device,
        )

        for block_id, rows in enumerate(sample.blocks):
            local_matrix = sample.matrix.index_select(0, rows).index_select(1, rows)
            eye = torch.eye(
                rows.numel(),
                dtype=sample.matrix.dtype,
                device=sample.matrix.device,
            )
            shifted = local_matrix + lambdas[block_id] * eye
            local_rhs = vec.index_select(0, rows)
            local_solution = torch.linalg.solve(
                shifted,
                local_rhs.unsqueeze(-1),
            ).squeeze(-1)
            weighted_solution = weights_by_block[block_id] * local_solution
            out.index_add_(0, rows, weighted_solution)
            covered[rows] = True

        fallback = sample.fallback_scales * vec
        out = torch.where(covered, out, fallback)
        return out

    def probe_loss(
        self,
        sample: LearnedSchwarzSample,
        probes: torch.Tensor,
        eps: float = 1e-30,
        *,
        parameter_mode: object = DEFAULT_LEARNED_SCHWARZ_PARAMETER_MODE,
    ) -> torch.Tensor:
        if probes.ndim == 1:
            probes = probes.unsqueeze(0)
        losses = []
        for probe in probes:
            z = self.apply(sample, probe, parameter_mode=parameter_mode)
            residual = sample.matrix.matmul(z) - probe
            losses.append(
                torch.sum(residual * residual)
                / (torch.sum(probe * probe) + float(eps))
            )
        return torch.stack(losses).mean()
class BoundaryCorrectionPreconditioner(nn.Module):
    """Fixed core block-Jacobi plus a learned additive boundary correction."""

    def __init__(
        self,
        *,
        block_feature_dim: int = BLOCK_FEATURE_DIM,
        row_feature_dim: int = ROW_FEATURE_DIM,
        correction_feature_dim: int = CORRECTION_FEATURE_DIM,
        hidden_dim: int = 64,
        correction_scale: float = 1.0,
        max_correction_ratio: float = 0.0,
        projection_mode: str = "none",
        projection_max_scale: float = 1.0,
        local_solution_gain_limit: float = 0.0,
        block_contribution_budget_ratio: float = 0.0,
        block_contribution_absolute_cap: float = 0.0,
    ) -> None:
        super().__init__()
        self.correction_scale = float(correction_scale)
        self.correction_feature_dim = int(correction_feature_dim)
        self.max_correction_ratio = float(max_correction_ratio)
        self.projection_mode = str(projection_mode)
        self.projection_max_scale = float(projection_max_scale)
        self.local_solution_gain_limit = float(local_solution_gain_limit)
        self.block_contribution_budget_ratio = float(block_contribution_budget_ratio)
        self.block_contribution_absolute_cap = float(block_contribution_absolute_cap)
        self.correction_mlp = nn.Sequential(
            nn.Linear(block_feature_dim * 2 + row_feature_dim + correction_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("block_feature_mean", torch.zeros(block_feature_dim))
        self.register_buffer("block_feature_std", torch.ones(block_feature_dim))
        self.register_buffer("row_feature_mean", torch.zeros(row_feature_dim))
        self.register_buffer("row_feature_std", torch.ones(row_feature_dim))
        self._initialize_zero_correction()

    def _initialize_zero_correction(self) -> None:
        last_layer = self.correction_mlp[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

    def set_feature_stats(
        self,
        *,
        block_mean: torch.Tensor,
        block_std: torch.Tensor,
        row_mean: torch.Tensor,
        row_std: torch.Tensor,
    ) -> None:
        self.block_feature_mean.copy_(block_mean.to(dtype=self.block_feature_mean.dtype, device=self.block_feature_mean.device))
        self.block_feature_std.copy_(block_std.to(dtype=self.block_feature_std.dtype, device=self.block_feature_std.device).clamp_min(1e-12))
        self.row_feature_mean.copy_(row_mean.to(dtype=self.row_feature_mean.dtype, device=self.row_feature_mean.device))
        self.row_feature_std.copy_(row_std.to(dtype=self.row_feature_std.dtype, device=self.row_feature_std.device).clamp_min(1e-12))

    def _normalize_block_features(self, features: torch.Tensor) -> torch.Tensor:
        mean = self.block_feature_mean.to(dtype=features.dtype, device=features.device)
        std = self.block_feature_std.to(dtype=features.dtype, device=features.device)
        return (features - mean) / std

    def _normalize_row_features(self, features: torch.Tensor) -> torch.Tensor:
        mean = self.row_feature_mean.to(dtype=features.dtype, device=features.device)
        std = self.row_feature_std.to(dtype=features.dtype, device=features.device)
        return (features - mean) / std

    def _build_correction_row_features(
        self,
        *,
        boundary_sample: LearnedSchwarzSample,
        rows: torch.Tensor,
        correction_input: torch.Tensor,
    ) -> torch.Tensor:
        local_values = correction_input.index_select(0, rows)
        abs_values = torch.abs(local_values)
        block_norm = torch.linalg.vector_norm(local_values).clamp_min(1e-30)
        block_mean_abs = abs_values.mean().clamp_min(1e-30)
        base_features = torch.stack(
            [
                torch.log(abs_values + 1e-30),
                torch.log(block_norm).expand_as(abs_values),
                abs_values / block_norm,
                abs_values / block_mean_abs,
            ],
            dim=-1,
        )
        if self.correction_feature_dim <= base_features.shape[1]:
            return base_features[:, : self.correction_feature_dim]
        padding = torch.zeros(
            (base_features.shape[0], self.correction_feature_dim - base_features.shape[1]),
            dtype=base_features.dtype,
            device=base_features.device,
        )
        return torch.cat([base_features, padding], dim=-1)

    def apply_core(self, core_sample: LearnedSchwarzSample, vec: torch.Tensor) -> torch.Tensor:
        matrix_size = int(core_sample.matrix.shape[0])
        out = torch.zeros(matrix_size, dtype=core_sample.matrix.dtype, device=core_sample.matrix.device)
        covered = torch.zeros(matrix_size, dtype=torch.bool, device=core_sample.matrix.device)
        for rows in core_sample.blocks:
            local_matrix = core_sample.matrix.index_select(0, rows).index_select(1, rows)
            local_rhs = vec.index_select(0, rows)
            local_solution = torch.linalg.solve(local_matrix, local_rhs.unsqueeze(-1)).squeeze(-1)
            out.index_copy_(0, rows, local_solution)
            covered[rows] = True
        fallback = core_sample.fallback_scales * vec
        return torch.where(covered, out, fallback)

    def predict_correction_coefficients(
        self,
        boundary_sample: LearnedSchwarzSample,
        correction_input: torch.Tensor,
    ) -> List[torch.Tensor]:
        normalized_block_features = self._normalize_block_features(boundary_sample.block_features)
        normalized_neighbor_features = self._normalize_block_features(boundary_sample.neighbor_block_features)
        coefficients: List[torch.Tensor] = []
        for block_id, rows in enumerate(boundary_sample.blocks):
            block_feat = normalized_block_features[block_id].expand(rows.numel(), -1)
            neighbor_feat = normalized_neighbor_features[block_id].expand(rows.numel(), -1)
            row_feat = self._normalize_row_features(boundary_sample.row_features[block_id])
            correction_feat = self._build_correction_row_features(
                boundary_sample=boundary_sample,
                rows=rows,
                correction_input=correction_input,
            )
            features = torch.cat([block_feat, neighbor_feat, row_feat, correction_feat], dim=-1)
            raw = self.correction_mlp(features).squeeze(-1)
            coefficients.append(self.correction_scale * torch.tanh(raw))
        return coefficients

    def _local_correction_contributions(
        self,
        boundary_sample: LearnedSchwarzSample,
        vec: torch.Tensor,
    ) -> List[tuple[torch.Tensor, torch.Tensor]]:
        contributions: List[tuple[torch.Tensor, torch.Tensor]] = []
        coefficients = self.predict_correction_coefficients(boundary_sample, vec)
        for rows, coeff in zip(boundary_sample.blocks, coefficients):
            local_matrix = boundary_sample.matrix.index_select(0, rows).index_select(1, rows)
            local_rhs = vec.index_select(0, rows)
            local_solution = torch.linalg.solve(local_matrix, local_rhs.unsqueeze(-1)).squeeze(-1)
            if self.local_solution_gain_limit > 0.0:
                rhs_norm = torch.linalg.vector_norm(local_rhs).clamp_min(1e-30)
                solution_norm = torch.linalg.vector_norm(local_solution)
                max_norm = float(self.local_solution_gain_limit) * rhs_norm
                scale = torch.clamp(max_norm / solution_norm.clamp_min(1e-30), max=1.0)
                local_solution = local_solution * scale
            contributions.append((rows, coeff * local_solution))
        return contributions

    def apply_correction(
        self,
        boundary_sample: LearnedSchwarzSample,
        vec: torch.Tensor,
        core_solution: Optional[torch.Tensor] = None,
        budget_reference: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        matrix_size = int(boundary_sample.matrix.shape[0])
        out = torch.zeros(matrix_size, dtype=boundary_sample.matrix.dtype, device=boundary_sample.matrix.device)
        contributions = self._local_correction_contributions(boundary_sample, vec)
        block_budget = None
        reference = budget_reference if budget_reference is not None else core_solution
        if self.block_contribution_budget_ratio > 0.0 and reference is not None and contributions:
            core_norm = torch.linalg.vector_norm(reference)
            block_budget = (
                float(self.block_contribution_budget_ratio)
                * core_norm
                / (float(len(contributions)) ** 0.5)
            )
        if self.block_contribution_absolute_cap > 0.0:
            absolute_budget = out.new_tensor(float(self.block_contribution_absolute_cap))
            block_budget = absolute_budget if block_budget is None else torch.minimum(block_budget, absolute_budget)
        for rows, contribution in contributions:
            if block_budget is not None:
                contribution_norm = torch.linalg.vector_norm(contribution)
                scale = torch.clamp(block_budget / contribution_norm.clamp_min(1e-30), max=1.0)
                contribution = contribution * scale
            out.index_add_(0, rows, contribution)
        return out

    def limit_correction(self, correction: torch.Tensor, core_solution: torch.Tensor) -> torch.Tensor:
        if self.max_correction_ratio <= 0.0:
            return correction
        correction_norm = torch.linalg.vector_norm(correction)
        core_norm = torch.linalg.vector_norm(core_solution).clamp_min(1e-30)
        max_norm = float(self.max_correction_ratio) * core_norm
        scale = torch.clamp(max_norm / correction_norm.clamp_min(1e-30), max=1.0)
        return correction * scale

    def project_correction_by_action(
        self,
        *,
        correction: torch.Tensor,
        correction_action: torch.Tensor,
        core_residual: torch.Tensor,
        eps: float = 1e-30,
    ) -> torch.Tensor:
        if self.projection_mode == "none":
            return correction
        if self.projection_mode != "residual_scalar":
            raise ValueError(f"Unsupported projection_mode: {self.projection_mode}")
        numerator = torch.sum(correction_action * core_residual)
        denominator = torch.sum(correction_action * correction_action) + float(eps)
        scale = numerator / denominator
        scale = torch.clamp(scale, min=0.0, max=float(self.projection_max_scale))
        return correction * scale

    def correction_diagnostics(
        self,
        core_sample: LearnedSchwarzSample,
        boundary_sample: LearnedSchwarzSample,
        vec: torch.Tensor,
        eps: float = 1e-30,
    ) -> Dict[str, float]:
        with torch.no_grad():
            core_solution = self.apply_core(core_sample, vec)
            core_residual = vec - core_sample.matrix.matmul(core_solution)
            raw_correction = self.apply_correction(
                boundary_sample,
                core_residual,
                core_solution,
                budget_reference=core_solution,
            )
            raw_action = core_sample.matrix.matmul(raw_correction)
            numerator = torch.sum(raw_action * core_residual)
            denominator = torch.sum(raw_action * raw_action) + float(eps)
            raw_alpha = numerator / denominator
            projected_correction = self.project_correction_by_action(
                correction=raw_correction,
                correction_action=raw_action,
                core_residual=core_residual,
                eps=eps,
            )
            limited_correction = self.limit_correction(projected_correction, core_solution)
            projected_action = core_sample.matrix.matmul(limited_correction)
            raw_action_norm = torch.linalg.vector_norm(raw_action)
            residual_norm = torch.linalg.vector_norm(core_residual)
            raw_cos = numerator / (raw_action_norm * residual_norm + float(eps))
            projected_action_norm = torch.linalg.vector_norm(projected_action)
            projected_cos = torch.sum(projected_action * core_residual) / (
                projected_action_norm * residual_norm + float(eps)
            )
            return {
                "core_solution_norm": float(torch.linalg.vector_norm(core_solution).cpu()),
                "core_residual_norm": float(residual_norm.cpu()),
                "raw_correction_norm": float(torch.linalg.vector_norm(raw_correction).cpu()),
                "projected_correction_norm": float(torch.linalg.vector_norm(projected_correction).cpu()),
                "limited_correction_norm": float(torch.linalg.vector_norm(limited_correction).cpu()),
                "raw_action_norm": float(raw_action_norm.cpu()),
                "projected_action_norm": float(projected_action_norm.cpu()),
                "raw_projection_alpha": float(raw_alpha.cpu()),
                "raw_projection_alpha_clipped": float(
                    torch.clamp(raw_alpha, min=0.0, max=float(self.projection_max_scale)).cpu()
                ),
                "raw_alignment_cos": float(raw_cos.cpu()),
                "projected_alignment_cos": float(projected_cos.cpu()),
            }

    def block_correction_diagnostics(
        self,
        boundary_sample: LearnedSchwarzSample,
        correction_input: torch.Tensor,
        eps: float = 1e-30,
    ) -> List[Dict[str, float]]:
        with torch.no_grad():
            coefficients = self.predict_correction_coefficients(boundary_sample, correction_input)
            rows_out: List[Dict[str, float]] = []
            for block_id, (rows, coeff) in enumerate(zip(boundary_sample.blocks, coefficients)):
                local_matrix = boundary_sample.matrix.index_select(0, rows).index_select(1, rows)
                local_rhs = correction_input.index_select(0, rows)
                local_solution = torch.linalg.solve(local_matrix, local_rhs.unsqueeze(-1)).squeeze(-1)
                raw_solution_norm = torch.linalg.vector_norm(local_solution)
                rhs_norm = torch.linalg.vector_norm(local_rhs)
                limited_solution = local_solution
                local_scale = local_solution.new_tensor(1.0)
                if self.local_solution_gain_limit > 0.0:
                    max_norm = float(self.local_solution_gain_limit) * rhs_norm.clamp_min(float(eps))
                    local_scale = torch.clamp(max_norm / raw_solution_norm.clamp_min(float(eps)), max=1.0)
                    limited_solution = local_solution * local_scale
                contribution = coeff * limited_solution
                rows_out.append(
                    {
                        "block_id": int(block_id),
                        "block_size": int(rows.numel()),
                        "local_rhs_norm": float(rhs_norm.cpu()),
                        "raw_local_solution_norm": float(raw_solution_norm.cpu()),
                        "limited_local_solution_norm": float(torch.linalg.vector_norm(limited_solution).cpu()),
                        "local_gain": float((raw_solution_norm / rhs_norm.clamp_min(float(eps))).cpu()),
                        "local_scale": float(local_scale.cpu()),
                        "coefficient_min": float(torch.min(coeff).cpu()),
                        "coefficient_max": float(torch.max(coeff).cpu()),
                        "coefficient_mean": float(torch.mean(coeff).cpu()),
                        "contribution_norm": float(torch.linalg.vector_norm(contribution).cpu()),
                    }
                )
            return rows_out

    def apply(
        self,
        core_sample: LearnedSchwarzSample,
        boundary_sample: LearnedSchwarzSample,
        vec: torch.Tensor,
    ) -> torch.Tensor:
        core_solution = self.apply_core(core_sample, vec)
        core_residual = vec - core_sample.matrix.matmul(core_solution)
        correction = self.apply_correction(
            boundary_sample,
            core_residual,
            core_solution,
            budget_reference=core_solution,
        )
        correction_action = core_sample.matrix.matmul(correction)
        correction = self.project_correction_by_action(
            correction=correction,
            correction_action=correction_action,
            core_residual=core_residual,
        )
        correction = self.limit_correction(correction, core_solution)
        return core_solution + correction

    def probe_loss(
        self,
        core_sample: LearnedSchwarzSample,
        boundary_sample: LearnedSchwarzSample,
        probes: torch.Tensor,
        eps: float = 1e-30,
        residual_floor: float = 1e-6,
        correction_weight: float = 1e-4,
        do_no_harm_weight: float = 0.0,
        do_no_harm_margin: float = 0.0,
        alignment_weight: float = 0.0,
        alignment_min_cos: float = 0.0,
    ) -> torch.Tensor:
        if probes.ndim == 1:
            probes = probes.unsqueeze(0)
        losses = []
        for probe in probes:
            with torch.no_grad():
                core_solution = self.apply_core(core_sample, probe)
                core_residual = probe - core_sample.matrix.matmul(core_solution)
            correction = self.apply_correction(
                boundary_sample,
                core_residual,
                core_solution,
                budget_reference=core_solution,
            )
            correction_action = core_sample.matrix.matmul(correction)
            correction = self.project_correction_by_action(
                correction=correction,
                correction_action=correction_action,
                core_residual=core_residual,
                eps=eps,
            )
            correction = self.limit_correction(correction, core_solution)
            correction_action = core_sample.matrix.matmul(correction)
            correction_residual = correction_action - core_residual
            probe_norm_sq = torch.sum(probe * probe)
            core_residual_norm_sq = torch.sum(core_residual * core_residual)
            fit_denom = core_residual_norm_sq + float(residual_floor) * probe_norm_sq + float(eps)
            fit_loss = torch.sum(correction_residual * correction_residual) / fit_denom
            core_solution_norm_sq = torch.sum(core_solution * core_solution)
            correction_loss = torch.sum(correction * correction) / (core_solution_norm_sq + float(eps))
            harm_ratio = torch.sum(correction_residual * correction_residual) / (core_residual_norm_sq + float(eps))
            harm_loss = torch.relu(harm_ratio - 1.0 - float(do_no_harm_margin))
            action_norm = torch.linalg.vector_norm(correction_action)
            residual_norm = torch.linalg.vector_norm(core_residual)
            alignment_cos = torch.sum(correction_action * core_residual) / (
                action_norm * residual_norm + float(eps)
            )
            alignment_loss = torch.relu(float(alignment_min_cos) - alignment_cos)
            losses.append(
                fit_loss
                + float(correction_weight) * correction_loss
                + float(do_no_harm_weight) * harm_loss
                + float(alignment_weight) * alignment_loss
            )
        return torch.stack(losses).mean()


def make_probe_matrix(
    *,
    matrix_size: int,
    linear_rhs: Optional[np.ndarray] = None,
    initial_residual: Optional[np.ndarray] = None,
    gaussian_count: int = 2,
    seed: int = 0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    rng = np.random.default_rng(int(seed))
    probes: List[np.ndarray] = [
        rng.standard_normal(int(matrix_size)).astype(np.float64)
        for _ in range(int(max(gaussian_count, 0)))
    ]
    if linear_rhs is not None:
        probes.append(np.asarray(linear_rhs, dtype=np.float64))
    if initial_residual is not None:
        probes.append(np.asarray(initial_residual, dtype=np.float64))
    if not probes:
        probes.append(rng.standard_normal(int(matrix_size)).astype(np.float64))
    return torch.as_tensor(np.stack(probes, axis=0), dtype=dtype, device=device)


def _build_neighbor_block_features(
    blocks: List[torch.Tensor],
    block_features: np.ndarray,
    matrix_size: int,
) -> np.ndarray:
    if len(blocks) == 0:
        return np.zeros((0, block_features.shape[1] if block_features.ndim == 2 else BLOCK_FEATURE_DIM), dtype=np.float64)
    row_to_blocks: List[List[int]] = [[] for _ in range(int(matrix_size))]
    for block_id, rows in enumerate(blocks):
        for row in rows.detach().cpu().numpy().astype(np.int64).tolist():
            if 0 <= int(row) < int(matrix_size):
                row_to_blocks[int(row)].append(int(block_id))

    neighbors = [set() for _ in blocks]
    for covering_blocks in row_to_blocks:
        if len(covering_blocks) <= 1:
            continue
        for block_id in covering_blocks:
            neighbors[block_id].update(other for other in covering_blocks if other != block_id)

    output = np.zeros_like(block_features, dtype=np.float64)
    for block_id, neighbor_ids in enumerate(neighbors):
        if neighbor_ids:
            output[block_id] = np.mean(block_features[sorted(neighbor_ids)], axis=0)
    return output


def _normalize_block_row_scores(
    *,
    rows_by_block: List[torch.Tensor],
    scores_by_block: List[torch.Tensor],
    matrix_size: int,
) -> List[torch.Tensor]:
    if not rows_by_block:
        return []
    device = scores_by_block[0].device
    dtype = scores_by_block[0].dtype
    denom = torch.zeros(int(matrix_size), dtype=dtype, device=device)
    exp_scores = [torch.exp(torch.clamp(scores, min=-30.0, max=30.0)) for scores in scores_by_block]
    for rows, exp_score in zip(rows_by_block, exp_scores):
        denom.index_add_(0, rows, exp_score)
    return [
        exp_score / denom.index_select(0, rows).clamp_min(1e-30)
        for rows, exp_score in zip(rows_by_block, exp_scores)
    ]


def _safe_log(value: float, eps: float = 1e-30) -> float:
    return float(np.log(float(abs(value)) + float(eps)))


def _role_flags(role: str) -> List[float]:
    normalized = str(role).lower()
    return [
        float(normalized == "internal_node"),
        float(normalized == "external_pin"),
        float(normalized == "boundary_shared_node"),
        float(normalized == "branch_current"),
        float(normalized not in {"internal_node", "external_pin", "boundary_shared_node", "branch_current"}),
    ]
