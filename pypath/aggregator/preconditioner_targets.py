import math
from typing import Any, Dict, List, Optional

import numpy as np


DEFAULT_PRECONDITIONER_TARGET_MODE = "diag_scale"
DEFAULT_PRECONDITIONER_EPS = 1e-30
PRECONDITIONER_NODE_FEATURE_DIM = 9


def _signed_log10_pair(value: float, eps: float = DEFAULT_PRECONDITIONER_EPS) -> List[float]:
    numeric = float(value)
    sign = 1.0 if numeric == 0.0 else float(np.sign(numeric))
    return [float(math.log10(abs(numeric) + float(eps))), sign]


def _row_kind_flags(row_kind: str) -> List[float]:
    normalized = str(row_kind).lower()
    return [
        float(normalized == "external"),
        float(normalized == "internal"),
        float(normalized == "branch"),
    ]


def infer_row_kind(node_name_lower: str) -> str:
    lowered = str(node_name_lower).lower()
    if lowered.endswith("#branch"):
        return "branch"
    if ("." in lowered) or ("#" in lowered):
        return "internal"
    return "external"


def jacobian_payload_to_matrix(payload: Any) -> Optional[np.ndarray]:
    if payload is None:
        return None
    if isinstance(payload, dict):
        real_part = np.asarray(payload.get("real", []), dtype=np.float64)
        imag_part = np.asarray(payload.get("imag", []), dtype=np.float64)
        if real_part.size == 0:
            return None
        return real_part + 1j * imag_part
    matrix = np.asarray(payload)
    if matrix.size == 0:
        return None
    return matrix


def extract_matrix_row_statistics(
    *,
    jacobian_payload: Any,
    matrix_index: int,
    matrix_size: int,
    node_name_lower: str,
    eps: float = DEFAULT_PRECONDITIONER_EPS,
) -> Dict[str, Any]:
    row_kind = infer_row_kind(node_name_lower)
    base_stats = {
        "matrix_index": int(matrix_index),
        "matrix_size": int(matrix_size),
        "row_kind": row_kind,
        "has_jacobian": False,
        "diag_real": None,
        "diag_imag": None,
        "diag_abs": None,
        "log10_abs_diag": None,
        "row_abs_sum": None,
        "offdiag_abs_sum": None,
        "diag_dominance": None,
    }

    matrix = jacobian_payload_to_matrix(jacobian_payload)
    if matrix is None or matrix.ndim != 2:
        return base_stats

    zero_idx = int(matrix_index) - 1
    if zero_idx < 0 or zero_idx >= matrix.shape[0] or zero_idx >= matrix.shape[1]:
        return base_stats

    row = np.asarray(matrix[zero_idx])
    diag_val = row[zero_idx]
    diag_abs = float(np.abs(diag_val))
    row_abs_sum = float(np.abs(row).sum())
    offdiag_abs_sum = float(max(row_abs_sum - diag_abs, 0.0))
    diag_dominance = float(diag_abs / max(offdiag_abs_sum, float(eps)))

    base_stats.update(
        {
            "has_jacobian": True,
            "diag_real": float(np.real(diag_val)),
            "diag_imag": float(np.imag(diag_val)),
            "diag_abs": diag_abs,
            "log10_abs_diag": float(math.log10(diag_abs + float(eps))),
            "row_abs_sum": row_abs_sum,
            "offdiag_abs_sum": offdiag_abs_sum,
            "diag_dominance": diag_dominance,
        }
    )
    return base_stats


def build_preconditioner_targets(
    *,
    matrix_stats: Dict[str, Any],
    matrix_aux_features: Optional[List[float]] = None,
    target_mode: str = DEFAULT_PRECONDITIONER_TARGET_MODE,
    eps: float = DEFAULT_PRECONDITIONER_EPS,
) -> Dict[str, Any]:
    diag_abs = matrix_stats.get("diag_abs")
    valid = bool(matrix_stats.get("has_jacobian")) and diag_abs is not None
    if valid:
        if target_mode == "diag_scale":
            target_log_diag_scale = float(-math.log10(float(diag_abs) + float(eps)))
        elif target_mode == "inv_sqrt_diag":
            target_log_diag_scale = float(math.log10(1.0 / math.sqrt(float(diag_abs) + float(eps))))
        else:
            raise ValueError(f"Unsupported preconditioner target mode: {target_mode}")
        return {
            "target_mode": str(target_mode),
            "target_log_diag_scale": target_log_diag_scale,
            "target_confidence": 1.0,
            "valid_mask": True,
            "target_source": "jacobian_diag",
        }

    matrix_aux = list(matrix_aux_features or [])
    residual_log = float(matrix_aux[3]) if len(matrix_aux) > 3 else 0.0
    rhsnew_log = float(matrix_aux[5]) if len(matrix_aux) > 5 else 0.0
    proxy_target = float(np.clip(residual_log - rhsnew_log, -6.0, 6.0))
    proxy_strength = abs(proxy_target)
    proxy_confidence = float(np.clip(proxy_strength / 6.0, 0.0, 0.35))
    return {
        "target_mode": "residual_rhs_proxy",
        "target_log_diag_scale": proxy_target,
        "target_confidence": proxy_confidence,
        "valid_mask": True,
        "target_source": "matrix_aux_proxy",
    }


def encode_preconditioner_node_features(
    matrix_stats: Dict[str, Any],
    *,
    eps: float = DEFAULT_PRECONDITIONER_EPS,
) -> List[float]:
    row_kind = str(matrix_stats.get("row_kind", "external"))
    diag_abs = matrix_stats.get("diag_abs")
    row_abs_sum = matrix_stats.get("row_abs_sum")
    offdiag_abs_sum = matrix_stats.get("offdiag_abs_sum")
    diag_dominance = matrix_stats.get("diag_dominance")

    if diag_abs is None:
        diag_log, diag_sign = 0.0, 1.0
    else:
        diag_log, diag_sign = _signed_log10_pair(float(diag_abs), eps=eps)

    row_sum_log = 0.0 if row_abs_sum is None else float(math.log10(float(row_abs_sum) + float(eps)))
    offdiag_sum_log = 0.0 if offdiag_abs_sum is None else float(math.log10(float(offdiag_abs_sum) + float(eps)))
    dominance_log = 0.0 if diag_dominance is None else float(math.log10(float(diag_dominance) + float(eps)))

    return [
        float(matrix_stats.get("has_jacobian", False)),
        *_row_kind_flags(row_kind),
        diag_log,
        diag_sign,
        row_sum_log,
        offdiag_sum_log,
        dominance_log,
    ]
