"""Shared linear-system contract for learned Schwarz training and deployment."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np


CHECKPOINT_SCHEMA_VERSION = 4
FEATURE_CONTRACT = "learned_schwarz_v1_abs_rhs_abs_initial_residual"
EFFECTIVE_MATRIX_CONTRACT = "jacobian_plus_gmin_identity"
INITIAL_RESIDUAL_FORMULA = "linear_rhs - effective_matrix @ initial_guess"
LOCAL_SHIFT_CONTRACT = "block_inf_norm_relative_floor_v1"
LOCAL_SHIFT_FLOOR_RELATIVE = 1e-6
INITIAL_GUESS_MODE_RHSOLD = "rhsold"
INITIAL_GUESS_MODE_ZERO = "zero"
INITIAL_GUESS_MODES = frozenset(
    {INITIAL_GUESS_MODE_RHSOLD, INITIAL_GUESS_MODE_ZERO}
)


def require_local_shift_floor_relative(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            "local_shift_floor_relative must be the finite float "
            f"{LOCAL_SHIFT_FLOOR_RELATIVE:.17g}"
        )
    try:
        relative = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "local_shift_floor_relative must be the finite float "
            f"{LOCAL_SHIFT_FLOOR_RELATIVE:.17g}"
        ) from exc
    if (
        not np.isfinite(relative)
        or relative != float(LOCAL_SHIFT_FLOOR_RELATIVE)
    ):
        raise ValueError(
            "local_shift_floor_relative must equal "
            f"{LOCAL_SHIFT_FLOOR_RELATIVE:.17g}"
        )
    return relative


def compute_effective_local_shift_floor(
    effective_local_matrix: Any,
    *,
    local_shift_floor_relative: float = LOCAL_SHIFT_FLOOR_RELATIVE,
) -> float:
    """Return the v4 floor for one already-effective local block matrix."""
    relative = require_local_shift_floor_relative(local_shift_floor_relative)
    raw = np.asarray(effective_local_matrix)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[0] != raw.shape[1]:
        raise ValueError(
            "effective_local_matrix must be a non-empty square matrix"
        )
    if np.iscomplexobj(raw):
        raise ValueError("effective_local_matrix must be real")
    try:
        matrix = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "effective_local_matrix must contain float64-compatible values"
        ) from exc
    if not np.all(np.isfinite(matrix)):
        raise ValueError("effective_local_matrix contains non-finite values")
    block_inf_norm = float(np.max(np.sum(np.abs(matrix), axis=1)))
    if not np.isfinite(block_inf_norm):
        raise ValueError("effective_local_matrix infinity norm is non-finite")
    block_scale = max(block_inf_norm, 1.0)
    floor = float(relative * block_scale)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("computed local shift floor is invalid")
    return floor


def require_initial_guess_mode(value: Any) -> str:
    if not isinstance(value, str) or value not in INITIAL_GUESS_MODES:
        raise ValueError(
            "initial_guess_mode must be one of "
            f"{sorted(INITIAL_GUESS_MODES)}"
        )
    return value


def require_real_finite_vector(
    value: Any,
    *,
    matrix_size: int,
    label: str,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional vector")
    if np.iscomplexobj(raw):
        raise ValueError(f"{label} must be real")
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a float64 vector") from exc
    if array.shape[0] != int(matrix_size):
        raise ValueError(
            f"{label} length {array.shape[0]} does not match "
            f"matrix size {int(matrix_size)}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    return array


def resolve_initial_guess(
    *,
    rhsold: Any,
    matrix_size: int,
    initial_guess_mode: str,
) -> np.ndarray:
    mode = require_initial_guess_mode(initial_guess_mode)
    if mode == INITIAL_GUESS_MODE_ZERO:
        return np.zeros(int(matrix_size), dtype=np.float64)
    return require_real_finite_vector(
        rhsold,
        matrix_size=int(matrix_size),
        label="rhsold",
    ).copy()


def compute_initial_residual(
    *,
    effective_matrix: Any,
    linear_rhs: Any,
    initial_guess: Any,
) -> np.ndarray:
    shape = getattr(effective_matrix, "shape", None)
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or int(shape[0]) == 0
        or int(shape[0]) != int(shape[1])
    ):
        raise ValueError("effective_matrix must be a non-empty square matrix")
    matrix_size = int(shape[0])
    rhs = require_real_finite_vector(
        linear_rhs,
        matrix_size=matrix_size,
        label="linear_rhs",
    )
    guess = require_real_finite_vector(
        initial_guess,
        matrix_size=matrix_size,
        label="initial_guess",
    )
    product = np.asarray(effective_matrix @ guess)
    if product.ndim != 1 or product.shape[0] != matrix_size:
        raise ValueError("effective_matrix @ initial_guess has invalid shape")
    if np.iscomplexobj(product):
        raise ValueError("effective_matrix @ initial_guess must be real")
    product = np.asarray(product, dtype=np.float64)
    if not np.all(np.isfinite(product)):
        raise ValueError("effective_matrix @ initial_guess contains non-finite values")
    residual = rhs - product
    if not np.all(np.isfinite(residual)):
        raise ValueError("initial_residual contains non-finite values")
    return residual


def validate_learned_schwarz_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    *,
    expected_initial_guess_mode: Optional[str] = None,
) -> str:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("learned Schwarz checkpoint must contain a mapping")
    if int(checkpoint.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "learned Schwarz checkpoint schema_version must be "
            f"{CHECKPOINT_SCHEMA_VERSION}"
        )
    if checkpoint.get("feature_contract") != FEATURE_CONTRACT:
        raise ValueError("learned Schwarz checkpoint feature_contract mismatch")
    if checkpoint.get("effective_matrix_contract") != EFFECTIVE_MATRIX_CONTRACT:
        raise ValueError(
            "learned Schwarz checkpoint effective_matrix_contract mismatch"
        )
    if checkpoint.get("initial_residual_formula") != INITIAL_RESIDUAL_FORMULA:
        raise ValueError(
            "learned Schwarz checkpoint initial_residual_formula mismatch"
        )
    if checkpoint.get("model_kind") == "learned_schwarz_v1":
        if checkpoint.get("local_shift_contract") != LOCAL_SHIFT_CONTRACT:
            raise ValueError(
                "learned Schwarz checkpoint local_shift_contract must equal "
                f"{LOCAL_SHIFT_CONTRACT}"
            )
        require_local_shift_floor_relative(
            checkpoint.get("local_shift_floor_relative")
        )
    mode = require_initial_guess_mode(checkpoint.get("initial_guess_mode"))
    if expected_initial_guess_mode is not None:
        expected = require_initial_guess_mode(expected_initial_guess_mode)
        if mode != expected:
            raise ValueError(
                "learned Schwarz checkpoint initial_guess_mode mismatch: "
                f"checkpoint={mode}, requested={expected}"
            )
    return mode
