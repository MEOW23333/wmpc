"""Export learned Schwarz V1 parameters for the native GMRES sidecar."""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pypath.preconditioner.sparse_learned_schwarz import (  # noqa: E402
    SparseLearnedSchwarzV1Preconditioner,
    extract_learned_schwarz_blocks,
    load_learned_schwarz_v1_model,
)
from pypath.utils.external_gmres_prototype import (  # noqa: E402
    _compute_node_map_hash,
)
from pypath.utils.ngspice_utils import (  # noqa: E402
    read_J_sparse,
    read_continuation_step,
    read_linear_system_corpus_step,
)


SCHEMA_VERSION = 5
PRECONDITIONER_MODE = "learned_schwarz_v1_sparse"
FEATURE_CONTRACT = "learned_schwarz_v1_abs_rhs_abs_initial_residual"
LOCAL_SHIFT_CONTRACT = "block_inf_norm_relative_floor_v1"
LOCAL_SHIFT_FLOOR_RELATIVE = 1e-6
INITIAL_RESIDUAL_CONTRACT = "linear_rhs - effective_matrix @ initial_guess"
INITIAL_RESIDUAL_NORM_ATOL = 1e-12
INITIAL_RESIDUAL_NORM_RTOL = 1e-9
INITIAL_GUESS_MODE_RHSOLD = "rhsold"
INITIAL_GUESS_MODE_ZERO = "zero"
INITIAL_GUESS_MODES = frozenset({INITIAL_GUESS_MODE_RHSOLD, INITIAL_GUESS_MODE_ZERO})
VECTOR_SHA256_HEADER = b"schema=pals_vector_f64_v1\n"
ROW_INDEX_BASE = 1
UNCOVERED_ROW_POLICY = "row_sum"
NATIVE_NEWTON_ITER_OFFSET = 1
NATIVE_MIN_BLOCK_SIZE = 2
NATIVE_MAX_BLOCK_SIZE = 32
WEIGHT_SUM_TOLERANCE = 1e-9
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STEP_METADATA_RE = re.compile(
    r"_time_(?P<time>[0-9.eE+-]+)"
    r"_gmin_(?:[0-9]+_)?(?P<gmin>[0-9.eE+-]+)"
    r"_iter_(?P<newton_iter>[0-9]+)(?:\.txt)?$"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_layout_sha256(
    *,
    matrix_size: int,
    block_count: int,
    total_block_rows: int,
    block_offsets: Sequence[int],
    block_rows: Sequence[int],
) -> str:
    """Hash the native layout contract using canonical UTF-8/LF text."""
    text = (
        "schema=learned_schwarz_layout_v1\n"
        f"matrix_size={int(matrix_size)}\n"
        f"block_count={int(block_count)}\n"
        f"total_block_rows={int(total_block_rows)}\n"
        f"block_offsets={','.join(str(int(v)) for v in block_offsets)}\n"
        f"block_rows={','.join(str(int(v)) for v in block_rows)}\n"
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_matrix_fingerprint(matrix: sp.spmatrix) -> str:
    """Hash a canonical real float64 CSR matrix."""
    csr = matrix.tocsr(copy=True)
    csr.sum_duplicates()
    csr.eliminate_zeros()
    csr.sort_indices()
    if csr.shape[0] == 0 or csr.shape[0] != csr.shape[1]:
        raise ValueError("matrix fingerprint requires a non-empty square matrix")
    if np.iscomplexobj(csr.data):
        imag = float(np.max(np.abs(csr.data.imag))) if csr.nnz else 0.0
        real = float(np.max(np.abs(csr.data.real))) if csr.nnz else 0.0
        if imag > 1e-14 * max(real, 1.0):
            raise ValueError(
                "matrix fingerprint requires a real matrix; "
                f"maximum imaginary magnitude is {imag}"
            )
        csr = csr.real
    csr = csr.astype(np.float64, copy=False)
    digest = hashlib.sha256()
    digest.update(b"schema=pals_csr_f64_v1\n")
    digest.update(
        (
            f"rows={int(csr.shape[0])}\n"
            f"cols={int(csr.shape[1])}\n"
            f"nnz={int(csr.nnz)}\n"
        ).encode("ascii")
    )
    digest.update(np.asarray(csr.indptr, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(csr.indices, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(csr.data, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def compute_float64_vector_sha256(
    vector: Any,
    *,
    label: str = "vector",
) -> str:
    """Hash ``header + length + little-endian float64 bytes`` canonically."""
    raw = np.asarray(vector)
    if raw.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional vector")
    if np.iscomplexobj(raw):
        raise ValueError(f"{label} must be a real float64 vector")
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a real float64 vector") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")

    digest = hashlib.sha256()
    digest.update(VECTOR_SHA256_HEADER)
    digest.update(f"length={int(array.shape[0])}\n".encode("ascii"))
    digest.update(np.asarray(array, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _require_float(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _int_array(value: Any, label: str) -> List[int]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an integer array")
    return [_require_int(v, f"{label}[{i}]") for i, v in enumerate(value)]


def _float_array(value: Any, label: str) -> List[float]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a numeric array")
    return [_require_float(v, f"{label}[{i}]") for i, v in enumerate(value)]


def _require_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 64-character SHA256")


def _require_initial_guess_mode(value: Any) -> str:
    if not isinstance(value, str) or value not in INITIAL_GUESS_MODES:
        raise ValueError("initial_guess_mode must be rhsold or zero")
    return value


def validate_native_learned_schwarz_sidecar(
    payload: Mapping[str, Any],
) -> None:
    """Validate every field and flattened-array invariant."""
    if not isinstance(payload, Mapping):
        raise ValueError("sidecar payload must be a mapping")
    if _require_int(payload.get("schema_version"), "schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("preconditioner_mode") != PRECONDITIONER_MODE:
        raise ValueError(f"preconditioner_mode must be {PRECONDITIONER_MODE}")
    if payload.get("feature_contract") != FEATURE_CONTRACT:
        raise ValueError(f"feature_contract must be {FEATURE_CONTRACT}")
    if payload.get("local_shift_contract") != LOCAL_SHIFT_CONTRACT:
        raise ValueError(
            f"local_shift_contract must be {LOCAL_SHIFT_CONTRACT}"
        )
    local_shift_floor_relative = _require_float(
        payload.get("local_shift_floor_relative"),
        "local_shift_floor_relative",
    )
    if local_shift_floor_relative != LOCAL_SHIFT_FLOOR_RELATIVE:
        raise ValueError(
            "local_shift_floor_relative must be "
            f"{LOCAL_SHIFT_FLOOR_RELATIVE}"
        )
    if _require_int(payload.get("row_index_base"), "row_index_base") != 1:
        raise ValueError("row_index_base must be 1")

    size = _require_int(payload.get("matrix_size"), "matrix_size")
    iteration = _require_int(payload.get("newton_iter"), "newton_iter")
    block_count = _require_int(payload.get("block_count"), "block_count")
    total_rows = _require_int(payload.get("total_block_rows"), "total_block_rows")
    if size <= 0:
        raise ValueError("matrix_size must be positive")
    if iteration < 1:
        raise ValueError("newton_iter must be one-based and at least 1")
    if block_count < 0 or total_rows < 0:
        raise ValueError("block counts cannot be negative")
    _require_float(payload.get("time"), "time")
    _require_float(payload.get("gmin"), "gmin")
    for label in (
        "node_map_hash",
        "matrix_fingerprint",
        "layout_sha256",
        "checkpoint_sha256",
        "linear_rhs_sha256",
        "initial_guess_sha256",
    ):
        _require_sha256(payload.get(label), label)
    if payload.get("initial_residual_contract") != INITIAL_RESIDUAL_CONTRACT:
        raise ValueError(
            "initial_residual_contract must be "
            f"{INITIAL_RESIDUAL_CONTRACT}"
        )
    initial_residual_norm_l2 = _require_float(
        payload.get("initial_residual_norm_l2"),
        "initial_residual_norm_l2",
    )
    if initial_residual_norm_l2 < 0.0:
        raise ValueError("initial_residual_norm_l2 must be non-negative")
    initial_residual_norm_atol = _require_float(
        payload.get("initial_residual_norm_atol"),
        "initial_residual_norm_atol",
    )
    if initial_residual_norm_atol != INITIAL_RESIDUAL_NORM_ATOL:
        raise ValueError(
            "initial_residual_norm_atol must be "
            f"{INITIAL_RESIDUAL_NORM_ATOL}"
        )
    initial_residual_norm_rtol = _require_float(
        payload.get("initial_residual_norm_rtol"),
        "initial_residual_norm_rtol",
    )
    if initial_residual_norm_rtol != INITIAL_RESIDUAL_NORM_RTOL:
        raise ValueError(
            "initial_residual_norm_rtol must be "
            f"{INITIAL_RESIDUAL_NORM_RTOL}"
        )
    initial_guess_mode = _require_initial_guess_mode(
        payload.get("initial_guess_mode")
    )
    if initial_guess_mode == INITIAL_GUESS_MODE_ZERO:
        zero_sha256 = compute_float64_vector_sha256(
            np.zeros(size, dtype=np.float64),
            label="zero initial guess",
        )
        if payload["initial_guess_sha256"] != zero_sha256:
            raise ValueError("initial_guess_sha256 does not match zero initial guess")
    if payload.get("uncovered_row_policy") != UNCOVERED_ROW_POLICY:
        raise ValueError(f"uncovered_row_policy must be {UNCOVERED_ROW_POLICY}")

    offsets = _int_array(payload.get("block_offsets"), "block_offsets")
    rows = _int_array(payload.get("block_rows"), "block_rows")
    lambdas = _float_array(payload.get("block_lambdas"), "block_lambdas")
    weights = _float_array(payload.get("block_row_weights"), "block_row_weights")
    if len(offsets) != block_count + 1:
        raise ValueError("block_offsets length must equal block_count + 1")
    if not offsets or offsets[0] != 0:
        raise ValueError("block_offsets must start at zero")
    if any(b < a for a, b in zip(offsets, offsets[1:])):
        raise ValueError("block_offsets must be monotonic")
    if offsets[-1] != total_rows:
        raise ValueError("block_offsets final value must equal total_block_rows")
    if len(rows) != total_rows or len(weights) != total_rows:
        raise ValueError(
            "block_rows and block_row_weights lengths must equal total_block_rows"
        )
    if len(lambdas) != block_count:
        raise ValueError("block_lambdas length must equal block_count")
    if any(row < 1 or row > size for row in rows):
        raise ValueError("block_rows contains an out-of-range matrix row")
    if any(value < 0.0 for value in lambdas):
        raise ValueError("block_lambdas must be finite and non-negative")
    if any(value < 0.0 or value > 1.0 for value in weights):
        raise ValueError("block_row_weights must lie in [0, 1]")

    for block_index in range(block_count):
        begin, end = offsets[block_index : block_index + 2]
        block_size = end - begin
        if not NATIVE_MIN_BLOCK_SIZE <= block_size <= NATIVE_MAX_BLOCK_SIZE:
            raise ValueError("each native block must contain between 2 and 32 rows")
        if len(rows[begin:end]) != len(set(rows[begin:end])):
            raise ValueError("a block cannot contain duplicate rows")

    if rows:
        sums = np.zeros(size, dtype=np.float64)
        row_indices = np.asarray(rows, dtype=np.int64) - 1
        covered = np.zeros(size, dtype=bool)
        covered[row_indices] = True
        np.add.at(
            sums,
            row_indices,
            np.asarray(weights, dtype=np.float64),
        )
        if not np.allclose(
            sums[covered],
            1.0,
            rtol=0.0,
            atol=WEIGHT_SUM_TOLERANCE,
        ):
            error = float(np.max(np.abs(sums[covered] - 1.0)))
            raise ValueError(
                "block_row_weights for each covered row must sum to one; "
                f"maximum error is {error}"
            )

    expected_layout = compute_layout_sha256(
        matrix_size=size,
        block_count=block_count,
        total_block_rows=total_rows,
        block_offsets=offsets,
        block_rows=rows,
    )
    if payload["layout_sha256"] != expected_layout:
        raise ValueError("layout_sha256 does not match the flattened block layout")


def _flatten_parameters(
    exported: Mapping[str, Any],
) -> Tuple[List[int], List[int], List[float], List[float]]:
    blocks = list(exported.get("blocks") or [])
    lambdas = np.asarray(exported.get("lambdas"), dtype=np.float64).reshape(-1)
    weights_by_block = list(exported.get("weights") or [])
    if lambdas.shape[0] != len(blocks):
        raise ValueError("exported lambda count does not match block count")
    if len(weights_by_block) != len(blocks):
        raise ValueError("exported weight count does not match block count")

    offsets = [0]
    flat_rows: List[int] = []
    flat_weights: List[float] = []
    for index, (raw_rows, raw_weights) in enumerate(
        zip(blocks, weights_by_block)
    ):
        rows = np.asarray(raw_rows, dtype=np.int64).reshape(-1)
        weights = np.asarray(raw_weights, dtype=np.float64).reshape(-1)
        if rows.shape[0] != weights.shape[0]:
            raise ValueError(
                f"exported block {index} row and weight lengths differ"
            )
        flat_rows.extend((rows + ROW_INDEX_BASE).tolist())
        flat_weights.extend(weights.tolist())
        offsets.append(len(flat_rows))
    return offsets, flat_rows, lambdas.tolist(), flat_weights


def build_native_learned_schwarz_sidecar_payload(
    *,
    preconditioner: SparseLearnedSchwarzV1Preconditioner,
    effective_matrix: sp.spmatrix,
    node_map: Mapping[str, int],
    checkpoint_path: str,
    time_value: float,
    gmin: float,
    newton_iter: int,
    linear_rhs: Sequence[float],
    initial_guess: Sequence[float],
    initial_residual: Sequence[float],
    initial_guess_mode: str,
) -> Dict[str, Any]:
    """Build and validate one sidecar v5 payload."""
    matrix = effective_matrix.tocsr()
    if matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("effective matrix must be non-empty and square")
    size = int(matrix.shape[0])
    linear_rhs_vector = _required_vector(linear_rhs, size, "linear_rhs")
    initial_guess_vector = _required_vector(initial_guess, size, "initial_guess")
    initial_residual_vector = _required_vector(
        initial_residual,
        size,
        "initial_residual",
    )
    resolved_initial_guess_mode = _require_initial_guess_mode(initial_guess_mode)
    if resolved_initial_guess_mode == INITIAL_GUESS_MODE_ZERO:
        if np.any(initial_guess_vector != 0.0):
            raise ValueError("zero initial_guess_mode requires an all-zero initial_guess")
        initial_guess_vector = np.zeros(size, dtype=np.float64)

    recomputed_initial_residual = linear_rhs_vector - np.asarray(
        matrix @ initial_guess_vector,
        dtype=np.float64,
    ).reshape(-1)
    if not np.array_equal(initial_residual_vector, recomputed_initial_residual):
        raise ValueError(
            "initial_residual must equal linear_rhs - effective_matrix @ initial_guess"
        )

    resolved_time = _require_float(time_value, "time")
    resolved_gmin = _require_float(gmin, "gmin")
    resolved_newton_iter = _require_int(newton_iter, "newton_iter")
    if resolved_newton_iter < 1:
        raise ValueError("newton_iter must be one-based and at least 1")
    if not isinstance(node_map, Mapping) or not node_map:
        raise ValueError("continuation step must contain a non-empty node_map")
    for name, raw_index in node_map.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"node_map index for {name!r} is not an integer") from exc
        if index < 0 or index > size:
            raise ValueError(
                f"node_map index for {name!r} lies outside [0, {size}]"
            )

    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    offsets, rows, lambdas, weights = _flatten_parameters(
        preconditioner.export_parameters()
    )
    block_count = len(lambdas)
    total_rows = len(rows)
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "preconditioner_mode": PRECONDITIONER_MODE,
        "feature_contract": FEATURE_CONTRACT,
        "local_shift_contract": LOCAL_SHIFT_CONTRACT,
        "local_shift_floor_relative": LOCAL_SHIFT_FLOOR_RELATIVE,
        "row_index_base": ROW_INDEX_BASE,
        "matrix_size": size,
        "node_map_hash": _compute_node_map_hash(dict(node_map)),
        "matrix_fingerprint": compute_matrix_fingerprint(matrix),
        "layout_sha256": compute_layout_sha256(
            matrix_size=size,
            block_count=block_count,
            total_block_rows=total_rows,
            block_offsets=offsets,
            block_rows=rows,
        ),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "linear_rhs_sha256": compute_float64_vector_sha256(
            linear_rhs_vector, label="linear_rhs"
        ),
        "initial_guess_sha256": compute_float64_vector_sha256(
            initial_guess_vector, label="initial_guess"
        ),
        "initial_residual_contract": INITIAL_RESIDUAL_CONTRACT,
        "initial_residual_norm_l2": float(np.linalg.norm(initial_residual_vector)),
        "initial_residual_norm_atol": INITIAL_RESIDUAL_NORM_ATOL,
        "initial_residual_norm_rtol": INITIAL_RESIDUAL_NORM_RTOL,
        "initial_guess_mode": resolved_initial_guess_mode,
        "time": resolved_time,
        "gmin": resolved_gmin,
        "newton_iter": resolved_newton_iter,
        "block_count": block_count,
        "total_block_rows": total_rows,
        "block_offsets": offsets,
        "block_rows": rows,
        "block_lambdas": lambdas,
        "block_row_weights": weights,
        "uncovered_row_policy": UNCOVERED_ROW_POLICY,
    }
    validate_native_learned_schwarz_sidecar(payload)
    return payload


def _resolve_step_metadata(
    *,
    step_path: str,
    time_value: Optional[float],
    gmin: Optional[float],
    newton_iter: Optional[int],
) -> Tuple[float, float, int]:
    step_name = Path(step_path).name
    match = STEP_METADATA_RE.search(step_name)
    parsed_time = float(match.group("time")) if match else None
    parsed_gmin = float(match.group("gmin")) if match else None
    parsed_iter = int(match.group("newton_iter")) if match else None
    if newton_iter is None and parsed_iter is not None:
        if step_name.startswith("continuation_circuit_"):
            raise ValueError(
                "continuation filenames encode a continuation-local step "
                "index, not the native Newton iteration; provide "
                "--newton-iter explicitly"
            )
        resolved_newton_iter = parsed_iter + NATIVE_NEWTON_ITER_OFFSET
    else:
        resolved_newton_iter = newton_iter
    resolved = (
        parsed_time if time_value is None else time_value,
        parsed_gmin if gmin is None else gmin,
        resolved_newton_iter,
    )
    if any(value is None for value in resolved):
        raise ValueError(
            "time, gmin and newton_iter must be encoded in the continuation "
            "filename or provided explicitly"
        )
    # niiter.c writes trajectory/corpus files before its iterno++ and calls
    # native GMRES afterwards. Therefore a regular filename suffix iter_000
    # belongs to the native solve with newton_iter=1. Continuation filenames
    # use a separate local counter and require an explicit native iteration.
    resolved_time = _require_float(resolved[0], "time")
    resolved_gmin = _require_float(resolved[1], "gmin")
    resolved_iter = _require_int(resolved[2], "newton_iter")
    if resolved_iter < 1:
        raise ValueError("newton_iter must be one-based and at least 1")
    return resolved_time, resolved_gmin, resolved_iter


def load_effective_matrix(jacobian_path: str, gmin: float) -> sp.csr_matrix:
    """Load the Jacobian and include gmin on the effective diagonal."""
    matrix = read_J_sparse(str(jacobian_path), matrix_format="csr").tocsr()
    if np.iscomplexobj(matrix.data):
        imag = float(np.max(np.abs(matrix.data.imag))) if matrix.nnz else 0.0
        real = float(np.max(np.abs(matrix.data.real))) if matrix.nnz else 0.0
        if imag > 1e-14 * max(real, 1.0):
            raise ValueError(
                "native learned Schwarz export requires a real Jacobian; "
                f"maximum imaginary magnitude is {imag}"
            )
        matrix = matrix.real
    matrix = matrix.astype(np.float64, copy=False)
    if matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Jacobian must be non-empty and square")
    gmin_value = _require_float(gmin, "gmin")
    if gmin_value != 0.0:
        matrix = matrix + sp.eye(
            matrix.shape[0], dtype=np.float64, format="csr"
        ) * gmin_value
    matrix = matrix.tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def _required_vector(value: Any, size: int, label: str) -> np.ndarray:
    raw = np.asarray([] if value is None else value)
    if raw.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional vector")
    if np.iscomplexobj(raw):
        raise ValueError(f"{label} must be a real float64 vector")
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a real float64 vector") from exc
    if array.shape[0] != size:
        raise ValueError(
            f"{label} length {array.shape[0]} does not match matrix size {size}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    return array


def write_native_learned_schwarz_sidecar(
    output_path: str,
    payload: Mapping[str, Any],
) -> None:
    validate_native_learned_schwarz_sidecar(payload)
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def export_sidecar_from_paths(
    *,
    jacobian_path: str,
    continuation_step: str,
    netlist_path: str,
    checkpoint_path: str,
    output_path: str,
    time_value: Optional[float] = None,
    gmin: Optional[float] = None,
    newton_iter: Optional[int] = None,
    initial_guess_mode: str = INITIAL_GUESS_MODE_RHSOLD,
    min_block_size: int = NATIVE_MIN_BLOCK_SIZE,
    max_block_size: int = NATIVE_MAX_BLOCK_SIZE,
    max_blocks: int = 0,
) -> Dict[str, Any]:
    """Create one sidecar from four recorded deployment inputs."""
    paths = {
        "jacobian": Path(jacobian_path).resolve(),
        "continuation step": Path(continuation_step).resolve(),
        "netlist": Path(netlist_path).resolve(),
        "checkpoint": Path(checkpoint_path).resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if min_block_size < 2 or max_block_size > 32:
        raise ValueError("native block sizes must remain within [2, 32]")
    if min_block_size > max_block_size:
        raise ValueError("min_block_size cannot exceed max_block_size")
    if max_blocks < 0:
        raise ValueError("max_blocks cannot be negative")
    initial_guess_mode = _require_initial_guess_mode(initial_guess_mode)

    time_value, gmin, newton_iter = _resolve_step_metadata(
        step_path=str(paths["continuation step"]),
        time_value=time_value,
        gmin=gmin,
        newton_iter=newton_iter,
    )
    matrix = load_effective_matrix(str(paths["jacobian"]), gmin)
    step = read_continuation_step(str(paths["continuation step"]))
    size = int(matrix.shape[0])
    linear_rhs = _required_vector(step.get("rhsnew"), size, "rhsnew")
    if initial_guess_mode == INITIAL_GUESS_MODE_RHSOLD:
        initial_guess = _required_vector(step.get("rhsold"), size, "rhsold")
    else:
        initial_guess = np.zeros(size, dtype=np.float64)
    initial_residual = _required_vector(
        linear_rhs - np.asarray(matrix @ initial_guess, dtype=np.float64).reshape(-1),
        size,
        "initial_residual",
    )
    node_map = step.get("node_map")
    if not isinstance(node_map, dict) or not node_map:
        raise ValueError("continuation step must contain a non-empty node_map")

    blocks, candidates, _ = extract_learned_schwarz_blocks(
        node_map=node_map,
        netlist_path=str(paths["netlist"]),
        max_block_size=int(max_block_size),
        min_block_size=int(min_block_size),
        max_blocks=int(max_blocks),
        matrix_size=size,
    )
    preconditioner = SparseLearnedSchwarzV1Preconditioner(
        matrix=matrix,
        blocks=blocks,
        block_candidates=candidates,
        model=load_learned_schwarz_v1_model(
            str(paths["checkpoint"]),
            initial_guess_mode=initial_guess_mode,
        ),
        linear_rhs=linear_rhs,
        initial_residual=initial_residual,
        gmin=gmin,
    )
    payload = build_native_learned_schwarz_sidecar_payload(
        preconditioner=preconditioner,
        effective_matrix=matrix,
        node_map=node_map,
        checkpoint_path=str(paths["checkpoint"]),
        time_value=time_value,
        gmin=gmin,
        newton_iter=newton_iter,
        linear_rhs=linear_rhs,
        initial_guess=initial_guess,
        initial_residual=initial_residual,
        initial_guess_mode=initial_guess_mode,
    )
    write_native_learned_schwarz_sidecar(output_path, payload)
    return payload


def export_sidecar_from_linear_system_corpus(
    *,
    system_path: str,
    jacobian_path: str,
    netlist_path: str,
    checkpoint_path: str,
    output_path: str,
    time_value: float,
    gmin: float,
    newton_iter: int,
    initial_guess_mode: str = INITIAL_GUESS_MODE_RHSOLD,
    min_block_size: int = NATIVE_MIN_BLOCK_SIZE,
    max_block_size: int = NATIVE_MAX_BLOCK_SIZE,
    max_blocks: int = 0,
) -> Dict[str, Any]:
    """Create one schema-5 sidecar from a live CKTload corpus snapshot."""
    paths = {
        "linear system corpus": Path(system_path).resolve(),
        "jacobian": Path(jacobian_path).resolve(),
        "netlist": Path(netlist_path).resolve(),
        "checkpoint": Path(checkpoint_path).resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if min_block_size < NATIVE_MIN_BLOCK_SIZE or max_block_size > NATIVE_MAX_BLOCK_SIZE:
        raise ValueError("native block sizes must remain within [2, 32]")
    if min_block_size > max_block_size:
        raise ValueError("min_block_size cannot exceed max_block_size")
    if max_blocks < 0:
        raise ValueError("max_blocks cannot be negative")

    resolved_time = _require_float(time_value, "time")
    resolved_gmin = _require_float(gmin, "gmin")
    resolved_newton_iter = _require_int(newton_iter, "newton_iter")
    if resolved_newton_iter < 1:
        raise ValueError("newton_iter must be one-based and at least 1")
    resolved_initial_guess_mode = _require_initial_guess_mode(initial_guess_mode)
    corpus = read_linear_system_corpus_step(str(paths["linear system corpus"]))
    meta = corpus.get("meta")
    if not isinstance(meta, Mapping):
        raise ValueError("linear system corpus is missing metadata")
    corpus_iteration = _require_int(meta.get("iteration"), "corpus iteration")
    if corpus_iteration + NATIVE_NEWTON_ITER_OFFSET != resolved_newton_iter:
        raise ValueError("corpus iteration does not match native Newton iteration")
    if _require_float(meta.get("time"), "corpus time") != resolved_time:
        raise ValueError("corpus time does not match requested time")
    if _require_float(meta.get("gmin"), "corpus gmin") != resolved_gmin:
        raise ValueError("corpus gmin does not match requested gmin")

    matrix = load_effective_matrix(str(paths["jacobian"]), resolved_gmin)
    size = int(matrix.shape[0])
    if _require_int(meta.get("matrix_size"), "corpus matrix_size") != size:
        raise ValueError("corpus matrix_size does not match Jacobian")
    linear_rhs = _required_vector(corpus.get("rhs"), size, "RHS")
    if resolved_initial_guess_mode == INITIAL_GUESS_MODE_RHSOLD:
        initial_guess = _required_vector(corpus.get("rhsold"), size, "RHSOLD")
    else:
        initial_guess = np.zeros(size, dtype=np.float64)
    initial_residual = _required_vector(
        linear_rhs - np.asarray(matrix @ initial_guess, dtype=np.float64).reshape(-1),
        size,
        "initial_residual",
    )
    node_map = corpus.get("node_map")
    if not isinstance(node_map, dict) or not node_map:
        raise ValueError("linear system corpus must contain a non-empty node_map")

    blocks, candidates, _ = extract_learned_schwarz_blocks(
        node_map=node_map,
        netlist_path=str(paths["netlist"]),
        max_block_size=int(max_block_size),
        min_block_size=int(min_block_size),
        max_blocks=int(max_blocks),
        matrix_size=size,
    )
    preconditioner = SparseLearnedSchwarzV1Preconditioner(
        matrix=matrix,
        blocks=blocks,
        block_candidates=candidates,
        model=load_learned_schwarz_v1_model(
            str(paths["checkpoint"]),
            initial_guess_mode=resolved_initial_guess_mode,
        ),
        linear_rhs=linear_rhs,
        initial_residual=initial_residual,
        gmin=resolved_gmin,
    )
    payload = build_native_learned_schwarz_sidecar_payload(
        preconditioner=preconditioner,
        effective_matrix=matrix,
        node_map=node_map,
        checkpoint_path=str(paths["checkpoint"]),
        time_value=resolved_time,
        gmin=resolved_gmin,
        newton_iter=resolved_newton_iter,
        linear_rhs=linear_rhs,
        initial_guess=initial_guess,
        initial_residual=initial_residual,
        initial_guess_mode=resolved_initial_guess_mode,
    )
    write_native_learned_schwarz_sidecar(output_path, payload)
    return payload

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export one native learned Schwarz sidecar v5."
    )
    parser.add_argument("--jacobian-path", required=True)
    parser.add_argument(
        "--continuation-step", "--step-path",
        dest="continuation_step", required=True,
    )
    parser.add_argument("--netlist-path", required=True)
    parser.add_argument(
        "--checkpoint", "--learned-schwarz-checkpoint",
        dest="checkpoint_path", required=True,
    )
    parser.add_argument(
        "--output", "--output-path", dest="output_path", required=True,
    )
    parser.add_argument("--time", dest="time_value", type=float, default=None)
    parser.add_argument("--gmin", type=float, default=None)
    parser.add_argument(
        "--newton-iter",
        type=int,
        default=None,
        help=(
            "native one-based Newton solve index; required for "
            "continuation_circuit_* inputs"
        ),
    )
    parser.add_argument(
        "--initial-guess-mode",
        choices=sorted(INITIAL_GUESS_MODES),
        default=INITIAL_GUESS_MODE_RHSOLD,
        help="native GMRES initial guess source (default: rhsold)",
    )
    parser.add_argument("--min-block-size", type=int, default=2)
    parser.add_argument("--max-block-size", type=int, default=32)
    parser.add_argument("--max-blocks", type=int, default=0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = export_sidecar_from_paths(
        jacobian_path=args.jacobian_path,
        continuation_step=args.continuation_step,
        netlist_path=args.netlist_path,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        time_value=args.time_value,
        gmin=args.gmin,
        newton_iter=args.newton_iter,
        initial_guess_mode=args.initial_guess_mode,
        min_block_size=args.min_block_size,
        max_block_size=args.max_block_size,
        max_blocks=args.max_blocks,
    )
    print(
        json.dumps(
            {
                "output_path": str(Path(args.output_path).resolve()),
                "schema_version": payload["schema_version"],
                "matrix_size": payload["matrix_size"],
                "block_count": payload["block_count"],
                "total_block_rows": payload["total_block_rows"],
                "layout_sha256": payload["layout_sha256"],
                "matrix_fingerprint": payload["matrix_fingerprint"],
            }
        )
    )


if __name__ == "__main__":
    main()
