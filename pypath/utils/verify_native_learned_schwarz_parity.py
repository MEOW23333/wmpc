#!/usr/bin/env python3
"""Compile and compare the native learned-Schwarz apply with Python."""

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pypath.utils.export_native_learned_schwarz_sidecar as sidecar_module
from pypath.utils.export_native_learned_schwarz_sidecar import (
    FEATURE_CONTRACT,
    INITIAL_GUESS_MODE_RHSOLD,
    SCHEMA_VERSION,
    compute_float64_vector_sha256,
    compute_layout_sha256,
    compute_matrix_fingerprint,
    validate_native_learned_schwarz_sidecar,
    write_native_learned_schwarz_sidecar,
)

DEFAULT_TOLERANCE = 1e-12
DRIVER_SOURCE = REPO_ROOT / "pypath/preconditioner/tests/native_learned_schwarz_driver.c"
NATIVE_SOURCE = REPO_ROOT / "src/maths/ni/ni_gmres_schwarz.c"
SPARSE_OUTPUT_SOURCE = REPO_ROOT / "src/maths/sparse/spoutput.c"


def _repo_path(raw_path, label):
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError(f"{label} must stay inside PALS: {path}")
    return path


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _effective_matrix(raw_matrix, gmin):
    matrix = sp.csr_matrix(np.asarray(raw_matrix, dtype=np.float64))
    matrix = matrix + sp.eye(matrix.shape[0], format="csr") * float(gmin)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def _handcrafted_case():
    raw_matrix = np.asarray(
        [
            [4.0, -1.0, 0.25, 0.0, 0.0],
            [2.0, 3.5, -0.75, 0.0, 0.0],
            [0.5, -1.25, 2.0, 1.5, 0.0],
            [0.0, 0.25, 3.0, -0.5, 1.0],
            [np.nextafter(1.0, 2.0), 0.0, 0.5, -2.0, 3.0],
        ],
        dtype=np.float64,
    )
    rhs = np.asarray([1.0, -2.0, 0.5, 3.0, -1.25])
    initial_guess = np.asarray([0.25, -0.5, 0.75, 0.125, -0.25])
    gmin = 0.125
    initial_residual = rhs - _effective_matrix(raw_matrix, gmin) @ initial_guess
    return {
        "name": "handcrafted_overlap_uncovered_gmin_lambda",
        "raw_matrix": raw_matrix,
        "rhs": rhs,
        "initial_guess": initial_guess,
        "initial_residual": np.asarray(initial_residual).reshape(-1),
        "initial_guess_mode": INITIAL_GUESS_MODE_RHSOLD,
        "time": 0.75,
        "gmin": gmin,
        "newton_iter": 1,
        "node_map_hash": hashlib.sha256(b"native-parity-node-map-v1").hexdigest(),
        "checkpoint_sha256": hashlib.sha256(
            b"native-parity-checkpoint-v1"
        ).hexdigest(),
        "block_offsets": [0, 3, 5],
        "block_rows": [1, 2, 3, 3, 4],
        "block_lambdas": [0.2, 0.05],
        "block_row_weights": [1.0, 1.0, 0.375, 0.625, 1.0],
    }


def _build_sidecar(case):
    raw_matrix = case["raw_matrix"]
    size = int(raw_matrix.shape[0])
    offsets = case["block_offsets"]
    rows = case["block_rows"]
    lambdas = case["block_lambdas"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "preconditioner_mode": "learned_schwarz_v1_sparse",
        "feature_contract": FEATURE_CONTRACT,
        "local_shift_contract": sidecar_module.LOCAL_SHIFT_CONTRACT,
        "local_shift_floor_relative": sidecar_module.LOCAL_SHIFT_FLOOR_RELATIVE,
        "row_index_base": 1,
        "matrix_size": size,
        "node_map_hash": case["node_map_hash"],
        "matrix_fingerprint": compute_matrix_fingerprint(
            _effective_matrix(raw_matrix, case["gmin"])
        ),
        "layout_sha256": compute_layout_sha256(
            matrix_size=size,
            block_count=len(lambdas),
            total_block_rows=len(rows),
            block_offsets=offsets,
            block_rows=rows,
        ),
        "checkpoint_sha256": case["checkpoint_sha256"],
        "linear_rhs_sha256": compute_float64_vector_sha256(
            case["rhs"], label="linear_rhs"
        ),
        "initial_guess_sha256": compute_float64_vector_sha256(
            case["initial_guess"], label="initial_guess"
        ),
        "initial_residual_contract": sidecar_module.INITIAL_RESIDUAL_CONTRACT,
        "initial_residual_norm_l2": float(
            np.linalg.norm(case["initial_residual"])
        ),
        "initial_residual_norm_atol": sidecar_module.INITIAL_RESIDUAL_NORM_ATOL,
        "initial_residual_norm_rtol": sidecar_module.INITIAL_RESIDUAL_NORM_RTOL,
        "initial_guess_mode": case["initial_guess_mode"],
        "time": case["time"],
        "gmin": case["gmin"],
        "newton_iter": case["newton_iter"],
        "block_count": len(lambdas),
        "total_block_rows": len(rows),
        "block_offsets": offsets,
        "block_rows": rows,
        "block_lambdas": lambdas,
        "block_row_weights": case["block_row_weights"],
        "uncovered_row_policy": "row_sum",
    }
    validate_native_learned_schwarz_sidecar(payload)
    return payload


def _write_driver_case(path, matrix, rhs, initial_guess, initial_residual):
    matrix = np.asarray(matrix, dtype=np.float64)
    vectors = [
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in (rhs, initial_guess, initial_residual)
    ]
    if any(matrix.shape != (vector.size, vector.size) for vector in vectors):
        raise ValueError("matrix and vector shapes do not match")
    lines = [str(vectors[0].size)]
    lines.extend(format(float(value), ".17g") for value in matrix.reshape(-1))
    for vector in vectors:
        lines.extend(format(float(value), ".17g") for value in vector)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _find_compiler(raw_compiler):
    candidates = (
        [raw_compiler]
        if raw_compiler
        else ["/opt/rh/devtoolset-9/root/usr/bin/gcc", "gcc", "cc"]
    )
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return str(Path(resolved).resolve())
        if Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise FileNotFoundError("no usable C compiler found")


def _compile_driver(compiler, executable):
    command = [
        compiler,
        "-std=gnu11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror=incompatible-pointer-types",
        "-Werror=implicit-function-declaration",
        "-Werror=int-conversion",
        "-I", str(REPO_ROOT / "release/src/include"),
        "-I", str(REPO_ROOT / "src/include"),
        "-I", str(REPO_ROOT / "src/maths/ni"),
        "-I", str(REPO_ROOT / "src/maths/sparse"),
        str(DRIVER_SOURCE),
        str(NATIVE_SOURCE),
        "-lm",
        "-o", str(executable),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    result = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-4000:],
        "warning_count": completed.stderr.count("warning:"),
        "passed": completed.returncode == 0,
    }
    if not result["passed"]:
        raise RuntimeError("C driver compilation failed:\n" + completed.stderr[-4000:])
    return result


def _run_driver(executable, sidecar_path, case_path, case):
    command = [
        str(executable),
        str(sidecar_path),
        str(case_path),
        format(float(case["time"]), ".17g"),
        format(float(case["gmin"]), ".17g"),
        str(int(case["newton_iter"])),
        case["node_map_hash"],
        case["initial_guess_mode"],
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("C driver returned no JSON: " + completed.stderr[-2000:])
    return {
        "returncode": completed.returncode,
        "stderr": completed.stderr,
        "payload": json.loads(lines[-1]),
    }


def _python_apply(matrix, rhs, payload):
    matrix = np.asarray(matrix, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    size = matrix.shape[0]
    out = np.zeros(size)
    covered = np.zeros(size, dtype=bool)
    offsets = payload["block_offsets"]
    rows = np.asarray(payload["block_rows"], dtype=np.int64) - 1
    weights = np.asarray(payload["block_row_weights"])
    lambdas = np.asarray(payload["block_lambdas"])
    gmin = payload["gmin"]
    for block_id, block_lambda in enumerate(lambdas):
        begin, end = offsets[block_id:block_id + 2]
        block_rows = rows[begin:end]
        local = matrix[np.ix_(block_rows, block_rows)].copy()
        local[np.diag_indices_from(local)] += gmin + block_lambda
        out[block_rows] += weights[begin:end] * np.linalg.solve(
            local, rhs[block_rows]
        )
        covered[block_rows] = True
    effective = matrix.copy()
    effective[np.diag_indices_from(effective)] += gmin
    row_sums = np.abs(effective).sum(axis=1)
    out[~covered] = rhs[~covered] / np.maximum(row_sums[~covered], 1e-30)
    counts = np.bincount(rows, minlength=size)
    details = {
        "overlap_rows_one_based": (np.flatnonzero(counts > 1) + 1).tolist(),
        "uncovered_rows_one_based": (np.flatnonzero(~covered) + 1).tolist(),
        "covered_rows": int(covered.sum()),
        "uncovered_rows": int((~covered).sum()),
        "fallback_row_sums": row_sums[~covered].tolist(),
    }
    return out, details


def _relative_l2(actual, expected):
    difference = np.linalg.norm(actual - expected)
    denominator = np.linalg.norm(expected)
    return float(difference if denominator == 0.0 else difference / denominator)


def _fingerprint_precision_diagnostic(matrix, gmin):
    live = _effective_matrix(matrix, gmin)
    live_hash = compute_matrix_fingerprint(live)
    source = SPARSE_OUTPUT_SOURCE.read_text(encoding="utf-8", errors="replace")

    matrix_real_17 = '"%d\\t%d\\t%-.17g\\n"' in source
    matrix_complex_17 = '"%d\\t%d\\t%-.17g\\t%-.17g\\n"' in source
    matrix_real_15 = '"%d\\t%d\\t%-.15g\\n"' in source
    matrix_complex_15 = '"%d\\t%d\\t%-.15g\\t%-.15g\\n"' in source
    current_uses_17 = matrix_real_17 and matrix_complex_17
    current_uses_15 = matrix_real_15 and matrix_complex_15
    current_digits = 17 if current_uses_17 else 15 if current_uses_15 else None

    def rounded_effective(digits):
        rounded = np.asarray(matrix, dtype=np.float64).copy()
        for row, column in zip(*np.nonzero(rounded)):
            rounded[row, column] = float(
                format(rounded[row, column], f".{digits}g")
            )
        return _effective_matrix(rounded, gmin)

    legacy_roundtrip = rounded_effective(15)
    current_roundtrip = (
        rounded_effective(current_digits)
        if current_digits is not None
        else legacy_roundtrip
    )
    current_hash = compute_matrix_fingerprint(current_roundtrip)
    legacy_hash = compute_matrix_fingerprint(legacy_roundtrip)
    current_difference = current_roundtrip.toarray() - live.toarray()
    legacy_difference = legacy_roundtrip.toarray() - live.toarray()
    current_safe = current_uses_17 and current_hash == live_hash
    legacy_safe = legacy_hash == live_hash
    return {
        "smpprint_source": str(SPARSE_OUTPUT_SOURCE.relative_to(REPO_ROOT)),
        "smpprint_matrix_significant_digits": current_digits,
        "smpprint_uses_15_significant_digits": current_uses_15,
        "smpprint_uses_17_significant_digits": current_uses_17,
        "live_effective_matrix_fingerprint": live_hash,
        "smpprint_roundtrip_effective_matrix_fingerprint": current_hash,
        "fingerprints_equal": live_hash == current_hash,
        "matrix_relative_l2_roundtrip_error": _relative_l2(
            current_roundtrip.toarray(), live.toarray()
        ),
        "matrix_max_abs_roundtrip_error": float(
            np.max(np.abs(current_difference))
        ),
        "bit_exact_current_smpprint_contract_safe": current_safe,
        "legacy_15_digit_roundtrip_effective_matrix_fingerprint": legacy_hash,
        "legacy_15_digit_fingerprints_equal": legacy_safe,
        "legacy_15_digit_matrix_relative_l2_roundtrip_error": _relative_l2(
            legacy_roundtrip.toarray(), live.toarray()
        ),
        "legacy_15_digit_matrix_max_abs_roundtrip_error": float(
            np.max(np.abs(legacy_difference))
        ),
        "bit_exact_old_smpprint_contract_safe": legacy_safe,
        "expected_precision_risk_reproduced": not legacy_safe,
        "passed": current_safe and not legacy_safe,
        "required_followup": (
            "Regenerate legacy 15-digit Jacobian corpora before enabling "
            "the live bit-exact fingerprint; current 17-digit matrix dumps "
            "are safe for binary64 round trips."
        ),
    }


def _newton_contract_diagnostic():
    regular = (
        "circuit_0_time_7.50000000000000000e-01_"
        "gmin_1.25000000000000000e-01_iter_000.txt"
    )
    continuation = (
        "continuation_circuit_0_time_7.50000000000000000e-01_"
        "gmin_1.25000000000000000e-01_iter_000.txt"
    )
    _, _, native_iter = sidecar_module._resolve_step_metadata(
        step_path=regular,
        time_value=None,
        gmin=None,
        newton_iter=None,
    )
    continuation_rejected = False
    continuation_error = ""
    try:
        sidecar_module._resolve_step_metadata(
            step_path=continuation,
            time_value=None,
            gmin=None,
            newton_iter=None,
        )
    except ValueError as exc:
        continuation_rejected = True
        continuation_error = str(exc)
    return {
        "recorded_filename_iter": 0,
        "native_gmres_newton_iter": native_iter,
        "continuation_requires_explicit_native_iter": continuation_rejected,
        "continuation_error": continuation_error,
        "passed": native_iter == 1 and continuation_rejected,
    }


def run_verification(tolerance=DEFAULT_TOLERANCE, compiler=""):
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    case = _handcrafted_case()
    payload = _build_sidecar(case)
    tmp_root = REPO_ROOT / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="native_schwarz_parity_", dir=tmp_root
    ) as temporary:
        workdir = Path(temporary)
        executable = workdir / "native_learned_schwarz_driver"
        case_path = workdir / "case.txt"
        valid_path = workdir / "valid.json"
        compile_result = _compile_driver(_find_compiler(compiler), executable)
        _write_driver_case(
            case_path,
            case["raw_matrix"],
            case["rhs"],
            case["initial_guess"],
            case["initial_residual"],
        )
        write_native_learned_schwarz_sidecar(str(valid_path), payload)

        valid_run = _run_driver(executable, valid_path, case_path, case)
        expected, coverage = _python_apply(
            case["raw_matrix"], case["rhs"], payload
        )
        actual = np.asarray(valid_run["payload"].get("output", []))
        if actual.shape == expected.shape:
            relative_error = _relative_l2(actual, expected)
            max_abs_error = float(np.max(np.abs(actual - expected)))
        else:
            relative_error = float("inf")
            max_abs_error = float("inf")
        valid_passed = (
            valid_run["returncode"] == 0
            and valid_run["payload"].get("create_ok") is True
            and valid_run["payload"].get("apply_ok") is True
            and relative_error <= tolerance
            and bool(coverage["overlap_rows_one_based"])
            and bool(coverage["uncovered_rows_one_based"])
        )

        invalid_cases = {}

        def check_invalid(name, mutated, expected_reason, runtime_case=None):
            path = workdir / f"{name}.json"
            path.write_text(
                json.dumps(mutated, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            native = _run_driver(
                executable,
                path,
                case_path,
                case if runtime_case is None else runtime_case,
            )
            case_passed = (
                native["returncode"] == 2
                and native["payload"].get("create_ok") is False
                and native["payload"].get("reason") == expected_reason
            )
            invalid_cases[name] = {
                "passed": case_passed,
                "expected_reason": expected_reason,
                "native_result": native["payload"],
            }

        bad_weights = copy.deepcopy(payload)
        bad_weights["block_row_weights"][2] += 0.125
        check_invalid(
            "bad_weights",
            bad_weights,
            "schwarz_weight_sum_mismatch",
        )
        missing_local_shift_contract = copy.deepcopy(payload)
        missing_local_shift_contract.pop("local_shift_contract")
        check_invalid(
            "missing_local_shift_contract",
            missing_local_shift_contract,
            "schwarz_missing_or_invalid_local_shift_contract",
        )
        missing_local_shift_floor_relative = copy.deepcopy(payload)
        missing_local_shift_floor_relative.pop("local_shift_floor_relative")
        check_invalid(
            "missing_local_shift_floor_relative",
            missing_local_shift_floor_relative,
            "schwarz_missing_or_invalid_local_shift_floor_relative",
        )
        mutations = (
            (
                "legacy_schema_v4",
                "schema_version",
                4,
                "schwarz_invalid_schema_version",
            ),
            (
                "feature_contract",
                "feature_contract",
                "wrong_contract",
                "schwarz_feature_contract_mismatch",
            ),
            (
                "local_shift_contract",
                "local_shift_contract",
                "wrong_contract",
                "schwarz_local_shift_contract_mismatch",
            ),
            (
                "local_shift_floor_relative",
                "local_shift_floor_relative",
                sidecar_module.LOCAL_SHIFT_FLOOR_RELATIVE * 2.0,
                "schwarz_local_shift_floor_relative_mismatch",
            ),
            (
                "row_index_base",
                "row_index_base",
                0,
                "schwarz_unsupported_row_index_base",
            ),
            (
                "layout_sha256",
                "layout_sha256",
                "0" * 64,
                "schwarz_layout_sha256_mismatch",
            ),
            (
                "matrix_fingerprint",
                "matrix_fingerprint",
                "0" * 64,
                "schwarz_matrix_fingerprint_mismatch",
            ),
            (
                "linear_rhs_sha256",
                "linear_rhs_sha256",
                "0" * 64,
                "schwarz_linear_rhs_sha256_mismatch",
            ),
            (
                "initial_guess_sha256",
                "initial_guess_sha256",
                "0" * 64,
                "schwarz_initial_guess_sha256_mismatch",
            ),
            (
                "initial_residual_contract",
                "initial_residual_contract",
                "wrong_contract",
                "schwarz_initial_residual_contract_mismatch",
            ),
            (
                "initial_residual_norm_l2",
                "initial_residual_norm_l2",
                float(np.linalg.norm(case["initial_residual"])) + 1.0,
                "schwarz_initial_residual_norm_mismatch",
            ),
            (
                "initial_guess_mode",
                "initial_guess_mode",
                "zero",
                "schwarz_initial_guess_mode_mismatch",
            ),
            (
                "time",
                "time",
                case["time"] + 0.125,
                "schwarz_time_mismatch",
            ),
            (
                "gmin",
                "gmin",
                case["gmin"] + 0.125,
                "schwarz_gmin_mismatch",
            ),
            (
                "node_map_hash",
                "node_map_hash",
                "0" * 64,
                "schwarz_node_map_hash_mismatch",
            ),
            (
                "newton_iter_mismatch",
                "newton_iter",
                case["newton_iter"] + 1,
                "schwarz_newton_iter_mismatch",
            ),
        )
        for name, field, value, expected_reason in mutations:
            mutated = copy.deepcopy(payload)
            mutated[field] = value
            check_invalid(name, mutated, expected_reason)

        zero_iter = copy.deepcopy(payload)
        zero_iter["newton_iter"] = 0
        zero_runtime_case = dict(case)
        zero_runtime_case["newton_iter"] = 0
        check_invalid(
            "newton_iter_both_zero",
            zero_iter,
            "schwarz_invalid_newton_iter",
            zero_runtime_case,
        )
        invalid_cases_passed = all(
            item["passed"] for item in invalid_cases.values()
        )

    fingerprint = _fingerprint_precision_diagnostic(
        case["raw_matrix"], case["gmin"]
    )
    newton = _newton_contract_diagnostic()
    passed = (
        compile_result["passed"]
        and valid_passed
        and invalid_cases_passed
        and newton["passed"]
        and fingerprint["passed"]
    )
    return {
        "schema_version": 1,
        "tool": "verify_native_learned_schwarz_parity",
        "status": "pass" if passed else "fail",
        "relative_l2_tolerance": tolerance,
        "maximum_relative_l2_error": relative_error,
        "compile": compile_result,
        "valid_operator_case": {
            "name": case["name"],
            "passed": valid_passed,
            "relative_l2_error": relative_error,
            "max_abs_error": max_abs_error,
            "python_output": expected.tolist(),
            "native_output": actual.tolist(),
            "coverage": coverage,
            "native_metrics": valid_run["payload"].get("metrics", {}),
            "covers_overlap_rows": bool(coverage["overlap_rows_one_based"]),
            "covers_uncovered_rows": bool(coverage["uncovered_rows_one_based"]),
            "covers_gmin_plus_lambda": (
                case["gmin"] != 0.0
                and all(value != 0.0 for value in case["block_lambdas"])
            ),
        },
        "invalid_sidecar_cases": invalid_cases,
        "newton_iteration_contract": {
            **newton,
            "zero_based_sidecar_and_runtime_rejected": invalid_cases[
                "newton_iter_both_zero"
            ]["passed"],
            "zero_based_native_result": invalid_cases[
                "newton_iter_both_zero"
            ]["native_result"],
        },
        "matrix_fingerprint_precision_contract": fingerprint,
        "overall_passed": passed,
    }


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compile a minimal C driver and verify native learned Schwarz "
            "against a deterministic Python reference."
        )
    )
    parser.add_argument(
        "--relative-l2-tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
    )
    parser.add_argument("--cc", default="")
    parser.add_argument("--output-json", default="")
    return parser


def main():
    args = _build_parser().parse_args()
    result = run_verification(
        tolerance=args.relative_l2_tolerance,
        compiler=args.cc,
    )
    if args.output_json:
        _write_json(_repo_path(args.output_json, "output path"), result)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if result["overall_passed"] else 1)


if __name__ == "__main__":
    main()
