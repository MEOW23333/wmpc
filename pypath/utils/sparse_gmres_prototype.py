import argparse
import json
import os
import re
import resource
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import gmres

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in os.sys.path:
    os.sys.path.insert(0, REPO_ROOT)

from pypath.utils.ngspice_utils import read_J_sparse, read_continuation_step
from pypath.preconditioner.linear_system_contract import compute_initial_residual
from pypath.preconditioner.learned_schwarz import (
    DEFAULT_LEARNED_SCHWARZ_PARAMETER_MODE,
    LEARNED_SCHWARZ_PARAMETER_MODES,
)
from pypath.precondition_construction.sparse import (
    SPARSE_SEMANTIC_MODES,
    build_preconditioner as _build_preconditioner,
    build_sparse_semantic_preconditioner as _build_sparse_semantic_preconditioner,
)




DEFAULT_RTOL = 1e-8
DEFAULT_ATOL = 1e-10
DEFAULT_RESTART = 30
DEFAULT_MAX_ITERS = 120
STEP_RE = re.compile(
    r"(?:continuation_)?circuit_(\d+)_time_([0-9.eE+-]+)_gmin_"
    r"(?:\d+_)?([0-9.eE+-]+)_iter_(\d+)\.txt$"
)


def _residual_ratio(residual: float, rhs_norm: float) -> float:
    return float(residual) / max(float(rhs_norm), 1e-30)


def _parse_modes(raw: str) -> List[str]:
    modes = []
    seen = set()
    for item in str(raw or '').split(','):
        mode = item.strip()
        if mode and mode not in seen:
            seen.add(mode)
            modes.append(mode)
    return modes or ['row_sum']


def _workpoint_key(
    *,
    circuit_id: int,
    time_value: float,
    gmin_value: float,
    iteration: int,
) -> Tuple[int, str, str, int]:
    return (
        int(circuit_id),
        f"{float(time_value):.17e}",
        f"{float(gmin_value):.17e}",
        int(iteration),
    )


def _load_workpoint_manifest(manifest_path: str) -> Dict[int, List[Dict[str, Any]]]:
    with open(manifest_path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    schema_version = payload.get('schema_version') if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ValueError('workpoint_manifest_schema_version_must_be_1')
    raw_workpoints = payload.get('workpoints')
    if not isinstance(raw_workpoints, list) or not raw_workpoints:
        raise ValueError('workpoint_manifest_requires_nonempty_workpoints')

    by_circuit: Dict[int, List[Dict[str, Any]]] = {}
    seen = set()
    for manifest_index, item in enumerate(raw_workpoints):
        if not isinstance(item, dict):
            raise ValueError('workpoint_manifest_item_must_be_object')
        try:
            raw_circuit_id = item['circuit_id']
            raw_time_value = item['time']
            raw_gmin_value = item['gmin_val']
            raw_iteration = item['iteration']
        except KeyError as exc:
            raise ValueError('workpoint_manifest_item_has_invalid_key_fields') from exc
        if (
            isinstance(raw_circuit_id, bool)
            or not isinstance(raw_circuit_id, int)
            or isinstance(raw_iteration, bool)
            or not isinstance(raw_iteration, int)
            or isinstance(raw_time_value, bool)
            or not isinstance(raw_time_value, (int, float))
            or isinstance(raw_gmin_value, bool)
            or not isinstance(raw_gmin_value, (int, float))
        ):
            raise ValueError('workpoint_manifest_item_has_invalid_key_fields')
        circuit_id = int(raw_circuit_id)
        time_value = float(raw_time_value)
        gmin_value = float(raw_gmin_value)
        iteration = int(raw_iteration)
        if (
            circuit_id < 0
            or iteration < 0
            or time_value < 0.0
            or gmin_value < 0.0
            or not np.isfinite(time_value)
            or not np.isfinite(gmin_value)
        ):
            raise ValueError('workpoint_manifest_item_has_nonfinite_or_negative_key_fields')
        key = _workpoint_key(
            circuit_id=circuit_id,
            time_value=time_value,
            gmin_value=gmin_value,
            iteration=iteration,
        )
        if key in seen:
            raise ValueError(f'workpoint_manifest_duplicate:{key}')
        seen.add(key)
        by_circuit.setdefault(circuit_id, []).append(
            {
                'key': key,
                'manifest_index': int(manifest_index),
            }
        )
    return by_circuit


def _find_steps(
    trajectory_dir: str,
    circuit_id: int,
    max_steps: int,
    *,
    requested_workpoints: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    root = Path(trajectory_dir)
    requested_by_key: Dict[Tuple[int, str, str, int], Dict[str, Any]] = {}
    if requested_workpoints is not None:
        for entry in requested_workpoints:
            key = tuple(entry.get('key', ()))
            if len(key) != 4 or int(key[0]) != int(circuit_id):
                raise ValueError('workpoint_manifest_circuit_mismatch')
            if key in requested_by_key:
                raise ValueError(f'workpoint_manifest_duplicate:{key}')
            requested_by_key[key] = entry

    out: List[Dict[str, Any]] = []
    found_keys = set()
    path_patterns = (
        f'circuit_{int(circuit_id)}_time_*_iter_*.txt',
        f'continuation_circuit_{int(circuit_id)}_time_*_iter_*.txt',
    )
    for path in sorted(
        path
        for pattern in path_patterns
        for path in root.glob(pattern)
    ):
        if path.name.endswith('_jac.txt'):
            continue
        match = STEP_RE.match(path.name)
        if not match:
            continue
        cid_raw, time_raw, gmin_raw, iter_raw = match.groups()
        time_value = float(time_raw)
        gmin_value = float(gmin_raw)
        iteration = int(iter_raw)
        key = _workpoint_key(
            circuit_id=int(cid_raw),
            time_value=time_value,
            gmin_value=gmin_value,
            iteration=iteration,
        )
        manifest_entry = requested_by_key.get(key)
        if requested_workpoints is not None and manifest_entry is None:
            continue
        if key in found_keys:
            if requested_workpoints is not None:
                raise ValueError(f'workpoint_manifest_trajectory_duplicate:{key}')
            continue
        jac_path = path.with_name(path.stem + '_jac.txt')
        if not jac_path.exists():
            if requested_workpoints is not None:
                raise ValueError(f'workpoint_manifest_jacobian_missing:{key}')
            continue
        step = {
            'circuit_id': int(cid_raw),
            'time': time_value,
            'gmin_val': gmin_value,
            'iteration': iteration,
            'step_path': str(path),
            'jacobian_path': str(jac_path),
        }
        if manifest_entry is not None:
            step['_workpoint_manifest_index'] = int(
                manifest_entry['manifest_index']
            )
        out.append(step)
        found_keys.add(key)

    if requested_workpoints is not None:
        missing = [key for key in requested_by_key if key not in found_keys]
        if missing:
            raise ValueError(f'workpoint_manifest_missing:{missing}')
        out.sort(key=lambda item: int(item['_workpoint_manifest_index']))
        return out

    out.sort(key=lambda item: (item['time'], item['gmin_val'], item['iteration']))
    return out[:max(1, int(max_steps))]
def _matrix_stats(matrix: sp.spmatrix) -> Dict[str, Any]:
    csr = matrix.tocsr()
    row_nnz = np.diff(csr.indptr)
    storage_bytes = int(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes)
    n = int(csr.shape[0])
    return {
        'matrix_size': n,
        'matrix_nnz': int(csr.nnz),
        'matrix_density': float(csr.nnz / max(n * n, 1)),
        'avg_nnz_per_row': float(csr.nnz / max(n, 1)),
        'median_nnz_per_row': float(np.median(row_nnz)) if row_nnz.size else 0.0,
        'max_nnz_per_row': int(row_nnz.max()) if row_nnz.size else 0,
        'zero_rows': int(np.count_nonzero(row_nnz == 0)),
        'csr_storage_bytes': storage_bytes,
        'dense_storage_bytes_estimate': int(n * n * 8),
    }



def _evaluate_scipy_sparse(step: Dict[str, Any], mode: str, args: argparse.Namespace) -> Dict[str, Any]:
    result: Dict[str, Any] = dict(step)
    result['backend'] = 'scipy_sparse'
    result['mode'] = mode
    result['learned_schwarz_parameter_mode'] = str(
        getattr(
            args,
            'learned_schwarz_parameter_mode',
            DEFAULT_LEARNED_SCHWARZ_PARAMETER_MODE,
        )
    )
    t0 = time.perf_counter()
    matrix = read_J_sparse(step['jacobian_path'], matrix_format='csr')
    original_matrix_dtype = str(matrix.dtype)
    imaginary_max_abs = (
        float(np.max(np.abs(matrix.data.imag)))
        if np.iscomplexobj(matrix.data) and matrix.nnz
        else 0.0
    )
    real_max_abs = float(np.max(np.abs(matrix.data.real))) if matrix.nnz else 0.0
    imaginary_tolerance = 1e-14 * max(real_max_abs, 1.0)
    result['input_matrix_dtype'] = original_matrix_dtype
    result['input_matrix_imaginary_max_abs'] = imaginary_max_abs
    result['input_matrix_imaginary_tolerance'] = imaginary_tolerance
    if imaginary_max_abs > imaginary_tolerance:
        raise ValueError(
            'scipy sparse GMRES requires a real Jacobian; '
            f'max imaginary value is {imaginary_max_abs}'
        )
    matrix = matrix.real.tocsr().astype(np.float64, copy=False)
    result['input_matrix_projected_to_real'] = bool(
        np.issubdtype(np.dtype(original_matrix_dtype), np.complexfloating)
    )
    result['read_sparse_matrix_s'] = float(time.perf_counter() - t0)
    if matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('empty or non-square sparse Jacobian')
    if bool(args.apply_gmin_diagonal) and float(step.get('gmin_val', 0.0)) != 0.0:
        matrix = matrix + sp.eye(matrix.shape[0], dtype=matrix.dtype, format='csr') * float(step['gmin_val'])
    result.update(_matrix_stats(matrix))

    t0 = time.perf_counter()
    payload = read_continuation_step(step['step_path'])
    result['read_step_s'] = float(time.perf_counter() - t0)
    rhs = np.asarray(payload.get('rhsnew', []), dtype=np.float64)
    rhsold = np.asarray(payload.get('rhsold', []), dtype=np.float64)
    if rhs.shape[0] != matrix.shape[0]:
        raise ValueError(f'rhs length {rhs.shape[0]} does not match matrix size {matrix.shape[0]}')
    x0 = rhsold if bool(args.use_rhsold_as_x0) and rhsold.shape[0] == matrix.shape[0] else None
    initial_guess = np.zeros_like(rhs) if x0 is None else np.asarray(x0, dtype=np.float64)
    initial_guess_mode = 'rhsold' if x0 is not None else 'zero'
    result['x0_source'] = initial_guess_mode
    result['x0_norm'] = float(np.linalg.norm(initial_guess))
    rhs_norm = float(np.linalg.norm(rhs))
    result['rhs_norm'] = rhs_norm
    initial_residual_vector = compute_initial_residual(
        effective_matrix=matrix,
        linear_rhs=rhs,
        initial_guess=initial_guess,
    )
    initial_raw_residual = float(np.linalg.norm(initial_residual_vector))
    result['initial_residual_norm'] = initial_raw_residual
    result['initial_residual_ratio'] = _residual_ratio(initial_raw_residual, rhs_norm)
    result['initial_residual_formula'] = 'linear_rhs - effective_matrix @ initial_guess'

    t0 = time.perf_counter()
    try:
        if mode in SPARSE_SEMANTIC_MODES:
            preconditioner, pc_info = _build_sparse_semantic_preconditioner(
                matrix,
                mode,
                step,
                payload,
                args,
                linear_rhs=rhs,
                initial_residual=initial_residual_vector,
                initial_guess_mode=initial_guess_mode,
            )
        else:
            preconditioner, pc_info = _build_preconditioner(matrix, mode)
        result['preconditioner_setup_s'] = float(time.perf_counter() - t0)
        result['preconditioner_info'] = pc_info
    except Exception as exc:
        result['preconditioner_setup_s'] = float(time.perf_counter() - t0)
        result['ok'] = False
        result['failure_stage'] = 'preconditioner_setup'
        result['reason'] = repr(exc)
        return result

    if mode == 'splu':
        t0 = time.perf_counter()
        try:
            if preconditioner is None:
                raise RuntimeError('splu direct baseline expected an LU solve operator')
            solution = np.asarray(preconditioner.matvec(rhs))
            solve_s = time.perf_counter() - t0
            raw_residual = float(np.linalg.norm(matrix.dot(solution) - rhs))
            true_rel_residual = _residual_ratio(raw_residual, rhs_norm)
            success_by_true_residual = bool(true_rel_residual < float(args.rtol) or raw_residual < float(args.atol))
            result.update({
                'ok': True,
                'backend': 'scipy_sparse',
                'mode': mode,
                'direct_solve_baseline': True,
                'gmres_used': False,
                'gmres_info': 0 if success_by_true_residual else 1,
                'iterations': 1,
                'gmres_restart': int(args.restart),
                'gmres_max_iters': int(args.max_iters),
                'solve_s': float(solve_s),
                'raw_residual_norm': raw_residual,
                'final_residual_norm': raw_residual,
                'true_rel_residual': true_rel_residual,
                'residual_ratio': true_rel_residual,
                'success_by_true_residual': success_by_true_residual,
                'success_by_residual_ratio': success_by_true_residual,
                'chosen_success': success_by_true_residual,
                'residual_success_conflict': False,
                'residual_history': [true_rel_residual],
                'residual_history_elapsed_s': [float(solve_s)],
                'residual_based_success': success_by_true_residual,
            })
            return result
        except Exception as exc:
            result.update({
                'ok': False,
                'backend': 'scipy_sparse',
                'mode': mode,
                'direct_solve_baseline': True,
                'gmres_used': False,
                'failure_stage': 'direct_solve',
                'reason': repr(exc),
                'solve_s': float(time.perf_counter() - t0),
            })
            return result

    iteration_count = [0]
    residual_history: List[float] = []
    residual_history_elapsed_s: List[float] = []
    solve_start_time: List[Optional[float]] = [None]

    def callback(value=None):
        iteration_count[0] += 1
        if value is not None:
            try:
                residual_history.append(float(value))
                if solve_start_time[0] is not None:
                    residual_history_elapsed_s.append(float(time.perf_counter() - solve_start_time[0]))
            except Exception:
                pass

    t0 = time.perf_counter()
    solve_start_time[0] = t0
    try:
        solution, info = gmres(
            matrix,
            rhs,
            x0=x0,
            M=preconditioner,
            rtol=float(args.rtol),
            atol=float(args.atol),
            restart=int(args.restart),
            maxiter=int(args.max_iters),
            callback=callback,
            callback_type='pr_norm',
        )
        solve_s = time.perf_counter() - t0
        raw_residual = float(np.linalg.norm(matrix.dot(solution) - rhs))
        true_rel_residual = _residual_ratio(raw_residual, rhs_norm)
        success_by_true_residual = bool(true_rel_residual < float(args.rtol) or raw_residual < float(args.atol))
        success_by_residual_ratio = bool(true_rel_residual < float(args.rtol) or raw_residual < float(args.atol))
        result.update({
            'ok': True,
            'backend': 'scipy_sparse',
            'mode': mode,
            'gmres_info': int(info),
            'iterations': int(iteration_count[0]),
            'gmres_restart': int(args.restart),
            'gmres_max_iters': int(args.max_iters),
            'solve_s': float(solve_s),
            'raw_residual_norm': raw_residual,
            'final_residual_norm': raw_residual,
            'true_rel_residual': true_rel_residual,
            'residual_ratio': true_rel_residual,
            'success_by_true_residual': success_by_true_residual,
            'success_by_residual_ratio': success_by_residual_ratio,
            'chosen_success': bool(success_by_true_residual or success_by_residual_ratio),
            'residual_success_conflict': bool(success_by_true_residual != success_by_residual_ratio),
            'residual_history': residual_history,
            'residual_history_elapsed_s': residual_history_elapsed_s,
            'residual_based_success': bool(success_by_true_residual or success_by_residual_ratio),
        })
    except Exception as exc:
        result.update({
            'ok': False,
            'backend': 'scipy_sparse',
            'mode': mode,
            'failure_stage': 'solve',
            'reason': repr(exc),
            'solve_s': float(time.perf_counter() - t0),
        })
    return result


def _petsc_type_name(value: Any) -> str:
    try:
        return str(value)
    except Exception:
        return repr(value)


def _petsc_reason_name(PETSc: Any, reason: int) -> str:
    try:
        for name in dir(PETSc.KSP.ConvergedReason):
            if name.isupper() and int(getattr(PETSc.KSP.ConvergedReason, name)) == int(reason):
                return name
    except Exception:
        pass
    return str(reason)


class _PetscRowSumContext:
    def __init__(self, inverse_scale: np.ndarray):
        self.inverse_scale = np.asarray(inverse_scale, dtype=np.float64)

    def apply(self, pc: Any, x: Any, y: Any) -> None:
        x_arr = x.getArray(readonly=True)
        y_arr = y.getArray(readonly=False)
        y_arr[...] = self.inverse_scale * x_arr


def _ensure_structural_diagonal(matrix: sp.spmatrix) -> tuple[sp.csr_matrix, int]:
    csr = matrix.tocsr(copy=True)
    n = int(csr.shape[0])
    present = np.zeros(n, dtype=bool)
    for row in range(n):
        start, end = int(csr.indptr[row]), int(csr.indptr[row + 1])
        if np.any(csr.indices[start:end] == row):
            present[row] = True
    missing = np.flatnonzero(~present)
    if missing.size:
        csr.setdiag(csr.diagonal())
        csr.sort_indices()
    return csr, int(missing.size)


def _csr_to_petsc_aij(matrix: sp.spmatrix, PETSc: Any) -> Any:
    csr = matrix.tocsr()
    csr.sort_indices()
    indptr = np.asarray(csr.indptr, dtype=PETSc.IntType)
    indices = np.asarray(csr.indices, dtype=PETSc.IntType)
    data = np.asarray(csr.data, dtype=PETSc.ScalarType)
    mat = PETSc.Mat().createAIJ(size=csr.shape, csr=(indptr, indices, data), comm=PETSc.COMM_SELF)
    mat.assemblyBegin()
    mat.assemblyEnd()
    return mat


def _configure_petsc_ksp(ksp: Any, pc: Any, mode: str, matrix: sp.spmatrix, PETSc: Any, args: argparse.Namespace) -> Dict[str, Any]:
    info: Dict[str, Any] = {'mode': mode, 'pc_mode': mode}
    if mode == 'identity':
        pc.setType(PETSc.PC.Type.NONE)
    elif mode in {'jacobi', 'jacobi_diagonal'}:
        pc.setType(PETSc.PC.Type.JACOBI)
    elif mode == 'row_sum':
        row_sum = np.asarray(abs(matrix).sum(axis=1)).reshape(-1)
        inv = 1.0 / np.maximum(row_sum, 1e-30)
        pc.setType(PETSc.PC.Type.PYTHON)
        pc.setPythonContext(_PetscRowSumContext(inv))
        info['pc_shell'] = 'row_sum'
    elif mode == 'ilu0':
        pc.setType(PETSc.PC.Type.ILU)
        try:
            pc.setFactorLevels(0)
            info['factor_levels'] = 0
        except Exception as exc:
            info['set_factor_levels_warning'] = repr(exc)
        _configure_petsc_factor_safety(pc, PETSc, args, info)
    elif mode == 'ilut':
        pc.setType(PETSc.PC.Type.ILU)
        fill_level = int(getattr(args, 'petsc_ilu_fill_level', 1))
        try:
            pc.setFactorLevels(fill_level)
            info['factor_levels'] = fill_level
        except Exception as exc:
            info['set_factor_levels_warning'] = repr(exc)
        info['note'] = 'PETSc ILU uses level-of-fill here; SciPy ilut uses threshold drop_tol/fill_factor.'
        _configure_petsc_factor_safety(pc, PETSc, args, info)
    elif mode == 'splu':
        pc.setType(PETSc.PC.Type.LU)
        _configure_petsc_factor_safety(pc, PETSc, args, info)
    else:
        raise ValueError(f'Unsupported PETSc mode: {mode}')
    return info


def _configure_petsc_factor_safety(pc: Any, PETSc: Any, args: argparse.Namespace, info: Dict[str, Any]) -> None:
    shift = float(getattr(args, 'petsc_factor_shift', 1e-12))
    zero_pivot = float(getattr(args, 'petsc_zero_pivot', 1e-14))
    info['factor_shift_nonzero'] = shift
    info['factor_zero_pivot'] = zero_pivot
    try:
        pc.setFactorShift(PETSc.Mat.FactorShiftType.NONZERO, shift)
    except Exception as exc:
        info['set_factor_shift_warning'] = repr(exc)
    try:
        pc.setFactorPivot(zeropivot=zero_pivot)
    except Exception as exc:
        info['set_factor_pivot_warning'] = repr(exc)


def _evaluate_petsc(step: Dict[str, Any], mode: str, args: argparse.Namespace) -> Dict[str, Any]:
    result: Dict[str, Any] = dict(step)
    result['backend'] = 'petsc'
    result['mode'] = mode
    try:
        import petsc4py  # type: ignore
        petsc4py.init([])
        from petsc4py import PETSc  # type: ignore
    except Exception as exc:
        return {
            **result,
            'ok': False,
            'failure_stage': 'backend_import',
            'reason': f'petsc4py unavailable: {exc!r}',
        }

    t0 = time.perf_counter()
    matrix = read_J_sparse(step['jacobian_path'], matrix_format='csr')
    result['read_sparse_matrix_s'] = float(time.perf_counter() - t0)
    if matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('empty or non-square sparse Jacobian')
    if bool(args.apply_gmin_diagonal) and float(step.get('gmin_val', 0.0)) != 0.0:
        matrix = matrix + sp.eye(matrix.shape[0], dtype=matrix.dtype, format='csr') * float(step['gmin_val'])
    if np.iscomplexobj(matrix.data):
        result['ok'] = False
        result['failure_stage'] = 'matrix_type'
        result['reason'] = 'complex PETSc matrices are not wired in this real-valued checkpoint'
        return result
    matrix = matrix.astype(np.float64).tocsr()
    result.update(_matrix_stats(matrix))
    if mode in {'ilu0', 'ilut', 'splu'}:
        factor_matrix, missing_diagonal_count = _ensure_structural_diagonal(matrix)
        result['petsc_factor_missing_diagonal_slots'] = int(missing_diagonal_count)
        result['petsc_factor_matrix_nnz'] = int(factor_matrix.nnz)
    else:
        factor_matrix = matrix

    t0 = time.perf_counter()
    payload = read_continuation_step(step['step_path'])
    result['read_step_s'] = float(time.perf_counter() - t0)
    rhs = np.asarray(payload.get('rhsnew', []), dtype=np.float64)
    rhsold = np.asarray(payload.get('rhsold', []), dtype=np.float64)
    if rhs.shape[0] != matrix.shape[0]:
        raise ValueError(f'rhs length {rhs.shape[0]} does not match matrix size {matrix.shape[0]}')
    rhs_norm = float(np.linalg.norm(rhs))
    result['rhs_norm'] = rhs_norm

    t0 = time.perf_counter()
    try:
        petsc_mat = _csr_to_petsc_aij(factor_matrix, PETSc)
        b = PETSc.Vec().createWithArray(np.asarray(rhs, dtype=PETSc.ScalarType), comm=PETSc.COMM_SELF)
        x_array = np.asarray(rhsold, dtype=PETSc.ScalarType).copy() if bool(args.use_rhsold_as_x0) and rhsold.shape[0] == rhs.shape[0] else np.zeros_like(rhs, dtype=PETSc.ScalarType)
        result['x0_source'] = 'rhsold' if bool(args.use_rhsold_as_x0) and rhsold.shape[0] == rhs.shape[0] else 'zero'
        result['x0_norm'] = float(np.linalg.norm(np.asarray(x_array)))
        x = PETSc.Vec().createWithArray(x_array, comm=PETSc.COMM_SELF)
        ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp.setOperators(petsc_mat)
        ksp_type = PETSc.KSP.Type.FGMRES if bool(args.petsc_flexible) else PETSc.KSP.Type.GMRES
        ksp.setType(ksp_type)
        try:
            ksp.setGMRESRestart(int(args.restart))
        except Exception as exc:
            result['gmres_restart_warning'] = repr(exc)
        ksp.setTolerances(rtol=float(args.rtol), atol=float(args.atol), max_it=int(args.max_iters))
        if bool(args.use_rhsold_as_x0) and rhsold.shape[0] == rhs.shape[0]:
            ksp.setInitialGuessNonzero(True)
        pc = ksp.getPC()
        pc_info = _configure_petsc_ksp(ksp, pc, mode, matrix, PETSc, args)
        ksp.setFromOptions()
        ksp.setUp()
        result['preconditioner_setup_s'] = float(time.perf_counter() - t0)
        result['preconditioner_info'] = pc_info
        result['petsc_ksp_type'] = _petsc_type_name(ksp.getType())
        result['petsc_pc_type'] = _petsc_type_name(pc.getType())
    except Exception as exc:
        result['preconditioner_setup_s'] = float(time.perf_counter() - t0)
        result['ok'] = False
        result['failure_stage'] = 'petsc_setup'
        result['reason'] = repr(exc)
        return result

    residual_history: List[float] = []

    def monitor(ksp_obj: Any, iteration: int, residual_norm: float) -> None:
        residual_history.append(float(residual_norm))

    try:
        ksp.setMonitor(monitor)
    except Exception as exc:
        result['monitor_warning'] = repr(exc)

    t0 = time.perf_counter()
    try:
        ksp.solve(b, x)
        solve_s = time.perf_counter() - t0
        solution = x.getArray(readonly=True).copy()
        raw_residual = float(np.linalg.norm(matrix.dot(solution) - rhs))
        reason = ksp.getConvergedReason()
        result.update({
            'ok': True,
            'solve_completed': True,
            'ksp_converged': bool(int(reason) > 0),
            'gmres_info': int(reason),
            'petsc_converged_reason': int(reason),
            'petsc_converged_reason_name': _petsc_reason_name(PETSc, int(reason)),
            'iterations': int(ksp.getIterationNumber()),
            'solve_s': float(solve_s),
            'raw_residual_norm': raw_residual,
            'residual_ratio': _residual_ratio(raw_residual, rhs_norm),
            'residual_history': residual_history,
            'residual_based_success': bool(_residual_ratio(raw_residual, rhs_norm) < float(args.rtol) or raw_residual < float(args.atol)),
        })
    except Exception as exc:
        result.update({
            'ok': False,
            'failure_stage': 'petsc_solve',
            'reason': repr(exc),
            'solve_s': float(time.perf_counter() - t0),
            'residual_history': residual_history,
        })
    finally:
        for obj_name in ['ksp', 'petsc_mat', 'b', 'x']:
            obj = locals().get(obj_name)
            try:
                if obj is not None:
                    obj.destroy()
            except Exception:
                pass
    return result


def evaluate_step(step: Dict[str, Any], mode: str, args: argparse.Namespace) -> Dict[str, Any]:
    started = time.perf_counter()
    if args.backend == 'petsc':
        result = _evaluate_petsc(step, mode, args)
    else:
        result = _evaluate_scipy_sparse(step, mode, args)
    result['total_s'] = float(time.perf_counter() - started)
    result['peak_rss_kb'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Sparse-native GMRES prototype for large ngspice Jacobian trajectories.')
    parser.add_argument('--trajectory-dir', required=True)
    parser.add_argument('--circuit-id', type=int, default=0)
    parser.add_argument(
        '--workpoint-manifest',
        default='',
        help='Schema-1 JSON workpoint list; when set, it selects exact trajectory steps.',
    )
    parser.add_argument('--backend', choices=['scipy_sparse', 'petsc'], default='scipy_sparse')
    parser.add_argument('--modes', default='row_sum,jacobi_diagonal,ilu0,ilut')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-steps', type=int, default=1)
    parser.add_argument('--restart', type=int, default=DEFAULT_RESTART)
    parser.add_argument('--max-iters', type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument('--rtol', type=float, default=DEFAULT_RTOL)
    parser.add_argument('--atol', type=float, default=DEFAULT_ATOL)
    parser.add_argument('--use-rhsold-as-x0', action='store_true', default=False)
    parser.add_argument('--no-rhsold-as-x0', dest='use_rhsold_as_x0', action='store_false')
    parser.add_argument('--apply-gmin-diagonal', action='store_true', default=True)
    parser.add_argument('--disable-gmin-diagonal', dest='apply_gmin_diagonal', action='store_false')
    parser.add_argument('--petsc-flexible', action='store_true', help='Use PETSc FGMRES instead of GMRES.')
    parser.add_argument('--petsc-factor-shift', type=float, default=1e-12, help='NONZERO diagonal shift amount for PETSc ILU/LU factorization.')
    parser.add_argument('--petsc-zero-pivot', type=float, default=1e-14, help='Zero-pivot tolerance for PETSc ILU/LU factorization.')
    parser.add_argument('--petsc-ilu-fill-level', type=int, default=1, help='Level-of-fill for PETSc ILU when mode=ilut.')
    parser.add_argument('--netlist-path', default='', help='Netlist path used by sparse semantic Schur modes.')
    parser.add_argument('--semantic-min-block-size', type=int, default=2)
    parser.add_argument('--semantic-max-block-size', type=int, default=64)
    parser.add_argument('--semantic-boundary-max-block-size', type=int, default=128)
    parser.add_argument('--semantic-max-blocks', type=int, default=0)
    parser.add_argument('--semantic-uncovered-policy', choices=['row_sum', 'jacobi_diagonal', 'identity'], default='row_sum')
    parser.add_argument('--semantic-coarse-max-condition', type=float, default=1.0e12)
    parser.add_argument('--semantic-coarse-rank-tol', type=float, default=1.0e-10)
    parser.add_argument('--sparse-schur-edge-budget', type=int, default=64)
    parser.add_argument('--local-schur-budget-multiplier', type=float, default=2.0)
    parser.add_argument('--sparse-schur-candidate-edge-limit', type=int, default=512)
    parser.add_argument('--sparse-schur-diagonal-shift', type=float, default=1e-8)
    parser.add_argument('--sparse-schur-factor-drop-tol', type=float, default=1e-4)
    parser.add_argument('--sparse-schur-factor-fill-factor', type=float, default=10.0)
    parser.add_argument('--sparse-schur-interface-solve', choices=['spilu', 'jacobi', 'jacobi_neumann1'], default='spilu')
    parser.add_argument('--sparse-schur-max-nnz', type=int, default=0)
    parser.add_argument('--sparse-schur-max-degree', type=int, default=0)
    parser.add_argument('--sparse-schur-max-exact-entries', type=int, default=0)
    parser.add_argument('--learning-sparse-schur-add-fraction', type=float, default=0.1)
    parser.add_argument('--learning-sparse-schur-add-min', type=int, default=1)
    parser.add_argument('--learning-sparse-schur-probe-restart', type=int, default=8)
    parser.add_argument('--learning-sparse-schur-probe-iterations', type=int, default=1)
    parser.add_argument('--learned-local-sparse-schur-checkpoint', default='')
    parser.add_argument('--learned-schwarz-checkpoint', default='')
    parser.add_argument(
        '--learned-schwarz-parameter-mode',
        choices=sorted(LEARNED_SCHWARZ_PARAMETER_MODES),
        default=DEFAULT_LEARNED_SCHWARZ_PARAMETER_MODE,
        help='Auditable learned Schwarz shift/overlap-weight ablation mode.',
    )
    args = parser.parse_args()

    workpoint_manifest_path = str(args.workpoint_manifest or '').strip()
    workpoint_manifest_by_circuit = (
        _load_workpoint_manifest(workpoint_manifest_path)
        if workpoint_manifest_path
        else {}
    )
    circuit_ids = (
        sorted(workpoint_manifest_by_circuit)
        if workpoint_manifest_by_circuit
        else [int(args.circuit_id)]
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = _parse_modes(args.modes)
    if "learned_schwarz_v1_sparse" in modes and not bool(args.apply_gmin_diagonal):
        raise ValueError(
            "learned_schwarz_v1_sparse requires --apply-gmin-diagonal"
        )

    steps_to_evaluate: List[Dict[str, Any]] = []
    for circuit_id in circuit_ids:
        steps = _find_steps(
            args.trajectory_dir,
            int(circuit_id),
            args.max_steps,
            requested_workpoints=(
                workpoint_manifest_by_circuit.get(int(circuit_id))
                if workpoint_manifest_by_circuit
                else None
            ),
        )
        if not steps:
            raise FileNotFoundError(
                'no sparse trajectory steps found in '
                f'{args.trajectory_dir} for circuit {int(circuit_id)}'
            )
        steps_to_evaluate.extend(steps)
    if workpoint_manifest_by_circuit:
        steps_to_evaluate.sort(
            key=lambda item: int(item['_workpoint_manifest_index'])
        )

    rows = []
    with (output_dir / 'per_step.jsonl').open('w', encoding='utf-8') as handle:
        for step in steps_to_evaluate:
            run_args = argparse.Namespace(**vars(args))
            run_args.circuit_id = int(step['circuit_id'])
            for mode in modes:
                row = evaluate_step(step, mode, run_args)
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                handle.flush()
                status = 'ok' if row.get('ok') else 'fail'
                ksp_reason = row.get(
                    'petsc_converged_reason_name',
                    row.get('petsc_converged_reason'),
                )
                print(
                    f"{status}\tbackend={args.backend}\tmode={mode}"
                    f"\tn={row.get('matrix_size')}\tnnz={row.get('matrix_nnz')}"
                    f"\titers={row.get('iterations')}"
                    f"\tratio={row.get('residual_ratio')}"
                    f"\tksp={ksp_reason}\treason={row.get('reason')}"
                )
    summary = {
        'backend': args.backend,
        'trajectory_dir': args.trajectory_dir,
        'circuit_id': int(args.circuit_id) if not workpoint_manifest_by_circuit else None,
        'circuit_ids': [int(value) for value in circuit_ids],
        'workpoint_manifest': workpoint_manifest_path or None,
        'exact_workpoint_selection': bool(workpoint_manifest_by_circuit),
        'learned_schwarz_parameter_mode': str(
            args.learned_schwarz_parameter_mode
        ),
        'modes': modes,
        'rows': rows,
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print('summary_json=' + str(output_dir / 'summary.json'))
if __name__ == '__main__':
    main()
