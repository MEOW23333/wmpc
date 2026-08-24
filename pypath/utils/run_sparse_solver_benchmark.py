import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pypath.precondition_construction import SPARSE_SEMANTIC_MODES  # noqa: E402
from pypath.utils.sparse_gmres_prototype import (  # noqa: E402
    _find_steps,
    _load_workpoint_manifest,
    evaluate_step,
)

LOCAL_SCHUR_MODE = 'local_sparse_schur_sparse'
SEMANTIC_MODES = SPARSE_SEMANTIC_MODES
THREAD_ENV = ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']

DATASET_PRESETS: Dict[str, Dict[str, Any]] = {
    'core10x10': {
        'trajectory_dir': 'experiments/core_block_repro_10x10/aggregator/trajectory',
        'netlist_dir': 'experiments/core_block_repro_10x10/aggregator/generated_netlists',
        'circuit_ids': '0-19',
        'max_steps': 1,
    },
    'data_coupler_10x100': {
        'trajectory_dir': 'experiments/scale_probe/data_coupler_10x100/trajectory',
        'netlist_dir': 'experiments/scale_probe/data_coupler_10x100/generated_netlists',
        'circuit_ids': '0',
        'max_steps': 1,
    },
    'pg100x100': {
        'trajectory_dir': 'experiments/pg_gen_v4_100x100_exact_schur/trajectory',
        'netlist_dir': 'experiments/pg_gen_v4_100x100_exact_schur/generated_netlists',
        'circuit_ids': '0',
        'max_steps': 1,
    },
    'precondition10x10': {
        'trajectory_dir': 'precondition_experiments/10x10/trajectory',
        'netlist_dir': 'precondition_experiments/10x10/generated_netlists',
        'circuit_ids': '0-19',
        'max_steps': 1,
    },
    'precondition10x100': {
        'trajectory_dir': 'precondition_experiments/10x100/trajectory',
        'netlist_dir': 'precondition_experiments/10x100/generated_netlists',
        'circuit_ids': '0-19',
        'max_steps': 1,
    },
    'precondition100x100': {
        'trajectory_dir': 'precondition_experiments/100x100/trajectory',
        'netlist_dir': 'precondition_experiments/100x100/generated_netlists',
        'circuit_ids': '0-19',
        'max_steps': 1,
    },
}


def _csv(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in str(raw or '').split(','):
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _ints(raw: str) -> List[int]:
    values: List[int] = []
    for part in _csv(raw):
        if '-' in part:
            lo, hi = part.split('-', 1)
            values.extend(range(int(lo), int(hi) + 1))
        else:
            values.append(int(part))
    return values


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _mean(values: Iterable[Any]) -> Optional[float]:
    vals = [_float_or_none(v) for v in values]
    vals = [v for v in vals if v is not None]
    return None if not vals else float(sum(vals) / len(vals))


def _slug(value: Any) -> str:
    return ''.join(ch if ch.isalnum() or ch in {'_', '-'} else '_' for ch in str(value)).strip('_') or 'x'


def _resolve_path(path: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return str(p)


def _resolve_netlist(args: argparse.Namespace, circuit_id: int) -> str:
    if args.netlist_path:
        return _resolve_path(args.netlist_path)
    if not args.netlist_dir:
        return ''
    root = Path(_resolve_path(args.netlist_dir))
    candidates = [
        root / f'{int(circuit_id)}.sp',
        root / f'circuit_{int(circuit_id)}.sp',
        root / f'power_grid_100x100_{int(circuit_id):02d}.sp',
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    matches = sorted(root.glob('*.sp'))
    if int(circuit_id) == 0 and len(matches) == 1:
        return str(matches[0])
    raise FileNotFoundError(f'cannot resolve netlist for circuit_id={circuit_id} under {root}')


def _selected_steps(args: argparse.Namespace) -> List[Dict[str, Any]]:
    trajectory_dir = _resolve_path(args.trajectory_dir)
    selected: List[Dict[str, Any]] = []
    manifest_path = str(getattr(args, 'workpoint_manifest', '') or '').strip()
    manifest_by_circuit = (
        _load_workpoint_manifest(_resolve_path(manifest_path))
        if manifest_path
        else {}
    )
    wanted_steps = None if str(args.steps).strip().lower() in {'', 'all'} else set(_ints(args.steps))
    circuit_ids = (
        sorted(int(value) for value in manifest_by_circuit)
        if manifest_by_circuit
        else _ints(args.circuit_ids)
    )
    for cid in circuit_ids:
        steps = _find_steps(
            trajectory_dir,
            int(cid),
            max(int(args.max_steps), 1000000),
            requested_workpoints=(
                manifest_by_circuit.get(int(cid))
                if manifest_by_circuit
                else None
            ),
        )
        per_circuit_count = 0
        for idx, step in enumerate(steps):
            if not manifest_by_circuit:
                if wanted_steps is not None and idx not in wanted_steps:
                    continue
                if wanted_steps is None and per_circuit_count >= int(args.max_steps):
                    break
            row = dict(step)
            row['selected_step_index'] = int(
                step.get('_workpoint_manifest_index', idx)
            )
            row['netlist_path'] = _resolve_netlist(args, int(cid))
            selected.append(row)
            per_circuit_count += 1
    return selected


def _task_args(args: argparse.Namespace, task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'backend': 'scipy_sparse',
        'restart': int(args.gmres_restart),
        'max_iters': int(args.gmres_max_iters),
        'rtol': float(args.rtol),
        'atol': float(args.atol),
        'use_rhsold_as_x0': bool(args.use_rhsold_as_x0),
        'apply_gmin_diagonal': bool(args.apply_gmin_diagonal),
        'netlist_path': task.get('netlist_path', ''),
        'trajectory_dir': _resolve_path(args.trajectory_dir),
        'circuit_id': int(task['circuit_id']),
        'semantic_min_block_size': int(args.semantic_min_block_size),
        'semantic_max_block_size': int(args.semantic_max_block_size),
        'semantic_boundary_max_block_size': int(args.semantic_boundary_max_block_size),
        'semantic_max_blocks': int(args.semantic_max_blocks),
        'semantic_uncovered_policy': args.semantic_uncovered_policy,
        'semantic_coarse_max_condition': float(args.semantic_coarse_max_condition),
        'semantic_coarse_rank_tol': float(args.semantic_coarse_rank_tol),
        'interface_basis_snapshot_path': args.interface_basis_snapshot_path,
        'interface_basis_max_condition': float(args.interface_basis_max_condition),
        'interface_basis_rank_tol': float(args.interface_basis_rank_tol),
        'interface_residual_test_space': getattr(args, 'interface_residual_test_space', 'range_action'),
        'interface_residual_guard_tolerance': float(getattr(args, 'interface_residual_guard_tolerance', 1.0e-10)),
        'sparse_schur_edge_budget': int(task.get('selected_edge_budget') or args.default_selected_edge_budget),
        'local_schur_budget_multiplier': float(args.local_schur_budget_multiplier),
        'sparse_schur_candidate_edge_limit': int(args.sparse_schur_candidate_edge_limit),
        'sparse_schur_diagonal_shift': float(args.sparse_schur_diagonal_shift),
        'sparse_schur_factor_drop_tol': float(args.sparse_schur_factor_drop_tol),
        'sparse_schur_factor_fill_factor': float(args.sparse_schur_factor_fill_factor),
        'sparse_schur_interface_solve': str(task.get('interface_solve_mode') or args.default_interface_solve_mode),
        'sparse_schur_max_nnz': int(args.sparse_schur_max_nnz),
        'sparse_schur_max_degree': int(args.sparse_schur_max_degree),
        'sparse_schur_max_exact_entries': int(args.sparse_schur_max_exact_entries),
        'learning_sparse_schur_add_fraction': float(args.learning_sparse_schur_add_fraction),
        'learning_sparse_schur_add_min': int(args.learning_sparse_schur_add_min),
        'learning_sparse_schur_probe_restart': int(args.learning_sparse_schur_probe_restart),
        'learning_sparse_schur_probe_iterations': int(args.learning_sparse_schur_probe_iterations),
        'learned_local_sparse_schur_checkpoint': args.learned_local_sparse_schur_checkpoint,
        'learned_schwarz_checkpoint': args.learned_schwarz_checkpoint,
    }


def _build_tasks(args: argparse.Namespace) -> List[Dict[str, Any]]:
    modes = _csv(args.modes)
    budgets = _ints(args.selected_edge_budgets)
    ifaces = _csv(args.interface_solve_modes)
    tasks: List[Dict[str, Any]] = []
    for step in _selected_steps(args):
        for mode in modes:
            if mode == LOCAL_SCHUR_MODE:
                for budget in budgets:
                    tasks.append({**step, 'mode': mode, 'selected_edge_budget': int(budget), 'interface_solve_mode': args.default_interface_solve_mode})
                if args.task8_first_round_grid:
                    for iface in ifaces:
                        if iface == args.default_interface_solve_mode:
                            continue
                        tasks.append({**step, 'mode': mode, 'selected_edge_budget': int(args.interface_anchor_budget), 'interface_solve_mode': iface})
                else:
                    existing = {(int(t['selected_edge_budget']), str(t['interface_solve_mode'])) for t in tasks if t.get('step_path') == step.get('step_path') and t.get('mode') == mode}
                    for budget in budgets:
                        for iface in ifaces:
                            key = (int(budget), str(iface))
                            if key not in existing:
                                tasks.append({**step, 'mode': mode, 'selected_edge_budget': int(budget), 'interface_solve_mode': iface})
            else:
                tasks.append({**step, 'mode': mode, 'selected_edge_budget': None, 'interface_solve_mode': None})
    return tasks


def _read_time_v(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = line.strip()
        try:
            if line.startswith('Maximum resident set size'):
                out['isolated_max_rss_kb'] = int(line.rsplit(':', 1)[1].strip())
            elif line.startswith('User time'):
                out['time_v_user_sec'] = float(line.rsplit(':', 1)[1].strip())
            elif line.startswith('System time'):
                out['time_v_system_sec'] = float(line.rsplit(':', 1)[1].strip())
            elif line.startswith('Elapsed (wall clock) time'):
                out['time_v_elapsed_raw'] = line.rsplit(': ', 1)[1].strip()
            elif line.startswith('Exit status'):
                out['time_v_exit_status'] = int(line.rsplit(':', 1)[1].strip())
        except Exception:
            pass
    return out


def _run_child(task_json: Path, output_json: Path) -> None:
    payload = json.loads(task_json.read_text(encoding='utf-8'))
    task = dict(payload['task'])
    args = SimpleNamespace(**payload['args'])
    step = {k: v for k, v in task.items() if k not in {'mode', 'selected_edge_budget', 'interface_solve_mode', 'netlist_path'}}
    row = evaluate_step(step, str(task['mode']), args)
    row['selected_step_index'] = int(task.get('selected_step_index', -1))
    row['selected_edge_budget'] = task.get('selected_edge_budget')
    row['interface_solve_mode'] = task.get('interface_solve_mode')
    row['resolved_netlist_path'] = task.get('netlist_path')
    output_json.write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')


def _run_one(index: int, task: Dict[str, Any], args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    task_dir = output_dir / 'per_task' / f"task_{index:04d}__c{int(task['circuit_id'])}__s{int(task.get('selected_step_index', -1))}__{_slug(task['mode'])}__b{task.get('selected_edge_budget') or 'na'}__{_slug(task.get('interface_solve_mode') or 'na')}"
    task_dir.mkdir(parents=True, exist_ok=True)
    task_json = task_dir / 'task.json'
    row_json = task_dir / 'row.json'
    stdout_log = task_dir / 'stdout.log'
    stderr_log = task_dir / 'stderr.log'
    time_log = task_dir / 'time_v.log'
    task_json.write_text(json.dumps({'task': task, 'args': _task_args(args, task)}, indent=2, ensure_ascii=False), encoding='utf-8')
    env = os.environ.copy()
    for key in THREAD_ENV:
        env[key] = str(int(args.blas_threads))
    cmd = [
        '/usr/bin/timeout', '-k', '30s', f'{int(args.timeout_per_mode_sec)}s',
        '/usr/bin/time', '-v', '-o', str(time_log),
        sys.executable, __file__, '--child-task-json', str(task_json), '--child-output-json', str(row_json),
    ]
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=float(args.timeout_per_mode_sec) + 120.0)
    except subprocess.TimeoutExpired as exc:
        proc = subprocess.CompletedProcess(cmd, 124, stdout=exc.stdout or '', stderr=exc.stderr or 'timeout')
    stdout_log.write_text(proc.stdout or '', encoding='utf-8', errors='ignore')
    stderr_log.write_text(proc.stderr or '', encoding='utf-8', errors='ignore')
    if row_json.exists():
        row = json.loads(row_json.read_text(encoding='utf-8'))
    else:
        row = {
            'ok': False,
            'mode': task['mode'],
            'circuit_id': int(task['circuit_id']),
            'selected_step_index': int(task.get('selected_step_index', -1)),
            'failure_stage': 'isolated_subprocess',
            'reason': (proc.stderr or '')[-2000:],
        }
    row.update(_read_time_v(time_log))
    row['isolated_task_index'] = int(index)
    row['isolated_returncode'] = int(proc.returncode)
    row['isolated_timed_out'] = bool(int(proc.returncode) in {124, 137})
    row['isolated_wall_time_sec'] = float(time.perf_counter() - started)
    row['isolated_stdout_log'] = str(stdout_log)
    row['isolated_stderr_log'] = str(stderr_log)
    row['selected_edge_budget'] = task.get('selected_edge_budget')
    row['interface_solve_mode'] = task.get('interface_solve_mode')
    return row


def _schur(row: Dict[str, Any]) -> Dict[str, Any]:
    pc = row.get('preconditioner_info') if isinstance(row.get('preconditioner_info'), dict) else {}
    schur = pc.get('schur') if isinstance(pc.get('schur'), dict) else {}
    return schur if isinstance(schur, dict) else {}


def _core(row: Dict[str, Any]) -> Dict[str, Any]:
    pc = row.get('preconditioner_info') if isinstance(row.get('preconditioner_info'), dict) else {}
    schur = _schur(row)
    core = schur.get('core') if isinstance(schur.get('core'), dict) else pc.get('core')
    return core if isinstance(core, dict) else {}


def _residual_coarse(row: Dict[str, Any]) -> Dict[str, Any]:
    pc = row.get('preconditioner_info')
    pc = pc if isinstance(pc, dict) else {}
    coarse = pc.get('interface_residual_coarse')
    return coarse if isinstance(coarse, dict) else {}


def _normalize(row: Dict[str, Any]) -> Dict[str, Any]:
    schur = _schur(row)
    core = _core(row)
    coarse = _residual_coarse(row)
    corrector = coarse.get('corrector')
    corrector = corrector if isinstance(corrector, dict) else {}
    guard = coarse.get('guard')
    guard = guard if isinstance(guard, dict) else {}
    mode = str(row.get('mode'))
    p_nnz = schur.get('P_nnz', schur.get('P_theta_nnz'))
    fill = schur.get('P_theta_factor_fill_nnz', schur.get('schur_factor_nnz'))
    return {
        'schema_version': 2,
        'solver_family': 'unified_sparse_solver_benchmark',
        'backend': row.get('backend'),
        'mode': mode,
        'circuit_id': row.get('circuit_id'),
        'selected_step_index': row.get('selected_step_index'),
        'time': row.get('time'),
        'gmin_val': row.get('gmin_val'),
        'newton_iteration': row.get('iteration'),
        'matrix_size': row.get('matrix_size'),
        'nnz_A': row.get('matrix_nnz'),
        'csr_storage_mb': None if row.get('csr_storage_bytes') is None else float(row['csr_storage_bytes']) / (1024.0 * 1024.0),
        'rhs_norm': row.get('rhs_norm'),
        'x0_type': row.get('x0_source'),
        'selected_edge_budget': row.get('selected_edge_budget'),
        'interface_solve_mode': row.get('interface_solve_mode'),
        'direct_solve_baseline': bool(row.get('direct_solve_baseline', False)),
        'gmres_used': row.get('gmres_used', True if mode != 'splu' else not bool(row.get('direct_solve_baseline', False))),
        'gmres_info': row.get('gmres_info'),
        'gmres_iterations': row.get('iterations'),
        'true_rel_residual': row.get('true_rel_residual', row.get('residual_ratio')),
        'residual_ratio': row.get('residual_ratio'),
        'chosen_success': row.get('chosen_success', row.get('residual_based_success')),
        'residual_based_success': row.get('residual_based_success'),
        'setup_time': row.get('preconditioner_setup_s'),
        'solve_time': row.get('solve_s'),
        'total_wall_time': row.get('total_s', row.get('isolated_wall_time_sec')),
        'peak_rss_kb': row.get('isolated_max_rss_kb', row.get('peak_rss_kb')),
        'status': 'timeout' if row.get('isolated_timed_out') else ('ok' if row.get('ok') else 'fail'),
        'timeout_hit': bool(row.get('isolated_timed_out', False)),
        'failure_stage': row.get('failure_stage'),
        'failure_reason': row.get('reason'),
        'core_blocks': core.get('block_count'),
        'core_rows': schur.get('core_rows', core.get('covered_rows')),
        'P_nnz': p_nnz,
        'P_density': schur.get('P_density', schur.get('P_theta_density')),
        'P_factor_fill_nnz': fill,
        'selected_local_constructed': schur.get('selected_local_constructed'),
        'full_schur_constructed': schur.get('full_schur_constructed', False),
        'schur_matrix_storage_bytes': schur.get('schur_matrix_storage_bytes'),
        'schur_factor_storage_bytes': schur.get('schur_factor_storage_bytes'),
        'interface_retained_bytes': schur.get('interface_retained_bytes'),
        'base_accounted_preconditioner_retained_bytes': coarse.get(
            'base_accounted_preconditioner_retained_bytes',
            schur.get('accounted_preconditioner_retained_bytes'),
        ),
        'accounted_preconditioner_retained_bytes': coarse.get(
            'accounted_preconditioner_retained_bytes',
            schur.get('accounted_preconditioner_retained_bytes'),
        ),
        'coarse_retained_bytes': coarse.get('coarse_retained_bytes', 0),
        'coarse_requested_rank': coarse.get('requested_rank'),
        'coarse_actual_rank': coarse.get('actual_rank'),
        'coarse_enabled': corrector.get('enabled'),
        'coarse_test_space': corrector.get('test_space'),
        'coarse_basis_condition': corrector.get('basis_condition'),
        'coarse_test_basis_condition': corrector.get(
            'test_basis_condition'
        ),
        'coarse_reduced_condition': corrector.get('reduced_condition'),
        'coarse_guard_accepted': guard.get('accepted'),
        'coarse_guard_ratio': guard.get('candidate_to_base_ratio'),
        'coarse_fallback_reason': corrector.get('fallback_reason'),
        'resolved_netlist_path': row.get('resolved_netlist_path'),
    }


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        'mode', 'circuit_id', 'selected_step_index', 'matrix_size', 'nnz_A',
        'selected_edge_budget', 'interface_solve_mode', 'direct_solve_baseline', 'gmres_used',
        'gmres_iterations', 'gmres_info', 'true_rel_residual', 'chosen_success',
        'setup_time', 'solve_time', 'total_wall_time', 'peak_rss_kb', 'status', 'timeout_hit',
        'P_nnz', 'P_factor_fill_nnz', 'failure_stage', 'failure_reason',
        'interface_retained_bytes', 'coarse_retained_bytes',
        'accounted_preconditioner_retained_bytes', 'coarse_actual_rank',
        'coarse_enabled', 'coarse_guard_accepted', 'coarse_guard_ratio',
    ]
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summary(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(str(row.get('mode')), []).append(row)
    mode_summary = []
    for mode, group in sorted(by_mode.items()):
        mode_summary.append({
            'mode': mode,
            'task_count': len(group),
            'ok_count': sum(1 for r in group if r.get('status') == 'ok'),
            'success_count': sum(1 for r in group if bool(r.get('chosen_success'))),
            'timeout_count': sum(1 for r in group if bool(r.get('timeout_hit'))),
            'avg_iterations': _mean(r.get('gmres_iterations') for r in group if r.get('gmres_iterations') is not None),
            'avg_true_rel_residual': _mean(r.get('true_rel_residual') for r in group if r.get('true_rel_residual') is not None),
            'avg_total_wall_time': _mean(r.get('total_wall_time') for r in group),
            'avg_peak_rss_kb': _mean(r.get('peak_rss_kb') for r in group),
            'coarse_enabled_count': sum(
                1 for r in group if r.get('coarse_enabled') is True
            ),
            'coarse_fallback_count': sum(
                1 for r in group if r.get('coarse_enabled') is False
            ),
            'avg_interface_retained_bytes': _mean(
                r.get('interface_retained_bytes') for r in group
            ),
            'avg_accounted_preconditioner_retained_bytes': _mean(
                r.get('accounted_preconditioner_retained_bytes')
                for r in group
            ),
        })
    return {
        'metadata': {
            'runner': 'run_sparse_solver_benchmark.py',
            'dataset_preset': args.dataset_preset,
            'trajectory_dir': args.trajectory_dir,
            'netlist_dir': args.netlist_dir,
            'netlist_path': args.netlist_path,
            'circuit_ids': args.circuit_ids,
            'max_steps': int(args.max_steps),
            'modes': _csv(args.modes),
            'task_count': len(rows),
            'workers': int(args.workers),
            'isolated_subprocess': True,
            'sparse_matrix_default': True,
        },
        'mode_summary': mode_summary,
    }


def _apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = DATASET_PRESETS.get(args.dataset_preset, {}) if args.dataset_preset else {}
    for key in ['trajectory_dir', 'netlist_dir', 'circuit_ids', 'max_steps']:
        if getattr(args, key) in {None, '', 0} and key in preset:
            setattr(args, key, preset[key])
    if not args.circuit_ids:
        args.circuit_ids = '0'
    if int(args.max_steps) <= 0:
        args.max_steps = 1
    if not args.trajectory_dir:
        raise ValueError('--trajectory-dir is required, either directly or through --dataset-preset')
    if any(mode in SEMANTIC_MODES for mode in _csv(args.modes)) and not (args.netlist_dir or args.netlist_path):
        raise ValueError('semantic modes require --netlist-dir or --netlist-path')
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description='Unified sparse linear solver benchmark for 10x10, 10x100, and 100x100 PALS systems.')
    parser.add_argument('--child-task-json', default='')
    parser.add_argument('--child-output-json', default='')
    parser.add_argument('--dataset-preset', choices=['', *DATASET_PRESETS.keys()], default='')
    parser.add_argument('--trajectory-dir', default='')
    parser.add_argument('--netlist-dir', default='')
    parser.add_argument('--netlist-path', default='')
    parser.add_argument('--workpoint-manifest', default='')
    parser.add_argument('--circuit-ids', default='')
    parser.add_argument('--steps', default='0')
    parser.add_argument('--max-steps', type=int, default=0)
    parser.add_argument('--modes', default='row_sum,semantic_cell_core_sparse,local_sparse_schur_sparse,splu')
    parser.add_argument('--selected-edge-budgets', default='512,1024,4096,16384')
    parser.add_argument('--interface-solve-modes', default='spilu,jacobi,jacobi_neumann1')
    parser.add_argument('--default-selected-edge-budget', type=int, default=4096)
    parser.add_argument('--default-interface-solve-mode', default='spilu')
    parser.add_argument('--interface-anchor-budget', type=int, default=4096)
    parser.add_argument('--task8-first-round-grid', action='store_true', default=True)
    parser.add_argument('--full-cartesian-grid', dest='task8_first_round_grid', action='store_false')
    parser.add_argument('--gmres-restart', type=int, default=100)
    parser.add_argument('--gmres-max-iters', type=int, default=500)
    parser.add_argument('--rtol', type=float, default=1e-8)
    parser.add_argument('--atol', type=float, default=1e-10)
    parser.add_argument('--timeout-per-mode-sec', type=float, default=1200.0)
    parser.add_argument('--blas-threads', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output-dir', required=False, default='')
    parser.add_argument('--use-rhsold-as-x0', action='store_true', default=False)
    parser.add_argument('--apply-gmin-diagonal', action='store_true', default=True)
    parser.add_argument('--disable-gmin-diagonal', dest='apply_gmin_diagonal', action='store_false')
    parser.add_argument('--semantic-min-block-size', type=int, default=2)
    parser.add_argument('--semantic-max-block-size', type=int, default=96)
    parser.add_argument('--semantic-boundary-max-block-size', type=int, default=192)
    parser.add_argument('--semantic-max-blocks', type=int, default=0)
    parser.add_argument('--semantic-uncovered-policy', default='row_sum')
    parser.add_argument('--semantic-coarse-max-condition', type=float, default=1.0e12)
    parser.add_argument('--semantic-coarse-rank-tol', type=float, default=1.0e-10)
    parser.add_argument('--interface-basis-snapshot-path', default='')
    parser.add_argument('--interface-basis-max-condition', type=float, default=1.0e12)
    parser.add_argument('--interface-basis-rank-tol', type=float, default=1.0e-10)
    parser.add_argument('--interface-residual-test-space', choices=['range_action', 'galerkin'], default='range_action')
    parser.add_argument('--interface-residual-guard-tolerance', type=float, default=1.0e-10)
    parser.add_argument('--local-schur-budget-multiplier', type=float, default=2.0)
    parser.add_argument('--sparse-schur-candidate-edge-limit', type=int, default=16384)
    parser.add_argument('--sparse-schur-diagonal-shift', type=float, default=1e-8)
    parser.add_argument('--sparse-schur-factor-drop-tol', type=float, default=1e-4)
    parser.add_argument('--sparse-schur-factor-fill-factor', type=float, default=10.0)
    parser.add_argument('--sparse-schur-max-nnz', type=int, default=0)
    parser.add_argument('--sparse-schur-max-degree', type=int, default=0)
    parser.add_argument('--sparse-schur-max-exact-entries', type=int, default=0)
    parser.add_argument('--learning-sparse-schur-add-fraction', type=float, default=0.0)
    parser.add_argument('--learning-sparse-schur-add-min', type=int, default=0)
    parser.add_argument('--learning-sparse-schur-probe-restart', type=int, default=8)
    parser.add_argument('--learning-sparse-schur-probe-iterations', type=int, default=1)
    parser.add_argument('--learned-schwarz-checkpoint', default='')
    parser.add_argument('--learned-local-sparse-schur-checkpoint', default='')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.child_task_json:
        _run_child(Path(args.child_task_json), Path(args.child_output_json))
        return
    args = _apply_preset(args)
    tasks = _build_tasks(args)
    if args.dry_run:
        print(json.dumps({'task_count': len(tasks), 'tasks': tasks}, indent=2, ensure_ascii=False))
        return
    if not args.output_dir:
        raise ValueError('--output-dir is required unless --dry-run is used')
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'planned_tasks.json').write_text(json.dumps({'task_count': len(tasks), 'tasks': tasks}, indent=2, ensure_ascii=False), encoding='utf-8')
    raw_rows: List[Dict[str, Any]] = []
    workers = min(max(int(args.workers), 1), max(len(tasks), 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(_run_one, idx, task, args, output_dir): (idx, task) for idx, task in enumerate(tasks)}
        for future in as_completed(future_map):
            idx, task = future_map[future]
            row = future.result()
            raw_rows.append(row)
            print(f"task={idx} mode={task['mode']} circuit={task['circuit_id']} step={task.get('selected_step_index')} success={row.get('chosen_success')} ratio={row.get('residual_ratio')} timeout={row.get('isolated_timed_out')}", flush=True)
    raw_rows.sort(key=lambda row: int(row.get('isolated_task_index', 0)))
    norm_rows = [_normalize(row) for row in raw_rows]
    _write_jsonl(output_dir / 'raw_rows.jsonl', raw_rows)
    _write_jsonl(output_dir / 'solver_outcome.jsonl', norm_rows)
    _write_csv(output_dir / 'result_table.csv', norm_rows)
    summary = _summary(norm_rows, args)
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('summary_json=' + str(output_dir / 'summary.json'))
    print('solver_outcome_jsonl=' + str(output_dir / 'solver_outcome.jsonl'))


if __name__ == '__main__':
    main()
