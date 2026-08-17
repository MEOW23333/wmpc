import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _mode_slug(mode: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in {'_', '-'} else '_' for ch in str(mode))


def _worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    from pypath.utils.sparse_gmres_prototype import evaluate_step

    step = payload['step']
    mode = payload['mode']
    step_index = int(payload['step_index'])
    args = SimpleNamespace(**payload['args'])
    job_dir = Path(payload['job_dir'])
    job_dir.mkdir(parents=True, exist_ok=True)
    out_path = job_dir / f"step_{step_index:03d}__{_mode_slug(mode)}.json"
    if out_path.exists() and not bool(payload.get('force', False)):
        try:
            row = json.loads(out_path.read_text())
            row['reused_existing_job'] = True
            return row
        except Exception:
            pass
    start = time.perf_counter()
    try:
        row = evaluate_step(dict(step), str(mode), args)
        row['parallel_job_ok'] = True
    except Exception as exc:
        row = dict(step)
        row.update({
            'backend': args.backend,
            'mode': str(mode),
            'ok': False,
            'parallel_job_ok': False,
            'failure_stage': 'parallel_worker',
            'reason': repr(exc),
        })
    row['step_index'] = step_index
    row['parallel_job_s'] = float(time.perf_counter() - start)
    tmp = out_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(row, ensure_ascii=False) + '\n')
    os.replace(tmp, out_path)
    return row


def _mean(values: List[float]) -> float | None:
    values = [float(v) for v in values if v is not None]
    return float(statistics.mean(values)) if values else None


def _median(values: List[float]) -> float | None:
    values = [float(v) for v in values if v is not None]
    return float(statistics.median(values)) if values else None


def _summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    modes = sorted({str(r.get('mode')) for r in rows})
    for mode in modes:
        group = [r for r in rows if str(r.get('mode')) == mode]
        ok_rows = [r for r in group if bool(r.get('ok'))]
        residual_success = [
            r for r in ok_rows
            if bool(r.get('residual_based_success')) or (
                r.get('residual_ratio') is not None and float(r.get('residual_ratio')) < 1e-8
            )
        ]
        out.append({
            'mode': mode,
            'sample_count': int(len(group)),
            'ok_count': int(len(ok_rows)),
            'converged_count': int(len(residual_success)),
            'failure_count': int(len(group) - len(ok_rows)),
            'avg_peak_rss_kb': _mean([r.get('peak_rss_kb') for r in group]),
            'avg_peak_rss_mb': None if _mean([r.get('peak_rss_kb') for r in group]) is None else _mean([r.get('peak_rss_kb') for r in group]) / 1024.0,
            'avg_iterations': _mean([r.get('iterations') for r in ok_rows]),
            'median_iterations': _median([r.get('iterations') for r in ok_rows]),
            'avg_residual_ratio': _mean([r.get('residual_ratio') for r in ok_rows]),
            'median_residual_ratio': _median([r.get('residual_ratio') for r in ok_rows]),
            'avg_setup_s': _mean([r.get('preconditioner_setup_s') for r in group]),
            'avg_solve_s': _mean([r.get('solve_s') for r in ok_rows]),
            'avg_total_s': _mean([r.get('total_s', r.get('parallel_job_s')) for r in group]),
        })
    return out


def _plot(summary_dir: Path, rows: List[Dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plot_dir = summary_dir / 'iteration_histograms'
    plot_dir.mkdir(parents=True, exist_ok=True)
    for mode in sorted({str(r.get('mode')) for r in rows}):
        vals = [int(r.get('iterations')) for r in rows if str(r.get('mode')) == mode and r.get('iterations') is not None]
        if not vals:
            continue
        plt.figure(figsize=(7, 4.5))
        bins = min(30, max(5, len(set(vals))))
        plt.hist(vals, bins=bins, edgecolor='black')
        plt.title(f'GMRES iteration distribution: {mode}')
        plt.xlabel('GMRES callback iterations')
        plt.ylabel('count')
        plt.tight_layout()
        plt.savefig(plot_dir / f'{_mode_slug(mode)}.png', dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory-dir', required=True)
    parser.add_argument('--circuit-id', type=int, default=0)
    parser.add_argument('--netlist-path', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--workers', type=int, default=200)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--modes', default='identity,row_sum,jacobi_diagonal,ilu0,ilut,splu,semantic_cell_core_sparse,local_sparse_schur_sparse,learned_sparse_schur_safe_add_sparse,learned_sparse_schur_safe_add_probe_sparse')
    parser.add_argument('--backend', default='scipy_sparse')
    parser.add_argument('--restart', type=int, default=30)
    parser.add_argument('--max-iters', type=int, default=120)
    parser.add_argument('--rtol', type=float, default=1e-8)
    parser.add_argument('--atol', type=float, default=1e-10)
    parser.add_argument('--semantic-min-block-size', type=int, default=2)
    parser.add_argument('--semantic-max-block-size', type=int, default=96)
    parser.add_argument('--semantic-boundary-max-block-size', type=int, default=192)
    parser.add_argument('--semantic-max-blocks', type=int, default=0)
    parser.add_argument('--semantic-uncovered-policy', default='row_sum')
    parser.add_argument('--sparse-schur-edge-budget', type=int, default=512)
    parser.add_argument('--local-schur-budget-multiplier', type=float, default=2.0)
    parser.add_argument('--sparse-schur-candidate-edge-limit', type=int, default=16384)
    parser.add_argument('--sparse-schur-diagonal-shift', type=float, default=1e-8)
    parser.add_argument('--sparse-schur-factor-drop-tol', type=float, default=1e-4)
    parser.add_argument('--sparse-schur-factor-fill-factor', type=float, default=10.0)
    parser.add_argument('--sparse-schur-interface-solve', default='spilu')
    parser.add_argument('--sparse-schur-max-nnz', type=int, default=0)
    parser.add_argument('--sparse-schur-max-degree', type=int, default=0)
    parser.add_argument('--sparse-schur-max-exact-entries', type=int, default=0)
    parser.add_argument('--learning-sparse-schur-add-fraction', type=float, default=0.1)
    parser.add_argument('--learning-sparse-schur-add-min', type=int, default=1)
    parser.add_argument('--learning-sparse-schur-probe-restart', type=int, default=8)
    parser.add_argument('--learning-sparse-schur-probe-iterations', type=int, default=1)
    parser.add_argument('--learned-local-sparse-schur-checkpoint', default='')
    parser.add_argument('--apply-gmin-diagonal', action='store_true', default=True)
    parser.add_argument('--use-rhsold-as-x0', action='store_true', default=False)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    from pypath.utils.sparse_gmres_prototype import _find_steps

    output_dir = Path(args.output_dir)
    job_dir = output_dir / 'jobs'
    output_dir.mkdir(parents=True, exist_ok=True)
    job_dir.mkdir(parents=True, exist_ok=True)
    steps = _find_steps(args.trajectory_dir, args.circuit_id, args.max_steps)
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]

    args_dict = vars(args).copy()
    args_dict['use_rhsold_as_x0'] = False
    tasks = []
    for i, step in enumerate(steps):
        for mode in modes:
            tasks.append({
                'step_index': i,
                'step': step,
                'mode': mode,
                'args': args_dict,
                'job_dir': str(job_dir),
                'force': bool(args.force),
            })

    meta = {
        'trajectory_dir': args.trajectory_dir,
        'circuit_id': int(args.circuit_id),
        'netlist_path': args.netlist_path,
        'workers_requested': int(args.workers),
        'workers_actual': int(args.workers),
        'max_steps': int(args.max_steps),
        'step_count': int(len(steps)),
        'modes': modes,
        'task_count': int(len(tasks)),
        'x0_policy': 'zero',
        'residual_definition': '||Jx-rhs||_2 / max(||rhs||_2, 1e-30)',
    }
    (output_dir / 'run_metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    rows: List[Dict[str, Any]] = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = [pool.submit(_worker, task) for task in tasks]
        for k, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            if k % 10 == 0 or k == len(futures):
                print(f'completed {k}/{len(futures)}', flush=True)

    rows.sort(key=lambda r: (str(r.get('mode')), int(r.get('step_index', -1))))
    summary = _summarize(rows)
    total_s = float(time.perf_counter() - start)
    payload = {
        'metadata': {**meta, 'wall_s': total_s},
        'summary': summary,
        'rows': rows,
    }
    (output_dir / 'summary.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    with (output_dir / 'per_job.jsonl').open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    _plot(output_dir, rows)

    lines = []
    lines.append('# 100x10 100-equation 200-worker preconditioner sweep')
    lines.append('')
    lines.append(f'- output_dir: {output_dir}')
    lines.append(f'- workers: {args.workers}')
    lines.append(f'- steps: {len(steps)}')
    lines.append(f'- modes: {len(modes)}')
    lines.append(f'- tasks: {len(tasks)}')
    lines.append(f'- wall_s: {total_s:.3f}')
    lines.append(f'- x0: zero')
    lines.append(f'- residual: ||Jx-rhs||_2 / max(||rhs||_2, 1e-30)')
    lines.append('')
    lines.append('| mode | samples | converged | failed | avg RSS MB | avg iters | median iters | avg residual | median residual | avg solve s |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for item in summary:
        def f(v):
            if v is None:
                return 'n/a'
            if abs(float(v)) >= 1e4 or abs(float(v)) < 1e-3:
                return f'{float(v):.3e}'
            return f'{float(v):.4f}'
        lines.append(
            f"| {item['mode']} | {item['sample_count']} | {item['converged_count']} | {item['failure_count']} | "
            f"{f(item.get('avg_peak_rss_mb'))} | {f(item.get('avg_iterations'))} | {f(item.get('median_iterations'))} | "
            f"{f(item.get('avg_residual_ratio'))} | {f(item.get('median_residual_ratio'))} | {f(item.get('avg_solve_s'))} |"
        )
    (output_dir / 'summary.md').write_text('\n'.join(lines) + '\n')
    print(output_dir / 'summary.json')
    print(output_dir / 'summary.md')


if __name__ == '__main__':
    main()

