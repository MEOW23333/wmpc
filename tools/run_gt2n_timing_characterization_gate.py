#!/usr/bin/env python3
"""GT2N standard-cell timing-table quantitative gate."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/ZhangLexin/PALS/wmpc')
EVIDENCE = Path('/home/ZhangLexin/PALS/wmpc-gt2n-compat/results/gt2n_ngspice_compat_20260903')
GT2N = EVIDENCE / 'GT2N'
LIB = GT2N / 'lib/tt/gt2_6t_w13_svt_tt_0p7v25c.lib'
MODEL = EVIDENCE / 'full_adapter_with4_20260903/model_card_osdi.sp'
CELLS = EVIDENCE / 'full_adapter_with4_20260903/cells_osdi.cdl'
OSDI = EVIDENCE / 'official_bsimcmg111_20260903/build/bsimcmg_v111_official.osdi'
NGSPICE = Path('/home/ZhangLexin/PALS/release/src/ngspice')
OUT = ROOT / 'results/gt2n_timing_characterization_gate_20260904'
CELL = 'gt2_6t_inv_x1_w13_svt'
VDD = 0.7
THRESHOLD = VDD / 2
SLEW_LO = VDD * 0.3
SLEW_HI = VDD * 0.7


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def balanced_block(text: str, start: int) -> str:
    opening = text.find('{', start)
    if opening < 0:
        raise ValueError('缺少块起始大括号')
    depth = 0
    for pos in range(opening, len(text)):
        if text[pos] == '{':
            depth += 1
        elif text[pos] == '}':
            depth -= 1
            if depth == 0:
                return text[opening + 1:pos]
    raise ValueError('大括号未闭合')


def get_cell_block(text: str, name: str) -> str:
    match = re.search(r'(?im)^\s*cell\s*\(\s*' + re.escape(name) + r'\s*\)', text)
    if not match:
        raise ValueError(f'未找到单元 {name}')
    return balanced_block(text, match.end())


def get_table(block: str, name: str) -> tuple[list[float], list[float], list[list[float]]]:
    match = re.search(r'(?im)^\s*' + re.escape(name) + r'\s*\([^)]*\)', block)
    if not match:
        raise ValueError(f'未找到表 {name}')
    table = balanced_block(block, match.end())
    indexes = re.findall(r'(?is)index_[12]\s*\(\s*"([^"]+)"\s*\)', table)
    if len(indexes) != 2:
        raise ValueError(f'{name} 的索引不完整')
    index_1 = [float(x.strip()) for x in indexes[0].split(',')]
    index_2 = [float(x.strip()) for x in indexes[1].split(',')]
    values_match = re.search(r'(?is)values\s*\((.*?)\)\s*;', table)
    if not values_match:
        raise ValueError(f'{name} 的数据不完整')
    rows = []
    for quoted in re.findall(r'"([^"]+)"', values_match.group(1)):
        rows.append([float(x.strip()) for x in quoted.replace('\\', '').split(',')])
    if len(rows) != len(index_1) or any(len(row) != len(index_2) for row in rows):
        raise ValueError(f'{name} 的矩阵形状不匹配')
    return index_1, index_2, rows


def fmt(value: float) -> str:
    return f'{value:.16g}'


def netlist(slew_ps: float, cap_pf: float, direction: str, reference_delay_ps: float, reference_transition_ps: float, job: Path) -> str:
    full_ramp = slew_ps * 1e-12 / 0.4
    t0 = 200e-12
    tstop = max(5e-9, t0 + full_ramp + (reference_delay_ps + reference_transition_ps + 2000.0) * 1e-12)
    if direction == 'fall':
        values = (VDD, VDD, 0.0, 0.0)
        delay = 'meas tran delay trig v(a) val=0.35 fall=1 targ v(y) val=0.35 rise=1'
        transition = 'meas tran transition trig v(y) val=0.21 rise=1 targ v(y) val=0.49 rise=1'
    else:
        values = (0.0, 0.0, VDD, VDD)
        delay = 'meas tran delay trig v(a) val=0.35 rise=1 targ v(y) val=0.35 fall=1'
        transition = 'meas tran transition trig v(y) val=0.49 fall=1 targ v(y) val=0.21 fall=1'
    maxstep = min(0.1e-12, full_ramp / 25.0)
    p = [fmt(x) for x in (0.0, values[0], t0, values[1], t0 + full_ramp, values[2], tstop, values[3])]
    return '\n'.join([
        '* GT2N timing-table quantitative gate',
        f'.include {MODEL}',
        f'.include {CELLS}',
        'VDD vdd 0 0.7',
        'VSS vss 0 0',
        f'V_A A 0 PWL({" ".join(p)})',
        f'CLOAD Y 0 {fmt(cap_pf * 1e-12)}',
        f'XU A Y vdd vss {CELL}',
        '.temp 25',
        '.options method=gear maxord=2 reltol=1e-5 abstol=1e-15 vntol=1e-6',
        '.control',
        f'pre_osdi {OSDI}',
        'set noaskquit',
        f'tran {fmt(maxstep)} {fmt(tstop)}',
        delay,
        transition,
        f'meas tran ymin min v(y) from={fmt(t0)} to={fmt(tstop)}',
        f'meas tran ymax max v(y) from={fmt(t0)} to={fmt(tstop)}',
        f'wrdata {job / "waveform.dat"} time v(a) v(y)',
        'quit',
        '.endc',
        '.end',
        ''
    ])


def metric(text: str, key: str) -> float | None:
    match = re.search(r'(?im)^\s*' + re.escape(key) + r'\s*=\s*([-+]?\d+(?:\.\d*)?(?:e[-+]?\d+)?)', text)
    return float(match.group(1)) if match else None


def run_one(task: dict) -> dict:
    job = OUT / 'jobs' / task['name']
    job.mkdir(parents=True, exist_ok=True)
    net = netlist(task['slew_ps'], task['cap_pf'], task['direction'], task['reference_delay_ps'], task['reference_transition_ps'], job)
    net_path = job / 'test.cir'
    log_path = job / 'ngspice.log'
    net_path.write_text(net)
    began = time.perf_counter()
    try:
        proc = subprocess.run(
            ['/usr/bin/time', '-f', 'WMPC_MAXRSS_KB=%M', str(NGSPICE), '-b', str(net_path)],
            cwd=job, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60,
        )
        output = proc.stdout
        code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or '') + '\nTIMEOUT\n'
        code = 124
    elapsed = time.perf_counter() - began
    log_path.write_text(output)
    values = {key: metric(output, key) for key in ('delay', 'transition', 'ymin', 'ymax')}
    warnings = sorted(set(re.findall(r'(?im)^.*(?:warning|error|non-convergence|singular|unknown parameter|ignored).*$', output)))
    rss_match = re.search(r'(?m)^WMPC_MAXRSS_KB=(\d+)$', output)
    finite = all(value is not None and math.isfinite(value) for value in values.values())
    solver_aborted = bool(re.search(r'(?im)(?:tran simulation\(s\) aborted|timestep too small|convergence failed)', output))
    solver_completed = code == 0 and not solver_aborted
    delay_ps = values['delay'] * 1e12 if values['delay'] is not None else None
    transition_ps = values['transition'] * 1e12 if values['transition'] is not None else None
    delay_error = abs(delay_ps - task['reference_delay_ps']) / task['reference_delay_ps'] if delay_ps is not None else None
    transition_error = abs(transition_ps - task['reference_transition_ps']) / task['reference_transition_ps'] if transition_ps is not None else None
    rails_ok = finite and values['ymin'] >= -0.01 * VDD and values['ymax'] <= 1.01 * VDD
    numerical_pass = solver_completed and finite and rails_ok and delay_error is not None and transition_error is not None and delay_error <= 0.10 and transition_error <= 0.10
    record = {
        **task, 'returncode': code, 'solver_completed': solver_completed, 'runtime_sec': elapsed,
        'max_rss_kb': int(rss_match.group(1)) if rss_match else None,
        'delay_ps': delay_ps, 'transition_ps': transition_ps,
        'delay_relative_error': delay_error, 'transition_relative_error': transition_error,
        'ymin_v': values['ymin'], 'ymax_v': values['ymax'], 'rails_ok': rails_ok,
        'finite_measurements': finite, 'warnings': warnings,
        'numerical_pass': numerical_pass, 'strict_pass': numerical_pass and not warnings,
        'netlist_sha256': digest(net_path), 'log_sha256': digest(log_path),
    }
    (job / 'result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    waveform = job / 'waveform.dat'
    if waveform.exists():
        waveform.unlink()
    return record


def main() -> None:
    global OSDI, NGSPICE, OUT
    parser = argparse.ArgumentParser(description='GT2N 表征时序定量闸门')
    parser.add_argument('--osdi', type=Path, default=OSDI, help='待加载的 OSDI 模型路径')
    parser.add_argument('--ngspice', type=Path, default=NGSPICE, help='ngspice 二进制路径')
    parser.add_argument('--output', type=Path, default=OUT, help='本次输出目录')
    parser.add_argument('--workers', type=int, default=32, help='最大并发进程数')
    parser.add_argument('--date', default='2026-09-04', help='写入汇总的实验日期')
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError('并发进程数必须为正')
    OSDI = args.osdi.resolve()
    NGSPICE = args.ngspice.resolve()
    OUT = args.output.resolve()
    for path in (LIB, MODEL, CELLS, OSDI, NGSPICE):
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUT.exists():
        raise FileExistsError(f'为保持可追溯性，输出目录必须不存在：{OUT}')
    OUT.mkdir(parents=True)
    cell_block = get_cell_block(LIB.read_text(errors='replace'), CELL)
    fall_delay = get_table(cell_block, 'cell_fall')
    rise_delay = get_table(cell_block, 'cell_rise')
    fall_transition = get_table(cell_block, 'fall_transition')
    rise_transition = get_table(cell_block, 'rise_transition')
    for table in (rise_delay, fall_transition, rise_transition):
        if table[0] != fall_delay[0] or table[1] != fall_delay[1]:
            raise ValueError('四张表的索引不一致')
    tasks = []
    for row, slew in enumerate(fall_delay[0]):
        for col, cap in enumerate(fall_delay[1]):
            tasks.append({
                'name': f'input_rise_s{row}_c{col}', 'direction': 'rise', 'slew_ps': slew, 'cap_pf': cap,
                'reference_delay_ps': fall_delay[2][row][col], 'reference_transition_ps': fall_transition[2][row][col],
                'reference_table': 'cell_fall/fall_transition',
            })
            tasks.append({
                'name': f'input_fall_s{row}_c{col}', 'direction': 'fall', 'slew_ps': slew, 'cap_pf': cap,
                'reference_delay_ps': rise_delay[2][row][col], 'reference_transition_ps': rise_transition[2][row][col],
                'reference_table': 'cell_rise/rise_transition',
            })
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(run_one, tasks))
    errors_delay = [r['delay_relative_error'] for r in results if r['delay_relative_error'] is not None]
    errors_transition = [r['transition_relative_error'] for r in results if r['transition_relative_error'] is not None]
    summary = {
        'experiment': 'GT2N timing-characterization quantitative gate', 'date': args.date,
        'cell': CELL, 'task_count': len(tasks), 'target_concurrency': args.workers, 'actual_max_workers': args.workers,
        'source_lib': str(LIB), 'source_lib_sha256': digest(LIB), 'model_adapter': str(MODEL),
        'model_adapter_sha256': digest(MODEL), 'cells_adapter': str(CELLS), 'cells_adapter_sha256': digest(CELLS),
        'osdi': str(OSDI), 'osdi_sha256': digest(OSDI), 'ngspice': str(NGSPICE), 'ngspice_sha256': digest(NGSPICE),
        'normal_exit_processes': sum(r['returncode'] == 0 for r in results),
        'converged_processes': sum(r['solver_completed'] for r in results),
        'numerical_pass_count': sum(r['numerical_pass'] for r in results),
        'strict_pass_count': sum(r['strict_pass'] for r in results),
        'warning_count': sum(bool(r['warnings']) for r in results),
        'delay_error_mean': sum(errors_delay) / len(errors_delay) if errors_delay else None,
        'delay_error_median': sorted(errors_delay)[len(errors_delay) // 2] if errors_delay else None,
        'delay_error_max': max(errors_delay) if errors_delay else None,
        'transition_error_mean': sum(errors_transition) / len(errors_transition) if errors_transition else None,
        'transition_error_median': sorted(errors_transition)[len(errors_transition) // 2] if errors_transition else None,
        'transition_error_max': max(errors_transition) if errors_transition else None,
        'elapsed_sec': time.perf_counter() - started,
        'matrix_dimension': None, 'matrix_nnz': None, 'linear_iterations': None, 'residual_definition': None,
        'results': results,
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: summary[key] for key in summary if key not in ('results',)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
