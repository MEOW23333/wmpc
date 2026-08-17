#!/usr/bin/env python3
"""冻结状态、固定 gmin 的预热单阶段重启式核验。

此入口只量化同一冻结状态下的预热因果差异；它不恢复完整动态 gmin
控制器状态，结果不得解释为完整续接部署收益。
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pypath.utils.ngspice_utils import NGSPICE_EXECUTABLE, read_continuation_step

SCHEMA_VERSION = 1
TRAJ_RE = re.compile(r"^circuit_(\d+)_time_([0-9.eE+-]+)_gmin_([0-9.eE+-]+)_iter_(\d+)\.txt$")
CONT_RE = re.compile(r"^continuation_circuit_(\d+)_time_([0-9.eE+-]+)_gmin_([0-9.eE+-]+)_iter_(\d+)\.txt$")
WARM_RE = re.compile(r"^segment_warmup_circuit_(\d+)_time_([0-9.eE+-]+)_gmin_([0-9.eE+-]+)_rhsold\.txt$")
RANGE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
ARMS = (("cold_row_sum", False), ("warm_row_sum", True))


def rp(raw, exists=False):
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError(f"path outside PALS: {path}")
    if exists and not path.exists():
        raise FileNotFoundError(path)
    return path


def sf(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sa(values):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1)).tobytes()).hexdigest()


def fk(value):
    return f"{float(value):.17e}"


def skey(circuit_id, time_value, gmin_value):
    return f"c{int(circuit_id)}__t{fk(time_value)}__g{fk(gmin_value)}"


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def wjson(path, value):
    write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def rjson(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def rjsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_ids(raw):
    result, seen = [], set()
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        match = RANGE_RE.match(token)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            step = 1 if start <= end else -1
            values = range(start, end + step, step)
        else:
            values = [int(token)]
        for value in values:
            if value not in seen:
                seen.add(value)
                result.append(value)
    if not result:
        raise ValueError("no circuit ids")
    return result


def frozen_state(path):
    payload = read_continuation_step(str(path))
    rhsold = np.asarray(payload.get("rhsold", []), dtype=np.float64).reshape(-1)
    state0 = np.asarray(payload.get("state0_in", []), dtype=np.float64).reshape(-1)
    if rhsold.size == 0 or state0.size == 0 or not np.all(np.isfinite(rhsold)) or not np.all(np.isfinite(state0)):
        raise ValueError(f"invalid frozen state: {path}")
    return rhsold, state0


def vector(path):
    values = np.asarray(np.loadtxt(path, dtype=np.float64), dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid vector: {path}")
    return values


def scan_traj(root, circuit_ids):
    chosen, skipped, wanted = {}, Counter(), set(circuit_ids)
    for path in sorted(root.glob("circuit_*_time_*_gmin_*_iter_*.txt")):
        match = TRAJ_RE.match(path.name)
        if not match:
            continue
        circuit_id, time_value, gmin_value, iteration = int(match.group(1)), float(match.group(2)), float(match.group(3)), int(match.group(4))
        if circuit_id not in wanted:
            continue
        if gmin_value <= 0:
            skipped["zero_gmin"] += 1
            continue
        key = skey(circuit_id, time_value, gmin_value)
        candidate = {"path": str(path.resolve()), "sha256": sf(path), "circuit_id": circuit_id, "time": time_value, "gmin_val": gmin_value, "iteration": iteration}
        old = chosen.get(key)
        if old is None or (iteration, candidate["path"]) < (old["iteration"], old["path"]):
            chosen[key] = candidate
    return chosen, skipped


def scan_warm(root, circuit_ids):
    selected, result = set(circuit_ids), {}
    for path in sorted(root.rglob("segment_warmup_circuit_*_rhsold.txt")):
        match = WARM_RE.match(path.name)
        if not match:
            continue
        circuit_id, time_value, gmin_value = int(match.group(1)), float(match.group(2)), float(match.group(3))
        if circuit_id not in selected:
            continue
        key = skey(circuit_id, time_value, gmin_value)
        candidate = {"path": str(path.resolve()), "sha256": sf(path), "circuit_id": circuit_id, "time": time_value, "gmin_val": gmin_value}
        old = result.get(key)
        if old and old["sha256"] != candidate["sha256"]:
            raise ValueError(f"conflicting warmup vector: {key}")
        if old is None or candidate["path"] < old["path"]:
            result[key] = candidate
    return result


def core(task):
    keys = ("schema_version", "task_id", "stage_key", "arm", "warmup_enabled", "circuit_id", "time", "gmin_val", "trajectory", "warmup_vector", "netlist", "expected_input_array_sha256", "expected_state0_array_sha256", "expected_input_size", "expected_state0_size", "gmres_env", "max_newton_steps", "residual_tol", "timeout_sec")
    return {key: task[key] for key in keys}


def git_snapshot():
    def call(args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False).stdout.strip()
    diff = subprocess.run(["git", "diff", "--binary", "--no-ext-diff"], cwd=REPO_ROOT, capture_output=True, check=False).stdout
    return {"head": call(["rev-parse", "HEAD"]), "branch": call(["branch", "--show-current"]), "status_lines": call(["status", "--short"]).splitlines(), "tracked_diff_sha256": hashlib.sha256(diff or b"").hexdigest()}


def provenance(items):
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"provenance must be label=path: {item}")
        label, raw_path = item.split("=", 1)
        path = rp(raw_path, True)
        if not label or label in result or not path.is_file():
            raise ValueError(f"invalid provenance: {item}")
        result[label] = {"path": str(path), "sha256": sf(path)}
    return result


def prepare(args):
    output, traj_root, warm_root, netlist_root = rp(args.output_dir), rp(args.trajectory_dir, True), rp(args.warmup_input_root, True), rp(args.netlist_dir, True)
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    circuit_ids = parse_ids(args.circuit_ids)
    trajectories, skipped = scan_traj(traj_root, circuit_ids)
    warmups, unavailable, selected = scan_warm(warm_root, circuit_ids), Counter(), []
    for circuit_id in circuit_ids:
        candidates = sorted((row for row in trajectories.values() if row["circuit_id"] == circuit_id), key=lambda row: (row["time"], -row["gmin_val"], row["iteration"]))
        accepted = 0
        for trajectory in candidates:
            key = skey(circuit_id, trajectory["time"], trajectory["gmin_val"])
            warmup, netlist = warmups.get(key), netlist_root / f"{circuit_id}.sp"
            if warmup is None:
                unavailable["warmup_vector_missing"] += 1
                continue
            try:
                entry_rhsold, state0, warm_rhsold = *frozen_state(Path(trajectory["path"])), vector(Path(warmup["path"]))
            except ValueError:
                unavailable["invalid_source_state"] += 1
                continue
            if warm_rhsold.size != entry_rhsold.size:
                unavailable["warmup_dimension_mismatch"] += 1
                continue
            if not netlist.is_file():
                unavailable["netlist_missing"] += 1
                continue
            selected.append({"stage_key": key, "trajectory": trajectory, "warmup_vector": warmup, "entry_sha": sa(entry_rhsold), "warm_sha": sa(warm_rhsold), "state_sha": sa(state0), "input_size": int(entry_rhsold.size), "state_size": int(state0.size), "netlist": {"path": str(netlist), "sha256": sf(netlist)}})
            accepted += 1
            if accepted >= args.max_workpoints_per_circuit:
                break
        if accepted == 0:
            unavailable[f"circuit_{circuit_id}_no_usable_workpoint"] += 1
    if not selected:
        raise RuntimeError("no usable frozen workpoint")
    output.mkdir(parents=True)
    gmres = {"NGSPICE_GMRES_MODE": "1", "NGSPICE_GMRES_PRECOND": "row_sum", "NGSPICE_GMRES_PRECOND_FALLBACK": "row_sum", "NGSPICE_GMRES_RESTART": str(args.gmres_restart), "NGSPICE_GMRES_MAX_ITERS": str(args.gmres_max_iters), "NGSPICE_GMRES_RTOL": fk(args.gmres_rtol), "NGSPICE_GMRES_ATOL": fk(args.gmres_atol), "NGSPICE_GMRES_USE_RHSOLD_X0": "1"}
    tasks = []
    for source in selected:
        for arm, warmup_enabled in ARMS:
            task_dir = output / "tasks" / source["stage_key"] / arm
            task = {"schema_version": SCHEMA_VERSION, "task_id": f"{source['stage_key']}__{arm}", "stage_key": source["stage_key"], "arm": arm, "warmup_enabled": warmup_enabled, "circuit_id": source["trajectory"]["circuit_id"], "time": source["trajectory"]["time"], "gmin_val": source["trajectory"]["gmin_val"], "trajectory": source["trajectory"], "warmup_vector": source["warmup_vector"], "netlist": source["netlist"], "expected_input_array_sha256": source["warm_sha"] if warmup_enabled else source["entry_sha"], "expected_state0_array_sha256": source["state_sha"], "expected_input_size": source["input_size"], "expected_state0_size": source["state_size"], "gmres_env": gmres, "max_newton_steps": args.max_newton_steps, "residual_tol": args.residual_tol, "timeout_sec": args.timeout_sec, "task_dir": str(task_dir)}
            task["task_fingerprint"] = canonical(core(task))
            task_dir.mkdir(parents=True)
            wjson(task_dir / "task.json", task)
            tasks.append(task)
    manifest = {"schema_version": SCHEMA_VERSION, "runner": "run_frozen_warmup_stage_benchmark.py", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "experiment_scope": "fixed_gmin_frozen_state_restart_microbenchmark", "not_valid_for": "complete_adaptive_gmin_continuation_claim", "output_dir": str(output), "trajectory_dir": str(traj_root), "warmup_input_root": str(warm_root), "netlist_dir": str(netlist_root), "circuit_ids": circuit_ids, "selected_workpoint_count": len(selected), "task_count": len(tasks), "source_skipped": dict(sorted(skipped.items())), "source_unavailable": dict(sorted(unavailable.items())), "provenance_files": provenance(args.provenance_path), "ngspice_executable": str(NGSPICE_EXECUTABLE), "ngspice_executable_sha256": sf(Path(NGSPICE_EXECUTABLE)), "runner_sha256": sf(Path(__file__)), "ngspice_utils_sha256": sf(REPO_ROOT / "pypath" / "utils" / "ngspice_utils.py"), "python": {"executable": sys.executable, "version": sys.version}, "git": git_snapshot()}
    manifest["manifest_fingerprint"] = canonical(manifest)
    wjson(output / "run_manifest.json", manifest)
    write(output / "planned_tasks.jsonl", "".join(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n" for task in tasks))
    print(f"run_manifest={output / 'run_manifest.json'}\nselected_workpoint_count={len(selected)}\ntask_count={len(tasks)}")
    return 0


def task_inputs(task):
    trajectory, warmup = rp(task["trajectory"]["path"], True), rp(task["warmup_vector"]["path"], True)
    if sf(trajectory) != task["trajectory"]["sha256"] or sf(warmup) != task["warmup_vector"]["sha256"]:
        raise ValueError("source file fingerprint mismatch")
    cold, state0 = frozen_state(trajectory)
    rhsold = vector(warmup) if task["warmup_enabled"] else cold
    if rhsold.size != task["expected_input_size"] or state0.size != task["expected_state0_size"] or sa(rhsold) != task["expected_input_array_sha256"] or sa(state0) != task["expected_state0_array_sha256"]:
        raise ValueError("source array contract mismatch")
    return rhsold, state0


def steps(runtime):
    result = []
    for path in sorted(runtime.glob("continuation_circuit_*_time_*_gmin_*_iter_*.txt")):
        match = CONT_RE.match(path.name)
        if not match:
            continue
        payload = read_continuation_step(str(path))
        residual = np.asarray(payload.get("residual", []), dtype=np.float64).reshape(-1)
        result.append({"path": str(path), "sha256": sf(path), "iteration": int(match.group(4)), "time": float(match.group(2)), "gmin_val": float(match.group(3)), "rhsold": np.asarray(payload.get("rhsold", []), dtype=np.float64).reshape(-1), "state0": np.asarray(payload.get("state0_in", []), dtype=np.float64).reshape(-1), "residual_norm": float(np.linalg.norm(residual)) if residual.size else None})
    return sorted(result, key=lambda row: row["iteration"])


def metric_errors(rows):
    errors = []
    if not rows:
        return ["gmres_metrics_missing"]
    for index, row in enumerate(rows):
        prefix = f"gmres_row_{index}"
        for key in ("gmres_success", "gmres_converged", "strict_true_relative_residual_pass"):
            if not bool(row.get(key)):
                errors.append(f"{prefix}_{key}_false")
        if bool(row.get("direct_fallback")):
            errors.append(f"{prefix}_direct_fallback")
        if row.get("requested_preconditioner_mode") != "row_sum" or row.get("executed_preconditioner_mode") != "row_sum":
            errors.append(f"{prefix}_preconditioner_mismatch")
    return errors


def run(args):
    task_path, task = rp(args.task_json, True), None
    task = rjson(task_path)
    if task.get("schema_version") != SCHEMA_VERSION or task.get("task_fingerprint") != canonical(core(task)):
        raise ValueError(f"task contract mismatch: {task_path}")
    task_dir, netlist = rp(task["task_dir"], True), rp(task["netlist"]["path"], True)
    if (task_dir / "outcome.json").exists() or (task_dir / "runtime").exists():
        raise FileExistsError(f"task output exists: {task_dir}")
    if sf(netlist) != task["netlist"]["sha256"]:
        raise ValueError("netlist fingerprint mismatch")
    rhsold, state0 = task_inputs(task)
    runtime = task_dir / "runtime"
    runtime.mkdir()
    rhsold_path, state0_path, metrics_path = runtime / "rhsold_input.txt", runtime / "state0_input.txt", runtime / "newton_metrics.jsonl"
    np.savetxt(rhsold_path, rhsold, fmt="%.17e")
    np.savetxt(state0_path, state0, fmt="%.17e")
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("NGSPICE_GMRES_") or key.startswith("PALS_ONLINE_SCHWARZ_"):
            env.pop(key, None)
    env.update({str(key): str(value) for key, value in task["gmres_env"].items()})
    env.update({"TRAJ": "0", "VALUE": "0", "CKT_ID": str(task["circuit_id"]), "WP_IN_PATH": str(rhsold_path), "CONTINUATION_MODE": "1", "CONTINUATION_START_ITER": "0", "CONTINUATION_MAX_STEPS": str(task["max_newton_steps"]), "CONTINUATION_DIR": str(runtime), "CONTINUATION_GMIN": fk(task["gmin_val"]), "CONTINUATION_STATE0_PATH": str(state0_path), "NGSPICE_GMRES_METRICS_PATH": str(metrics_path), "PALS_NIITER_TRACE": "1"})
    start, timed_out, returncode, stdout, stderr = time.perf_counter(), False, None, "", ""
    try:
        completed = subprocess.run([str(NGSPICE_EXECUTABLE), "-b", str(netlist)], capture_output=True, text=True, check=False, timeout=task["timeout_sec"], env=env)
        returncode, stdout, stderr = int(completed.returncode), completed.stdout or "", completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    write(task_dir / "stdout.log", stdout)
    write(task_dir / "stderr.log", stderr)
    captured, metrics, errors = steps(runtime), rjsonl(metrics_path), []
    if timed_out:
        errors.append("timeout")
    elif returncode != 0:
        errors.append(f"ngspice_exit_nonzero:{returncode}")
    if not captured:
        errors.append("continuation_steps_missing")
    for index, step in enumerate(captured):
        if fk(step["time"]) != fk(task["time"]) or fk(step["gmin_val"]) != fk(task["gmin_val"]):
            errors.append(f"step_{index}_workpoint_mismatch")
        if step["rhsold"].size != rhsold.size or step["state0"].size != state0.size or not np.all(np.isfinite(step["rhsold"])) or not np.all(np.isfinite(step["state0"])):
            errors.append(f"step_{index}_frozen_state_invalid")
    rhsold_match = bool(captured) and sa(captured[0]["rhsold"]) == sa(rhsold)
    state0_match = bool(captured) and sa(captured[0]["state0"]) == sa(state0)
    if not rhsold_match:
        errors.append("captured_rhsold_not_equal_to_frozen_input")
    if not state0_match:
        errors.append("captured_state0_not_equal_to_frozen_input")
    final_residual = captured[-1]["residual_norm"] if captured else None
    if final_residual is None or not np.isfinite(final_residual) or final_residual > task["residual_tol"]:
        errors.append("final_residual_not_converged")
    errors.extend(metric_errors(metrics))
    outcome = {"schema_version": SCHEMA_VERSION, "task_id": task["task_id"], "task_fingerprint": task["task_fingerprint"], "stage_key": task["stage_key"], "arm": task["arm"], "warmup_enabled": task["warmup_enabled"], "circuit_id": task["circuit_id"], "time": task["time"], "gmin_val": task["gmin_val"], "experiment_scope": "fixed_gmin_frozen_state_restart_microbenchmark", "not_valid_for": "complete_adaptive_gmin_continuation_claim", "status": "ok" if not errors else "failed", "strict_success": not errors, "failure_classes": sorted(set(errors)), "elapsed_sec": time.perf_counter() - start, "timed_out": timed_out, "returncode": returncode, "captured_newton_steps": len(captured), "final_residual_norm": final_residual, "captured_rhsold_matches_frozen_input": rhsold_match, "captured_state0_matches_frozen_input": state0_match, "gmres_metrics_count": len(metrics), "gmres_total_iterations": sum(int(row.get("gmres_iterations") or 0) for row in metrics), "runtime_dir": str(runtime), "metrics_path": str(metrics_path), "steps": [{key: row[key] for key in ("iteration", "time", "gmin_val", "residual_norm", "path", "sha256")} for row in captured]}
    wjson(task_dir / "outcome.json", outcome)
    print(json.dumps(outcome, ensure_ascii=False))
    return 0 if outcome["strict_success"] else 1


def average(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def summarize(args):
    output = rp(args.output_dir, True)
    outcomes = [rjson(path) | {"outcome_path": str(path)} for path in sorted((output / "tasks").glob("*/*/outcome.json"))]
    by_stage = defaultdict(dict)
    for row in outcomes:
        by_stage[row["stage_key"]][row["arm"]] = row
    pairs = []
    for key, arms in sorted(by_stage.items()):
        cold, warm = arms.get("cold_row_sum"), arms.get("warm_row_sum")
        if cold is None or warm is None:
            pairs.append({"stage_key": key, "status": "missing_arm"})
        elif not cold["strict_success"] or not warm["strict_success"]:
            pairs.append({"stage_key": key, "status": "arm_not_strict_success", "cold_failure_classes": cold["failure_classes"], "warm_failure_classes": warm["failure_classes"]})
        elif (cold["circuit_id"], fk(cold["time"]), fk(cold["gmin_val"])) != (warm["circuit_id"], fk(warm["time"]), fk(warm["gmin_val"])):
            pairs.append({"stage_key": key, "status": "workpoint_mismatch"})
        else:
            pairs.append({"stage_key": key, "status": "strict_paired", "circuit_id": cold["circuit_id"], "time": cold["time"], "gmin_val": cold["gmin_val"], "cold_captured_newton_steps": cold["captured_newton_steps"], "warm_captured_newton_steps": warm["captured_newton_steps"], "saved_captured_newton_steps": cold["captured_newton_steps"] - warm["captured_newton_steps"], "cold_gmres_total_iterations": cold["gmres_total_iterations"], "warm_gmres_total_iterations": warm["gmres_total_iterations"], "saved_gmres_iterations": cold["gmres_total_iterations"] - warm["gmres_total_iterations"]})
    strict = [row for row in pairs if row["status"] == "strict_paired"]
    summary = {"schema_version": SCHEMA_VERSION, "experiment_scope": "fixed_gmin_frozen_state_restart_microbenchmark", "not_valid_for": "complete_adaptive_gmin_continuation_claim", "outcome_count": len(outcomes), "strict_task_count": sum(row["strict_success"] for row in outcomes), "task_failure_class_counts": dict(sorted(Counter(error for row in outcomes for error in row["failure_classes"]).items())), "pair_status_counts": dict(sorted(Counter(row["status"] for row in pairs).items())), "strict_pair_count": len(strict), "mean_saved_captured_newton_steps": average(row["saved_captured_newton_steps"] for row in strict), "mean_saved_gmres_iterations": average(row["saved_gmres_iterations"] for row in strict)}
    aggregate = output / "aggregate"
    write(aggregate / "per_task.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in outcomes))
    write(aggregate / "strict_pairs.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in pairs))
    wjson(aggregate / "summary.json", summary)
    write(aggregate / "report.md", "# 冻结状态固定 gmin 预热单阶段重启式核验\n\n此结果不能外推为完整自适应 gmin 续接结果。\n\n" + "\n".join(f"- {key}：{value}" for key, value in summary.items()) + "\n")
    print(f"summary={aggregate / 'summary.json'}\nstrict_pair_count={len(strict)}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--run-task", action="store_true")
    modes.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--task-json", default="")
    parser.add_argument("--trajectory-dir", default="")
    parser.add_argument("--warmup-input-root", default="")
    parser.add_argument("--netlist-dir", default="")
    parser.add_argument("--circuit-ids", default="")
    parser.add_argument("--max-workpoints-per-circuit", type=int, default=10)
    parser.add_argument("--max-newton-steps", type=int, default=100)
    parser.add_argument("--residual-tol", type=float, default=1e-8)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--gmres-restart", type=int, default=100)
    parser.add_argument("--gmres-max-iters", type=int, default=500)
    parser.add_argument("--gmres-rtol", type=float, default=1e-8)
    parser.add_argument("--gmres-atol", type=float, default=1e-10)
    parser.add_argument("--provenance-path", action="append", default=[])
    args = parser.parse_args()
    if args.prepare_only:
        missing = [key for key in ("output_dir", "trajectory_dir", "warmup_input_root", "netlist_dir", "circuit_ids") if not getattr(args, key)]
        if missing:
            parser.error("--prepare-only missing: " + ", ".join(missing))
        return prepare(args)
    if args.run_task:
        if not args.task_json:
            parser.error("--run-task requires --task-json")
        return run(args)
    if not args.output_dir:
        parser.error("--summarize-only requires --output-dir")
    return summarize(args)



if __name__ == "__main__":
    raise SystemExit(main())
