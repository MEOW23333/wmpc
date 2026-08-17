#!/usr/bin/env python3
"""Prepare, run, or summarize the four-arm warmup/Schwarz benchmark."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pypath.utils.ngspice_utils import NGSPICE_EXECUTABLE, run_ngspice_joint_case


SCHEMA_VERSION = 2
WARMUP_FILE_RE = re.compile(
    r"^segment_warmup_circuit_(\d+)_time_([0-9.eE+-]+)_gmin_([0-9.eE+-]+)_rhsold\.txt$"
)
RANGE_TOKEN_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
ARM_DEFINITIONS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "cold_row_sum",
        "warmup_enabled": False,
        "logical_preconditioner": "row_sum",
        "requires_native_schwarz": False,
    },
    {
        "name": "warm_row_sum",
        "warmup_enabled": True,
        "logical_preconditioner": "row_sum",
        "requires_native_schwarz": False,
    },
    {
        "name": "cold_learned_schwarz",
        "warmup_enabled": False,
        "logical_preconditioner": "learned_schwarz_v1_sparse",
        "requires_native_schwarz": True,
    },
    {
        "name": "warm_learned_schwarz",
        "warmup_enabled": True,
        "logical_preconditioner": "learned_schwarz_v1_sparse",
        "requires_native_schwarz": True,
    },
)
ONLINE_SIDECAR_MODE_ONESHOT = "oneshot_v1"
ONLINE_GMRES_ENV_RESERVED = frozenset(
    {
        "NGSPICE_GMRES_MODE",
        "NGSPICE_GMRES_PRECOND",
        "NGSPICE_GMRES_PRECOND_FALLBACK",
        "NGSPICE_GMRES_PRECOND_SIDECAR_SCOPE",
        "NGSPICE_GMRES_PRECOND_SIDECAR_PATH",
        "NGSPICE_GMRES_METRICS_PATH",
        "NGSPICE_GMRES_USE_RHSOLD_X0",
    }
)


def _resolve_online_common(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    mode = str(args.online_sidecar_mode or "").strip()
    if not mode:
        return None
    if mode != ONLINE_SIDECAR_MODE_ONESHOT:
        raise ValueError(f"unsupported online sidecar mode: {mode}")
    checkpoint = _repo_path(args.learned_schwarz_checkpoint, must_exist=True)
    generator = _repo_path(args.online_generator, must_exist=True)
    if not generator.is_file():
        raise FileNotFoundError(generator)
    if int(args.online_timeout_sec) < 1 or int(args.online_timeout_sec) > 300:
        raise ValueError("online timeout must lie in [1, 300] seconds")
    if not 2 <= int(args.online_min_block_size) <= int(args.online_max_block_size) <= 32:
        raise ValueError("online block sizes must lie in [2, 32]")
    if int(args.online_max_blocks) < 0:
        raise ValueError("online max blocks cannot be negative")
    return {
        "mode": mode,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "generator_path": str(generator),
        "generator_sha256": _sha256_file(generator),
        "timeout_sec": int(args.online_timeout_sec),
        "min_block_size": int(args.online_min_block_size),
        "max_block_size": int(args.online_max_block_size),
        "max_blocks": int(args.online_max_blocks),
        "retain_input": True,
    }


def _parse_circuit_ids(raw_value: str) -> List[int]:
    circuit_ids: List[int] = []
    seen = set()
    for raw_token in str(raw_value or "").split(","):
        token = raw_token.strip()
        if not token:
            continue
        match = RANGE_TOKEN_RE.match(token)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            step = 1 if start <= end else -1
            values = range(start, end + step, step)
        else:
            values = [int(token)]
        for value in values:
            if value not in seen:
                seen.add(value)
                circuit_ids.append(int(value))
    if not circuit_ids:
        raise ValueError("no circuit ids were selected")
    return circuit_ids


def _repo_path(raw_path: str, *, must_exist: bool = False) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError(f"path must stay inside PALS: {path}")
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    _atomic_write_text(path, text)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def _git_output(args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return (completed.stdout or "").strip()


def _git_snapshot() -> Dict[str, Any]:
    status = _git_output(["status", "--short"])
    diff = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    ).stdout
    return {
        "head": _git_output(["rev-parse", "HEAD"]),
        "branch": _git_output(["branch", "--show-current"]),
        "status_lines": status.splitlines() if status else [],
        "tracked_diff_sha256": hashlib.sha256(diff or b"").hexdigest(),
    }


def _scan_warmup_inputs(
    warmup_input_root: Optional[Path],
    circuit_ids: Sequence[int],
) -> Dict[int, List[Dict[str, Any]]]:
    selected = {int(value) for value in circuit_ids}
    by_circuit: Dict[int, Dict[Tuple[str, str], Dict[str, Any]]] = {
        circuit_id: {} for circuit_id in selected
    }
    if warmup_input_root is None:
        return {circuit_id: [] for circuit_id in selected}

    for path in sorted(warmup_input_root.rglob("segment_warmup_circuit_*_rhsold.txt")):
        match = WARMUP_FILE_RE.match(path.name)
        if not match:
            continue
        circuit_id = int(match.group(1))
        if circuit_id not in selected:
            continue
        time_val = float(match.group(2))
        gmin_val = float(match.group(3))
        time_key = f"{time_val:.17e}"
        gmin_key = f"{gmin_val:.17e}"
        entry = {
            "circuit_id": circuit_id,
            "time": time_val,
            "gmin_val": gmin_val,
            "time_key": time_key,
            "gmin_key": gmin_key,
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
        }
        key = (time_key, gmin_key)
        previous = by_circuit[circuit_id].get(key)
        if previous is not None and previous["sha256"] != entry["sha256"]:
            raise ValueError(
                "conflicting warmup inputs for "
                f"circuit={circuit_id}, time={time_key}, gmin={gmin_key}"
            )
        if previous is None or entry["path"] < previous["path"]:
            by_circuit[circuit_id][key] = entry

    return {
        circuit_id: [
            entries[key]
            for key in sorted(entries, key=lambda item: (float(item[0]), -float(item[1])))
        ]
        for circuit_id, entries in by_circuit.items()
    }


def _load_warmup_inputs(entries: Sequence[Dict[str, Any]]) -> Dict[Tuple[float, float], np.ndarray]:
    warmup_inputs: Dict[Tuple[float, float], np.ndarray] = {}
    for entry in entries:
        path = _repo_path(str(entry["path"]), must_exist=True)
        actual_sha256 = _sha256_file(path)
        expected_sha256 = str(entry.get("sha256") or "")
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ValueError(f"warmup input fingerprint mismatch: {path}")
        values = np.asarray(np.loadtxt(path, dtype=np.float64), dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid warmup vector: {path}")
        key = (float(entry["time"]), float(entry["gmin_val"]))
        if key in warmup_inputs:
            raise ValueError(f"duplicate warmup key: {key}")
        warmup_inputs[key] = values
    return warmup_inputs


def _load_env_json(path: str) -> Dict[str, str]:
    if not path:
        return {}
    payload = _read_json(_repo_path(path, must_exist=True))
    result: Dict[str, str] = {}
    for key, value in payload.items():
        name = str(key)
        if not name.startswith("NGSPICE_GMRES_"):
            raise ValueError(f"not a GMRES environment variable: {key}")
        if name in ONLINE_GMRES_ENV_RESERVED:
            raise ValueError(
                "the task builder owns this GMRES setting: " + name
            )
        result[name] = str(value)
    return result
def _base_gmres_env(args: argparse.Namespace, native_preconditioner: str) -> Dict[str, str]:
    return {
        "NGSPICE_GMRES_MODE": "1",
        "NGSPICE_GMRES_PRECOND": str(native_preconditioner),
        "NGSPICE_GMRES_PRECOND_FALLBACK": "row_sum",
        "NGSPICE_GMRES_RESTART": str(int(args.gmres_restart)),
        "NGSPICE_GMRES_MAX_ITERS": str(int(args.gmres_max_iters)),
        "NGSPICE_GMRES_RTOL": f"{float(args.rtol):.17e}",
        "NGSPICE_GMRES_ATOL": f"{float(args.atol):.17e}",
        "NGSPICE_GMRES_USE_RHSOLD_X0": "1" if args.use_rhsold_x0 else "0",
    }


def _task_fingerprint_payload(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": task["schema_version"],
        "circuit_id": task["circuit_id"],
        "arm": task["arm"],
        "warmup_enabled": task["warmup_enabled"],
        "logical_preconditioner": task["logical_preconditioner"],
        "native_preconditioner": task["native_preconditioner"],
        "requires_native_schwarz": task["requires_native_schwarz"],
        "requires_online_sidecar": task["requires_online_sidecar"],
        "netlist_sha256": task.get("netlist_sha256"),
        "warmup_entries": [
            {
                "time_key": entry["time_key"],
                "gmin_key": entry["gmin_key"],
                "sha256": entry["sha256"],
            }
            for entry in task.get("warmup_entries", [])
        ],
        "gmres_env": task["gmres_env"],
        "online_sidecar": task.get("online_sidecar"),
        "timeout_sec": task["timeout_sec"],
    }
def _build_tasks(
    args: argparse.Namespace,
    output_dir: Path,
    circuit_ids: Sequence[int],
    netlist_dir: Path,
    warmup_by_circuit: Dict[int, List[Dict[str, Any]]],
    *,
    online_common: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    schwarz_extra_env = _load_env_json(args.schwarz_gmres_env_json)
    tasks: List[Dict[str, Any]] = []
    for circuit_id in circuit_ids:
        netlist_path = netlist_dir / f"{int(circuit_id)}.sp"
        netlist_sha256 = _sha256_file(netlist_path) if netlist_path.exists() else None
        for arm in ARM_DEFINITIONS:
            blockers: List[str] = []
            if not netlist_path.exists():
                blockers.append("netlist_missing")
            warmup_entries = (
                list(warmup_by_circuit.get(int(circuit_id), []))
                if arm["warmup_enabled"]
                else []
            )
            if arm["warmup_enabled"] and not warmup_entries:
                blockers.append("warmup_inputs_missing")

            requires_native_schwarz = bool(arm["requires_native_schwarz"])
            if requires_native_schwarz:
                native_preconditioner = str(args.native_schwarz_precond or "").strip()
                if native_preconditioner != "learned_schwarz_v1_sparse":
                    blockers.append("native_schwarz_mode_must_be_learned_schwarz_v1_sparse")
                    native_preconditioner = native_preconditioner or "UNAVAILABLE"
                if online_common is None:
                    blockers.append("online_sidecar_mode_required")
            else:
                native_preconditioner = "row_sum"

            task_dir = output_dir / "tasks" / f"circuit_{int(circuit_id):06d}" / str(arm["name"])
            gmres_env = _base_gmres_env(args, native_preconditioner)
            if requires_native_schwarz:
                gmres_env.update(schwarz_extra_env)
            gmres_env["NGSPICE_GMRES_PRECOND"] = native_preconditioner
            gmres_env["NGSPICE_GMRES_METRICS_PATH"] = str(task_dir / "newton_metrics.jsonl")

            online_sidecar: Optional[Dict[str, Any]] = None
            if requires_native_schwarz and online_common is not None:
                online_root = task_dir / "online_sidecar"
                input_dir = online_root / "input"
                sidecar_dir = online_root / "sidecars"
                status_dir = online_root / "status"
                logs_dir = online_root / "logs"
                sidecar_template = sidecar_dir / "sidecar_iter_{iter}_time_{time}_gmin_{gmin}.json"
                event_log = logs_dir / "events.jsonl"
                gmres_env["NGSPICE_GMRES_PRECOND_SIDECAR_SCOPE"] = "per_step"
                gmres_env["NGSPICE_GMRES_PRECOND_SIDECAR_PATH"] = str(sidecar_template)
                online_env = {
                    "PALS_ONLINE_SCHWARZ_MODE": str(online_common["mode"]),
                    "PALS_ONLINE_SCHWARZ_REPO_ROOT": str(REPO_ROOT),
                    "PALS_ONLINE_SCHWARZ_GENERATOR": str(online_common["generator_path"]),
                    "PALS_ONLINE_SCHWARZ_CHECKPOINT": str(online_common["checkpoint_path"]),
                    "PALS_ONLINE_SCHWARZ_NETLIST": str(netlist_path),
                    "PALS_ONLINE_SCHWARZ_INPUT_DIR": str(input_dir),
                    "PALS_ONLINE_SCHWARZ_SIDECAR_DIR": str(sidecar_dir),
                    "PALS_ONLINE_SCHWARZ_STATUS_DIR": str(status_dir),
                    "PALS_ONLINE_SCHWARZ_EVENT_LOG": str(event_log),
                    "PALS_ONLINE_SCHWARZ_TIMEOUT_MS": str(
                        int(online_common["timeout_sec"]) * 1000
                    ),
                    "PALS_ONLINE_SCHWARZ_MIN_BLOCK_SIZE": str(
                        int(online_common["min_block_size"])
                    ),
                    "PALS_ONLINE_SCHWARZ_MAX_BLOCK_SIZE": str(
                        int(online_common["max_block_size"])
                    ),
                    "PALS_ONLINE_SCHWARZ_MAX_BLOCKS": str(
                        int(online_common["max_blocks"])
                    ),
                }
                online_sidecar = {
                    **online_common,
                    "root": str(online_root),
                    "input_dir": str(input_dir),
                    "sidecar_dir": str(sidecar_dir),
                    "status_dir": str(status_dir),
                    "logs_dir": str(logs_dir),
                    "event_log": str(event_log),
                    "sidecar_template": str(sidecar_template),
                    "env": online_env,
                }

            task: Dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "task_id": f"c{int(circuit_id)}__{arm['name']}",
                "circuit_id": int(circuit_id),
                "arm": str(arm["name"]),
                "warmup_enabled": bool(arm["warmup_enabled"]),
                "logical_preconditioner": str(arm["logical_preconditioner"]),
                "requires_native_schwarz": requires_native_schwarz,
                "requires_online_sidecar": bool(requires_native_schwarz),
                "native_preconditioner": native_preconditioner,
                "netlist_dir": str(netlist_dir),
                "netlist_path": str(netlist_path),
                "netlist_sha256": netlist_sha256,
                "warmup_entries": warmup_entries,
                "gmres_env": gmres_env,
                "online_sidecar": online_sidecar,
                "timeout_sec": int(args.timeout_sec),
                "task_dir": str(task_dir),
                "preflight_blockers": blockers,
                "runnable": not blockers,
            }
            task["task_fingerprint"] = _canonical_sha256(
                _task_fingerprint_payload(task)
            )
            tasks.append(task)
    return tasks
def prepare(args: argparse.Namespace) -> int:
    output_dir = _repo_path(args.output_dir)
    netlist_dir = _repo_path(args.netlist_dir, must_exist=True)
    warmup_root = (
        _repo_path(args.warmup_input_root, must_exist=True)
        if args.warmup_input_root
        else None
    )
    circuit_ids = _parse_circuit_ids(args.circuit_ids)
    warmup_by_circuit = _scan_warmup_inputs(warmup_root, circuit_ids)
    online_common = _resolve_online_common(args)
    tasks = _build_tasks(
        args,
        output_dir,
        circuit_ids,
        netlist_dir,
        warmup_by_circuit,
        online_common=online_common,
    )

    if (output_dir / "run_manifest.json").exists():
        raise FileExistsError(f"run manifest already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        task_dir = _repo_path(str(task["task_dir"]))
        task_dir.mkdir(parents=True, exist_ok=True)
        online_sidecar = task.get("online_sidecar")
        if isinstance(online_sidecar, dict):
            for key in ("input_dir", "sidecar_dir", "status_dir", "logs_dir"):
                path = _repo_path(str(online_sidecar[key]))
                if not path.is_relative_to(task_dir):
                    raise ValueError(f"online path must stay inside task: {path}")
                path.mkdir(parents=True, exist_ok=True)
            _write_json(
                _repo_path(str(online_sidecar["root"])) / "session_manifest.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "task_id": task["task_id"],
                    "task_fingerprint": task["task_fingerprint"],
                    "online_sidecar": online_sidecar,
                },
            )
        _write_json(task_dir / "task.json", task)

    warmup_manifest = {
        "schema_version": SCHEMA_VERSION,
        "warmup_input_root": str(warmup_root) if warmup_root else None,
        "circuits": {
            str(circuit_id): warmup_by_circuit.get(circuit_id, [])
            for circuit_id in circuit_ids
        },
    }
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "runner": "run_joint_warmup_schwarz_benchmark.py",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_dir": str(output_dir),
        "netlist_dir": str(netlist_dir),
        "circuit_ids": list(circuit_ids),
        "arms": [dict(item) for item in ARM_DEFINITIONS],
        "task_count": len(tasks),
        "runnable_task_count": sum(1 for task in tasks if task["runnable"]),
        "blocked_task_count": sum(1 for task in tasks if not task["runnable"]),
        "native_schwarz_precond": args.native_schwarz_precond or None,
        "ngspice_executable": str(NGSPICE_EXECUTABLE),
        "ngspice_executable_sha256": (
            _sha256_file(Path(NGSPICE_EXECUTABLE))
            if Path(NGSPICE_EXECUTABLE).exists()
            else None
        ),
        "runner_sha256": _sha256_file(Path(__file__)),
        "ngspice_utils_sha256": _sha256_file(REPO_ROOT / "pypath" / "utils" / "ngspice_utils.py"),
        "git": _git_snapshot(),
    }
    run_manifest["manifest_fingerprint"] = _canonical_sha256(run_manifest)

    _write_json(output_dir / "run_manifest.json", run_manifest)
    _write_json(output_dir / "warmup_manifest.json", warmup_manifest)
    _write_jsonl(output_dir / "planned_tasks.jsonl", tasks)
    print(f"run_manifest={output_dir / 'run_manifest.json'}")
    print(f"planned_tasks={output_dir / 'planned_tasks.jsonl'}")
    print(f"task_count={len(tasks)}")
    print(f"runnable_task_count={run_manifest['runnable_task_count']}")
    print(f"blocked_task_count={run_manifest['blocked_task_count']}")
    return 0


def _verify_native_method(
    task: Dict[str, Any],
    metrics_rows: Sequence[Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    if not metrics_rows:
        return False, "native_gmres_metrics_missing"
    expected = str(task["native_preconditioner"])
    requires_online = bool(task.get("requires_online_sidecar"))
    for row in metrics_rows:
        executed = row.get(
            "executed_preconditioner_mode",
            row.get("executed_preconditioner", row.get("precond")),
        )
        if executed is None:
            return False, "native_gmres_metrics_missing_executed_mode"
        if str(executed) != expected:
            return False, f"native_preconditioner_mismatch:{executed}"
        if requires_online:
            if not bool(row.get("online_sidecar_enabled")):
                return False, "online_sidecar_not_enabled"
            if not bool(row.get("online_sidecar_success")):
                return False, "online_sidecar_generation_failed"
        if bool(row.get("direct_fallback")) or str(
            row.get("event", "")
        ).startswith("fallback"):
            return False, "native_direct_fallback_used"
    return True, None
def _failure_class(native: Dict[str, Any], task: Dict[str, Any], injection_count: int) -> Optional[str]:
    if native.get("timed_out"):
        return "timeout"
    if not native.get("success"):
        reason = str(native.get("reason") or "")
        if reason.startswith("netlist_not_found"):
            return "netlist_missing"
        return "ngspice_exit_nonzero"
    if task.get("warmup_enabled") and injection_count <= 0:
        return "warmup_not_injected"
    return None


def run_task(args: argparse.Namespace) -> int:
    task_json = _repo_path(args.task_json, must_exist=True)
    task = _read_json(task_json)
    if int(task.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported task schema: {task.get('schema_version')}")
    expected_fingerprint = _canonical_sha256(_task_fingerprint_payload(task))
    if task.get("task_fingerprint") != expected_fingerprint:
        raise ValueError(f"task fingerprint mismatch: {task_json}")

    task_dir = _repo_path(str(task["task_dir"]))
    task_dir.mkdir(parents=True, exist_ok=True)
    blockers = list(task.get("preflight_blockers") or [])
    if blockers:
        outcome = {
            "schema_version": SCHEMA_VERSION,
            "task_id": task["task_id"],
            "circuit_id": task["circuit_id"],
            "arm": task["arm"],
            "status": "blocked",
            "strict_success": False,
            "failure_class": "preflight_blocked",
            "preflight_blockers": blockers,
            "task_fingerprint": task["task_fingerprint"],
        }
        _write_json(task_dir / "outcome.json", outcome)
        print(json.dumps(outcome, ensure_ascii=False))
        return 2

    netlist_path = _repo_path(str(task["netlist_path"]), must_exist=True)
    if _sha256_file(netlist_path) != task.get("netlist_sha256"):
        raise ValueError(f"netlist fingerprint mismatch: {netlist_path}")
    warmup_inputs = (
        _load_warmup_inputs(task.get("warmup_entries", []))
        if task.get("warmup_enabled")
        else {}
    )
    online_sidecar = task.get("online_sidecar")
    online_env: Dict[str, str] = {}
    online_event_path: Optional[Path] = None
    if bool(task.get("requires_online_sidecar")):
        if not isinstance(online_sidecar, dict):
            raise ValueError("learned Schwarz task is missing online sidecar config")
        for key in ("input_dir", "sidecar_dir", "status_dir", "logs_dir"):
            path = _repo_path(str(online_sidecar[key]), must_exist=True)
            if not path.is_relative_to(task_dir):
                raise ValueError(f"online path must stay inside task: {path}")
            if any(path.iterdir()):
                raise FileExistsError(f"refusing to reuse online outputs: {path}")
        online_event_path = _repo_path(str(online_sidecar["event_log"]))
        if not online_event_path.is_relative_to(task_dir):
            raise ValueError(
                f"online event path must stay inside task: {online_event_path}"
            )
        online_env = {
            str(key): str(value)
            for key, value in dict(online_sidecar.get("env") or {}).items()
        }
        if not online_env:
            raise ValueError("online sidecar environment is empty")
    _write_json(
        task_dir / "env.json",
        {
            "task_fingerprint": task["task_fingerprint"],
            "gmres_env": task["gmres_env"],
            "online_sidecar": online_sidecar,
            "online_env": online_env,
        },
    )

    metrics_path = _repo_path(str(task["gmres_env"]["NGSPICE_GMRES_METRICS_PATH"]))
    if not metrics_path.is_relative_to(task_dir):
        raise ValueError(f"metrics path must stay inside task directory: {metrics_path}")
    stale_outputs = [
        path
        for path in (
            metrics_path,
            task_dir / "outcome.json",
            task_dir / "native_result.json",
            task_dir / "segment_stage_stats.tsv",
            task_dir / "segment_stage_residuals.tsv",
            online_event_path,
        )
        if path is not None
        if path.exists()
    ]
    if stale_outputs:
        raise FileExistsError(
            "refusing to reuse task outputs: "
            + ", ".join(str(path) for path in stale_outputs)
        )

    native = run_ngspice_joint_case(
        val_dir=str(task_dir),
        netlist_dir=str(_repo_path(str(task["netlist_dir"]), must_exist=True)),
        real_ckt_id=int(task["circuit_id"]),
        warmup_inputs=warmup_inputs,
        case=str(task["task_id"]),
        gmres_env=dict(task["gmres_env"]),
        extra_env=online_env or None,
        timeout=int(task["timeout_sec"]),
        task_dir=str(task_dir),
        inherit_gmres_env=False,
    )
    _atomic_write_text(task_dir / "stdout.log", str(native.pop("stdout", "")))
    _atomic_write_text(task_dir / "stderr.log", str(native.pop("stderr", "")))

    stages = list(native.get("stages") or [])
    injection_count = sum(1 for stage in stages if stage.get("injected"))
    converged_stage_count = sum(1 for stage in stages if stage.get("converged"))
    captured_positive_gmin_newton_iters = sum(
        int(stage.get("iters") or 0)
        for stage in stages
    )
    failure_class = _failure_class(native, task, injection_count)
    metrics_parse_error = None
    try:
        metrics_rows = _read_jsonl(metrics_path) if metrics_path.exists() else []
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        metrics_rows = []
        metrics_parse_error = repr(exc)
        if failure_class is None:
            failure_class = "gmres_metrics_parse_failed"
    method_verified, method_verification_reason = _verify_native_method(
        task,
        metrics_rows,
    )
    if failure_class is None and not method_verified:
        failure_class = method_verification_reason or "native_method_verification_failed"
    strict_success = bool(failure_class is None and method_verified)
    outcome = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task["task_id"],
        "circuit_id": int(task["circuit_id"]),
        "arm": task["arm"],
        "status": "ok" if failure_class is None else "failed",
        "strict_success": strict_success,
        "method_verified": method_verified,
        "method_verification_reason": method_verification_reason,
        "failure_class": failure_class,
        "task_fingerprint": task["task_fingerprint"],
        "warmup_enabled": bool(task["warmup_enabled"]),
        "logical_preconditioner": task["logical_preconditioner"],
        "native_preconditioner": task["native_preconditioner"],
        "stage_count": len(stages),
        "converged_stage_count": converged_stage_count,
        "warmup_injection_count": injection_count,
        "captured_newton_scope": "positive_gmin_stage_stats_only",
        "captured_positive_gmin_newton_iters": captured_positive_gmin_newton_iters,
        "total_newton_iters": len(metrics_rows) if metrics_rows else None,
        "elapsed_sec": native.get("elapsed_sec"),
        "returncode": native.get("returncode"),
        "timed_out": native.get("timed_out"),
        "gmres_metrics_path": str(metrics_path),
        "gmres_metrics_count": len(metrics_rows),
        "gmres_metrics_parse_error": metrics_parse_error,
        "online_sidecar_enabled": bool(task.get("requires_online_sidecar")),
        "online_event_path": str(online_event_path) if online_event_path else None,
        "online_generation_success_count": sum(
            1 for row in metrics_rows if bool(row.get("online_sidecar_success"))
        ),
        "online_generation_failure_count": sum(
            1 for row in metrics_rows
            if bool(task.get("requires_online_sidecar"))
            and not bool(row.get("online_sidecar_success"))
        ),
        "online_generation_seconds_total": sum(
            float(row.get("online_sidecar_generation_seconds") or 0.0)
            for row in metrics_rows
        ),
        "online_generation_seconds_mean": (
            sum(float(row.get("online_sidecar_generation_seconds") or 0.0) for row in metrics_rows)
            / len(metrics_rows)
            if metrics_rows and bool(task.get("requires_online_sidecar"))
            else 0.0
        ),
        "total_gmres_iterations": sum(
            int(row.get("gmres_iterations", row.get("iter", 0)) or 0)
            for row in metrics_rows
        ),
        "native_result_path": str(task_dir / "native_result.json"),
    }
    _write_json(task_dir / "native_result.json", native)
    _write_json(task_dir / "outcome.json", outcome)
    print(json.dumps(outcome, ensure_ascii=False))
    return 0 if failure_class is None else 1


def _load_outcomes(output_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted((output_dir / "tasks").glob("circuit_*/*/outcome.json")):
        row = _read_json(path)
        row["outcome_path"] = str(path)
        rows.append(row)
    return rows


def _interaction_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_circuit: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        by_circuit.setdefault(int(row["circuit_id"]), {})[str(row["arm"])] = row
    required = {item["name"] for item in ARM_DEFINITIONS}
    interactions: List[Dict[str, Any]] = []
    for circuit_id, arm_rows in sorted(by_circuit.items()):
        if set(arm_rows) < required:
            continue
        if any(arm_rows[name].get("status") != "ok" for name in required):
            continue
        values = {
            name: arm_rows[name].get("total_newton_iters")
            for name in required
        }
        if any(value is None for value in values.values()):
            continue
        interaction = (
            float(values["warm_learned_schwarz"])
            - float(values["warm_row_sum"])
            - float(values["cold_learned_schwarz"])
            + float(values["cold_row_sum"])
        )
        interactions.append(
            {
                "circuit_id": circuit_id,
                "metric": "total_newton_iters",
                "interaction": interaction,
                "values": values,
            }
        )
    return interactions


def summarize(args: argparse.Namespace) -> int:
    output_dir = _repo_path(args.output_dir, must_exist=True)
    planned_path = output_dir / "planned_tasks.jsonl"
    planned_count = 0
    if planned_path.exists():
        planned_count = sum(
            1
            for line in planned_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    rows = _load_outcomes(output_dir)
    status_counts = Counter(str(row.get("status") or "missing") for row in rows)
    failure_counts = Counter(
        str(row["failure_class"])
        for row in rows
        if row.get("failure_class")
    )
    arm_summary: Dict[str, Dict[str, Any]] = {}
    for arm in [item["name"] for item in ARM_DEFINITIONS]:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        arm_summary[arm] = {
            "outcome_count": len(arm_rows),
            "ok_count": sum(1 for row in arm_rows if row.get("status") == "ok"),
            "strict_success_count": sum(1 for row in arm_rows if row.get("strict_success")),
            "blocked_count": sum(1 for row in arm_rows if row.get("status") == "blocked"),
            "failed_count": sum(1 for row in arm_rows if row.get("status") == "failed"),
            "total_newton_iters": sum(
                int(row.get("total_newton_iters") or 0)
                for row in arm_rows
                if row.get("status") == "ok"
            ),
        }
    interactions = _interaction_rows(rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "runner": "run_joint_warmup_schwarz_benchmark.py",
        "planned_task_count": planned_count,
        "outcome_count": len(rows),
        "missing_outcome_count": max(planned_count - len(rows), 0),
        "status_counts": dict(sorted(status_counts.items())),
        "failure_class_counts": dict(sorted(failure_counts.items())),
        "arm_summary": arm_summary,
        "newton_interaction_count": len(interactions),
    }
    aggregate_dir = output_dir / "aggregate"
    _write_jsonl(aggregate_dir / "per_task.jsonl", rows)
    _write_jsonl(aggregate_dir / "interaction.jsonl", interactions)
    _write_json(aggregate_dir / "summary.json", summary)
    print(f"summary={aggregate_dir / 'summary.json'}")
    print(f"outcome_count={len(rows)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run-task", action="store_true")
    mode.add_argument("--summarize-only", action="store_true")

    parser.add_argument("--output-dir", default="")
    parser.add_argument("--task-json", default="")
    parser.add_argument("--netlist-dir", default="")
    parser.add_argument("--warmup-input-root", default="")
    parser.add_argument("--circuit-ids", default="")
    parser.add_argument(
        "--native-schwarz-precond",
        default="",
        help="Leave empty until the native C implementation is available.",
    )
    parser.add_argument(
        "--schwarz-gmres-env-json",
        default="",
        help="Optional JSON object with non-path NGSPICE_GMRES_* overrides.",
    )
    parser.add_argument(
        "--online-sidecar-mode",
        choices=("", ONLINE_SIDECAR_MODE_ONESHOT),
        default="",
        help="Required for learned Schwarz arms.",
    )
    parser.add_argument("--learned-schwarz-checkpoint", default="")
    parser.add_argument(
        "--online-generator",
        default="pypath/utils/generate_online_learned_schwarz_sidecar.py",
    )
    parser.add_argument("--online-timeout-sec", type=int, default=60)
    parser.add_argument("--online-min-block-size", type=int, default=2)
    parser.add_argument("--online-max-block-size", type=int, default=32)
    parser.add_argument("--online-max-blocks", type=int, default=0)
    parser.add_argument("--timeout-sec", type=int, default=10000)
    parser.add_argument("--gmres-restart", type=int, default=100)
    parser.add_argument("--gmres-max-iters", type=int, default=500)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--use-rhsold-x0", action="store_true", default=True)
    parser.add_argument(
        "--zero-x0",
        dest="use_rhsold_x0",
        action="store_false",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.prepare_only:
        if not args.output_dir or not args.netlist_dir or not args.circuit_ids:
            parser.error("--prepare-only requires --output-dir, --netlist-dir, and --circuit-ids")
        return prepare(args)
    if args.run_task:
        if not args.task_json:
            parser.error("--run-task requires --task-json")
        return run_task(args)
    if not args.output_dir:
        parser.error("--summarize-only requires --output-dir")
    return summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
