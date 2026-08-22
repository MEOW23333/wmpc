#!/usr/bin/env python3
"""校验 WMPC 最小可复现数据集的文件、维数、哈希和工作点契约。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pypath.utils.ngspice_utils import read_J, read_continuation_step

TRAJ_RE = re.compile(r"^circuit_(\d+)_time_([0-9.eE+-]+)_gmin_([0-9.eE+-]+)_iter_(\d+)\.txt$")
WARM_RE = re.compile(r"^segment_warmup_circuit_(\d+)_time_([0-9.eE+-]+)_gmin_([0-9.eE+-]+)_rhsold\.txt$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_key(circuit_id: int, time_value: float, gmin_value: float) -> tuple[int, str, str]:
    return int(circuit_id), f"{float(time_value):.17e}", f"{float(gmin_value):.17e}"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"不是对象：{path}")
    return payload


def validate(root: Path, require_native: bool) -> dict[str, Any]:
    root = root.resolve()
    summary_path = root / "generation_summary.json"
    manifest_path = root / "workpoint_manifest.json"
    netlist_dir = root / "generated_netlists"
    trajectory_dir = root / "trajectory"
    warmup_dir = root / "warmup"
    for path in (summary_path, manifest_path, netlist_dir, trajectory_dir, warmup_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    summary = load_json(summary_path)
    manifest = load_json(manifest_path)
    circuits = list(summary.get("circuits") or [])
    workpoints = list(manifest.get("workpoints") or [])
    netlists = sorted(netlist_dir.glob("*.sp"))
    trajectory_paths = []
    jacobian_paths = []
    for path in sorted(trajectory_dir.glob("circuit_*.txt")):
        if path.name.endswith("_jac.txt"):
            jacobian_paths.append(path)
        else:
            trajectory_paths.append(path)
    warmup_paths = sorted(warmup_dir.glob("segment_warmup_*.txt"))

    errors: list[str] = []
    expected_circuits = int(summary.get("num_circuits", len(circuits)))
    if len(circuits) != expected_circuits:
        errors.append(f"circuits_count:{len(circuits)}!={expected_circuits}")
    if len(netlists) != expected_circuits:
        errors.append(f"netlist_count:{len(netlists)}!={expected_circuits}")
    if len(trajectory_paths) != len(jacobian_paths):
        errors.append(f"trajectory_jacobian_count:{len(trajectory_paths)}!={len(jacobian_paths)}")
    if len(trajectory_paths) != len(workpoints):
        errors.append(f"trajectory_manifest_count:{len(trajectory_paths)}!={len(workpoints)}")

    manifest_by_path = {str(row.get("step_sha256")): row for row in workpoints}
    seen_workpoints: set[tuple[int, str, str, int]] = set()
    dims: set[int] = set()
    for trajectory_path in trajectory_paths:
        match = TRAJ_RE.match(trajectory_path.name)
        if not match:
            errors.append(f"trajectory_name_invalid:{trajectory_path.name}")
            continue
        circuit_id = int(match.group(1))
        time_value = float(match.group(2))
        gmin_value = float(match.group(3))
        iteration = int(match.group(4))
        key = (circuit_id, f"{time_value:.17e}", f"{gmin_value:.17e}", iteration)
        if key in seen_workpoints:
            errors.append(f"duplicate_workpoint:{trajectory_path.name}")
        seen_workpoints.add(key)
        payload = read_continuation_step(str(trajectory_path))
        rhsold = np.asarray(payload.get("rhsold", []), dtype=np.float64).reshape(-1)
        rhsnew = np.asarray(payload.get("rhsnew", []), dtype=np.float64).reshape(-1)
        if rhsold.size == 0 or rhsnew.size == 0 or rhsold.size != rhsnew.size:
            errors.append(f"trajectory_vector_contract:{trajectory_path.name}")
            continue
        if not np.all(np.isfinite(rhsold)) or not np.all(np.isfinite(rhsnew)):
            errors.append(f"trajectory_nonfinite:{trajectory_path.name}")
        dims.add(int(rhsold.size))
        jacobian_path = trajectory_dir / (trajectory_path.stem + "_jac.txt")
        if not jacobian_path.exists():
            errors.append(f"jacobian_missing:{trajectory_path.name}")
        else:
            matrix = read_J(str(jacobian_path))
            if matrix.shape != (rhsold.size, rhsold.size):
                errors.append(f"jacobian_shape:{jacobian_path.name}:{matrix.shape}!={(rhsold.size, rhsold.size)}")
        step_hash = sha256(trajectory_path)
        row = manifest_by_path.get(step_hash)
        if row is None:
            errors.append(f"manifest_step_hash_missing:{trajectory_path.name}")
        elif key != (
            int(row.get("circuit_id", -1)),
            f"{float(row.get('time', float('nan'))):.17e}",
            f"{float(row.get('gmin_val', float('nan'))):.17e}",
            int(row.get("iteration", -1)),
        ):
            errors.append(f"manifest_workpoint_mismatch:{trajectory_path.name}")

    warmup_keys: set[tuple[int, str, str]] = set()
    warmup_dims: set[int] = set()
    for warmup_path in warmup_paths:
        match = WARM_RE.match(warmup_path.name)
        if not match:
            errors.append(f"warmup_name_invalid:{warmup_path.name}")
            continue
        key = canonical_key(int(match.group(1)), float(match.group(2)), float(match.group(3)))
        if key in warmup_keys:
            errors.append(f"duplicate_warmup_stage:{warmup_path.name}")
        warmup_keys.add(key)
        values = np.asarray(np.loadtxt(warmup_path, dtype=np.float64), dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            errors.append(f"warmup_vector_invalid:{warmup_path.name}")
        warmup_dims.add(int(values.size))

    if len(dims) > 1:
        errors.append(f"mixed_matrix_dimensions:{sorted(dims)}")
    if warmup_dims and dims and not warmup_dims.issubset(dims):
        errors.append(f"warmup_dimensions_not_in_trajectory:{sorted(warmup_dims)}:{sorted(dims)}")
    if require_native and not trajectory_paths:
        errors.append("native_data_missing")

    return {
        "schema_version": 1,
        "root": str(root),
        "valid": not errors,
        "errors": errors,
        "circuit_count": len(circuits),
        "netlist_count": len(netlists),
        "trajectory_count": len(trajectory_paths),
        "jacobian_count": len(jacobian_paths),
        "warmup_count": len(warmup_paths),
        "warmup_unique_stage_count": len(warmup_keys),
        "matrix_dimensions": sorted(dims),
        "manifest_workpoint_count": len(workpoints),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--require-native", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    result = validate(Path(args.root).expanduser(), bool(args.require_native))
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(text, end="")
    if args.json_out:
        path = Path(args.json_out).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
