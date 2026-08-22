#!/usr/bin/env python3
"""生成 WMPC 的最小可重复数据集。

流程：生成无工艺库依赖的二极管/电阻电路，调用带 WMPC 导出的原生程序，
把线性系统导出转换成统一的 continuation 轨迹、Jacobian、warmup 向量和
带哈希的工作点清单。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pypath.utils.ngspice_utils import (
    read_linear_system_corpus_step,
    run_ngspice_linear_system_corpus,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError(f"path must stay inside WMPC: {path}")
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def netlist_text(circuit_id: int, node_count: int, seed: int, tran_step: float, tran_stop: float) -> str:
    rng = np.random.default_rng(int(seed) + int(circuit_id) * 10007)
    lines = [
        f"* WMPC reproducible circuit {circuit_id}",
        ".title WMPC_REPRODUCIBLE_CIRCUIT",
        ".options gmin=1e-12 reltol=1e-5 abstol=1e-12",
        "VDD vdd 0 1.0",
        "VIN vin 0 PULSE(0 1 0 1e-12 1e-12 2e-12 4e-12)",
        ".model DWMPC D(Is=1e-14 N=1.2 Rs=2)",
    ]
    previous = "vin"
    for index in range(int(node_count)):
        node = f"n{index}"
        resistance = float(rng.uniform(100.0, 1500.0))
        capacitance = float(rng.uniform(0.05e-12, 0.5e-12))
        lines.append(f"R{index} {previous} {node} {resistance:.12g}")
        lines.append(f"C{index} {node} 0 {capacitance:.12g}")
        lines.append(f"D{index} {node} 0 DWMPC")
        previous = node
    lines.extend(
        [
            f"RLOAD {previous} 0 {float(rng.uniform(500.0, 5000.0)):.12g}",
            f".tran {float(tran_step):.17e} {float(tran_stop):.17e}",
            ".control",
            "run",
            "quit",
            ".endc",
            ".end",
            "",
        ]
    )
    return "\n".join(lines)


def vector_text(values: Iterable[float]) -> str:
    return "".join(f"{float(value):.17e}\n" for value in values)


def continuation_text(payload: Dict[str, Any]) -> str:
    def section(name: str, values: Iterable[float]) -> List[str]:
        return [name, *[f"{float(value):.17e}" for value in values]]

    lines: List[str] = []
    lines += section("OLD", payload.get("rhsold", []))
    lines += section("STATE0_IN", payload.get("state0", []))
    lines += section("NEW", payload.get("rhs", []))
    lines += section("STATE0_OUT", payload.get("state0", []))
    lines += section("RES", payload.get("raw_residual", []))
    lines += section("WP_OUT", payload.get("rhs", []))
    lines.append("NODE_MAP")
    for name, index in sorted(
        (payload.get("node_map") or {}).items(),
        key=lambda item: int(item[1]),
    ):
        lines.append(f"{name} {int(index)}")
    lines.append("")
    return "\n".join(lines)


def convert_step(
    *,
    source_path: Path,
    source_jacobian: Path,
    trajectory_dir: Path,
    warmup_dir: Path,
    circuit_id: int,
    time_value: float,
    gmin_value: float,
    iteration: int,
) -> Dict[str, Any]:
    payload = read_linear_system_corpus_step(str(source_path))
    rhsold = np.asarray(payload.get("rhsold", []), dtype=np.float64)
    rhs = np.asarray(payload.get("rhs", []), dtype=np.float64)
    if rhsold.size == 0 or rhs.size == 0:
        raise ValueError(f"empty rhs/rhsold in {source_path}")
    if rhsold.size != rhs.size:
        raise ValueError(f"rhs dimension mismatch in {source_path}")
    filename = (
        f"circuit_{circuit_id}_time_{time_value:.17e}_gmin_{gmin_value:.17e}"
        f"_iter_{iteration:03d}.txt"
    )
    trajectory_path = trajectory_dir / filename
    jacobian_path = trajectory_dir / (filename[:-4] + "_jac.txt")
    warmup_path = warmup_dir / (
        f"segment_warmup_circuit_{circuit_id}_time_{time_value:.17e}"
        f"_gmin_{gmin_value:.17e}_rhsold.txt"
    )
    trajectory_path.write_text(continuation_text(payload), encoding="utf-8")
    shutil.copyfile(source_jacobian, jacobian_path)
    warmup_path.write_text(vector_text(rhsold), encoding="utf-8")
    return {
        "circuit_id": int(circuit_id),
        "time": float(time_value),
        "gmin_val": float(gmin_value),
        "iteration": int(iteration),
        "step_sha256": sha256(trajectory_path),
        "jacobian_sha256": sha256(jacobian_path),
        "warmup_sha256": sha256(warmup_path),
        "netlist_sha256": None,
        "trajectory_path": str(trajectory_path),
        "jacobian_path": str(jacobian_path),
        "warmup_path": str(warmup_path),
        "matrix_size": int(rhs.size),
        "rhsold_norm": float(np.linalg.norm(rhsold)),
        "rhs_norm": float(np.linalg.norm(rhs)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-circuits", type=int, default=1)
    parser.add_argument("--node-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--tran-step", type=float, default=1e-12)
    parser.add_argument("--tran-stop", type=float, default=8e-12)
    args = parser.parse_args()

    if int(args.num_circuits) <= 0 or int(args.node_count) < 2:
        raise ValueError("num-circuits must be positive and node-count must be at least 2")
    if float(args.tran_step) <= 0 or float(args.tran_stop) <= 0 or float(args.tran_stop) < float(args.tran_step):
        raise ValueError("tran-step and tran-stop must be positive, with tran-stop >= tran-step")
    root = repo_path(args.output_dir)
    netlists = root / "generated_netlists"
    runtime = root / "runtime"
    trajectory = root / "trajectory"
    warmup = root / "warmup"
    failed = root / "failed_netlists"
    for path in (netlists, runtime, trajectory, warmup, failed):
        path.mkdir(parents=True, exist_ok=True)

    circuits: List[Dict[str, Any]] = []
    workpoints: List[Dict[str, Any]] = []
    for circuit_id in range(int(args.num_circuits)):
        netlist_path = netlists / f"{circuit_id}.sp"
        netlist_path.write_text(
            netlist_text(circuit_id, int(args.node_count), int(args.seed), float(args.tran_step), float(args.tran_stop)),
            encoding="utf-8",
        )
        record: Dict[str, Any] = {
            "circuit_id": int(circuit_id),
            "netlist_path": str(netlist_path),
            "netlist_sha256": sha256(netlist_path),
            "status": "generated",
            "step_count": 0,
            "reason": None,
        }
        circuits.append(record)
        if args.skip_native:
            continue
        result = run_ngspice_linear_system_corpus(
            val_dir=str(runtime),
            netlist_dir=str(netlists),
            real_ckt_id=int(circuit_id),
            case=f"wmpc_reproducible_c{circuit_id}",
            timeout=int(args.timeout_sec),
        )
        if not result.get("steps"):
            record["status"] = "failed"
            record["reason"] = result.get("reason") or "linear_system_export_no_steps"
            continue
        for step in result["steps"]:
            converted = convert_step(
                source_path=Path(str(step["filepath"])),
                source_jacobian=Path(str(step["jacobian_filepath"])),
                trajectory_dir=trajectory,
                warmup_dir=warmup,
                circuit_id=int(circuit_id),
                time_value=float(step["time"]),
                gmin_value=float(step["gmin_val"]),
                iteration=int(step["iteration"]),
            )
            converted["netlist_sha256"] = record["netlist_sha256"]
            workpoints.append(converted)
        record["step_count"] = len(result["steps"])
        record["status"] = "ok" if result.get("success") else "partial"
        record["reason"] = result.get("reason")

    manifest_rows = []
    for index, item in enumerate(workpoints):
        manifest_rows.append(
            {
                "circuit_id": item["circuit_id"],
                "time": item["time"],
                "gmin_val": item["gmin_val"],
                "iteration": item["iteration"],
                "step_sha256": item["step_sha256"],
                "jacobian_sha256": item["jacobian_sha256"],
                "netlist_sha256": item["netlist_sha256"],
                "manifest_index": index,
            }
        )
    write_json(
        root / "workpoint_manifest.json",
        {"schema_version": 1, "workpoints": manifest_rows},
    )
    write_json(
        root / "generation_summary.json",
        {
            "schema_version": 1,
            "seed": int(args.seed),
            "num_circuits": int(args.num_circuits),
            "node_count": int(args.node_count),
            "tran_step": float(args.tran_step),
            "tran_stop": float(args.tran_stop),
            "native_skipped": bool(args.skip_native),
            "circuits": circuits,
            "workpoint_count": len(workpoints),
            "trajectory_count": len([p for p in trajectory.glob("circuit_*.txt") if not p.name.endswith("_jac.txt")]),
            "jacobian_count": len(list(trajectory.glob("circuit_*_jac.txt"))),
            "warmup_count": len(list(warmup.glob("segment_warmup_*.txt"))),
            "warmup_unique_stage_count": len({(item["circuit_id"], item["time"], item["gmin_val"]) for item in workpoints}),
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(root),
                "circuit_count": len(circuits),
                "workpoint_count": len(workpoints),
                "native_skipped": bool(args.skip_native),
            },
            ensure_ascii=False,
        )
    )
    return 0 if args.skip_native or workpoints else 1


if __name__ == "__main__":
    raise SystemExit(main())
