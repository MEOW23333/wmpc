"""为接口低秩基生成同电路、因果历史误差快照。

该工具只读取测试工作点之前的同电路较小 gmin 工作点。每个历史点用
全阶直接解得到参考解，再用固定的局部稀疏舒尔预条件子做若干次阻尼
预条件残差迭代，保存接口误差列。结果只作为离线 POD 基训练数据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pypath.precondition_construction.sparse import (  # noqa: E402
    SparseLocalSchurPreconditioner,
    SparseSemanticBlockJacobi,
    _extract_semantic_blocks_sparse,
)
from pypath.utils.ngspice_utils import read_J_sparse, read_continuation_step  # noqa: E402
from pypath.utils.sparse_gmres_prototype import (  # noqa: E402
    _find_steps,
    _workpoint_key,
)


def _resolve_netlist(root: Path, circuit_id: int) -> str:
    candidates = [
        root / f"{int(circuit_id)}.sp",
        root / f"circuit_{int(circuit_id)}.sp",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    matches = sorted(root.glob("*.sp"))
    if int(circuit_id) == 0 and len(matches) == 1:
        return str(matches[0])
    raise FileNotFoundError(f"netlist_not_found:{circuit_id}")


def _matrix_and_payload(step: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    matrix = read_J_sparse(step["jacobian_path"], matrix_format="csr")
    if np.iscomplexobj(matrix.data):
        imag = float(np.max(np.abs(matrix.data.imag))) if matrix.nnz else 0.0
        real = float(np.max(np.abs(matrix.data.real))) if matrix.nnz else 0.0
        if imag > 1.0e-12 * max(real, 1.0):
            raise ValueError("history_jacobian_has_non_negligible_imaginary_part")
        matrix = matrix.real
    matrix = matrix.astype(np.float64).tocsr()
    gmin = float(step.get("gmin_val", 0.0))
    if gmin != 0.0:
        matrix = matrix + gmin * sp.eye(
            matrix.shape[0], dtype=np.float64, format="csr"
        )
    payload = read_continuation_step(step["step_path"])
    return matrix, payload


def _build_local_schur(
    matrix: Any,
    payload: Dict[str, Any],
    netlist_path: str,
    *,
    edge_budget: int,
    factor_drop_tol: float,
    factor_fill_factor: float,
) -> Tuple[SparseLocalSchurPreconditioner, np.ndarray]:
    node_map = payload.get("node_map")
    if not isinstance(node_map, dict) or not node_map:
        raise ValueError("history_payload_missing_node_map")
    core_blocks, _ = _extract_semantic_blocks_sparse(
        matrix=matrix,
        node_map=node_map,
        netlist_path=netlist_path,
        semantic_mode="cell_core",
        max_block_size=96,
        min_block_size=2,
        max_blocks=0,
    )
    boundary_blocks, _ = _extract_semantic_blocks_sparse(
        matrix=matrix,
        node_map=node_map,
        netlist_path=netlist_path,
        semantic_mode="cell_core_plus_onehop_boundary",
        max_block_size=192,
        min_block_size=2,
        max_blocks=0,
    )
    core = SparseSemanticBlockJacobi(matrix, core_blocks, uncovered_policy="row_sum")
    preconditioner = SparseLocalSchurPreconditioner(
        matrix=matrix,
        core=core,
        boundary_blocks=boundary_blocks,
        strategy="topk_abs",
        edge_budget=int(edge_budget),
        budget_multiplier=2.0,
        candidate_edge_limit=16384,
        diagonal_shift=1.0e-8,
        factor_drop_tol=float(factor_drop_tol),
        factor_fill_factor=float(factor_fill_factor),
        interface_solve_mode="spilu",
        probe_rhs=None,
        probe_x0=None,
        node_map=node_map,
        max_schur_nnz=0,
        max_degree=0,
        max_exact_entries=0,
    )
    return preconditioner, np.asarray(preconditioner.interface_rows, dtype=np.int64)


def _one_history_snapshots(
    step: Dict[str, Any],
    *,
    netlist_path: str,
    expected_interface_rows: np.ndarray,
    edge_budget: int,
    iterations: int,
    damping: float,
    factor_drop_tol: float,
    factor_fill_factor: float,
    snapshot_kind: str = "iterative_error",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    matrix, payload = _matrix_and_payload(step)
    rhs = np.asarray(payload.get("rhsnew", []), dtype=np.float64)
    rhsold = np.asarray(payload.get("rhsold", []), dtype=np.float64)
    if rhs.shape[0] != matrix.shape[0]:
        raise ValueError("history_rhs_dimension_mismatch")
    preconditioner, interface_rows = _build_local_schur(
        matrix,
        payload,
        netlist_path,
        edge_budget=edge_budget,
        factor_drop_tol=factor_drop_tol,
        factor_fill_factor=factor_fill_factor,
    )
    if not np.array_equal(interface_rows, expected_interface_rows):
        raise ValueError("history_interface_rows_not_aligned")
    x = (
        rhsold.copy()
        if rhsold.shape[0] == rhs.shape[0]
        else np.zeros_like(rhs, dtype=np.float64)
    )
    initial_residual = rhs - matrix.dot(x)
    direct_rhs = (
        initial_residual
        if snapshot_kind == "post_schwarz_error"
        else rhs
    )
    try:
        reference = np.asarray(
            splu(matrix.tocsc()).solve(direct_rhs),
            dtype=np.float64,
        )
        direct_mode = "splu"
    except Exception as exc:
        if snapshot_kind == "post_schwarz_error":
            raise RuntimeError(
                "post_schwarz_error_requires_sparse_direct_solve"
            ) from exc
        reference = np.asarray(
            np.linalg.lstsq(matrix.toarray(), rhs, rcond=None)[0]
        )
        direct_mode = "dense_lstsq"
    columns: List[np.ndarray] = []
    residual_norms: List[float] = []
    def real_correction(residual: np.ndarray) -> np.ndarray:
        correction = np.asarray(preconditioner.apply(residual))
        if np.iscomplexobj(correction):
            imag = float(np.max(np.abs(correction.imag))) if correction.size else 0.0
            real = float(np.max(np.abs(correction.real))) if correction.size else 0.0
            if imag > 1.0e-10 * max(real, 1.0):
                raise ValueError("history_preconditioner_complex_output")
            correction = correction.real
        correction = np.asarray(correction, dtype=np.float64)
        if not np.all(np.isfinite(correction)):
            raise ValueError("history_preconditioner_nonfinite_output")
        return correction
    if snapshot_kind == "post_schwarz_error":
        residual_norms.append(float(np.linalg.norm(initial_residual)))
        base_correction = real_correction(initial_residual)
        error = reference[interface_rows] - base_correction[interface_rows]
        if np.all(np.isfinite(error)) and float(np.linalg.norm(error)) > 0.0:
            columns.append(np.asarray(error, dtype=np.float64))
    elif snapshot_kind == "iterative_error":
        for _ in range(max(int(iterations), 1)):
            error = reference[interface_rows] - x[interface_rows]
            if np.all(np.isfinite(error)) and float(np.linalg.norm(error)) > 0.0:
                columns.append(np.asarray(error, dtype=np.float64))
            residual = rhs - matrix.dot(x)
            residual_norms.append(float(np.linalg.norm(residual)))
            x = x + float(damping) * real_correction(residual)
            if not np.all(np.isfinite(x)) or float(np.linalg.norm(x)) > 1.0e100:
                break
    else:
        raise ValueError(f"unsupported_snapshot_kind:{snapshot_kind}")
    snapshots = (
        np.column_stack(columns)
        if columns
        else np.zeros((expected_interface_rows.shape[0], 0), dtype=np.float64)
    )
    if snapshots.shape[1]:
        norms = np.linalg.norm(snapshots, axis=0)
        keep = np.isfinite(norms) & (norms > 1.0e-30)
        snapshots = snapshots[:, keep] / norms[keep]
    return snapshots, {
        "step_path": str(step["step_path"]),
        "snapshot_kind": snapshot_kind,
        "gmin_val": float(step.get("gmin_val", 0.0)),
        "iteration": int(step.get("iteration", -1)),
        "direct_solver": direct_mode,
        "snapshot_count": int(snapshots.shape[1]),
        "snapshot_columns_normalized": True,
        "residual_norms": residual_norms,
    }


def _target_for_circuit(
    steps: Sequence[Dict[str, Any]],
    workpoint: Dict[str, Any],
) -> Dict[str, Any]:
    target_key = _workpoint_key(
        circuit_id=int(workpoint["circuit_id"]),
        time_value=float(workpoint["time"]),
        gmin_value=float(workpoint["gmin_val"]),
        iteration=int(workpoint["iteration"]),
    )
    for step in steps:
        key = _workpoint_key(
            circuit_id=int(step["circuit_id"]),
            time_value=float(step["time"]),
            gmin_value=float(step["gmin_val"]),
            iteration=int(step["iteration"]),
        )
        if key == target_key:
            return step
    raise ValueError(f"target_step_not_found:{target_key}")


def _target_interface(
    step: Dict[str, Any],
    *,
    netlist_path: str,
) -> np.ndarray:
    matrix, payload = _matrix_and_payload(step)
    _, rows = _build_local_schur(
        matrix,
        payload,
        netlist_path,
        edge_budget=4096,
        factor_drop_tol=1.0e-4,
        factor_fill_factor=10.0,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate causal same-circuit interface error snapshots."
    )
    parser.add_argument("--trajectory-dir", required=True)
    parser.add_argument("--netlist-dir", required=True)
    parser.add_argument("--workpoint-manifest", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--history-limit", type=int, default=8)
    parser.add_argument("--iterations-per-history", type=int, default=8)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--edge-budget", type=int, default=4096)
    parser.add_argument("--factor-drop-tol", type=float, default=1.0e-4)
    parser.add_argument("--factor-fill-factor", type=float, default=10.0)
    parser.add_argument(
        "--snapshot-kind",
        choices=["iterative_error", "post_schwarz_error"],
        default="iterative_error",
    )
    args = parser.parse_args()

    trajectory_dir = str(Path(args.trajectory_dir).resolve())
    netlist_dir = Path(args.netlist_dir).resolve()
    manifest_path = str(Path(args.workpoint_manifest).resolve())
    manifest_payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    arrays: Dict[str, np.ndarray] = {}
    report: Dict[str, Any] = {
        "schema_version": 1,
        "generator": "generate_interface_error_snapshots.py",
        "trajectory_dir": trajectory_dir,
        "netlist_dir": str(netlist_dir),
        "workpoint_manifest": manifest_path,
        "history_limit": int(args.history_limit),
        "iterations_per_history": int(args.iterations_per_history),
        "damping": float(args.damping),
        "snapshot_kind": str(args.snapshot_kind),
        "circuits": {},
    }
    for workpoint in manifest_payload["workpoints"]:
        cid = int(workpoint["circuit_id"])
        if str(cid) in report["circuits"]:
            continue
        netlist_path = _resolve_netlist(netlist_dir, cid)
        all_steps = _find_steps(
            trajectory_dir,
            cid,
            1_000_000,
            requested_workpoints=None,
        )
        target = _target_for_circuit(all_steps, workpoint)
        interface_rows = _target_interface(target, netlist_path=netlist_path)
        history = [
            step
            for step in all_steps
            if float(step["time"]) == float(target["time"])
            and int(step["iteration"]) == int(target["iteration"])
            and float(step["gmin_val"]) < float(target["gmin_val"]) - 1.0e-15
        ]
        history.sort(key=lambda item: float(item["gmin_val"]), reverse=True)
        history = history[: max(int(args.history_limit), 0)]
        columns: List[np.ndarray] = []
        source_records: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        for history_step in history:
            try:
                snapshot, source = _one_history_snapshots(
                    history_step,
                    netlist_path=netlist_path,
                    expected_interface_rows=interface_rows,
                    edge_budget=int(args.edge_budget),
                    iterations=int(args.iterations_per_history),
                    damping=float(args.damping),
                    factor_drop_tol=float(args.factor_drop_tol),
                    factor_fill_factor=float(args.factor_fill_factor),
                    snapshot_kind=str(args.snapshot_kind),
                )
                if snapshot.shape[1]:
                    columns.extend(
                        [snapshot[:, idx] for idx in range(snapshot.shape[1])]
                    )
                source_records.append(source)
            except Exception as exc:
                skipped.append(
                    {"step_path": str(history_step["step_path"]), "reason": repr(exc)}
                )
        merged = (
            np.column_stack(columns)
            if columns
            else np.zeros((interface_rows.shape[0], 0), dtype=np.float64)
        )
        arrays[f"circuit_{cid}"] = merged
        report["circuits"][str(cid)] = {
            "target_step_path": str(target["step_path"]),
            "target_gmin_val": float(target["gmin_val"]),
            "interface_count": int(interface_rows.size),
            "history_candidate_count": int(len(history)),
            "history_used_count": int(len(source_records)),
            "snapshot_count": int(merged.shape[1]),
            "source_records": source_records,
            "skipped_history": skipped,
        }

    output_npz = Path(args.output_npz).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    report["snapshot_file_sha256"] = hashlib.sha256(output_npz.read_bytes()).hexdigest()
    report["circuit_count"] = int(len(arrays))
    report["total_snapshot_count"] = int(
        sum(int(array.shape[1]) for array in arrays.values())
    )
    output_manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
