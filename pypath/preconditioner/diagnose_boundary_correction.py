import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pypath.preconditioner.learned_schwarz import (
    BoundaryCorrectionPreconditioner,
    build_learned_schwarz_sample,
    make_probe_matrix,
)
from pypath.preconditioner.train_learned_schwarz import (
    _load_samples,
    _make_core_arnoldi_probe_matrix,
)
from pypath.preconditioner.linear_system_contract import INITIAL_GUESS_MODE_RHSOLD, INITIAL_GUESS_MODES, validate_learned_schwarz_checkpoint_contract


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def _load_boundary_model(checkpoint_path: str, initial_guess_mode: str) -> BoundaryCorrectionPreconditioner:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    validate_learned_schwarz_checkpoint_contract(checkpoint, expected_initial_guess_mode=initial_guess_mode)
    model_kind = str(checkpoint.get("model_kind", ""))
    if model_kind != "boundary_correction_v1":
        raise ValueError(f"expected boundary_correction_v1 checkpoint, got {model_kind}")
    model_args = dict(checkpoint.get("model_args") or {})
    model_args.setdefault("correction_feature_dim", 0)
    model_args.setdefault("projection_mode", "none")
    model_args.setdefault("projection_max_scale", 1.0)
    model = BoundaryCorrectionPreconditioner(**model_args).to(dtype=torch.float64)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    numeric_keys = [
        "core_solution_norm",
        "raw_projection_alpha",
        "raw_projection_alpha_clipped",
        "raw_alignment_cos",
        "projected_alignment_cos",
        "raw_correction_norm",
        "projected_correction_norm",
        "limited_correction_norm",
        "raw_action_norm",
        "projected_action_norm",
        "core_residual_norm",
    ]
    summary: Dict[str, Any] = {"probe_count": len(rows)}
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows if key in row], dtype=np.float64)
        if values.size == 0:
            continue
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_median"] = float(np.median(values))
        summary[f"{key}_min"] = float(np.min(values))
        summary[f"{key}_max"] = float(np.max(values))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose learned boundary correction projection behavior.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--netlist-dir", required=True)
    parser.add_argument("--trajectory-dir", required=True)
    parser.add_argument("--circuit-ids", default="0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-tag", default="boundary_projection_diagnostics")
    parser.add_argument("--block-mode", default="cell_core_plus_onehop_boundary")
    parser.add_argument("--core-block-mode", default="cell_core")
    parser.add_argument("--boundary-block-mode", default="cell_core_plus_onehop_boundary")
    parser.add_argument("--max-block-size", type=int, default=32)
    parser.add_argument("--min-block-size", type=int, default=2)
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument("--max-total-block-nnz", type=int, default=0)
    parser.add_argument("--uncovered-row-policy", choices=["identity", "jacobi", "row_sum"], default="row_sum")
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--positive-gmin-only", action="store_true")
    parser.add_argument("--max-steps-per-circuit", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--gaussian-probes", type=int, default=1)
    parser.add_argument("--arnoldi-probes", type=int, default=0)
    parser.add_argument("--include-block-diagnostics", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-guess-mode", choices=sorted(INITIAL_GUESS_MODES), default=INITIAL_GUESS_MODE_RHSOLD)
    parser.add_argument("--disable-gmin-diagonal", action="store_true")
    parser.set_defaults(model_kind="boundary_correction_v1")
    args = parser.parse_args()
    if args.disable_gmin_diagonal:
        raise ValueError("schema v3 diagnostics require the gmin diagonal")

    torch.set_default_dtype(torch.float64)
    model = _load_boundary_model(args.checkpoint, args.initial_guess_mode)
    sample_payloads = _load_samples(args)
    if not sample_payloads:
        raise RuntimeError("no samples were loaded for diagnostics")

    output_dir = os.path.join(args.output_dir, args.run_tag)
    os.makedirs(output_dir, exist_ok=True)
    per_probe_path = os.path.join(output_dir, "projection_diagnostics.jsonl")
    per_block_path = os.path.join(output_dir, "block_projection_diagnostics.jsonl")
    rows: List[Dict[str, Any]] = []

    block_handle = open(per_block_path, "w", encoding="utf-8") if args.include_block_diagnostics else None
    try:
        with open(per_probe_path, "w", encoding="utf-8") as handle:
            for sample_idx, payload in enumerate(sample_payloads):
                core_sample = build_learned_schwarz_sample(
                    matrix=payload["matrix"],
                    plan=payload["core_plan"],
                    linear_rhs=payload["linear_rhs"],
                    initial_residual=payload["initial_residual"],
                    gmin=payload["gmin"],
                    dtype=torch.float64,
                )
                boundary_sample = build_learned_schwarz_sample(
                    matrix=payload["matrix"],
                    plan=payload["boundary_plan"],
                    linear_rhs=payload["linear_rhs"],
                    initial_residual=payload["initial_residual"],
                    gmin=payload["gmin"],
                    dtype=torch.float64,
                )
                probes = make_probe_matrix(
                    matrix_size=int(payload["matrix"].shape[0]),
                    linear_rhs=payload["linear_rhs"],
                    initial_residual=payload["initial_residual"],
                    gaussian_count=int(args.gaussian_probes),
                    seed=int(args.seed) + sample_idx,
                    dtype=torch.float64,
                )
                if int(args.arnoldi_probes) > 0:
                    arnoldi_probes = _make_core_arnoldi_probe_matrix(
                        matrix=payload["matrix"],
                        linear_rhs=payload["linear_rhs"],
                        initial_residual=payload["initial_residual"],
                        core_plan=payload["core_plan"],
                        probe_count=int(args.arnoldi_probes),
                        seed=int(args.seed) + sample_idx + 1000003,
                        dtype=torch.float64,
                    )
                    if arnoldi_probes.numel() > 0:
                        probes = torch.cat([probes, arnoldi_probes], dim=0)
                for probe_idx, probe in enumerate(probes):
                    row = {
                        "circuit_id": int(payload["circuit_id"]),
                        "iteration": int(payload["iteration"]),
                        "probe_idx": int(probe_idx),
                        "matrix_size": int(payload["matrix"].shape[0]),
                    }
                    row.update(model.correction_diagnostics(core_sample, boundary_sample, probe))
                    rows.append(row)
                    handle.write(json.dumps(row, default=_json_default) + "\n")
                    if block_handle is not None:
                        with torch.no_grad():
                            core_solution = model.apply_core(core_sample, probe)
                            core_residual = probe - core_sample.matrix.matmul(core_solution)
                            block_rows = model.block_correction_diagnostics(boundary_sample, core_residual)
                        for block_row in block_rows:
                            block_row.update(
                                {
                                    "circuit_id": int(payload["circuit_id"]),
                                    "iteration": int(payload["iteration"]),
                                    "probe_idx": int(probe_idx),
                                }
                            )
                            block_handle.write(json.dumps(block_row, default=_json_default) + "\n")
    finally:
        if block_handle is not None:
            block_handle.close()

    by_circuit: Dict[str, Any] = {}
    for circuit_id in sorted({int(row["circuit_id"]) for row in rows}):
        by_circuit[str(circuit_id)] = _summarize([row for row in rows if int(row["circuit_id"]) == circuit_id])
    summary = {
        "checkpoint": args.checkpoint,
        "run_tag": args.run_tag,
        "sample_count": len(sample_payloads),
        "probe_count": len(rows),
        "overall": _summarize(rows),
        "by_circuit": by_circuit,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)

    print(f"summary={summary_path}")
    print(f"per_probe={per_probe_path}")
    if args.include_block_diagnostics:
        print(f"per_block={per_block_path}")


if __name__ == "__main__":
    main()
