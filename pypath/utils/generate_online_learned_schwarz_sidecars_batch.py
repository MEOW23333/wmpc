#!/usr/bin/env python3
"""在同一 Python 进程中批量生成在线 learned Schwarz 侧车。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pypath.utils.export_native_learned_schwarz_sidecar import (
    export_sidecar_from_linear_system_corpus,
)

SYSTEM_RE = re.compile(
    r"^linear_system_circuit_(?P<circuit>\d+)_time_(?P<time>[0-9.eE+-]+)"
    r"_gmin_(?P<gmin>[0-9.eE+-]+)_iter_(?P<iteration>\d+)\.txt$"
)


def _repo_path(raw: str, *, must_exist: bool = False) -> Path:
    path = Path(raw).expanduser().resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError(f"path must stay inside PALS: {path}")
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\\n")
    os.replace(tmp, path)


def _discover(input_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for system_path in sorted(input_dir.glob("linear_system_circuit_*_iter_*.txt")):
        match = SYSTEM_RE.match(system_path.name)
        if not match:
            continue
        jacobian_path = system_path.with_name(system_path.stem + "_jac.txt")
        if not jacobian_path.is_file():
            rows.append({"system_path": str(system_path), "error": "jacobian_missing"})
            continue
        rows.append(
            {
                "system_path": system_path,
                "jacobian_path": jacobian_path,
                "circuit_id": int(match.group("circuit")),
                "time": float(match.group("time")),
                "gmin": float(match.group("gmin")),
                "corpus_iteration": int(match.group("iteration")),
                "newton_iter": int(match.group("iteration")) + 1,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--netlist-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--initial-guess-mode", choices=("rhsold", "zero"), default="rhsold")
    parser.add_argument("--min-block-size", type=int, default=2)
    parser.add_argument("--max-block-size", type=int, default=32)
    parser.add_argument("--max-blocks", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    input_dir = _repo_path(args.input_dir, must_exist=True)
    output_dir = _repo_path(args.output_dir)
    netlist = _repo_path(args.netlist_path, must_exist=True)
    checkpoint = _repo_path(args.checkpoint, must_exist=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    discovered = _discover(input_dir)
    if int(args.max_items) > 0:
        discovered = discovered[: int(args.max_items)]
    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for item in discovered:
        if "error" in item:
            rows.append({**item, "success": False})
            continue
        system_path = Path(item["system_path"])
        jacobian_path = Path(item["jacobian_path"])
        output_path = output_dir / (
            f"sidecar_iter_{item['newton_iter']}_time_{item['time']:.17e}"
            f"_gmin_{item['gmin']:.17e}.json"
        )
        if output_path.exists() and not args.overwrite:
            rows.append({**item, "success": False, "error": "output_exists", "output_path": str(output_path)})
            continue
        t0 = time.perf_counter()
        try:
            payload = export_sidecar_from_linear_system_corpus(
                system_path=str(system_path),
                jacobian_path=str(jacobian_path),
                netlist_path=str(netlist),
                checkpoint_path=str(checkpoint),
                output_path=str(output_path),
                time_value=float(item["time"]),
                gmin=float(item["gmin"]),
                newton_iter=int(item["newton_iter"]),
                initial_guess_mode=str(args.initial_guess_mode),
                min_block_size=int(args.min_block_size),
                max_block_size=int(args.max_block_size),
                max_blocks=int(args.max_blocks),
            )
            rows.append({
                **item,
                "success": True,
                "generation_seconds": time.perf_counter() - t0,
                "output_path": str(output_path),
                "output_bytes": output_path.stat().st_size,
                "output_sha256": _sha256(output_path),
                "matrix_size": int(payload["matrix_size"]),
                "block_count": int(payload["block_count"]),
            })
        except Exception as exc:
            rows.append({**item, "success": False, "generation_seconds": time.perf_counter() - t0, "error": repr(exc)[:300]})
    summary = {
        "schema_version": 1,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "netlist_path": str(netlist),
        "netlist_sha256": _sha256(netlist),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "item_count": len(rows),
        "success_count": sum(bool(row.get("success")) for row in rows),
        "failure_count": sum(not bool(row.get("success")) for row in rows),
        "elapsed_seconds": time.perf_counter() - started,
        "rows_path": str(output_dir / "batch_rows.jsonl"),
    }
    _write_jsonl(output_dir / "batch_rows.jsonl", rows)
    (output_dir / "batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
