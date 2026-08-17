#!/usr/bin/env python3
"""Generate one schema-5 learned-Schwarz sidecar from a live corpus snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pypath.utils.export_native_learned_schwarz_sidecar import (
    INITIAL_GUESS_MODES,
    export_sidecar_from_linear_system_corpus,
)


STATUS_SCHEMA_VERSION = 1


def _repo_path(raw_path: str, *, label: str, must_exist: bool) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError(f"{label} must stay inside PALS: {path}")
    if must_exist and not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return max(value, 0) * 1024


def _failure_reason(exc: BaseException) -> str:
    if isinstance(exc, FileNotFoundError):
        return "input_not_found"
    if isinstance(exc, ValueError):
        return "input_invalid"
    if isinstance(exc, OSError):
        return "io_error"
    return "generator_exception"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one online learned Schwarz sidecar from one CKTload snapshot."
    )
    parser.add_argument("--system-path", required=True)
    parser.add_argument("--jacobian-path", required=True)
    parser.add_argument("--netlist-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--time", dest="time_value", required=True, type=float)
    parser.add_argument("--gmin", required=True, type=float)
    parser.add_argument("--newton-iter", required=True, type=int)
    parser.add_argument(
        "--initial-guess-mode",
        choices=sorted(INITIAL_GUESS_MODES),
        required=True,
    )
    parser.add_argument("--min-block-size", type=int, default=2)
    parser.add_argument("--max-block-size", type=int, default=32)
    parser.add_argument("--max-blocks", type=int, default=0)
    return parser


def _status_base(args: argparse.Namespace, started: float) -> Dict[str, Any]:
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "generator": "generate_online_learned_schwarz_sidecar.py",
        "system_path": str(Path(args.system_path).resolve()),
        "jacobian_path": str(Path(args.jacobian_path).resolve()),
        "output_path": str(Path(args.output_path).resolve()),
        "time": float(args.time_value),
        "gmin": float(args.gmin),
        "newton_iter": int(args.newton_iter),
        "initial_guess_mode": str(args.initial_guess_mode),
        "generation_seconds": float(time.perf_counter() - started),
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def main() -> int:
    args = _build_parser().parse_args()
    started = time.perf_counter()
    status_path: Path | None = None
    try:
        system_path = _repo_path(args.system_path, label="system_path", must_exist=True)
        jacobian_path = _repo_path(args.jacobian_path, label="jacobian_path", must_exist=True)
        netlist_path = _repo_path(args.netlist_path, label="netlist_path", must_exist=True)
        checkpoint_path = _repo_path(args.checkpoint, label="checkpoint", must_exist=True)
        output_path = _repo_path(args.output_path, label="output_path", must_exist=False)
        status_path = _repo_path(args.status_path, label="status_path", must_exist=False)
        payload = export_sidecar_from_linear_system_corpus(
            system_path=str(system_path),
            jacobian_path=str(jacobian_path),
            netlist_path=str(netlist_path),
            checkpoint_path=str(checkpoint_path),
            output_path=str(output_path),
            time_value=float(args.time_value),
            gmin=float(args.gmin),
            newton_iter=int(args.newton_iter),
            initial_guess_mode=str(args.initial_guess_mode),
            min_block_size=int(args.min_block_size),
            max_block_size=int(args.max_block_size),
            max_blocks=int(args.max_blocks),
        )
        status = _status_base(args, started)
        status.update(
            {
                "success": True,
                "failure_reason": "",
                "matrix_size": int(payload["matrix_size"]),
                "block_count": int(payload["block_count"]),
                "total_block_rows": int(payload["total_block_rows"]),
                "sidecar_bytes": int(output_path.stat().st_size),
                "sidecar_sha256": _sha256_file(output_path),
                "matrix_fingerprint": str(payload["matrix_fingerprint"]),
                "layout_sha256": str(payload["layout_sha256"]),
            }
        )
        _atomic_write_json(status_path, status)
        print(json.dumps(status, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        status = _status_base(args, started)
        status.update(
            {
                "success": False,
                "failure_reason": _failure_reason(exc),
                "error": str(exc)[:240],
            }
        )
        if status_path is not None:
            try:
                _atomic_write_json(status_path, status)
            except OSError:
                pass
        print(json.dumps(status, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
