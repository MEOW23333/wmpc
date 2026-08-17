"""训练数据、模型和环境的可复现来源记录。"""
from __future__ import annotations
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

SCHEMA_VERSION = 1
_CHUNK_SIZE = 1024 * 1024

def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def _absolute(path: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.fspath(path))

def _relative(path: str, repo_root: Optional[str]) -> Optional[str]:
    if not repo_root:
        return None
    try:
        return os.path.relpath(path, _absolute(repo_root))
    except ValueError:
        return None

def fingerprint_path(path: str | os.PathLike[str], *, repo_root: Optional[str] = None) -> Dict[str, Any]:
    absolute = _absolute(path)
    result: Dict[str, Any] = {
        "path": absolute,
        "relative_path": _relative(absolute, repo_root),
        "exists": os.path.exists(absolute),
    }
    if not result["exists"]:
        return result
    if os.path.isfile(absolute):
        stat_result = os.stat(absolute)
        result.update(kind="file", size=int(stat_result.st_size), sha256=sha256_file(absolute))
        return result
    if os.path.isdir(absolute):
        files = []
        for child in sorted(Path(absolute).rglob("*")):
            if child.is_file():
                child_path = str(child)
                files.append({
                    "relative_path": os.path.relpath(child_path, absolute),
                    "size": int(child.stat().st_size),
                    "sha256": sha256_file(child_path),
                })
        manifest = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        result.update(kind="directory", file_count=len(files), sha256=hashlib.sha256(manifest).hexdigest(), files=files)
        return result
    result["kind"] = "other"
    return result

def _git_metadata(repo_root: str) -> Dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", "-C", repo_root, *args], stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return ""
    status = run("status", "--short")
    return {"branch": run("branch", "--show-current"), "commit": run("rev-parse", "HEAD"), "dirty": bool(status), "status_short": status.splitlines()}

def environment_metadata() -> Dict[str, Any]:
    result: Dict[str, Any] = {"python_executable": _absolute(sys.executable), "python_version": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "scipy", "torch"):
        try:
            module = __import__(name)
            result[f"{name}_version"] = getattr(module, "__version__", None)
        except Exception as exc:
            result[f"{name}_version"] = None
            result[f"{name}_import_error"] = repr(exc)
    try:
        import torch
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["cuda_device_count"] = int(torch.cuda.device_count())
    except Exception:
        result["cuda_available"] = False
        result["cuda_device_count"] = 0
    return result

def build_training_provenance(*, repo_root: str, source_data_path: str, netlist_dir: str, proposer_paths: Sequence[str] = (), split_manifest_path: Optional[str] = None, code_paths: Sequence[str] = (), metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    root = _absolute(repo_root)
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": root,
        "git": _git_metadata(root),
        "environment": environment_metadata(),
        "paths": {
            "source_data": fingerprint_path(source_data_path, repo_root=root),
            "netlist_dir": fingerprint_path(netlist_dir, repo_root=root),
            "proposer_models": [fingerprint_path(path, repo_root=root) for path in sorted({_absolute(x) for x in proposer_paths})],
            "split_manifest": fingerprint_path(split_manifest_path, repo_root=root) if split_manifest_path else None,
            "code": [fingerprint_path(path, repo_root=root) for path in sorted({_absolute(x) for x in code_paths})],
        },
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    return payload

def attach_artifact_fingerprints(payload: Mapping[str, Any], artifacts: Iterable[str], *, repo_root: Optional[str] = None) -> Dict[str, Any]:
    result = dict(payload)
    result["artifacts"] = [fingerprint_path(path, repo_root=repo_root) for path in artifacts]
    return result

def write_provenance(path: str, payload: Mapping[str, Any]) -> None:
    target = _absolute(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    temporary = f"{target}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, target)
