#!/usr/bin/env python3
"""检查 WMPC 在目标服务器上的 Python、科学计算、CUDA 和原生程序环境。"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path


def module_version(name: str):
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    return {"available": True, "version": str(getattr(module, "__version__", "unknown"))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ngspice", default=os.environ.get("WMPC_NGSPICE_EXECUTABLE", ""))
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-native", action="store_true")
    args = parser.parse_args()

    result = {
        "python": {"executable": sys.executable, "version": sys.version},
        "modules": {name: module_version(name) for name in ("numpy", "scipy", "torch", "torch_geometric")},
        "cuda": {"available": False, "device_count": 0, "devices": []},
        "native": {"path": args.ngspice or None, "available": False},
        "errors": [],
    }
    try:
        import torch

        available = bool(torch.cuda.is_available())
        result["cuda"]["available"] = available
        result["cuda"]["device_count"] = int(torch.cuda.device_count()) if available else 0
        result["cuda"]["devices"] = [torch.cuda.get_device_name(i) for i in range(int(torch.cuda.device_count()))] if available else []
    except Exception as exc:
        result["cuda"]["error"] = repr(exc)

    if args.ngspice:
        native = Path(args.ngspice).expanduser()
        result["native"]["path"] = str(native)
        result["native"]["available"] = native.is_file() and os.access(native, os.X_OK)
    else:
        result["native"]["path"] = shutil.which("ngspice")
        result["native"]["available"] = bool(result["native"]["path"])

    required_modules = ("numpy", "scipy", "torch", "torch_geometric")
    for name in required_modules:
        if not result["modules"][name]["available"]:
            result["errors"].append(f"module_missing:{name}")
    if args.require_cuda and not result["cuda"]["available"]:
        result["errors"].append("cuda_unavailable")
    if args.require_native and not result["native"]["available"]:
        result["errors"].append("native_executable_unavailable")
    result["valid"] = not result["errors"]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
