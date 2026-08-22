#!/usr/bin/env python3
"""WMPC 最小 warmup 训练和向量导出工具。

训练目标是根据某个工作点的 rhsold、时间、gmin 和迭代编号预测下一状态向量。
这是一个可独立复现的基线训练器，不等同于旧 PALS 的完整双头 warmup 模型。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pypath.utils.ngspice_utils import read_continuation_step


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


def scalar_from_name(name: str, token: str, default: float = 0.0) -> float:
    import re

    match = re.search(token + r"_([0-9.eE+-]+)", name)
    return float(match.group(1)) if match else float(default)


def load_rows(trajectory_dir: Path) -> List[Dict[str, Any]]:
    import re

    pattern = re.compile(
        r"^circuit_(\d+)_time_([0-9.eE+-]+)_gmin_([0-9.eE+-]+)_iter_(\d+)\.txt$"
    )
    rows: List[Dict[str, Any]] = []
    for path in sorted(trajectory_dir.glob("circuit_*.txt")):
        match = pattern.match(path.name)
        if not match:
            continue
        payload = read_continuation_step(str(path))
        rhsold = np.asarray(payload.get("rhsold", []), dtype=np.float64)
        # continuation 文件中的 STATE0_OUT 可能是完整状态向量，
        # 而 rhsold 属于线性系统维度。训练基线必须选择同维目标。
        candidates = (
            payload.get("wp_out", []),
            payload.get("rhsnew", []),
            payload.get("state0_out", []),
        )
        target = np.asarray([], dtype=np.float64)
        for candidate in candidates:
            value = np.asarray(candidate, dtype=np.float64)
            if value.size and value.size == rhsold.size:
                target = value
                break
        if rhsold.size == 0 or target.size == 0:
            continue
        rows.append(
            {
                "path": path,
                "circuit_id": int(match.group(1)),
                "time": float(match.group(2)),
                "gmin": float(match.group(3)),
                "iteration": int(match.group(4)),
                "rhsold": rhsold,
                "target": target,
            }
        )
    return rows


class WarmupMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def make_features(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("no usable continuation rows")
    dimension = int(rows[0]["rhsold"].size)
    features: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    for row in rows:
        if int(row["rhsold"].size) != dimension:
            continue
        scalar = np.asarray(
            [
                float(np.log10(abs(row["gmin"]) + 1e-30)),
                float(row["time"]),
                float(row["iteration"]),
            ],
            dtype=np.float64,
        )
        features.append(np.concatenate([row["rhsold"], scalar]))
        targets.append(row["target"])
    if not features:
        raise ValueError("all rows have incompatible dimensions")
    return np.stack(features), np.stack(targets)


def train(args: argparse.Namespace) -> int:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    rows = load_rows(repo_path(args.trajectory_dir))
    features, targets = make_features(rows)
    if features.shape[0] < 2:
        raise ValueError("warmup training needs at least two usable rows")
    permutation = np.random.default_rng(int(args.seed)).permutation(features.shape[0])
    split = max(1, int(round(features.shape[0] * (1.0 - float(args.validation_ratio)))))
    train_indices = permutation[:split]
    validation_indices = permutation[split:]
    if validation_indices.size == 0:
        validation_indices = train_indices[-1:]
        train_indices = train_indices[:-1]

    feature_mean = features[train_indices].mean(axis=0)
    feature_std = features[train_indices].std(axis=0)
    feature_std[feature_std < 1e-12] = 1.0
    target_mean = targets[train_indices].mean(axis=0)
    target_std = targets[train_indices].std(axis=0)
    target_std[target_std < 1e-12] = 1.0

    x_train = torch.as_tensor((features[train_indices] - feature_mean) / feature_std, dtype=torch.float64)
    y_train = torch.as_tensor((targets[train_indices] - target_mean) / target_std, dtype=torch.float64)
    x_valid = torch.as_tensor((features[validation_indices] - feature_mean) / feature_std, dtype=torch.float64)
    y_valid = torch.as_tensor((targets[validation_indices] - target_mean) / target_std, dtype=torch.float64)

    model = WarmupMLP(x_train.shape[1], y_train.shape[1], int(args.hidden_dim)).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    history: List[Dict[str, float]] = []
    for epoch in range(int(args.epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = torch.mean((model(x_train) - y_train) ** 2)
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = torch.mean((model(x_valid) - y_valid) ** 2)
        row = {
            "epoch": float(epoch + 1),
            "train_mse": float(train_loss.detach()),
            "validation_mse": float(valid_loss.detach()),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "warmup_mlp.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_args": {
                "input_dim": int(x_train.shape[1]),
                "output_dim": int(y_train.shape[1]),
                "hidden_dim": int(args.hidden_dim),
            },
            "model_state_dict": model.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "target_mean": target_mean,
            "target_std": target_std,
            "dimension": int(y_train.shape[1]),
            "train_args": vars(args),
            "sample_count": int(features.shape[0]),
            "train_count": int(train_indices.size),
            "validation_count": int(validation_indices.size),
            "history": history,
        },
        checkpoint,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "sample_count": int(features.shape[0]),
                "train_count": int(train_indices.size),
                "validation_count": int(validation_indices.size),
                "final_train_mse": history[-1]["train_mse"],
                "final_validation_mse": history[-1]["validation_mse"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"checkpoint={checkpoint}")
    return 0


def generate(args: argparse.Namespace) -> int:
    checkpoint_path = repo_path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = WarmupMLP(**payload["model_args"]).double()
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    feature_mean = np.asarray(payload["feature_mean"], dtype=np.float64)
    feature_std = np.asarray(payload["feature_std"], dtype=np.float64)
    target_mean = np.asarray(payload["target_mean"], dtype=np.float64)
    target_std = np.asarray(payload["target_std"], dtype=np.float64)
    dimension = int(payload["dimension"])
    rows = load_rows(repo_path(args.trajectory_dir))
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    # warmup 注入以 (电路、时间、gmin) 阶段为单位。若同一阶段存在多个
    # Newton 迭代，只导出最小迭代编号，避免后续样本静默覆盖入口向量。
    entry_rows: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (
            int(row["circuit_id"]),
            f"{float(row['time']):.17e}",
            f"{float(row['gmin']):.17e}",
        )
        previous = entry_rows.get(key)
        if previous is None or int(row["iteration"]) < int(previous["iteration"]):
            entry_rows[key] = row
    for key in sorted(entry_rows, key=lambda item: (item[0], float(item[1]), -float(item[2]))):
        row = entry_rows[key]
        rhsold = row["rhsold"]
        if rhsold.size != dimension:
            continue
        features = np.concatenate(
            [
                rhsold,
                np.asarray(
                    [
                        np.log10(abs(row["gmin"]) + 1e-30),
                        row["time"],
                        row["iteration"],
                    ],
                    dtype=np.float64,
                ),
            ]
        )
        with torch.no_grad():
            normalized = model(
                torch.as_tensor(
                    ((features - feature_mean) / feature_std)[None, :],
                    dtype=torch.float64,
                )
            ).numpy()[0]
        prediction = normalized * target_std + target_mean
        filename = (
            f"segment_warmup_circuit_{row['circuit_id']}_time_{row['time']:.17e}"
            f"_gmin_{row['gmin']:.17e}_rhsold.txt"
        )
        path = output_dir / filename
        np.savetxt(path, prediction, fmt="%.17e")
        manifest.append(
            {
                "circuit_id": row["circuit_id"],
                "time": row["time"],
                "gmin_val": row["gmin"],
                "iteration": row["iteration"],
                "path": str(path),
                "sha256": sha256(path),
                "dimension": int(prediction.size),
                "prediction_norm": float(np.linalg.norm(prediction)),
            }
        )
    (output_dir / "warmup_manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "checkpoint": str(checkpoint_path), "entries": manifest},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "count": len(manifest)}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--trajectory-dir", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--validation-ratio", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.set_defaults(handler=train)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--trajectory-dir", required=True)
    generate_parser.add_argument("--checkpoint", required=True)
    generate_parser.add_argument("--output-dir", required=True)
    generate_parser.set_defaults(handler=generate)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
