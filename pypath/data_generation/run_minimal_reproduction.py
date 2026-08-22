#!/usr/bin/env python3
"""从空目录串联 WMPC 的最小可复现实验链路。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = os.environ.get("WMPC_PYTHON", sys.executable)


def inside(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError(f"路径必须位于 WMPC 内：{path}")
    return path


def run(label: str, args: Sequence[str], *, allow_failure: bool = False) -> int:
    command = [str(item) for item in args]
    print(f"[{label}] {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, env=os.environ.copy(), check=False)
    if completed.returncode and not allow_failure:
        raise RuntimeError(f"{label} 失败，退出码 {completed.returncode}")
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results/runs/minimal_reproduction")
    parser.add_argument("--num-circuits", type=int, default=1)
    parser.add_argument("--node-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--circuit-id", type=int, default=0)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--preconditioner-epochs", type=int, default=5)
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--tran-step", type=float, default=1e-12)
    parser.add_argument("--tran-stop", type=float, default=8e-12)
    parser.add_argument("--run-joint", action="store_true", help="运行四臂联合评测；通用测试网表可能因缺少正 gmin 阶段而仅产生诊断结果")
    parser.add_argument("--joint-timeout-sec", type=int, default=120)
    parser.add_argument("--joint-online-timeout-sec", type=int, default=30)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--skip-environment-check", action="store_true")
    args = parser.parse_args()
    if int(args.num_circuits) <= 0 or not 0 <= int(args.circuit_id) < int(args.num_circuits):
        parser.error("--circuit-id must be within [0, --num-circuits)")
    if float(args.tran_step) <= 0 or float(args.tran_stop) < float(args.tran_step):
        parser.error("--tran-step and --tran-stop must be positive, with tran-stop >= tran-step")

    root = inside(args.output_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"输出目录非空，为避免覆盖请更换 --output-root：{root}")
    root.mkdir(parents=True, exist_ok=True)
    data = root / "data"
    results = root / "results"
    generated = data / "generated"
    warmup_train = results / "warmup_train"
    warmup_vectors = results / "warmup_vectors"
    preconditioner = results / "preconditioner"
    sparse_eval = results / "sparse_eval"
    frozen_eval = results / "warmup_frozen"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    os.environ.update(env)

    if not args.skip_environment_check:
        environment_command = [PYTHON, "pypath/data_generation/check_environment.py"]
        native_hint = os.environ.get("WMPC_NGSPICE_EXECUTABLE", "")
        if native_hint:
            environment_command += ["--ngspice", native_hint]
        if args.require_cuda:
            environment_command.append("--require-cuda")
        if not args.skip_native:
            environment_command.append("--require-native")
        run("环境自检", environment_command)

    run(
        "数据生成",
        [
            PYTHON,
            "pypath/data_generation/generate_reproducible_dataset.py",
            "--output-dir",
            str(generated),
            "--num-circuits",
            str(args.num_circuits),
            "--node-count",
            str(args.node_count),
            "--seed",
            str(args.seed),
            "--timeout-sec",
            str(args.timeout_sec),
            "--tran-step",
            str(args.tran_step),
            "--tran-stop",
            str(args.tran_stop),
        ]
        + (["--skip-native"] if args.skip_native else []),
    )
    run(
        "数据契约校验",
        [
            PYTHON,
            "pypath/data_generation/validate_reproducible_dataset.py",
            "--root",
            str(generated),
        ]
        + ([] if args.skip_native else ["--require-native"])
        + ["--json-out", str(generated / "validation.json")],
    )
    if args.skip_native:
        print(json.dumps({"status": "网表已生成，原生数据未生成；后续训练阶段跳过", "output_root": str(root)}, ensure_ascii=False))
        return 0

    trajectory = generated / "trajectory"
    netlists = generated / "generated_netlists"
    run(
        "暖启动训练",
        [
            PYTHON,
            "pypath/data_generation/warmup_tools.py",
            "train",
            "--trajectory-dir",
            str(trajectory),
            "--output-dir",
            str(warmup_train),
            "--epochs",
            str(args.warmup_epochs),
            "--hidden-dim",
            "32",
            "--seed",
            str(args.seed),
        ],
    )
    run(
        "暖启动导出",
        [
            PYTHON,
            "pypath/data_generation/warmup_tools.py",
            "generate",
            "--trajectory-dir",
            str(trajectory),
            "--checkpoint",
            str(warmup_train / "warmup_mlp.pt"),
            "--output-dir",
            str(warmup_vectors),
        ],
    )
    run(
        "预条件子训练",
        [
            PYTHON,
            "pypath/preconditioner/train_learned_schwarz.py",
            "--netlist-dir",
            str(netlists),
            "--trajectory-dir",
            str(trajectory),
            "--circuit-ids",
            str(args.circuit_id),
            "--output-dir",
            str(preconditioner),
            "--run-tag",
            f"circuit{args.circuit_id}_step0",
            "--step-offset",
            "0",
            "--max-steps-per-circuit",
            "1",
            "--max-samples",
            "1",
            "--epochs",
            str(args.preconditioner_epochs),
            "--lr",
            "1e-4",
            "--hidden-dim",
            "64",
            "--gaussian-probes",
            "2",
            "--initial-guess-mode",
            "rhsold",
            "--block-mode",
            "generic",
            "--seed",
            str(args.seed),
        ],
    )
    checkpoint = preconditioner / f"circuit{args.circuit_id}_step0" / "learned_schwarz_v1.pt"
    run(
        "稀疏 GMRES 验证",
        [
            PYTHON,
            "pypath/utils/sparse_gmres_prototype.py",
            "--trajectory-dir",
            str(trajectory),
            "--circuit-id",
            str(args.circuit_id),
            "--max-steps",
            "1",
            "--backend",
            "scipy_sparse",
            "--modes",
            "row_sum,learned_schwarz_v1_sparse",
            "--output-dir",
            str(sparse_eval),
            "--netlist-path",
            str(netlists / f"{args.circuit_id}.sp"),
            "--learned-schwarz-checkpoint",
            str(checkpoint),
            "--apply-gmin-diagonal",
            "--use-rhsold-as-x0",
        ],
    )
    run(
        "冻结暖启动准备",
        [
            PYTHON,
            "pypath/utils/run_frozen_warmup_stage_benchmark.py",
            "--prepare-only",
            "--include-zero-gmin",
            "--output-dir",
            str(frozen_eval),
            "--trajectory-dir",
            str(trajectory),
            "--warmup-input-root",
            str(warmup_vectors),
            "--netlist-dir",
            str(netlists),
            "--circuit-ids",
            str(args.circuit_id),
            "--max-workpoints-per-circuit",
            "1",
        ],
    )
    for task in sorted(frozen_eval.glob("tasks/*/*/task.json")):
        run("冻结暖启动任务", [PYTHON, "pypath/utils/run_frozen_warmup_stage_benchmark.py", "--run-task", "--task-json", str(task)], allow_failure=True)
    run("冻结暖启动汇总", [PYTHON, "pypath/utils/run_frozen_warmup_stage_benchmark.py", "--summarize-only", "--output-dir", str(frozen_eval)])

    joint_eval = results / "joint"
    if args.run_joint:
        run(
            "联合评测准备",
            [
                PYTHON,
                "pypath/utils/run_joint_warmup_schwarz_benchmark.py",
                "--prepare-only",
                "--output-dir",
                str(joint_eval),
                "--netlist-dir",
                str(netlists),
                "--warmup-input-root",
                str(warmup_vectors),
                "--circuit-ids",
                str(args.circuit_id),
                "--native-schwarz-precond",
                "learned_schwarz_v1_sparse",
                "--online-sidecar-mode",
                "oneshot_v1",
                "--learned-schwarz-checkpoint",
                str(checkpoint),
                "--timeout-sec",
                str(args.joint_timeout_sec),
                "--online-timeout-sec",
                str(args.joint_online_timeout_sec),
            ],
        )
        for task in sorted(joint_eval.glob("tasks/*/*/task.json")):
            run("联合评测任务", [PYTHON, "pypath/utils/run_joint_warmup_schwarz_benchmark.py", "--run-task", "--task-json", str(task)], allow_failure=True)
        run("联合评测汇总", [PYTHON, "pypath/utils/run_joint_warmup_schwarz_benchmark.py", "--summarize-only", "--output-dir", str(joint_eval)])

    manifest = {
        "schema_version": 1,
        "status": "completed",
        "output_root": str(root),
        "data_root": str(generated),
        "warmup_checkpoint": str(warmup_train / "warmup_mlp.pt"),
        "preconditioner_checkpoint": str(checkpoint),
        "sparse_evaluation": str(sparse_eval / "summary.json"),
        "frozen_warmup_evaluation": str(frozen_eval / "aggregate" / "summary.json"),
        "joint_evaluation": str(joint_eval / "aggregate" / "summary.json") if args.run_joint else None,
        "joint_requested": bool(args.run_joint),
    }
    (root / "reproduction_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
