# WMPC

WMPC 是当前 PALS 课题的精简、可复现实验项目。只保留原生求解器源码、稀疏学习型 Schwarz 预条件子、侧车工具、预热与联合评测入口，以及小型结果证据。

## 当前可核验结论

- 稀疏算子与稠密参考算子的相对二范数误差为 2.28116602572133e-33。
- 预条件器增量常驻内存节省为 58.4985%；整进程峰值内存节省只有 7.3475%，不能混写。
- 固定块 Schwarz 在冻结正 gmin 工作点上将平均 GMRES 迭代从 26.0 降至 8.1，20/20 严格成功。
- 三种规模共 60 个稀疏任务中 57 个严格成功，失败集中在 100×100 长尾任务。
- 当前 warmup 历史平均减少 Newton 迭代的结果存在数据绑定缺口，不能作为论文主结论。

## 目录

- src：带稀疏 GMRES、Schwarz 和安全侧车接口的原生求解器源码。
- pypath/preconditioner：稀疏学习型 Schwarz、块划分、训练和线性系统契约。
- pypath/utils：侧车导出、批量生成、原生评测和结果核验入口。
- results：检查点、侧车和小型结果摘要，不包含轨迹、缓存或整批训练数据。
- docs：研究状态、实验计划、证据来源和投稿判断。

## 最小验证

进入 /home/ZhangLexin/PALS/wmpc 后，使用 PALS_env 环境运行：

    PYTHONPATH=. /home/ZhangLexin/miniconda3/envs/PALS_env/bin/python3 pypath/preconditioner/tests/test_sparse_learned_schwarz_v1.py

该测试应报告 20/20 通过。结果文件的完整性可用以下命令复核：

    sha256sum -c results/SHA256SUMS.txt

原生程序重新编译时使用本目录的构建骨架，生成的二进制放在 build/src/ngspice。实验先决条件写在 docs/实验计划_2026_08_17.md。

## 研究边界

WMPC 当前是 稀疏预条件子已实现、联合端到端收益待补证的研究版本。不得把预条件器增量内存节省写成整进程内存节省，也不得把历史 warmup 结果写成当前严格复现实验结果。

## 从空目录复现

本项目现在包含数据生成脚本，不需要预先复制旧项目的 `experiments/`、`pals_data/` 或 `precondition_experiments/`。在具备 C 编译器、CUDA 驱动和 GPU 的服务器上执行：

```bash
git clone git@github.com:MEOW23333/wmpc.git
cd wmpc
export PYTHONPATH=$PWD
export WMPC_PYTHON=/home/ZhangLexin/miniconda3/envs/PALS_env/bin/python3
./autogen.sh
mkdir build && cd build
../configure --without-x --disable-xspice --disable-openmp --disable-cider --with-readline=no
make -j8
cd ..
export WMPC_NGSPICE_EXECUTABLE=$PWD/build/src/ngspice
```

依赖安装有两种口径：

- 使用仓库锁定清单：`$WMPC_PYTHON -m pip install -r requirements-cu12-lock.txt`；该清单用于固定一套 CUDA 12 环境，不要求与目标服务器驱动完全相同。
- 使用目标服务器现有 CUDA 环境：先按服务器 CUDA 版本安装 `torch` 和 `torch-geometric`，再执行 `$WMPC_PYTHON -m pip install -r requirements-minimal.txt`；不要把锁定文件中的 NVIDIA 运行库版本强行覆盖服务器驱动。

一键完成网表生成、原生轨迹与 Jacobian 导出、warmup 训练与导出、预条件子训练、稀疏 GMRES 验证和冻结 warmup 验证：

```bash
$WMPC_PYTHON pypath/data_generation/run_minimal_reproduction.py \
  --output-root results/runs/minimal_reproduction \
  --num-circuits 1 --node-count 4 --seed 7 --circuit-id 0 \
  --timeout-sec 120 --warmup-epochs 10 --preconditioner-epochs 5
```

若只需生成网表而暂不运行原生程序，可加 `--skip-native`。若要显式运行四臂联合评测，可加 `--run-joint`；该评测要求原生程序支持联合续接和在线侧车，通用测试网表可能因没有正 gmin 阶段而只得到诊断性失败，不能把这种结果当作算法失败。

生成结果默认位于 `results/runs/`，其中：

- `data/generated/generation_summary.json`：电路数、工作点数、轨迹数、Jacobian 数和 warmup 阶段数。
- `data/generated/workpoint_manifest.json`：每个工作点的时间、gmin、迭代编号和文件指纹。
- `data/generated/validation.json`：数据完整性、维数、Jacobian 对齐和哈希校验。
- `results/warmup_train/`、`results/warmup_vectors/`：warmup 检查点、训练摘要和向量清单。
- `results/preconditioner/`：预条件子检查点及训练摘要。
- `results/sparse_eval/summary.json`：稀疏 GMRES 的维数、非零元、内存估计、耗时、迭代和严格残差。
- `results/warmup_frozen/aggregate/summary.json`：冻结状态 warmup 对照。
- `reproduction_manifest.json`：本次运行的入口、产物和联合评测状态。

数据完整性可单独复核：

```bash
$WMPC_PYTHON pypath/data_generation/validate_reproducible_dataset.py \
  --root results/runs/minimal_reproduction/data/generated --require-native
```

## 分步入口

- 生成：`pypath/data_generation/generate_reproducible_dataset.py`
- warmup 训练和导出：`pypath/data_generation/warmup_tools.py train|generate`
- 预条件子训练：`pypath/preconditioner/train_learned_schwarz.py`
- 稀疏线性求解验证：`pypath/utils/sparse_gmres_prototype.py`
- 冻结 warmup 验证：`pypath/utils/run_frozen_warmup_stage_benchmark.py`
- 四臂联合评测：`pypath/utils/run_joint_warmup_schwarz_benchmark.py`

更完整的执行说明见 `docs/最小复现流程_2026_08_21.md`。所有论文级指标都必须在当前网表、节点映射、检查点和原生二进制一致的条件下重新生成；最小通用电路只用于验证链路，不代表最终性能结论。
