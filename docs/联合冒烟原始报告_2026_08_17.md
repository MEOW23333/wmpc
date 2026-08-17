# 联合 warmup／Schwarz 冒烟预检报告

日期：2026 年 8 月 17 日  
运行器：`pypath/utils/run_joint_warmup_schwarz_benchmark.py`  
实验目录：`pals_data/runs/joint_warmup_schwarz_preflight_20260817_full500_online/`

## 目的

只验证联合运行器的输入绑定、warmup 注入门禁、原生预条件子模式、在线侧车路径和严格失败分类。不把本次结果作为论文性能主表，也不对缺失任务补零或插值。

## 准备阶段

- 四臂任务数：12 个（3 条线路 × 4 个臂）。
- 网表：历史 full500 生成网表，逐线路记录 SHA-256。
- warmup 输入：旧 `warmup_repro_20260709_from_aggregator` 目录，逐文件记录 SHA-256。
- 学习检查点：`learned_schwarz_crossgmin_retraining_20260816/stage_a_crossgmin_schema4_e5_lr5e-5_seed0/learned_schwarz_v1.pt`。
- 准备结果：12/12 任务可执行；没有未记录的预检阻断。

## 已运行结果

| 任务 | 阶段收敛 | warmup 注入 | 总 Newton 阶段迭代 | 总 GMRES 迭代 | 耗时 | 严格结果 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `c2__cold_row_sum` | 11/11 | 0 | 337 | 52,536 | 6.772 秒 | 失败：直接法回退 |
| `c2__warm_row_sum` | 11/11 | 0 | 337 | 52,536 | 6.751 秒 | 失败：warmup 未注入 |

两个 `row_sum` 臂的 Newton 阶段数完全一致，说明旧向量没有形成有效 warmup 因果对照。冷臂的 GMRES 过程中存在直接法回退，故即使最终阶段收敛，也不满足严格线性求解门槛。

## 学习 Schwarz 臂的预算终止

`c2__cold_learned_schwarz` 在约 260 秒后手动终止，独立记录于该任务目录的 `manual_abort.json`，不进入性能汇总。

终止前已有 119 条原生线性指标：

- GMRES 严格收敛：27 条；
- 直接法回退：92 条，回退率 `77.31%`；
- 在线侧车累计生成时间：约 `254.34` 秒，平均 `2.14` 秒/步；
- 已观测平均块覆盖率：`94.17%`；
- 预条件器增量峰值估算平均约 `61 KiB`；
- 侧车生成器本身报告的峰值常驻内存约 `398 MiB`。

这说明侧车可以成功生成，且部分工作点的稀疏预条件子能够收敛；但“每个 Newton 步启动一个独立 Python 生成器”的部署路径时间成本过高，当前不能用于大规模联合主实验。

## 批量预生成复核

新增批量入口后，在同一输入目录的 120 个线性系统工作点上一次性生成侧车：总耗时 2.896 秒，成功 120/120。随机抽查 5 个侧车，矩阵指纹、布局指纹、向量指纹、块参数和文件大小与逐步生成版本完全一致。

这证明了“预先批量生成、原生阶段只读取侧车”的路线可行，预计能消除逐 Newton 步启动 Python 的主要开销；但原生 `precomputed` 读取模式尚未接入四臂重跑，因此当前只作为实现验证，不计入端到端收益。

## 严格结论

1. 运行器能拒绝直接法回退和零 warmup 注入，不会把它们伪装成成功。
2. 旧 warmup 向量不能证明当前 full500 网表上的 Newton 节省。
3. 当前学习侧车路线的首要问题是在线部署成本和回退率，而不是内存算子本身。
4. 后续应先实现侧车预生成/缓存或持久化生成服务，再做联合大规模实验；在此之前不报告学习 Schwarz 的端到端加速。

## 产物

- `run_manifest.json`
- `planned_tasks.jsonl`
- `aggregate/summary.json`
- `tasks/circuit_000002/cold_row_sum/outcome.json`
- `tasks/circuit_000002/warm_row_sum/outcome.json`
- `tasks/circuit_000002/cold_learned_schwarz/manual_abort.json`
- `tasks/circuit_000002/cold_learned_schwarz/newton_metrics.jsonl`
