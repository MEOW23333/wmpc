# 语义粗空间先导实验结果

日期：2026 年 8 月 21 日
项目：`wmpc`
实验目录：`results/semantic_coarse_space_pilot_20260821`

## 一、结论

本轮实验验证了接口完整性和全阶精度，但没有验证当前粗空间构造能够改善现有语义 Schwarz。

在 20 个冻结正 `gmin` 工作点上：

- `semantic_cell_core_sparse`：平均 GMRES 迭代 8.5；
- `semantic_coarse_r1_sparse`：平均 21.7；
- `semantic_coarse_r2_sparse`：平均 21.7；
- 三种语义方案均严格成功 20/20，真实相对残差均小于 `1e-8`；
- 两层粗空间相对于语义块基线的 20/20 个配对工作点都变差；
- 粗空间构造使总耗时和峰值内存增加，不能宣称当前实现带来端到端加速。

因此，当前“局部矩阵最小奇异模态直接拼接成全局粗空间”的方案不应进入学习型提议器训练，也不应作为论文正面结果。

## 二、实现内容

新增：

- `pypath/preconditioner/semantic_coarse_space.py`
- `pypath/preconditioner/tests/test_semantic_coarse_space.py`

接入：

- `semantic_coarse_r1_sparse`
- `semantic_coarse_r2_sparse`

方法细节：

1. 局部层复用 `cell_core` 标准单元语义块 Schwarz；
2. 粗空间候选来自 `cell_core_plus_onehop_boundary` 块的局部有效矩阵；
3. 每块选最小奇异值对应的右奇异向量；
4. 共享边界按覆盖次数做分区统一；
5. 全局正交化后形成 \(Z\)；
6. 使用全阶加性校正：

\[
M_{\mathrm{two}}^{-1}r
=M_{\mathrm{Schwarz}}^{-1}r+
Z(Z^\mathsf{T}A_{\mathrm{eff}}Z)^{-1}Z^\mathsf{T}
\left(r-A_{\mathrm{eff}}M_{\mathrm{Schwarz}}^{-1}r\right).
\]

粗空间守卫包括有限性、秩、条件数和运行时输出检查；失败时回退到局部 Schwarz。

## 三、实验设置

输入为历史 PALS 数据，只读使用，不复制到 `wmpc`：

- 轨迹：`/home/ZhangLexin/PALS/pals_data/runs/warmup_migration_79cab9f07_full500_20260710/aggregator/trajectory`
- 网表：`/home/ZhangLexin/PALS/pals_data/runs/warmup_migration_79cab9f07_full500_20260710/aggregator/generated_netlists`
- 工作点清单：`/home/ZhangLexin/PALS/pals_data/runs/positive_gmin_full500_row_sum_screen_20260816/workpoint_manifest.json`

固定参数：

- 工作点：20；
- 方案：`row_sum`、`semantic_cell_core_sparse`、`semantic_coarse_r1_sparse`、`semantic_coarse_r2_sparse`；
- 初值：`rhsold`；
- GMRES 重启长度：100；
- 最大外层迭代：500；
- 相对残差阈值：`1e-8`；
- 绝对残差阈值：`1e-10`；
- 粗矩阵条件数上限：`1e12`；
- 并行度：4；
- 每个任务超时：120 秒；
- 线程数：1。

共 80 个独立任务，所有任务均正常结束，无超时。

## 四、主要结果

| 方案 | 严格成功 | 平均 GMRES | 中位 GMRES | 平均真实相对残差 | 平均构造时间（秒） | 平均求解时间（秒） | 平均总时间（秒） | 平均峰值内存（千字节） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `row_sum` | 20/20 | 26.0 | 26.5 | 2.93e-9 | 0.00019 | 0.00300 | 0.01108 | 385789 |
| `semantic_cell_core_sparse` | 20/20 | 8.5 | 8.0 | 9.51e-10 | 0.00521 | 0.00183 | 0.01483 | 386048 |
| `semantic_coarse_r1_sparse` | 20/20 | 21.7 | 22.0 | 2.85e-9 | 0.01159 | 0.00497 | 0.02430 | 386399 |
| `semantic_coarse_r2_sparse` | 20/20 | 21.7 | 22.0 | 2.95e-9 | 0.01250 | 0.00541 | 0.02573 | 386663 |

相对于 `row_sum`，语义块基线平均减少 GMRES 约 67.3%；粗空间一阶和二阶仅减少约 16.5%。相对于语义块基线，粗空间两种阶数平均增加 13.2 次 GMRES，20 个工作点全部变差。

## 五、粗空间守卫与内存

| 方案 | 守卫通过 | 平均实际秩 | 条件数平均值 | 条件数最大值 | 粗空间保留字节平均值 | 运行时回退 |
|---|---:|---:|---:|---:|---:|---:|
| `semantic_coarse_r1_sparse` | 20/20 | 17.0 | 36.28 | 144.74 | 49439 | 0 |
| `semantic_coarse_r2_sparse` | 20/20 | 34.0 | 126.98 | 144.74 | 104176 | 0 |

粗空间本身没有出现秩亏、非有限值或运行时回退。问题在于模态方向和校正强度，而不是数值守卫失效。

整进程峰值内存没有下降：粗空间一阶比语义块基线平均增加约 351 千字节，二阶增加约 615 千字节。该结果不能替代“整进程内存节省超过 50%”的目标。

## 六、流程审计

第一次 80 任务运行漏传 `--use-rhsold-as-x0`，实际使用零初值；该批结果已完整保存在：

`results/semantic_coarse_space_pilot_20260821_zero_x0_diagnostic`

该批只作为流程审计，不进入上述正式统计。随后使用 `rhsold` 重新完成 80 个正式任务。

单元测试：

```text
Ran 22 tests in 0.399s
OK
```

## 七、研究判断

当前负结果说明：把局部矩阵的最小奇异模态直接视为全局慢误差模式并不可靠。初步推断有三点：

1. 局部奇异模态描述的是块内代数方向，不一定对应跨单元的低频物理模式；
2. 重叠边界的局部模态经过简单分区后，仍可能破坏全局耦合比例；
3. 未学习或校准的加性粗校正强度可能改变原有 Schwarz 的有效谱结构。

下一步若继续，应先做一个新的、独立确认的先导：比较常数/图拉普拉斯低频基、误差快照 POD 基和现有接口 Schur 低秩基；只有其中一种在固定工作点上优于 `semantic_cell_core_sparse`，才值得继续做提议器学习。直接无源 MOR 暂不启动。

## 八、追溯信息

- 实验时 `HEAD`：`8113c4e5cd509288f5a6d54ec445a5ff49976a99`；
- Python：`/home/ZhangLexin/miniconda3/envs/PALS_env/bin/python3`，版本 3.12.3；
- 工作点清单哈希：`f8611b48b2692d602758d849b1bba59e27f69d0d6a99956fc96444a4294702cd`；
- 线路 0 网表哈希：`a7e3e7f66c2cfbcf19f2bcef743b1fe3a4e98455c5383931d54596975ab12d56`；
- 粗空间模块哈希：`ab1d41675a55ea871e8eab04aa8fbde1da206caa2859f86f00d9cd293e9ff17b`；
- 结果原始记录：`raw_rows.jsonl`，80 条；
- 结果汇总：`summary.json`；
- 零初值诊断批次：`results/semantic_coarse_space_pilot_20260821_zero_x0_diagnostic`，不纳入正式统计。

实验期间工作树另有与本实验无关的未提交文件；本轮未重置、覆盖或提交这些文件。
