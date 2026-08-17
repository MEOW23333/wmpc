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

    /home/ZhangLexin/miniconda3/envs/PALS_env/bin/python3 -m pytest pypath/preconditioner/tests/test_sparse_learned_schwarz_v1.py

原生程序重新编译时使用本目录的构建骨架，生成的二进制放在 release/src/ngspice。实验先决条件写在 docs/实验计划_2026_08_17.md。

## 研究边界

WMPC 当前是 稀疏预条件子已实现、联合端到端收益待补证的研究版本。不得把预条件器增量内存节省写成整进程内存节省，也不得把历史 warmup 结果写成当前严格复现实验结果。
