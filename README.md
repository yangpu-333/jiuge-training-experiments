# 九格4B训练与微调优化

在八卡BI-V150、CoreX5.0上，完成九格4B三种并行配置的端到端对照与数值、保存恢复验证。**TP4/PP2 CPT吞吐提升16.73%、完整任务耗时减少14.33%，均值和95%区间下界均达到目标。**

固定条件：BF16、S512、MBS1/GBS16、DP1、seed1234，关闭激活重计算。每次50步，计时覆盖启动、加载、训练、完整checkpoint保存、fsync和进程退出。数据预处理、预先编译和退出后的独立校验不计时，正式测试不启用Profiler。

| 并行配置 | 任务 | 吞吐提升 | 耗时减少 | 目标状态 |
|---|---|---:|---:|---|
| TP2/PP4 | CPT | 15.42% | 13.36% | 均值达标，区间未达标 |
| TP2/PP4 | LoRA | 15.65% | 13.53% | 未达标 |
| TP4/PP2 | **CPT** | **16.73%** | **14.33%** | **均值及区间达标** |
| TP4/PP2 | LoRA | 11.87% | 10.61% | 未达标 |
| TP8/PP1 | CPT | 8.52% | 7.85% | 未达标 |
| TP8/PP1 | LoRA | 8.81% | 8.10% | 未达标 |

目标：CPT吞吐+15%／耗时−12%，LoRA吞吐+17%／耗时−15%。表中均相对各布局同轮的**适配local基线**：原生attention配九格reference RMSNorm及必要语义适配，不是完全未经修改的官方模型。保留全部慢样本，整体目标尚未全部完成，不直接外推长训练收益。

优化组合覆盖RMSNorm、SwiGLU、LoRA线性与TP路径、CPT状态组装。新增编译池配置在TP2/PP4三臂中分别带来CPT +4.45%、LoRA +4.01%的稳定端到端增量，主要减少主机启动和退出等待；不能把其他布局的总收益全部归给该设置。

本轮共84次正式任务、14项十步整模型数值审计、12组原生保存恢复对照；19,200组配对rank-step检查和统计均经独立CPU复算。公开证据包已去除内部账号、容器和SSH标识，张量指纹及数值保持不变。

## 复现入口

```bash
git clone https://github.com/yangpu-333/jiuge-training-experiments.git
cd jiuge-training-experiments
python reproduction/20261116/verify.py --full
python scripts/check_links.py
```

普通电脑可核验包和成员SHA、整模型指纹、恢复状态、逐步标量及20,000次bootstrap。GPU重跑需要同型号八卡CoreX5环境以及原模型/语料；完整代码、固定来源、校验值、环境和执行命令见[八卡复现指南](REPRODUCE_20261116.md)。模型权重和语料正文按上游来源取得，不随代码仓库转发；运行结果会受存储与主机负载影响。

- [详细报告与目标差距](REPORT_20261116.md)
- [六组精确结果JSON](results/20261116/stage2-all-layouts-e2e-summary.json)／[CSV](results/20261116/stage2-all-layouts-e2e-summary.csv)
- [公开证据包及SHA](results/20261116/PACKAGES.json)
- [冻结训练源码和六组配置](reproduction/20261116)
- [历史四卡报告](WEEKLY_REPORT_20261110.md)／[历史四卡复现指南](REPRODUCE.md)

八卡结果统计截至2026年11月16日；发布整理于11月17日。历史四卡16.38%使用CoreX4.4与不同存储条件，独立保留，不与本轮跨环境拼接。
