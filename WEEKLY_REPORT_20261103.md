# 九格训练优化周报

统计周期：2026-10-28—2026-11-03；主要实验集中在 10 月 30 日至 11 月 2 日。

## 本周完成的工作

- **统一实验配置与复现条件。** 采用 [jiuge-training](https://github.com/ZhouBencheng/jiuge-training) 训练框架，固定源码 `9fabd71df4897f08267df5431aa480ab9ef70a75`，使用九格 4B、四卡 BI-V150、DP1/TP2/PP2、BF16、S512、MBS1/GBS16、seed 1234，关闭激活重计算。实验、依赖、日志和 trace 均保存在 `/private/yp/jiuge-training/`。
- **完成 RMSNorm 数值对齐及融合。** 原始 Triton 路径未通过整模型判据；定位到 BF16 舍入及反向梯度累加顺序后，保留 `x` 与 `x.float()` 两条反向边，逐项将统计量和行归约迁移到 Triton，再融合前向统计/输出和反向行归约/双梯度输出。LoRA、CPT 均通过原十步整模型判据，未放宽阈值。
- **优化 LoRA 与 CPT 执行路径。** LoRA 引入 C++ A/B autograd 和 A+低秩 TP 路径合并，降低主机调用与计算图开销；CPT 融合 norm weight 梯度 products。已选候选完成对应短程数值及原生 checkpoint 恢复检查。
- **建立 trace 和配对测速流程。** 使用 PyTorch Profiler 分析四 rank 时间线；性能另以无 profiler、无全量审计的入口测量。保留失败候选：scale/add 融合未形成一致整步收益，后续 dw 归约合并也未达到预设稳定性条件，均未推广。
- **核验 Megatron 原生算子基线。** 固定 MCore commit `5be9626709af2722333bf54797c954c09edeada3`。完整原生 RMSNorm/attention 对照在 LoRA、CPT 上均未通过原数值门槛；LoRA 消融将差异定位到 RMSNorm 路径。另建立“原生 attention + 九格 reference RMSNorm + 九格语义适配”的可比基线，两任务均通过原判据及与优化版的直接数值检查。

## 性能结果

最新对照为上述**适配基线**与当前优化版，每任务四组 AB/BA，每臂预热 5 步、测量 20 步；每步取四 rank 最慢同步 `train_step` 耗时。

| 任务 | 适配基线（秒/步） | 优化版（秒/步） | 速度提升 | 步时下降 |
|---|---:|---:|---:|---:|
| LoRA | 6.1532 | 5.6704 | 8.51% | 7.85% |
| CPT | 5.2059 | 4.8885 | 6.49% | 6.10% |

两任务四组均为优化版更快，各步 loss 按原门槛通过，测量窗口未记录到二代 GC。结果是固定配置下的探索性训练步时，尚未按正式端到端规范验收；也不能称为相对完整原生 local spec 的有效加速。

单项实验曾测到：RMSNorm 融合使 LoRA/CPT 速度提高约 2.8%/2.6%，LoRA C++ A/B 约 2.5%，A+TP 合并约 2.2%，CPT products 融合约 0.8%。这些实验使用不同阶段的直接基线，**百分比不能相加**，总收益以上表为准。

## 性能分析与复盘

1. **主要收益来自归一化融合和主机执行路径简化。** LoRA A+TP 合并后的 trace 中，GPU GEMM、通信 kernel 数量未变，每 rank autograd 节点减少 1,984 个；说明收益不只来自 GPU kernel 数量下降。
2. **剩余大块成本尚未改变。** 大 GEMM、attention、TP 通信和 PP 同步继续执行。早期 trace 中 LoRA rank0 每步约有 18,816 个 GEMM kernel、4,053 个通信 kernel；这些是当时采样计数，不是当前整步的精确时间占比。小算子局部加速会被整模型其他工作摊薄。
3. **融合必须实际降低调用成本。** 早期 Triton 归约版虽减少 kernel，norm 前向主机调用反而多约 60–76 ms/步；后续融合才改善该路径。scale/add 融合也出现“kernel 减少但整步收益不稳定”，不能只凭 kernel 数选方案。
4. **测量波动限制了小收益的可信度。** LoRA 最新测速期间节点 IO 压力较高，且监测未覆盖第一组开头；节点 PSI 不能视为本训练进程的等待比例。CPT 后续归约合并候选未达到预设稳定性条件，说明亚百分比收益需要谨慎判断。
5. **数值一致性是有效对照的前提。** 完整原生 RMSNorm 对照与九格 reference 在 CPT 第 1/10 步梯度 relative L2 约为 58.80%/20.97%，累计主参数更新约为 39.96%；这些是当前短程训练路径差异，不能据此判断长期模型质量。适配基线通过检查后，才进行了上表的性能比较。

## 下一步计划

| 优先级 | 工作 | 验收方式 |
|---|---|---|
| P0 | 分析原生 RMSNorm 与九格路径的精度转换、权重相乘及反向舍入边界，明确完整原生基线的必要适配 | 固定真实输入的前反向检查，再做 LoRA/CPT 原十步整模型判据；不放宽门槛 |
| P0 | 在 IO 压力较低的窗口复测当前适配基线与优化版，补齐阶段遥测 | AB/BA 配对，保留全部样本，记录节点压力、每 rank 步时和 GC；报告波动范围 |
| P1 | 依据当前版本 trace 优先筛查 LoRA A/B 小 GEMM及低秩通信调用成本 | 局部筛查 → 原整模型数值检查 → 配对测速；按实际整步改善选候选 |
| P1 | 验证 PP 通信调度与计算通信重叠的可行性 | 先验证四卡收发、重复步和超时稳定性，再做整模型数值及性能检查 |
| P2 | 完善端到端测量规范，补充更长训练与吞吐测量 | 明确加载、数据、保存等计时边界，保持基线和候选同口径 |

本周已形成可复现的数值检查、trace 和候选筛选流程，并取得约 6%–9% 的整步速度提升。下一阶段优先解决基线语义和剩余主机/通信成本，避免继续围绕微小 kernel 数量变化优化。

## 相关记录

- 当前基线边界与结果：[LOCAL_SPEC_BASELINE_PLAN_20261102.md](LOCAL_SPEC_BASELINE_PLAN_20261102.md)。
- 详细实验汇总： [local_spec_baseline_20261102/README.md](local_spec_baseline_20261102/README.md)。
- 优化和 trace：[FUSED_RMSNORM_20261031.md](FUSED_RMSNORM_20261031.md)、[LORA_TP_20261102.md](LORA_TP_20261102.md)、[NORM_DW_20261101.md](NORM_DW_20261101.md)、[TRACE_DIAGNOSIS_20261031.md](TRACE_DIAGNOSIS_20261031.md)。

- 仓库内容与复现边界：[REPOSITORY_CONTENTS.md](REPOSITORY_CONTENTS.md)。
