# Megatron local 算子对照：4B LoRA

配置：BI-V150 四卡，TP2、PP2、DP1，BF16、S512、MBS1、GBS16、seed 1234，关闭重计算。源码与审计边界见 [LOCAL_SPEC_BASELINE_PLAN_20261102.md](../LOCAL_SPEC_BASELINE_PLAN_20261102.md)。

## 完整原生算子的数值边界

在保留九格模型语义适配的前提下，采用 MCore local 的 `torch.nn.RMSNorm` 和 `DotProductAttention`，LoRA 十步审计相对九格 reference 未通过原数值门槛。step 1/10 全梯度 relative L2 分别约 20.51%/18.82%，累计 FP32 主参数更新约 15.80%；输入、RNG 与初始状态控制通过。优化版直接对比这套原生算子的原门槛也未通过，因此不能称下面的结果是相对完整原生算子模型的有效加速。

消融中，原生 attention + 九格 reference RMSNorm 与九格 reference 通过十步检查，所检查的梯度和主参数更新逐元一致；原生 RMSNorm + 九格 attention 未通过，四 rank 初始主参数、第 1/10 步梯度和最终主参数哈希与两种原生算子并用时一致。这一固定 LoRA 配置的差异定位到 RMSNorm 路径，不能推广到未经验证的配置。

## 通过数值门槛的适配基线对比

基线为 MCore 原生 `DotProductAttention`、九格 reference RMSNorm 和九格语义适配、原有 LoRA 线性实现；优化版为融合 RMSNorm 和 C++ LoRA A+TP/B 实现。两者十步直接数值审计通过，所检查梯度、累计主参数更新和最终模型逐元一致。此对照不能称为完整官方 local spec，收益也包含多项优化，不能全部归因于单个算子。

四组 AB/BA 配对，每臂预热 5 步、测量 20 步，以每步四 rank 最慢同步耗时为样本：

| 组 | 基线秒/步 | 优化版秒/步 | 基线/优化版 |
|---|---:|---:|---:|
| 1 | 6.128965 | 5.638097 | 1.0871× |
| 2 | 6.153797 | 5.698394 | 1.0799× |
| 3 | 6.089795 | 5.624157 | 1.0828× |
| 4 | 6.240233 | 5.720847 | 1.0908× |
| 平均 | 6.153197 | 5.670374 | 1.0851× |

平均步时减少约 7.85%，相同工作量的速度比约 1.085×。各步 loss 按原门槛通过，测量期间没有记录到二代 GC。结果为当前实验条件下的探索性训练步时，非正式端到端规范。

节点 IO PSI 监测在第一组已开始后才启动，采样到的 `some avg10` 中位数约 71%，范围约 8.7%–88%；这是节点级压力，并非本训练进程的 IO 等待比例，无法单凭该指标确定干扰来源或幅度。建议在较空闲环境复测。

远端原始汇总：`/private/yp/jiuge-training/checks/20261102-local-ablation/attention-only/performance/performance-summary.json`，SHA-256 `0f42ecc3630f2cbc3b343ac670a52f424260f7985453ba0bcd6125d10963ed70`。各 rank 原始记录、控制脚本、数值审计及 IO 采样均留在 `/private/yp/jiuge-training/`。

## CPT 完整原生算子审计

相同四卡配置下，完整原生 RMSNorm 与原生 attention、加九格语义适配的 CPT 十步训练及无损快照导出完成。相对九格 reference，输入、初始状态和历史 reference 控制通过，但原数值门槛未通过：40 个 rank-step 中 32 个 loss/gradient-norm 检查未通过；第 1/10 步全梯度 relative L2 为 58.80%/20.97%，累计主参数更新为 39.96%，最终完整模型为 0.0431%。逐元素读取及哈希校验证据见 `/private/yp/jiuge-training/checks/20261102-official-local-cpt/numerical/cpt-comparison.json`，SHA-256 `4aa81f94a5187ed4c526b7050f4fdea44abf5abdc7e13ee4371126c34b94b12b`。

原生算子与优化版直接复核也未通过：以原生算子为分母，第 1/10 步梯度 relative L2 为 52.62%/20.46%，累计主参数更新为 39.96%，最终完整模型为 0.0431%，输入和初始状态控制通过。直接复核结果为同目录 `cpt-direct-comparison-v2.json`，SHA-256 `06408e11ef60adfcd06cf29031c7aa63c98ca417dcc8337d5511abb555eb4ee1`。当前不能报告相对完整原生基线的有效性能提升；CPT 的适配基线必须先独立通过原数值判据。

## CPT 适配基线数值与测速

原生 attention、九格 reference RMSNorm 的 CPT 适配基线相对九格 reference 通过原十步门槛：第 1/10 步完整梯度、累计主参数更新、最终模型均逐元素一致。适配基线与融合 RMSNorm 优化版直接比较也通过原门槛：第 1/10 步梯度 relative L2 分别为 `4.94e-7`/`2.84e-7`，累计主参数更新为 `3.32e-6`，最终模型逐元素一致。审计路径 `/private/yp/jiuge-training/checks/20261102-adapted-local-cpt/numerical/`，两份结果 SHA-256 依次为 `d1294d8d844d59d456abc0a6bbf4f246b5701e692f16f47bd7291c4e94f03d83`、`9e63fc1d6b969de1f1b77c5b94990ed3bf91d9e3134db97b90183f3bedd0f262`。

四组 AB/BA 配对、每臂预热 5 步并测量 20 步，按每步四卡最慢同步耗时统计：

| 组 | 适配基线秒/步 | 优化版秒/步 | 基线/优化版 |
|---|---:|---:|---:|
| 1 | 5.202214 | 4.910841 | 1.0593× |
| 2 | 5.165138 | 4.893448 | 1.0555× |
| 3 | 5.240658 | 4.875663 | 1.0749× |
| 4 | 5.215403 | 4.874214 | 1.0700× |
| 平均 | 5.205853 | 4.888542 | 1.0649× |

平均步时减少约 6.10%，速度比约 1.0649×；loss 原门槛通过、测量期间无二代 GC。该结果是原生 attention 加九格 RMSNorm 的适配基线，不能称为相对完整原生 RMSNorm 的有效加速；这是探索性训练步时，非正式端到端规范。汇总路径 `/private/yp/jiuge-training/checks/20261102-adapted-local-cpt/performance/performance-summary.json`，SHA-256 `4ba300300345ca18b175f9bb6d1f216666efad899c700da34d982f94a7d96cc0`。
