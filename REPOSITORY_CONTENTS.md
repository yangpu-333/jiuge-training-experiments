# 仓库内容与复现边界

本仓库保存九格 4B 在四卡 BI-V150 上的训练优化记录、关键代码快照和本地已留存的结果摘要。首页为 [2026-11-03 周报](WEEKLY_REPORT_20261103.md)。

| 内容 | 入口 |
|---|---|
| 固定训练配置与测量约定 | [统一配置](UNIFIED_CONFIG_20261030.md) |
| 最新 LoRA/CPT 数值、配对性能与基线边界 | [实验汇总](local_spec_baseline_20261102/README.md) |
| Megatron 原生算子基线审查 | [基线分析](LOCAL_SPEC_BASELINE_PLAN_20261102.md) |
| RMSNorm 数值对齐与 Triton 迁移 | [数值修复](NUMERIC_FIX_20261031.md)、[归约迁移](TRITON_REDUCTIONS_20261031.md) |
| RMSNorm 融合 | [报告](FUSED_RMSNORM_20261031.md)、[实现](fused_norm_20261031/fused_split_norm.py) |
| LoRA C++ autograd | [报告](LORA_CPP_AUTOGRAD_20261101.md)、[实现](lora_linear_20261101/lora_linear_cpp.py) |
| LoRA A+TP 合并 | [报告](LORA_TP_20261102.md)、[实现](lora_tp_20261101/lora_tp_linear.py) |
| CPT products 融合 | [报告](NORM_DW_20261101.md)、[实现](norm_dw_20261101/fused_norm_dw.py) |
| 未推广的候选 | [scale/add](SCALE_ADD_20261101.md)、[dw 归约合并](norm_dw_reduce_20261102/README.md) |
| Trace 性能分析 | [分析报告](TRACE_DIAGNOSIS_20261031.md) |
| 短程原生恢复验证 | [恢复报告](CANDIDATE_NATIVE_RESTORE_20261101.md) |
| 仓库内相对链接检查 | [检查脚本](scripts/check_links.py) |

`results/` 保存报告引用的本地摘要；最新四组 LoRA/CPT 结果、原始产物位置与哈希记录在实验汇总中。模型权重、完整 checkpoint、大型原始 profiler trace 和远端全部运行目录仍保留在实验环境，未全部镜像到本仓库。

代码为各次实验的关键补丁与检查脚本快照，需结合固定版本的 jiuge-training、Megatron-LM、CoreX/PyTorch/Triton 及实验依赖运行；本仓库尚未整理为可独立安装的训练发行包。本文和周报的速度提升均指通过既定短程数值判据的适配基线比较，不代表对完整原生 RMSNorm 基线的有效加速，也不代表长期收敛验收。

具体源码版本和运行条件见统一配置与各实验报告。引用不同轮次的单项收益时应保留各自基线，不能相加。
