# 仓库内容与复现边界

本仓库保存九格4B四卡BI-V150训练优化代码和证据。首页为[本周报告：11月4日—11月10日](WEEKLY_REPORT_20261110.md)，最新CPT固定50步完整任务吞吐+16.38%、耗时−14.08%。

## 最新端到端结果与复现

| 内容 | 入口 |
|---|---|
| 离线复算及四卡重新测量命令 | [完整复现指南](REPRODUCE.md) |
| 六轮原始耗时、目标、统计区间、完整证据包 | [最新证据](results/e2e-20261110/README.md) |
| 18次正式运行的配置、命令、源码manifest和日志 | [运行记录](results/e2e-20261110/runs) |
| 本轮使用的入口、算子及依赖源码 | [源码目录](reproduction/20261110/tools)、[28份工具SHA256](reproduction/20261110/TOOLS_SHA256.json) |
| 新增GPU分块状态组装 / direct DP1 | [coalesced](reproduction/20261110/tools/coalesced_state_20261109.py)、[direct](reproduction/20261110/tools/dp1_checkpoint_pack_20261107.py) |
| 融合RMSNorm及双反向边 | [dw融合](reproduction/20261110/tools/fused_norm_dw.py)、[split norm](reproduction/20261110/tools/fused_split_norm.py)、[原生归约次序](reproduction/20261110/tools/native_order_reductions.py) |
| SwiGLU融合和绑定 | [一次性绑定](reproduction/20261110/tools/swiglu_fast_patch_20261106.py)、[构建器](reproduction/20261110/tools/swiglu_order_patch_20261105.py)、[C++](reproduction/20261110/tools/swiglu_cpp_20261105.cpp)、[CoreX](reproduction/20261110/tools/swiglu_order_20261105.cu) |
| 适配local基线 | [选择器](reproduction/20261110/tools/local_spec_ablation_patch.py)、[语义适配](reproduction/20261110/tools/local_spec_patch.py) |
| 完整保存及原生恢复 | [共同封装](reproduction/20261110/tools/checkpoint_round_common_v2_20261109.py)、[恢复审计](reproduction/20261110/tools/integrity_coalesced_v2_20261109.py) |
| 环境与输入来源、哈希 | [环境](reproduction/20261110/environment/environment.lock.json)、[模型](reproduction/20261110/inputs/MODEL_SOURCE.json)、[数据](reproduction/20261110/inputs/data-manifest.original.json) |
| 发布完整性检查 | [manifest](publication-manifest.json)、[发布验证](scripts/validate_publication_20261110.py) |

工具源码保持实测字节。训练框架及Megatron-LM从公开固定commit取得，本仓库提供全部本轮算子补丁和运行入口。大权重及checkpoint不上传，按指南从各自上游取得；数值及恢复检查由新鲜环境重新执行。历史归档和诊断脚本作为来源链保留，不是新实验入口；`checkpoint_diagnostic`仅使用其fingerprint辅助函数，不运行其历史主程序。

## 历史训练步时资料

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

历史目录是各次关键补丁快照，最新完整任务复现请使用上方指南和`reproduction/20261110`，不要混用旧轮次入口。所有结果均保留各自计时边界；不代表对完整未经适配官方原生RMSNorm模型的有效加速，也不代表长期收敛验收。最新CPT结论不替代LoRA新目标验收。

具体源码版本和运行条件见统一配置与各实验报告。引用不同轮次的单项收益时应保留各自基线，不能相加。


## 2026-11-16 八卡结果

- [展示首页](README.md)
- [三配置完整报告](REPORT_20261116.md)
- [八卡重跑与离线复算](REPRODUCE_20261116.md)
- [源码、配置和输入指纹](reproduction/20261116)
- [公开证据包SHA](results/20261116/PACKAGES.json)
