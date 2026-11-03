# 融合RMSNorm版整模型trace

远端`/private/yp/jiuge-training/checks/20261101-fused-fulltrace/`，固定四卡4B/TP2PP2/BF16/S512/MBS1/GBS16/关闭重计算。两任务各5步预热+1步四rank PyTorch Profiler，八份trace；不记录shape/stack/memory，形状由模块metadata提供。

`annotate_lora_trace.py`只在采集步添加前向record_function，LoRA算术保持原顺序，dropout限定当前0；base/A/TP/B/scale/add分开。反向没有hook或新Function；分析器通过现有序列号追溯。采集后六步训练标量逐rank对照已验收融合版，检查LoRA更新有效。

`analyze_lora_components.py`按CPU嵌套区间标记前向、Sequence number追溯反向，runtime correlation关联普通GPU kernel。NCCL异步launch无External id时，用GPU NCCL注释同stream区间+External id追溯CPU标记。dropout=0的无反向节点操作排除出序列映射。每rank验证A/B各1984次、低秩TP通信1984次，无歧义序列和CPU交叉。

Host self是记录区间减子区间，含等待和调度；跨线程可能重叠。GPU累计时间也不等于关键路径可省时间。profiler与组件标记有显著扰动，仅用于归因，不替代无profiler配对测速。

失败分析器及中间结果保留于远端stage，原trace不改写。主要结论见上层`FUSED_FULLTRACE_20261101.md`。A/B autograd简化的局部筛查另存`checks/20261101-lora-linear/`；Python版本局部数值对齐但隔离速度倒退，暂未进入整模型验证。
