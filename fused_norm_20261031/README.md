# RMSNorm融合：保留两条反向边

实际产物：`/private/yp/jiuge-training/checks/20261031-fused-norm/`。固定四卡BI-V150、九格4B、DP1/TP2/PP2、BF16、S512、MBS1/GBS16、seed1234、关闭重计算。

`fused_split_norm.py`前向统计与输出合成一次Triton发射，反向行VJP与独立BF16 direct/FP32 variance输出合成一次发射。外部`x.float()`的autograd边保留；dw仍是原三kernel。前向借用已有R缓冲区暂存并volatile读取FP32均值，防止融合后跨舍入边界优化；没有独立mean缓冲区。反向无GW/DOT/coefficient临时张量。

前三个诊断prototype出现统计量/舍入差异，源码、报告、编译IR均保留；修正版在125个实际捕获VJP和8个固定合成样本上，输出、direct、FP32 variance、dw、残差及冻结权重dx与当前Triton逐元素一致。该局部检查不替代整模型原门槛。

`fused_numeric_training.py`复用无损引用存储及完整审计；`compare_fused_norm.py`仅从既有审查器替换stage和candidate路径，计算及判据不变。`run_fused_numerical.py`复制原protocol并事先记录哈希，审查LoRA/CPT完整10步，还核对捕获张量内容SHA与当前reductions-full-v1一致。历史reference及其引用链必须保留。

数值通过后才运行`profile_fused_seams.py`确认两种任务分支的实际kernel路径，再运行`run_fused_pairs.py`。每任务四对独立启动、相邻current/fused交错，顺序AB/BA/AB/BA；每组5步预热+10步计时。两臂均使用相同轻量host阶段计数，移除全量审计和checkpoint写盘；profiler单独运行。`current`是reductions-full-v1，不是原始仓库未验收融合后端。

`summarize_fused.py`检查同对配置与15步训练标量一致、汇总配对差值和host调用成本。测速是探索性同配置train_step对照，不是用户正式端到端规范验收。源码与Megatron固定checkout不修改；所有新脚本在远端tools和本地镜像中保存。
