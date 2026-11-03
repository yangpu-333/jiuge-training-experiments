# RMSNorm融合验证与配对测速（2026-10-31）

固定四卡BI-V150、九格4B、DP1/TP2/PP2、BF16、S512、MBS1/GBS16、seed1234、关闭重计算；CoreX4.4/Torch2.7.1/Triton3.1沿用当前环境。当前版为reductions-full-v1，融合候选为fused-split-v1。

结果：LoRA/CPT通过未放宽的原整模型数值门槛，捕获张量内容SHA与当前版一致；每任务四对测速均为融合版更快，平均吞吐分别改善2.81%/2.62%。norm前向host调用合计分别下降约49.5/48.6ms/步。峰值allocated显存不变：LoRA约5.952GiB，CPT约22.018GiB。该收益仅限本次探索性配对测量。

## 实现与舍入边界

前向统计量与输出合成一次Triton发射；反向行归约与BF16 direct、FP32 variance双输出合成一次发射。外部`x.float()`输入边保留，框架分别累加两条反向分支；norm weight梯度原三kernel保持。删除独立mean、GW、DOT、coefficient临时张量。

前三个局部prototype出现很小的统计量舍入差异，没有进入整模型实验。缩小布局和显式RN算术未消除差异。最终版在同一前向kernel内借用既有512元素FP32 R缓冲区物化mean，经barrier及volatile读取后计算r并覆盖R；没有增加独立mean分配。观察到的差异随该物化边界消失；不据此断言唯一的底层编译优化原因。所有失败prototype、原始指标与编译IR保留。

## 数值验收

原protocol逐字复制、SHA一致，审查器只调整产物路径。133组局部样本（125组真实捕获+8组固定合成）的输出、双梯度、dw、残差及冻结权重dx，与当前Triton版逐元素一致。随后LoRA/CPT分别进行完整10步输入/RNG/初始状态控制、完整梯度、累计FP32更新和最终模型审查。

|任务|step1完整梯度relL2|step10完整梯度relL2|累计更新relL2|最终模型relL2|原判据|
|---|---:|---:|---:|---:|---|
|lora|0|0|0|0|通过|
|cpt|4.9393302e-07|2.8413032e-07|3.3226542e-06|0|通过|

两任务捕获的初始/最终主参数、step1/10梯度schema内容SHA及最终命名模型参数SHA均与当前reductions-full-v1一致，无新增捕获数值漂移。LoRA冻结基座未变；CPT对reference的微小残差沿用当前版已有norm weight梯度路径。存储继续使用无损reference复用，全元素重建审查；未重复写原生optimizer checkpoint，必须保留引用链。

## PyTorch Profiler路径核验

独立实际形状`[512,1,2560]`的前反向trace，包含两条输入边；LoRA冻结norm weight、CPT训练norm weight。此处是算子路径核验，不作为整模型吞吐测量。

|分支|当前版GPU kernel数|融合版GPU kernel数|
|---|---:|---:|
|lora|7|5|
|cpt|10|8|

融合版的`fused_statistics_output`与`fused_row_dual_dx`各调用1次，原`forward_statistics/given_r/backward_row/split_dx`调用均为0。原始小型trace位于`checks/20261031-fused-norm/operator-{current|fused}-{lora|cpt}.json`。

## 配对测速

每任务四对相邻独立启动，顺序AB/BA/AB/BA；每组预热5步、采样10步，共16组。两臂相同轻量host阶段计数，不带全量数值审计或profiler，实际命令移除checkpoint写盘。每步取四rank同步train_step耗时最大值，再取均值；吞吐按8192输入tokens/step计算（LoRA包含loss mask屏蔽的输入token）。

|任务|当前版均值(s)|融合版均值(s)|吞吐比 fused/current|融合更快的配对数|norm前向host ms/步 rank0（当前→融合）|
|---|---:|---:|---:|---:|---|
|lora|5.8626|5.7025|1.0281x|4/4|219.2 → 169.7|
|cpt|5.0539|4.9251|1.0262x|4/4|242.6 → 194.1|

|任务/配对|顺序|当前(s)|融合(s)|节省(ms)|吞吐比|
|---|---|---:|---:|---:|---:|
|lora/1|current→fused|5.8588|5.7258|133.1|1.0232x|
|lora/2|fused→current|5.8358|5.7162|119.6|1.0209x|
|lora/3|current→fused|5.8760|5.7012|174.8|1.0307x|
|lora/4|fused→current|5.8797|5.6667|212.9|1.0376x|
|cpt/1|current→fused|5.0664|4.9771|89.3|1.0180x|
|cpt/2|fused→current|5.0404|4.9043|136.2|1.0278x|
|cpt/3|current→fused|5.0655|4.9143|151.2|1.0308x|
|cpt/4|fused→current|5.0433|4.9049|138.4|1.0282x|

各配对有效配置完全一致、15步训练loss/grad_norm/skipped等标量一致；所有运行四rank完成、无跳步、loss/grad_norm有限，LoRA适配器有效更新。阶段host时间包含发射和内部等待，未另加阶段末设备同步，不能当GPU独占时间；它与整步指标也不能直接相加。

计时不含加载、step外日志和checkpoint写盘；包含train_step内前反向、通信和optimizer。本轮为固定配置探索性配对对照，四对样本不等于正式端到端验收或长训练/恢复验收。reference与当前Triton对照保留，原始仓库未验收融合实现未因此通过。

## 入口与证据

本地镜像：[实现与实验入口](fused_norm_20261031/README.md)、[完整配对与数值汇总](results/fused-norm-summary-20261031.json)。原protocol SHA256为`a49da90bbcfe5cc67bbe97047f3b0008c09e7fcf8261174e90803f7612371fce`。

数值入口`tools/fused_numeric_training.py`配合本轮runner；测速入口`tools/fused_performance_training.py`，`PERF_ARM=triton`且`PERF_FUSED=1`选择融合，`PERF_FUSED=0`选择当前版。每rank有`fusion-install-rankN.json`记录实际安装与代码SHA。原始CLI不会自动安装外部候选。

协议/审查/配对数据均在`checks/20261031-fused-norm/`；完整运行位于`runs/4b-{task}-tp2-pp2-triton-no-recompute-s512-gbs16-fused-split-v1`及`runs/4b-{task}-tp2pp2-s512-fused-pair{1..4}-{current|fused}-v1`。汇总为`fused-summary.json`；原数值门槛和PP同步均未更改。
