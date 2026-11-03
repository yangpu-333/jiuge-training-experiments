# 三版 trace 归因与下一步优化（2026-10-31）

当前结论：减少的主要是小型逐元素/归约kernel，但LoRA的低秩线性层、TP通信、PP同步和主机调度没有随之减少；同时完整Triton版的norm前向主机调用变慢，抵消了部分GPU侧节省。CPT有前反向收益，但收益量级接近跨轮波动，尚不能认定完整Triton归约稳定优于混合版。

本轮只读重分析原24份trace，另完成reference/full端点的LoRA/CPT四组阶段补测。所有实际产物在 `/private/yp/jiuge-training/checks/20261031-trace-diagnosis/`；固定四卡BI-V150、4B、DP1/TP2/PP2、BF16、S512、MBS1/GBS16、seed1234、关闭重计算。CoreX4.4、Torch2.7.1、Triton3.1沿用上一轮；没有改模型算法、并行拓扑或PP同步设置。

## 1. LoRA的主要工作没有被norm替换减少

下表是带profiler的单步、rank0分类；“kernel累计时间”不是整步wall time。

|指标|LoRA reference|LoRA mixed|LoRA full|CPT reference|CPT full|
|---|---:|---:|---:|---:|---:|
|整步kernel数|93,999|90,063|79,279|75,667|61,779|
|GEMM kernel数|18,816|18,816|18,816|8,928|8,928|
|GEMM累计时间(ms)|1,195.7|1,190.6|1,190.0|950.6|943.1|
|通信kernel数|4,053|4,053|4,053|2,086|2,086|
|通信活跃区间并集(ms)|1,364.0|1,161.5|1,186.0|1,220.0|1,008.6|
|PP相关cudaDeviceSynchronize次数|17|17|17|17|17|

rank2/3也保持相同结构：LoRA GEMM18,880次、通信4,117次，CPT GEMM8,976次、通信2,134次。三版的这些计数都没有改变。

对应实现为固定源码 `source/src/jiuge_training/lora.py:ParallelLoRA.forward`：基座linear后，另外执行A/B低秩linear、scale和add，并按column/row语义执行低秩TP copy-backward或reduce-forward。冻结基座参数仍需传播输入梯度。因此，LoRA参数少并不意味着当前训练实现的kernel或通信更少；本配置的LoRA GEMM与通信次数约为CPT两倍。norm替换没有优化这些路径。

## 2. GPU执行量下降，时间线空档基本没变

使用全部kernel、memcpy、memset的区间并集计算GPU活跃时间，避免重复相加；设备窗口覆盖CPU train_step标记至最后设备事件结束。

|任务/版本|trace窗口(ms)|GPU活跃并集(ms)|无已记录设备活动(ms)|计算/通信重叠(ms)|
|---|---:|---:|---:|---:|
|LoRA reference|10,504.7|4,003.1|6,501.6|0.0|
|LoRA mixed|10,146.0|3,634.7|6,511.3|0.0|
|LoRA full|10,019.8|3,539.6|6,480.2|0.0|
|CPT reference|8,413.3|4,024.4|4,388.9|0.0|
|CPT mixed|8,731.4|3,645.5|5,085.8|0.0|
|CPT full|7,816.7|3,487.3|4,329.4|0.0|

以上只代表profiler扰动后的单步，**不能把LoRA约6.5秒的trace空档当作无profiler运行中可消除的时间**。各rank的计算/通信并集交集在这24份trace中均为0；当前执行没有展示计算通信重叠。`CUDA_DEVICE_MAX_CONNECTIONS=1`沿用现有环境，但未经独立实验，不能把无重叠单独归因于该变量。

CPU launch与GPU事件按correlation关联后，LoRA full至少约1,157ms空档具有“下一次launch在空档开始之后才发生”的证据；reference约1,165ms，mixed约1,145ms。该指标只覆盖成功关联的下一事件，不覆盖全部空档，也不能区分Python、线程调度、driver背压或其他依赖。

PP同步的CPU调用路径为`nccl:coalesced`，固定MCore `megatron/core/pipeline_parallel/p2p_communication.py:416`在`batch_p2p_comm && batch_p2p_sync`时执行`torch.cuda.synchronize()`，源码明确用于规避批量收发的竞态。rank0的这些同步累计CPU等待为LoRA reference/mixed/full约886/705/769ms，CPT reference/mixed/full约701/632/605ms。**同步等待和通信kernel可能覆盖同一依赖，不能相加，也不能直接当作删同步就能节省的时间。**

## 3. 无profiler补测：LoRA收益被主机调用开销抵消

四组独立15步，前5步预热、后10步取均值；只增加轻量主机阶段计数，不额外插入设备同步。阶段时间包含发射和内部等待，未在各阶段末同步，不能当GPU独占时间。表内阶段为rank0；整步为四rank同一步最大同步耗时的均值。

|指标|LoRA reference|LoRA full|CPT reference|CPT full|
|---|---:|---:|---:|---:|
|同步整步(s)|6.0083|6.0016|5.3393|5.0509|
|前反向host区间(s)|5.9972|5.9911|5.2349|4.9494|
|optimizer.step host区间(ms)|8.9|8.2|99.6|99.2|
|norm前向host调用合计(ms)|151.6|225.9|177.0|241.3|
|单rank进程CPU时间(s)|6.2521|6.2424|5.5430|5.2460|

其他rank也观察到相同norm前向方向：LoRA reference151ms左右、full218–227ms；CPT reference177–185ms、full239–245ms。也就是说，GPU kernel变少的同时，norm前向调用合计增加约60–76ms/步。整步进程CPU时间并未在LoRA中实质下降，CPT则随前反向一起下降。CPU时间含进程内所有线程和driver相关执行，不能解释为纯Python时间或机器整体CPU占用。

这直接支持“GPU小算子节省被主机调用成本部分抵消”的解释；但本轮没有进一步把norm调用时间拆成纯Python、allocator和driver各自耗时。源码中full路径仍有custom autograd包装、逐次assert、临时tensor创建与两次前向Triton launch。trace中FullReductionRMSNormBackward记录区间的CPU self合计约317ms，比mixed约196ms更大；包含子操作的总区间却由mixed536ms降至full407ms，说明主机路径的成本结构发生变化，不能仅按kernel数推测收益。

LoRA的optimizer只有约9ms，不是当前优先瓶颈；CPT这次改善也主要来自前反向，optimizer约100ms基本不变。四组补测均检查有效配置、四rank完成、无跳步、LoRA有效更新，15步loss/grad_norm与历史同臂结果全部一致。没有用标量一致替代已有的全量数值判据。

## 4. CPT为何还不能称稳定收益

原三轮full步时均值为5.0639、5.3268、5.1572秒，跨轮范围263ms；reference范围35ms、mixed范围200ms。full与reference三轮总均值差145ms，与mixed仅差42ms，均小于候选的跨轮范围。新阶段补测中reference/full为5.3393/5.0509秒，进一步表明收益估计对采样窗口敏感，不能用这一组替换原三轮结果。

trace中CPT full的GEMM数量与执行量基本不变，优化来自更小的归约/逐元素与主机路径，通信等待仍占明显区间；这类小收益容易受主机调度和跨rank等待变化影响。已有数据能确认“改动没有消除这些波动路径、候选波动量级足以淹没小收益”，**不能唯一证明原第二轮变慢由某次CPU抢占、GIL、driver或网络异常导致**，原计时轮次没有同步记录足够的CPU/通信阶段遥测。后续应以相邻配对采样验证，而不是从单份带profiler的trace硬推原计时抖动原因。

此外，full相对reference的收益包含此前split-v4的变化，不全是本次统计量/行归约替换贡献；评价新增替换应以mixed作直接对照。

## 5. 下一步的具体顺序与验收

1. **先做保留反向图的norm融合候选，目标是降低主机调用与临时内存成本。** 前向把`forward_statistics + given_r`合成一次launch，保留外部`x.float()`边；反向把`backward_row + split_dx`合成一次launch，仍返回独立BF16 direct与FP32 variance。去掉只用于诊断而生产未消费的mean/dot中间张量写回和对应临时分配。保持既定原生归约顺序、BF16中间舍入、FP32 reciprocal乘法及FMA设置；不把两条反向结果合并。
2. **LoRA整模型优化优先转向低秩分支。** 单独测A/B小GEMM、scale/add以及低秩TP collective的CPU/设备成本；先试保持原舍入次序的scale/add融合和减少包装/发射，再评估低秩小张量collective的合并或调度。TP2PP2、rank8/alpha16/dropout0保持不变；通信布局变化同样要重新走整模型数值门槛。
3. **PP同步与计算通信重叠单列兼容性课题。** 本轮不更改。若后续验证`batch_p2p_sync`或非batch P2P路径，要在当前CoreX4.4/Torch2.7.1上先做四rank收发完整性、重复步和超时/挂起检查，再做严格整模型数值对照。同步存在源码竞态依据，不能因版本号新或trace等待大就直接删掉。
4. **测速改为相邻配对，并记录阶段遥测。** 在同任务内按mixed/full/full/mixed等顺序交错，多次独立启动，保持样本、seed、预热和有效步数一致；报告配对差值与跨启动范围。主机阶段/进程CPU/上下文切换作归因辅助；profiler另跑。正式端到端标准仍按用户后续规范。

建议下一次先实现第1项，作为当前数值已验收候选的局部迭代；LoRA的更大收益空间则在第2项。验收顺序是：既有局部/VJP/残差舍入样本检查→同配置CPT/LoRA完整10步原数值门槛→不带审计的配对测速。只有host调用和同步整步实际改善才记收益，kernel数减少只是辅助指标；不预先承诺百分比。

## 证据与复现

- 上轮原始证据：[三版性能报告](PERFORMANCE_COMPARISON_20261031.md)，源summary SHA256为`efa490cbe17506d846ca407d2fc0830078f2d0247c018d7915eec475f2aa427b`。
- [诊断汇总JSON](results/trace-diagnosis-summary-20261031.json)、[24份时间线汇总](results/trace-timeline-summary-20261031.json)；远端同名内容分别为`checks/20261031-trace-diagnosis/diagnosis-summary.json`、`timeline-summary.json`。逐文件保留原trace SHA并校验，不重写原trace。
- [分析与补测代码](trace_diagnosis_20261031/README.md)。CPU嵌套解析的24份结果均无区间交叉；区间并集的重叠、相邻、空区间样例检查通过。CPU self按线程计算，不能跨线程相加成wall time。
- 第一版阶段包装在首步前因未转发`schedule_pg_collection`退出；失败日志、协议与源码副本保留，v2独立目录重做四组。补测时暂停本次CPU解析进程，结束后恢复，相关记录均保留。没有删除历史证据或checkpoint引用链。
