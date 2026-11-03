# C++ A/B与新增epilogue整模型trace（2026-11-01）

固定四卡BI-V150、4B、TP2PP2、BF16、S512、MBS1/GBS16、关闭重计算。当前C++ A/B+融合norm与新增scale/add融合各5步预热+1步四rank PyTorch Profiler，共8份trace。两臂预热末统一full GC、单记成本，自动GC保留；采样步均无gen2事件。六步训练标量与已验证基线一致。

## kernel变少，但设备epilogue时间没有稳定改善

|rank|当前kernel数|融合kernel数|当前scale+add GPU时间和ms|融合epilogue GPU时间和ms|
|---|---:|---:|---:|---:|
|0|77311|75327|46.099|48.771|
|1|77294|75310|50.252|51.485|
|2|78290|76306|46.672|49.988|
|3|78273|76289|53.505|48.550|

每rank准确少1984次发射，融合kernel的1984次都完整归入epilogue。但3个rank的epilogue GPU时间和反而更高，单次trace不能证明稳定回退，也没有稳定设备收益证据。可进一步检查自定义标量kernel的访存/转换路径，本轮没有硬件计数器证据确认具体瓶颈；不能只凭合成一次发射就假设快于两个原生vectorized算子。

rank0 A/B GPU时间和405.37→407.21ms，GEMM数量及形状没有减少。B backward仍约219.7/220.8ms，是A/B中最大项。低秩TP每rank仍1984次、8KiB逻辑payload，rank0其GPU时间和118.95→161.90ms。collective kernel时长包含等待对端与流时序，且profiler会扰动rank间进度；不能把这次变化认定为链路带宽回退、可直接去除的通信成本或融合版实际慢步的唯一原因。

不带profiler的两对LoRA为+0.50%和-0.86%，合并约-0.18%吞吐，CPT未变算法负对照约-0.69%。结合[配对报告](SCALE_ADD_20261101.md)，保留C++ A/B基线，不推广新增epilogue。下一项先筛查CPT norm products并入逐行反向、保留两级dw归约，再视新证据评估低秩TP主机路径。

## 归因完整性

C++ forward的ATen事件没有sequence，而backward CppNode有；在采样步读取输出grad_fn序号并写profiler-only marker，已用隔离profile核验。每rank A/B forward各1984、backward CppNode各1984、低秩通信1984均覆盖；CPU区间crossing和歧义sequence均为0。

v1分析发现原生C++ `cudaLaunchKernel`有correlation与CPU scope，但无External id，导致融合kernel未归因。v2保留原方法，并补充launch correlation对应的CPU嵌套scope；原失败源文件/日志保留，同一批trace未修改或重跑。原native NCCL GPU annotation fallback也保留。

forward marker包含额外元数据读取成本，尤其融合epilogue多一次sequence读取；不要用其CPU self差值当作无profiler主机收益。GPU时间和也可能重叠，不等于关键路径墙钟；本文不从trace计算加速比。

证据：`checks/20261101-candidate-trace/{current,fused}/components-summary-v2.json`、`complete.json`与原始trace SHA；本地镜像`results/candidate-trace-{current,fused}-20261101.json`，代码`candidate_trace_20261101/`。
