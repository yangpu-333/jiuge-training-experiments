# 三版性能对比（2026-10-31）

固定 BI-V150 四卡、九格4B、DP1/TP2/PP2、BF16、S512、MBS1、GBS16、seed1234、关闭重计算。只替换 RMSNorm；reference 是现有九格适配参考实现。

结论：LoRA没有稳定加速证据，三版重复范围重叠；完整Triton版的平均步时略高于reference。CPT的mixed/full版平均吞吐分别比reference高约1.98%/2.80%，但重复范围重叠；完整Triton版相对mixed均值仅改善约0.80%，不能认定为稳定收益。候选的峰值allocated显存均下降：LoRA约0.60GiB，CPT约0.95GiB。reference继续保留，下一步可按正式测量规范增加采样，并检查host发射、通信等待和归约kernel自身耗时。

## 无 profiler 测量

每任务每版独立启动三轮；每轮5步预热、10步计时，轮换版本顺序。step前后设备同步，采用四rank同一步耗时最大值后取均值。吞吐按8192输入tokens/step计算，LoRA包括被loss mask屏蔽的输入token。

|任务|版本|平均步时(s)|三轮均值范围(s)|输入tokens/s|相对reference加速|峰值allocated(GiB)|
|---|---|---:|---:|---:|---:|---:|
|lora|reference|6.0065|5.8558–6.1024|1363.9|1.0000x|6.553|
|lora|mixed|5.9840|5.9349–6.0661|1369.0|1.0038x|5.944|
|lora|triton|6.0254|5.9563–6.1391|1359.6|0.9969x|5.952|
|cpt|reference|5.3278|5.3070–5.3422|1537.6|1.0000x|22.965|
|cpt|mixed|5.2243|5.1402–5.3399|1568.0|1.0198x|22.018|
|cpt|triton|5.1826|5.0639–5.3268|1580.7|1.0280x|22.018|

计时覆盖train_step内部的数据获取、前反向、通信和optimizer，排除模型加载、step外日志及checkpoint写盘。测量从第6步重置显存峰值。该采样方法包含同步开销，是探索性同条件对照，不是正式端到端验收。

## PyTorch Profiler

每任务每版另启一组，5步预热后采1步，四rank全部抓取CPU+CUDA trace；关闭shape/stack/memory记录。

|任务/版本|四rank kernel数量|四rank kernel累计时长(ms)|
|---|---|---|
|lora/reference|93999, 93982, 95426, 95409|3932.62, 4003.82, 3783.20, 3904.61|
|lora/mixed|90063, 90046, 91394, 91377|3583.52, 3687.20, 3537.49, 3569.61|
|lora/triton|79279, 79262, 80306, 80289|3488.22, 3708.16, 3437.14, 3477.58|
|cpt/reference|75667, 75651, 76786, 76769|3952.33, 3904.53, 3853.80, 3929.62|
|cpt/mixed|72691, 72675, 73762, 73745|3593.82, 3983.13, 4052.03, 3575.00|
|cpt/triton|61779, 61763, 62674, 62657|3435.14, 3501.31, 3467.72, 3419.35|

原始trace位置：`runs/4b-{task}-tp2pp2-s512-perf-{arm}-trace-r1/trace/rank-{0..3}.json`，可用Chrome/Perfetto trace viewer打开。`checks/20261031-performance/summary.json`记录每rank前20耗时kernel、候选norm kernel、通信kernel以及trace SHA256。

kernel时长相加可能含重叠，不能当作wall time；trace有探针开销，不参与加速比计算。全模型aten计数不是RMSNorm专属计数，不据此报告reference的norm独占耗时。

四rank合计kernel启动次数相对reference，full版在LoRA约减少15.7%、CPT约减少18.3%。但启动次数下降不直接等于相同比例的整步加速。以full版rank0为例，LoRA的`forward_statistics`/`backward_row`分别累计16.53/23.95ms；CPT分别16.37/33.53ms。CPT的norm weight梯度三个kernel合计约32.06ms；LoRA的norm weight冻结，无该反向分支kernel。完整明细见summary，以上仅为带profiler单步内的累计GPU kernel时长。

同一rank0 trace中，full版LoRA的AllReduce及SendRecv累计约1.18秒，CPT约1.01秒，仍是明显开销。通信kernel可能包含等待且与其他流重叠，不能将这些数值直接解释为可消除的时间。根据当前计时与trace，后续优化应先确认关键路径和host发射间隙；此处只是排查方向，不认定已经定位了唯一瓶颈。

## 检查与边界

24组运行均检查四rank完成、无跳步、loss/grad_norm有限，LoRA检查适配器有效更新。配置核对model、parallel及除产物路径外的task字段一致。新的15步运行仅记录训练标量，没有重复做全量张量数值验收；此前10步数值证据保持不变。

三版含义：mixed=split-v4；triton=reductions-full-v1；两者均保留外部FP32 cast与两条反向边。原始仓库融合Triton失败证据没有被覆盖。本轮不验证长训练、原生checkpoint恢复，也不适用于其他形状、软件栈或GPU。

lora 三轮15步训练标量观察：loss全部一致=True，grad_norm全部一致=True。这不是完整梯度或参数对齐的替代判据。

cpt 三轮15步训练标量观察：loss全部一致=True，grad_norm全部一致=True。这不是完整梯度或参数对齐的替代判据。

全部24份trace均检查CPU步标记与GPU kernel存在；trace同时含CPU和GPU同名注释，汇总仅选择`cat=user_annotation, ph=X`的CPU区间。首次汇总误把两类注释一起计数而停止，修正后重新分析原始文件通过，未重跑或改写trace。原失败日志及修正后的日志分别保留为`checks/performance-20261031-analysis.log`和`checks/performance-20261031-analysis-v2.log`。

交付时四卡空闲，容器保留，source/MCore的tracked diff均为空。源码commit分别为`9fabd71df4897f08267df5431aa480ab9ef70a75`、`5be9626709af2722333bf54797c954c09edeada3`。汇总SHA256：`efa490cbe17506d846ca407d2fc0830078f2d0247c018d7915eec475f2aa427b`。
