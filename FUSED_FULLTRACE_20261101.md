# 融合版整模型trace与LoRA组件归因（2026-11-01）

固定四卡BI-V150、九格4B、DP1/TP2/PP2、BF16、S512、MBS1/GBS16、seed1234、关闭重计算，实际工作区仍为`/private/yp/jiuge-training/`。本轮使用数值已验收的`fused-split-v1` RMSNorm。

LoRA/CPT各5步预热后采集第6步，四rank共8份PyTorch Profiler整模型trace。CPU+CUDA，关闭shape/stack/memory记录；用模块metadata记录实际A/B形状。两任务六步训练标量均与上一轮融合版一致，LoRA适配器有效更新。只增加record_function标记，没有反向hook或算术变化；trace耗时不能作为训练吞吐。

## 直接测到的组件成本

下表为LoRA rank0单步，GPU为累计kernel时长；CPU为按线程嵌套减去子区间后的记录区间self时间，包含等待/调度，不是纯CPU周期。跨线程区间可重叠，不能相加当wall time。

|组件|GPU次数|GPU累计(ms)|CPU self区间(ms)|
|---|---:|---:|---:|
|A前向|1984 GEMM|70.51|204.10|
|A反向|3952 GEMM|86.55|563.33|
|B前向|1984 GEMM|27.05|202.36|
|B反向|3968 GEMM|218.78|993.90|
|scale前向|1984|16.44|145.10|
|add前向|1984|29.25|112.99|
|scale反向|1984|17.18|80.05|
|低秩TP前向（row）|992 all-reduce|23.32|365.27|
|低秩TP反向（column）|992 all-reduce|23.85|279.29|

四rank的A/B前向各1984次、低秩TP总1984次均完整对齐；无CPU区间交叉或前向序列号歧义。A反向少16个输入梯度GEMM符合冻结输入的首层路径。基座linear、低秩TP及PP通信分别归属，不把所有all-reduce混为LoRA额外通信。

低秩张量实际为`[512,1,8]` BF16，即每次逻辑payload 8192 bytes。column是copy-forward/reduce-backward，row是reduce-forward/identity-backward。它们可以有较高host调用成本，但本次GPU累计时间约47ms；不能把host区间与GPU时间相加，也不能直接把它们当成可删除的通信时间或线速流量。

实际A/B权重形状：`[8,2560]→[1536,8]`（QKV）、`[8,1280]→[2560,8]`（attention proj）、`[8,2560]→[6400,8]`（fc1）、`[8,3200]→[2560,8]`（fc2）。每个rank31层、每层4个适配器、每步16个microbatch。B反向中fc1投影累计约96.0ms，是低秩GEMM中最明显的单个投影方向；B反向整体约219ms。

## 融合版全模型规模与边界

|任务/rank|kernel数|trace窗口(ms)|设备活跃并集(ms)|
|---|---:|---:|---:|
|LoRA/0|77311|10148.6|3293.2|
|LoRA/1|77294|10148.6|3577.2|
|LoRA/2|78290|10149.6|3306.0|
|LoRA/3|78273|10129.6|3473.3|
|CPT/0|59795|7792.1|3611.4|
|CPT/1|59779|7792.0|3665.8|
|CPT/2|60658|7772.8|3431.1|
|CPT/3|60641|7792.7|3422.3|

融合norm前向/反向kernel均存在，旧的四个独立统计/输出/行归约/dx kernel均为0。窗口受profiler及额外组件标记扰动，不能据此重新估算LoRA/CPT加速比。计时收益沿用已独立完成的[四对对照](FUSED_RMSNORM_20261031.md)。

## 优化顺序及已开始的筛查

1. 优先检查A/B的autograd主机路径，而非只看kernel数量。A/B GPU累计约403ms，反向CPU记录区间更突出，尤其B反向。尝试保留原生BF16 GEMM、原操作数布局及dW→dX发射顺序，减少transpose/view等autograd节点。
2. scale/add前向累计约45.7ms GPU、1984次各一kernel，适合作为下一项可控融合候选。必须保留scale结果的BF16中间舍入，不能直接用FP32 FMA跨过该边界；反向及整模型关联仍走原数值门槛。
3. 低秩TP以小payload和host发射为主要检查方向；跨分支合并需要先证明依赖与BF16归约语义，不能直接删通信。PP同步保持现有竞态兼容设置。

已按用户持续优化要求执行第一轮局部筛查：Python自定义A/B linear在48组局部形状/幅度/输入grad条件下输出与VJP逐元素一致，实际反向GEMM布局/顺序也已捕获核对。但隔离前反向耗时普遍比native慢约12%–15%，最大B形状约持平，说明Python回调代价抵消了节点简化；该版本未进入整模型训练或作为性能候选。下一步检查不引入Python反向回调的轻量C++ autograd封装，再以局部成本筛查决定是否进入原整模型判据。

## 关联方法与证据

前向按`jiuge_detail/{lora_A,lora_B,lora_scale,lora_add,lora_TP,base_linear}/...`标记，反向通过现有autograd Sequence number关联；没有替换反向图。dropout=0不产生反向节点，但会携带下一次操作的Sequence number，已从序列号字典排除。NCCL异步launch缺少CPU External id时，使用同stream、完整包含kernel区间的GPU NCCL注释及其External id关联CPU标记。所有rank均验证1984次低秩通信完整覆盖，未按LL/SIMPLE kernel名猜归属。

第一版分析器未处理NCCL异步归属及dropout的无节点序列号，校验未通过；失败源码、日志和中间结果保留。修正版8份全部通过，原trace未重写。

产物：`checks/20261101-fused-fulltrace/`；原trace：`runs/4b-{lora|cpt}-tp2pp2-s512-fused-fulltrace-detail-20261101-v1/trace/rank-{0..3}.json`。本地[完整组件汇总](results/fused-fulltrace-components-20261101.json)与[标记/分析代码](fused_trace_20261101/README.md)。
