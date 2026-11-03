# LoRA A/B C++ autograd简化（2026-11-01）

固定BI-V150四卡、4B、DP1/TP2/PP2、BF16、S512、MBS1/GBS16、seed1234、关闭重计算。RMSNorm保持已验收的fused-split-v1。基线是native F.linear LoRA，候选是lora-linear-cpp-v2。

本轮结果：LoRA/CPT十步原数值门槛通过，捕获内容SHA均与fused-split-v1一致；LoRA四对均更快，均值5.8850→5.7431秒、平均吞吐+2.47%。配对范围+1.03%至+5.22%，存在约一秒慢步且不排除任何样本，后续正在用不改策略的GC观测解释波动。CPT未改算法，单对负对照-0.79%，不作为收益或回归结论。

## 为什么做这项

依据融合版整模型trace，A/B前反向GPU累计约403ms，B反向CPU记录区间明显；scale/add前向GPU约46ms。先消除A/B transpose/view等反向节点，保留原BF16 GEMM，而不是仅追求更少GPU kernel。

PythonFunction首版48组局部数值逐元一致，但隔离成本普遍倒退12%–15%，未进入整模型。C++首版在主要形状局部改善9%–14%，v2增加pybind11 GIL释放，在48组局部数值及成本筛查通过后进入整模型。源码与筛查结果均保留。

## 实现边界

C++autograd Function前向仍用at::linear，反向先dW=g.T@x，再dX=g@W，匹配实际捕获的native操作数布局、stride和发射顺序。仅替换LoRA A/B；scale/add、低秩TP映射、基座linear和RMSNorm两条反向边不改。独立extension编译至/private/yp/jiuge-training/cache/torch-extensions/lora-linear-cpp-v2/，未修改Torch/CoreX。

## 原数值判据

|任务|step1梯度relL2|step10梯度relL2|累计更新relL2|最终模型relL2|
|---|---:|---:|---:|---:|
|lora|0|0|0|0|
|cpt|4.9393302e-07|2.8413032e-07|3.3226542e-06|0|

lora 原门槛通过；与融合基线捕获内容SHA一致=True，差异项=[]。

cpt 原门槛通过；与融合基线捕获内容SHA一致=True，差异项=[]。

协议与原numerical-v1一致；初始/输入/RNG/完整梯度/FP32累计更新/最终模型全量审查。十步数值不泛化到长训练或原生恢复。无损张量包依赖reference，不能单独删除引用链。

## 无profiler配对结果

LoRA四对AB/BA交错独立启动，每组预热5步、计时10步；CPT未改算法，仅一对负对照，不宣称CPT收益。相同轻量host计数，去除全量审计与checkpoint写盘。取每步四rank同步耗时最大值后求均值。

|任务|基线均值(s)|CPP均值(s)|吞吐比|更快配对|说明|
|---|---:|---:|---:|---:|---|
|lora|5.8850|5.7431|1.0247x|4/4|优化对象|
|cpt|5.0256|5.0658|0.9921x|0/1|未改动负对照，仅一对|

|任务/配对|基线(s)|CPP(s)|节省(ms)|吞吐比|
|---|---:|---:|---:|---:|
|lora/1|5.8094|5.7500|59.4|1.0103x|
|lora/2|5.8393|5.7627|76.6|1.0133x|
|lora/3|5.8520|5.7201|131.9|1.0231x|
|lora/4|6.0393|5.7396|299.7|1.0522x|
|cpt/1|5.0256|5.0658|-40.2|0.9921x|

每对配置和15步训练标量一致。探索性train_step计时排除加载、step外日志与保存，包含前反向、通信、optimizer；不等于正式端到端验收。

## 节点/设备路径核验

独立实际形状A/B前反向PyTorch Profiler：

|形状/版本|GPU kernel数|autograd engine节点数|
|---|---:|---:|
|A/native|3|6|
|A/cpp|3|3|
|B/native|3|6|
|B/cpp|3|3|

这项优化保留设备GEMM数量，减少的是主机autograd节点；独立trace不用于计算整模型加速比。

## 入口与证据

产物checks/20261101-lora-linear/；源码tools/lora_linear_cpp.cpp、lora_linear_cpp.py、lora_linear_patch.py。数值入口tools/lora_cpp_numeric.py；性能入口tools/lora_cpp_perf.py，PERF_LORA_CPP=1/0选择候选/融合norm+native linear基线。每rank记录binary与fused norm SHA；原始CLI不自动启用。

汇总cpp-summary.json，局部/图证明local-linear-cpp-probe.json与cpp-graph-profile.json，原协议numerical/protocol.json。仍保留reference及此前融合基线；长期训练、原生恢复及其他形状未验收。
