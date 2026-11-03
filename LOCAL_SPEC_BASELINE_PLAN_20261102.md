# Megatron local spec 对照边界（源码核对）

固定Megatron commit `5be9626709af2722333bf54797c954c09edeada3` 的 `get_gpt_layer_local_spec` 调用 `get_gpt_layer_local_submodules`。在RMSNorm配置下，`LocalSpecProvider.layer_norm`选择`WrappedTorchNorm`，`core_attention`选择`DotProductAttention`；线性层为`ColumnParallelLinear`和`RowParallelLinear`，BDA为`get_bias_dropout_add`。这是源码选择，运行时实际实现仍需单独验证。

九格固定源码commit `9fabd71df4897f08267df5431aa480ab9ef70a75` 的`model.py`自行构造spec：4B fm9g使用FP32 Q/K rotary路径、`HFCompatibleAttention`、可配置的九格RMSNorm和`scaled_bias_dropout_add`。这些与官方spec并非相同对象；应核对精度转换、attention scaling/mask/softmax与残差缩放语义，再设计独立baseline。仅替换类并成功训练不能证明模型语义等价。

源码层面已确认一个应独立验证的舍入边界：九格`HFCompatibleAttention`先将Q/K转换为value dtype，执行`torch.matmul`后再除以`sqrt(head_dim)`；固定MCore的`DotProductAttention`将`1/sqrt(head_dim)`作为`torch.baddbmm`的alpha。低精度下两种计算顺序可能不同，不能仅凭代数等价宣称数值相同。需要在实际BI-V150、BF16和本模型shape下用固定Q/K/V比较前向及反向，再决定是报告原生spec差异，还是引入并明确标注必要的九格语义适配。

归一化也需要独立核验：固定MCore的`WrappedTorchNorm`在RMSNorm配置下实例化`torch.nn.RMSNorm`；九格reference显式将输入转FP32计算均方与rsqrt，再把归一化值转换回输入dtype后乘weight。两者算术与梯度是否满足九格判据必须实测，不能只凭类名相同判断。

已读取4B resolved config：62层、`scale_depth=1.4`、hidden=2560、40个attention heads、4个KV heads。九格spec据此向BDA注入`1.4/sqrt(62)`的缩放，源码在残差相加之前计算`dropout(x)*scale`；原生local spec选择标准BDA。构造同一九格模型的原生算子对照时，必须单独保留或验证这一模型语义，不能在未核验前直接使用标准BDA并称模型相同。

建议在现有v3审计完成且GPU空闲后，先记录local spec实际模块和参数映射，分别用固定输入验证RMSNorm、attention与BDA前向/反向，保持用户4B四卡配置和原数值门槛。若原生算子不满足九格语义，记录最小必需适配及独立对照名称，再做原协议整模型数值和配对测速。当前没有官方local spec完整模型性能结论，也不把已有+2.23%称为相对该基线。

已准备两个隔离探针：固定捕获和合成输入的`torch.nn.functional.rms_norm`（`WrappedTorchNorm`的RMSNorm运算）与九格reference前反向对比；以及TP2实际score矩阵形状下`bmm`后除法与`baddbmm alpha`的随机输入前反向对比。后者仅检查score边界，不代表完整attention或GQA语义。探针已排在v3配对测速及选定候选原生恢复之后，须GPU空闲才运行；BDA和完整模块验证仍待后续。

隔离探针已完成。RMSNorm共125份实际捕获与4组合成输入：仅实际捕获中，output/dx/dw的单例最大relative L2分别约0.003585/0.003359/0.020163，三个字段都有逐元素差异且均finite。这不是原十步整模型判据，也不能从单算子结果直接得出模型训练是否通过。score边界在`[20,512,64]`、head_dim=64的3组合成输入下，scores/dQ/dK逐元一致；此处scale为1/8，仅验证该分界和输入，未验证完整attention、GQA或其他模型。原始数据与输入/源码哈希见`checks/20261102-local-spec-probe/`，本地摘要`results/local-spec-{norm-comparison,attention-score}-20261102.json`。下一步先确认官方norm与九格reference的误差来源及必要适配，再逐层验证实际attention和残差分支。

四rank完整attention core探针采用TP2PP2、Q `[512,1,20,64]`、K/V `[512,1,2,64]`、causal mask、dropout 0，保留九格FP32 rotary后的BF16转换；该次进程启动时遇到节点高I/O压力，最终完成。整模型LoRA审计使用MCore `torch.nn.RMSNorm`与`DotProductAttention`原生算子，明确保留九格Q/K转换、残差缩放、embedding/output及加载适配；原十步判据不变，审计类继承原生RMSNorm forward以核对模块身份，不修改固定源码。它是九格语义适配的local算子对照，不能声称整个模型未经适配。CPT原始全量审计最坏空间约56GiB，当前剩余约72GiB；将先看LoRA对齐及实际变化，再设计可保留证据的CPT存储方案。

上述四rank完整attention core探针现已完成：3种合成尺度下，MCore `DotProductAttention`与九格eager attention的输出及Q/K/V梯度均逐元一致；适用范围是指定形状、causal mask、dropout=0及显式Q/K转换。探针、源码哈希及四rank结果保存在`checks/20261102-local-full-attention/`，本地汇总`results/local-spec-full-attention-20261102.json`。依赖任务已启动上述LoRA原生算子整模型数值审计；官方相对性能仍未知。

LoRA官方算子整模型十步已完成，使用官方`torch.nn.RMSNorm`和`DotProductAttention`，保留九格Q/K dtype转换及残差、embedding/output和权重适配。相对九格reference，输入、RNG和初始状态控制通过，但step1/10完整梯度relative L2约20.51%/18.82%、累计FP32主参数更新约15.80%，超过原协议门槛；原始报告`checks/20261102-official-local/numerical/lora-comparison.json`。优化版直接相对该基线的梯度relative L2约20.53%/18.44%、累计更新15.81%，同样不通过；故按预注册条件未进行同口径加速测速，不能声称已知相对官方基线的有效加速。

直接审查v1复用了只读取canonical reference的最终模型汇总器，导致最终模型字段不能代表官方基线与优化版直接比较；梯度、loss和主参数直接比较不受影响。v1证据保留，v2重新从两臂各自无损重建最终模型，使用相同阈值，结果另存`lora-direct-comparison-v2.json`。为定位原因，后续将保留原门槛分别运行`attention-only`（原生attention+九格reference norm）和`norm-only`（原生norm+九格reference attention）LoRA整模型消融；不能把其中通过的适配版误称完整官方原生基线。

消融已完成：attention-only 与九格 reference 的十步检查通过，所检查梯度和累计主参数更新逐元一致；norm-only 未通过，四 rank 的初始主参数、第 1/10 步梯度和最终主参数哈希均与两种原生算子并用时一致。在当前固定 LoRA 配置下，差异定位到原生 RMSNorm 路径。attention-only 与优化版的直接数值检查通过，目前正在做四组 AB/BA 配对测速；该对照应称为“原生 attention + 九格 reference RMSNorm 的适配基线”，不是完整原生 local spec。节点 IO 压力监测与结果一同保留，不能将节点级 PSI 当作训练进程的等待占比。

CPT 完整原生算子审计排在 LoRA 配对之后；以原有判据检查，并使用经真实梯度和大张量验证的无损参考异或压缩、逐张量哈希与剩余空间保护保存证据。审计完成后再直接比较 CPT 优化版；数值未通过时不得报告同口径有效加速。排队脚本初次引用了错误的汇总文件位置，尚未触发 CPT 即已修正，原预注册和日志保留。

LoRA 四组 AB/BA 测速已完成：适配基线平均 6.153197 秒/步，优化版 5.670374 秒/步，速度比 1.085148×、步时降低约 7.85%；四组比值依次为 1.0871、1.0799、1.0828、1.0908，loss 检查通过，测量期间无二代 GC。节点 IO 压力较高，监测从第一组开始之后才启动，须在较空闲环境复测。详细对照边界和结果见 `checks/20261102-local-ablation/attention-only/README.md`。CPT 训练和压缩导出已完成，原始数值门槛复核正在进行。

CPT 完整原生算子相对 reference 与优化版直接比较均未通过原数值门槛；CPT 原生 attention 加九格 reference RMSNorm 适配基线与 reference、优化版的两项原门槛检查均通过。四组配对平均 5.205853→4.888542 秒/步，速度比 1.064909×，各组为 1.0593、1.0555、1.0749、1.0700，loss 检查通过、测量期间无二代 GC。不能将其称为相对完整原生 RMSNorm 的有效加速。原始结果见 `checks/20261102-adapted-local-cpt/`，对照边界与哈希见 `checks/20261102-local-ablation/attention-only/README.md`。
