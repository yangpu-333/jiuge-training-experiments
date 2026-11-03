# RMSNorm反向图修复与整模型验证

2026-10-31。实际工作区 `/private/yp/jiuge-training/`。总证据：[split-numeric-evidence-20261031.json](results/split-numeric-evidence-20261031.json)，SHA256 `7f8f747cd9748b6fa8d17ad8deff6533305b91845d6bf342b1ccf35de9223dbd`。

**保留两条输入梯度分支的混合候选 `split-v4` 已通过CPT和LoRA的整模型数值门槛。** 四卡TP2PP2、真实4B、BF16、S512、MBS1/GBS16、关闭激活重计算，分别完成10步；判据沿用上一轮，没有放宽。LoRA所捕获的完整梯度和主参数内容SHA一致；CPT只保留很小的norm weight梯度及FP32主参数差异，所有数值门槛通过。

该候选明确包含Torch统计量和行归约，**不代表原始纯Triton后端已通过**。原始后端及先前失败候选的记录见[上一轮报告](NUMERIC_ALIGNMENT_20261031.md)，证据保持不变。本轮没有正式性能验收。

## 1. 关键发现：BF16反向图的梯度合并顺序

reference里，输入 `x` 同时进入直接归一化分支、`x.float()`统计量分支，以及模型的残差分支。统计量分支的FP32梯度经过独立的cast反向边，再参与 `x` 处的BF16梯度累加。

原来的单个自定义autograd Function先把两个norm梯度合成一个BF16结果，再与残差梯度相加。它改变了低精度浮点加法的分组方式。即便数学导数相同、局部norm梯度相同，外层残差图也可能得到不同结果。

为单独验证这个因素，`probe_gradient_graph.py` 构造了一个局部反向**完全调用reference**的Function：

| 情形 | 合并norm反向的Function | 分开返回两个梯度分支 |
|---|---|---|
| 孤立norm，seed1234/2026 | 输出、dx、dw数值一致 | 同左 |
| 接上残差，seed1234 | dx相对L2 `0.00261970`，388,645个元素不同 | dx无差异 |
| 接上残差，seed2026 | dx相对L2 `0.00264648`，393,044个元素不同 | dx无差异 |

每例比较1,310,720个输入梯度元素，输出和dw仍一致。该实验没有使用Triton来计算局部反向，因此把这一差异与Triton局部导数误差分离开了。它也解释了为什么上一轮仅修正局部BF16舍入仍不能使整模型对齐。

本轮选择保留reference的反向图语义，以满足既定对齐目标；没有据此判断其他累加顺序的长期训练质量。

## 2. 真实激活与上游梯度检查

另跑一个明确标注的CPT reference诊断步，保留4B/TP2PP2/BF16/S512/MBS1/GBS16/关闭重计算。只在此诊断命令移除原生checkpoint保存，采集第一个microbatch的125个逻辑RMSNorm模块的输入、weight及实际上游梯度。四rank的完整逐步审计与上一轮reference第一步严格一致。

随后在全部125组数据上计算完整VJP，不抽样：

| 检查项 | 结果 |
|---|---|
| 前向输出 | 全部数值一致 |
| 直接输入梯度分量 | 全部数值一致 |
| FP32方差梯度分量及其BF16 cast | 全部数值一致 |
| 加入残差后的输入梯度 | 全部数值一致 |
| Triton dw | 14个模块各有1个元素不同；最大单模块相对L2 `7.33e-6`，全部有限 |

对照分开计算 `x` 与独立FP32输入 `xf` 的reference梯度。残差诊断使用由实际上游梯度确定的附加梯度，不消耗新RNG。真实捕获数据约0.66GB，保留为两个rank文件，内容SHA记入总证据。

## 3. 候选实现边界

`tools/split_graph_patch.py` 在固定工程外安装混合实现：

| 部分 | 实现 |
|---|---|
| 前向FP32均方值、rsqrt | Torch，与reference计算序列相同 |
| 前向归一化、BF16舍入、乘weight | Triton |
| 反向weight乘法、行VJP归约及标量系数 | Torch |
| 两个输入梯度分量的逐点计算 | Triton，分别返回BF16直接梯度和FP32方差梯度 |
| dw | 固定工程已有Triton实现 |
| 反向图 | 在Function外计算 `x.float()`，把 `x`、`x.float()` 作为独立输入，保留cast反向边 |

没有在自定义反向里提前合并两个输入梯度。实际125组检查及下述整模型结果验证了该结构在本轮条件下复现参考轨迹。

实现限定当前BF16输入/weight、`[512,1,2560]`；未验证其他序列长度、micro batch、dtype或模型。原始工程及MCore的跟踪源码保持不变。CLI仍用 `--backend triton` 选择模块，但实际worker显式安装上述混合补丁，并记录实际命令及全部依赖SHA；直接运行原始CLI不会自动启用此候选。

## 4. 固定条件和数值门槛

- 工程 `9fabd71df4897f08267df5431aa480ab9ef70a75`；MCore `5be9626709af2722333bf54797c954c09edeada3`。
- 4×BI-V150、DP1/TP2/PP2；4B resolved模型及原S512中文Alpaca数据；BF16、MBS1/GBS16、seed1234、重计算false。
- CPT：Adam、LR1e-5、weight decay0.01、clip1、distributed optimizer=true；LoRA：r8/alpha16/dropout0、基座冻结、LR1e-4、distributed optimizer=false。
- 两任务分别从相同HF权重开始，复用上一轮固定reference对照，运行10步并保存原生最终checkpoint。
- 协议SHA仍为 `a49da90bbcfe5cc67bbe97047f3b0008c09e7fcf8261174e90803f7612371fce`：每步loss误差≤`0.001+0.001×|reference|`、梯度范数相对误差≤1%；第1/10步完整梯度相对L2≤1%、cosine≥0.9999，并检查每个梯度张量；累计更新相对L2≤5%、cosine≥0.995。

初始参数、实际microbatch、RNG、消耗样本数和LoRA冻结基座按原协议严格控制。完整梯度在归约后、裁剪前读取；累计更新为最终FP32主参数减相同的初始主参数。所有张量按块用CPU FP64比较全部元素。

## 5. 整模型结果

下表为相对L2的原始比例，**不是百分数**。

| 指标 | CPT | LoRA |
|---|---:|---:|
| 10步完成、0 skipped、全部有限 | 通过 | 通过 |
| 初始状态、输入、RNG、样本计数控制 | 通过 | 通过 |
| 10步loss与reference精确复现 | 是 | 是 |
| 10步梯度范数与reference精确复现 | 是 | 是 |
| 第1步完整梯度相对L2 | **`4.93933e-7`** | **0** |
| 第10步完整梯度相对L2 | **`2.84130e-7`** | **0** |
| 第1/10步失败梯度张量数 | 0 / 0 | 0 / 0 |
| 10步累计更新相对L2 | **`3.32265e-6`** | **0** |
| 最终模型张量相对L2 | 0 | 0 |
| **既定整模型数值门槛** | **通过** | **通过** |

CPT每次完整梯度比较覆盖750个rank本地optimizer张量、4,318,028,800个元素；LoRA覆盖992个张量、22,474,752个元素。汇总包含实际TP复制norm和PP共享embedding副本，不是模型独立参数量或MCore去重后的裁剪范数。

LoRA另核对了3,968对初始/最终FP32主参数及第1/10步梯度张量的内容SHA，全部一致；最终named model parameter哈希也一致，冻结基座未改变。

CPT的第1/10步残余梯度差异均限于norm weight，其他梯度张量一致。累计更新中有48个norm张量的56个FP32元素不同，最大绝对差 `6.61612e-6`，在原门槛内；最终BF16模型张量数值一致。没有把CPT表述成全状态逐字节一致。

本轮比较最终全部模型张量、FP32主参数更新、scheduler及审计状态；没有逐元素比较Adam moment，也没有验证候选的原生恢复。结果适用于本次种子、数据、布局和10步范围，不是长期训练质量验收。

## 6. 产物、复用与下一阶段

远端入口：`/private/yp/jiuge-training/checks/20261031-numeric-fix/`。

- `gradient-graph-probe.json`：局部反向严格相同仍因残差图产生差异的独立证据。
- `capture-validation.json`、`captured-partials-review.json`：真实捕获的参考轨迹复现及125组VJP比较。
- `protocol.json`、`preregistration.json`：不变门槛、候选边界、运行顺序及源码/工具哈希。
- `lora-comparison-split-v4.json`、`cpt-comparison-split-v4.json`：完整逐张量审查。
- `lora-content-hash-parity.json`：LoRA捕获梯度/主参数和最终模型参数的内容SHA核对。
- `runs/*numeric-split-v4/`：实际配置、命令、逐步审计、完整梯度和主参数快照、原生最终checkpoint。
- `runs/4b-cpt-tp2-pp2-reference-no-recompute-s512-gbs16-backward-capture-v1/`：125组真实激活和上游梯度。

本地代码镜像为 `experiments/jiuge_training/numeric_fix_20261031/`，实际运行工具为远端 `tools/`。带审计的训练入口为 `numeric_training_split.py`，批量入口为 `run_numeric_split.py`；已有结果拒绝覆盖，审计入口不用于正式测速。

后续保留reference为基线，`split-v4`作为已通过本轮数值门槛的混合候选。若继续追求纯Triton，需要在保留两个反向分支的基础上，逐项复现原生归约的累加顺序，再把统计量及行VJP归约移回Triton；每次更换仍须沿用同一数值门槛。正式性能测量需使用去除审计的入口，并按用户后续规范执行。

结束时四卡0%/约68MiB，无GPU进程，容器保留。账户网盘约870G已用、155G可用；新一轮大checkpoint需先规划容量。本轮没有删除历史失败、对照或候选产物。
