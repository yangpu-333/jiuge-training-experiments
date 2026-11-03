# RMSNorm统计量与行归约的Triton替换

2026-10-31。工作区 `/private/yp/jiuge-training/`。总证据：[triton-reductions-evidence-20261031.json](results/triton-reductions-evidence-20261031.json)，SHA256 `aa8507ba500a6e16175507ed6effd1706a595673e220647b9704e511ac69e608`。

**混合版中的Torch前向统计量、反向行归约及相应标量计算已逐项替换为Triton。两个阶段、CPT/LoRA共四组10步整模型对照全部通过原数值判据。** 与已验证混合版split-v4相比，所捕获的完整梯度、主参数和最终模型参数内容SHA均一致，没有新增数值漂移。

两条输入反向边始终保留。Torch仍负责外部 `x.float()` cast、自动微分梯度累加和内存管理；本次替换仅针对RMSNorm，attention/GEMM等仍是原框架路径。固定训练工程源码未改，新实现通过明确记录SHA的外部候选入口启用。原始仓库的融合单分支Triton实现没有因此获得验收。

## 1. 不变的实验条件

| 项目 | 设置 |
|---|---|
| 模型 / 设备 | 真实FM9G-4B，4×BI-V150 |
| 分布式 | **DP1 / TP2 / PP2** |
| 精度 / 序列 / 批次 | **BF16 / S512 / MBS1 / GBS16**，16次梯度累积 |
| 激活重计算 | **false**，每次核对有效配置 |
| 种子 / 数据 | 1234；原4B resolved模型和S512中文Alpaca预处理数据 |
| CPT | Adam、LR1e-5、weight decay0.01、clip1、distributed optimizer=true |
| LoRA | r8/alpha16/dropout0，基座冻结；LR1e-4、distributed optimizer=false |
| 工程 / MCore | `9fabd71df4897f08267df5431aa480ab9ef70a75` / `5be9626709af2722333bf54797c954c09edeada3` |
| 环境 | CoreX/driver4.4.0；运行时Torch2.7.1、Triton3.1.0；本机warp size=64 |

复用原始numeric-v1固定reference，四个候选均从相同HF权重独立开始。没有更换参考轨迹、模型、数据、监督mask、优化器或随机数策略。reference的九格适配/MCore实现边界沿用[统一配置](UNIFIED_CONFIG_20261030.md)。

两个阶段的协议均逐字节复制原协议，SHA仍为 `a49da90bbcfe5cc67bbe97047f3b0008c09e7fcf8261174e90803f7612371fce`：每步loss误差≤`0.001+0.001×|reference|`、梯度范数相对误差≤1%；第1/10步完整梯度相对L2≤1%、cosine≥0.9999，并执行原逐张量检查；累计更新相对L2≤5%、cosine≥0.995。初始状态、实际microbatch、RNG和样本计数仍严格控制。

## 2. 逐项替换的实现

### 第一阶段：前向统计量

用短PyTorch Profiler窗口确认当前原生归约布局，并读取本机已安装的`Reduce.cuh`。本形状每行使用64线程、每线程4路累加器；按原顺序累加后，依次合并4路结果，再执行相邻两两合并。

Triton实现显式复现该顺序，保留FP32均方值、乘倒数、加eps和rsqrt的舍入边界。当前CoreX语言接口没有`tl.gather`，最终使用连续的二维reshape及两元素reduce；失败原型与日志另行保留。没有使用NVIDIA专有指令或把warp32假设套到本设备。

第一阶段只替换前向统计量；反向行VJP仍沿用split-v4的Torch路径，以单独验证前向改动。

### 第二阶段：反向行VJP及标量系数

将weight乘法的BF16中间舍入、逐行dot以及标量系数计算移入Triton。行dot使用同样的4路累加和合并顺序；BF16中间量显式保留舍入。

标量系数另做中间量诊断：乘方、前两次乘法与参考一致，但直接除以2560会产生少量FP32末位差异。实测当前Torch标量除法在这里等效于**乘FP32倒数**；最终显式采用相同路径，没有通过放宽容差吸收差异。相关原型、CPU FP64舍入对照与宽指数用例都已保留。

最终`full`候选的前向统计量、反向行VJP及其标量算术均由Triton执行，dw继续使用已有Triton实现。`x`和Function外的`x.float()`仍作为独立输入，分别返回BF16直接梯度与FP32方差梯度，保持[反向图修复](NUMERIC_FIX_20261031.md)的累加语义。

实现仍限定当前BI-V150/CoreX环境、BF16输入/weight和`[512,1,2560]`；不泛化其他形状、模型、dtype或软件版本。

## 3. 局部与实际调用路径检查

前向mean/r和反向gw/dot/系数在125组真实捕获数据及8组固定应力用例上，最终均与当前参考数值一致。每个阶段另在全部125组实际输入、weight、上游梯度上，检查完整VJP和残差图：

| 检查项 | 前向替换阶段 | 完整替换阶段 |
|---|---|---|
| 前向输出 | 全部一致 | 全部一致 |
| 两个输入梯度分量、FP32方差梯度及其BF16 cast | 全部一致 | 全部一致 |
| 接入残差后的输入梯度 | 全部一致 | 全部一致 |
| dw | 保留原有14个末位差异 | 相同，最大单模块相对L2约`7.33e-6` |

用独立PyTorch Profiler窗口核对实际RMSNorm前向+反向调用，输入/weight均可训练、形状与当前任务相同，编译先预热。下表是CPU算子调用数，不是耗时：

| 路径 | mean | rsqrt | sum | pow | div | mul |
|---|---:|---:|---:|---:|---:|---:|
| 混合split-v4 | 1 | 1 | 1 | 2 | 1 | 4 |
| 仅前向统计量替换 | 0 | 0 | 1 | 1 | 1 | 4 |
| **完整替换** | **0** | **0** | **0** | **0** | **0** | **0** |

完整替换窗口实际执行`forward_statistics`、`given_r`、`backward_row`、`split_dx`及三个dw Triton kernel。剩余3个原生kernel是外部cast和梯度累加，符合保留两条反向边的设计。该窗口验证执行路径，没有正式性能结论。

## 4. 四组整模型结果

以下为相对L2的原始比例，**不是百分数**：

| 阶段 | 任务 | 第1步完整梯度 | 第10步完整梯度 | 10步累计更新 | 原门槛 |
|---|---|---:|---:|---:|---|
| 前向统计量替换 | LoRA | 0 | 0 | 0 | 通过 |
| 前向统计量替换 | CPT | `4.93933e-7` | `2.84130e-7` | `3.32265e-6` | 通过 |
| **完整替换** | **LoRA** | **0** | **0** | **0** | **通过** |
| **完整替换** | **CPT** | **`4.93933e-7`** | **`2.84130e-7`** | **`3.32265e-6`** | **通过** |

四组均完成10步、0 skipped；实际输入、初始参数、RNG、样本计数及LoRA冻结基座控制通过。每步loss和梯度范数精确复现reference；所有逐张量梯度检查通过，最终全部模型张量数值一致且有限。

第1/10步梯度、初始/最终FP32主参数以及最终named model parameter的内容SHA，也分别与已通过验证的混合split-v4一致。因此，本次替换没有新增该已捕获范围内的数值差异。CPT相对原始reference的微小norm weight梯度/主参数差异，仍是此前已有dw路径的差异。

CPT每次完整梯度比较覆盖750个rank本地optimizer张量、4,318,028,800个元素；LoRA覆盖992个张量、22,474,752个元素。汇总包含实际TP复制norm和PP共享embedding副本，不是模型独立参数量。

## 5. 无损存储与审查方式

启动时个人网盘约剩150G，不重复生成大体积optimizer checkpoint。本轮保留相同数值检查内容，调整了产物存储方式：

- 实际GPU张量先完整复制并计算内容SHA；与固定reference完全相同的张量记录引用，发生变化的张量完整另存，不做抽样或量化。
- CPU审查时重建每个候选张量，校验参考和候选的全部内容哈希，再按原算法用CPU FP64逐块比较全部元素。
- 最终完整模型状态和scheduler在训练结束后导出，实际consumed samples=160。实际启动命令移除原生`--save`，不另存没有参与本轮比较的optimizer大状态；配置计划命令和实际命令均保留。
- 格式测试覆盖符号零、相同/变化张量、参考被篡改、候选被篡改和模型标量结构，全部通过。

四组导出的审计张量逻辑总量约138.81GB；其中另存的变化张量约8.40MB，另有引用元数据、日志和报告。这个数值不是整个实验目录大小，也不包含已存在的参考文件。

**无损包依赖固定的numeric-v1参考快照和最终模型checkpoint；整理存储时必须保留引用链。** 引用文件、schema SHA和大小/mtime约束均记入总证据。容器内可用`tools/compact_tensor_pack.py`的`restore(path)`重建张量。mtime只用于检测文件是否变化，不用于推断实验活动时间。

本轮没有逐元素比较Adam moment，没有新增候选原生恢复验收，也不是长期训练质量或正式端到端性能验收。

## 6. 产物与后续入口

远端统一入口：`/private/yp/jiuge-training/checks/20261031-triton-reductions/`。

- `native-order-final.json`、`reduction-partials.json`：完整局部检查。
- `coefficient-probe.json`、`division-probe.json`：标量舍入路径定位。
- `forward/`、`full/`：各阶段不变协议、预登记、命令、逐任务全量审查和资源日志。
- `operator-seams-summary.json`、`operator-seams-*.json`：Profiler路径核验及原始trace。
- `REDUCTION_EVIDENCE_20261031.json`：完整总证据。
- `runs/*reductions-forward-v1/`、`runs/*reductions-full-v1/`：实际配置、逐步审计、无损张量包、最终模型导出和完成标志。

实际实现为远端`tools/native_order_reductions.py`和`tools/reduction_patch.py`；本地镜像在 `experiments/jiuge_training/triton_reductions_20261031/`。数值审计入口为`tools/compact_numeric_training.py`，必须明确`REDUCTION_STAGE=full`以启用完整替换；批量入口为`tools/run_reduction_stage.py --stage full --task cpt|lora`，会拒绝覆盖本轮已有目录。原始CLI的`--backend triton`不会自动安装本候选。

reference继续保留为对照，后续实验候选可使用已通过本轮数值门槛的完整替换版。正式测速需要去除审计与张量导出，并按用户后续测量规范执行；候选原生保存恢复也需单独验证。

结束时四卡空闲、无GPU进程，容器保留；个人网盘约876G已用、149G可用。没有删除此前的失败或已验收产物。
