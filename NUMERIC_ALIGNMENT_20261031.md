# 4B 整模型 reference / Triton 数值对照

2026-10-31。实验工作区：`/private/yp/jiuge-training/`。完整证据：[numeric-alignment-evidence-20261031.json](results/numeric-alignment-evidence-20261031.json)，SHA256：`25fe50a574a4c32eb54c58a328a66c0ba07c3093f2a9320bd179443bfb894250`。

**本轮对照已完成，原始 Triton 后端的 CPT、LoRA 均未通过预先固定的整模型工程数值门槛。后续继续使用 reference 基线。** 两个任务都完成了 10 步，没有 OOM、NaN 或 skipped update；失败项是数值对照。独立的反向舍入候选和混合统计诊断版也未通过，均未替换工作基线。

## 1. 固定配置与比较范围

| 项目 | 本轮设置 |
|---|---|
| 模型与设备 | 真实 FM9G-4B，4×BI-V150，每卡32GiB |
| 分布式 | **DP1 / TP2 / PP2** |
| 精度、序列与批次 | **BF16，S512，MBS1，GBS16**，16次梯度累积 |
| 激活重计算 | **false**，有效配置逐次核对 |
| 数据与初始化 | 已验收的4B resolved模型、S512中文Alpaca预处理数据，seed1234 |
| CPT | 全参数；Adam、LR1e-5、weight decay0.01、clip1，distributed optimizer=true |
| LoRA | r8、alpha16、dropout0，基座冻结；Adam、LR1e-4，distributed optimizer=false |
| 步数 | reference / 原始 Triton 各10步，每个任务独立从相同HF权重开始 |
| 工程 / MCore | `9fabd71df4897f08267df5431aa480ab9ef70a75` / `5be9626709af2722333bf54797c954c09edeada3` |
| 环境 | CoreX/IX-ML/driver4.4.0；运行时字符串 torch2.7.1、Triton3.1.0；Python3.12.3、transformers4.57.6 |

固定工程及MCore的跟踪源码均未改。原始 `reference/triton` 开关只切换 RMSNorm；其余仍是固定 MCore 与九格适配路径，包括 PyTorch eager attention。该 reference 的实现边界沿用[统一配置记录](UNIFIED_CONFIG_20261030.md)。

本轮每步读取实际 microbatch、完整精度 loss、梯度范数和 RNG。在原生梯度归约及 `prepare_grads` 完成后、裁剪前，保存第1、10步全部FP32主参数梯度；另保存初始/最终主参数，比较第10步累计更新 `final - initial`。比较最终原生checkpoint的全部模型张量和scheduler。

所有张量按块遍历全部元素，用CPU FP64计算误差，不抽样。CPT每次梯度比较覆盖750个rank本地optimizer张量、4,318,028,800个元素；LoRA覆盖992个张量、22,474,752个元素。汇总包含实际存储的TP复制norm和PP共享embedding副本，不是去重后的MCore裁剪范数，也不是模型独立参数量。

本轮不逐元素比较Adam moment，不导出全部logits或逐token loss。前向逐层诊断的窗口和范围见第4节。

## 2. 运行前固定的门槛

`checks/20261031-numeric/protocol.json` 在训练启动前记录时间与SHA256；其内容哈希为 `a49da90bbcfe5cc67bbe97047f3b0008c09e7fcf8261174e90803f7612371fce`。后续诊断候选沿用相同门槛，没有因失败调整阈值。

- 初始模型及FP32主参数、实际输入、每步RNG、消耗样本数严格相同；0 skipped update；LoRA冻结基座保持逐字节相同。
- 10步的loss绝对误差≤`0.001 + 0.001×|reference|`；梯度范数相对误差≤1%。
- 第1、10步完整梯度拼接向量：相对L2≤1%，cosine≥0.9999。
- 参考范数≥全局拼接范数`1e-6`的单个梯度张量：相对L2≤5%、cosine≥0.995；更小张量的绝对误差范数≤全局参考范数`1e-6`。
- 第10步FP32主参数累计更新：相对L2≤5%，cosine≥0.995；最终全部模型张量有限。

这些是短程工程筛查标准，不是训练质量验收，也不是单算子BF16 `atol=rtol=0.02` 的延伸。

## 3. 原始后端的整模型结果

| 指标 | CPT | LoRA |
|---|---:|---:|
| 10步完整运行 | 通过 | 通过 |
| 初始状态、输入、RNG等严格控制 | 通过 | 通过 |
| reference与上轮reference逐步审计严格复现 | 通过 | 通过 |
| loss门槛 | 全部通过 | step4、7未通过 |
| 最大loss绝对误差 | 0.00233555 | 0.00690043 |
| 梯度范数门槛失败步 | 4、6、7 | 4、5、6、10 |
| 最大梯度范数相对差异 | 5.3466% | 5.9513% |
| **step1完整梯度相对L2** | **21.1256%** | **6.5801%** |
| step1梯度cosine | 0.977586 | 0.997837 |
| step1失败梯度张量数 | 716 / 750 | 452 / 992 |
| **step10完整梯度相对L2** | **11.1168%** | **14.5361%** |
| step10梯度cosine | 0.993875 | 0.989886 |
| step10失败梯度张量数 | 687 / 750 | 740 / 992 |
| **10步累计更新相对L2** | **26.5928%** | **11.2267%** |
| 累计更新cosine | 0.964629 | 0.993697 |
| 最终模型张量全部有限 | 通过 | 通过 |
| **整模型工程门槛** | **未通过** | **未通过** |

四rank的初始模型/主参数哈希、实际tokens/labels/positions/loss mask、步前/步后Python/NumPy/CPU/CUDA/MCore tracker RNG及样本计数均相同。LoRA冻结基座的初始与最终哈希相同，fresh B更新检查通过。两组新reference的全部逐步审计与上一轮同任务reference严格一致，观测器没有改变已验证的参考轨迹。

第10步loss仍很接近：CPT为reference `1.5325084924697876`、Triton `1.5319017171859741`；LoRA为 `1.4227726459503174` / `1.4220649003982544`。这两个末步标量都通过loss门槛，不能替代梯度和更新验证。

最终模型权重整体相对L2只有CPT约0.03439%、LoRA约0.00938%，但累计更新差异分别达26.59%、11.23%。预训练权重占主导，因此本轮明确以更新量另设门槛。

## 4. 差异定位与独立候选

### BF16反向中间舍入

隔离RMSNorm使用H2560、eps1e-5、`[512,1,2560]`，覆盖两个随机种子和由第一条实际训练tokens及原始embedding重建的输入。原始输入梯度相对L2误差约0.247%–0.259%；显式用整数位操作保留BF16中间舍入后，误差降到0–`1.18e-7`，最多2个元素不同。实际embedding用例的tokens哈希与训练记录匹配。

独立消融表明：只设置`enable_fp_fusion=False`没有改变原始误差；显式BF16舍入在fusion开/关时都改善结果。这是对当前CoreX环境中中间舍入行为的局部证据；尚未通过编译器IR分析确认根因，不能泛化为所有CoreX版本或所有dtype的结论。

`strict-v2`仅在外部worker替换RMSNorm的dx kernel，原始前向和dw不动，限定BF16/H2560。LoRA完整10步及全量审查仍未通过：

| 候选 | step1梯度相对L2 | step10梯度相对L2 | 累计更新相对L2 | 结果 |
|---|---:|---:|---:|---|
| 原始Triton | 6.5801% | 14.5361% | 11.2267% | 未通过 |
| strict-v2：显式BF16 dx舍入 | 6.6286% | 13.1016% | 10.0036% | 未通过 |

因此局部dx改善不足以证明整模型对齐。

### 相同真实激活的前向统计量

另跑一个明确标注的reference LoRA诊断步，仍为4B/TP2PP2/BF16/S512/MBS1/GBS16/重计算关闭。每个RMSNorm只观测第一个microbatch，TP rank0所在的两个进程合计覆盖125个逻辑模块；额外计算不接入训练图。该诊断步在四rank的输入、loss、梯度范数及RNG与原reference第一步严格相同。只在此诊断命令移除原生保存参数，避免生成无用大checkpoint。

在同一份reference输入和weight上：

| 算法 | 有输出差异的模块 | 不同元素 / 比较元素 | 最大单模块相对L2 |
|---|---:|---:|---:|
| 原始Triton | **38 / 125** | **916 / 163,840,000** | `9.57e-5` |
| 提供Torch统计量，再用Triton原前向算术 | 0 / 125 | 0 / 163,840,000 | 0 |
| 提供Torch统计量，再显式BF16舍入 | 0 / 125 | 0 / 163,840,000 | 0 |

这把本窗口的前向差异定位到FP32统计量路径；FP32的微小归约差异可能跨过BF16舍入边界。它没有证明所有microbatch、所有训练步的logits逐元素一致。

### 混合统计诊断版

`stats-v3`明确使用**Torch FP32统计量 + Triton前向算术 + strict dx + 原始Triton dw**。它是独立混合诊断版，不能标为原始纯Triton后端通过。

LoRA第一步loss与reference相同；完整审查结果仍未通过：step1完整梯度相对L2 **1.9796%**、cosine0.999806，step10 **12.5624%**、cosine0.992198，累计更新 **8.5346%**、cosine0.996370。第1步没有单张量失败，但拼接梯度仍超过1%门槛；第10步有677个梯度张量失败。初始状态、输入、RNG及冻结基座控制通过。

该候选预先规定“先LoRA，完整审查通过才扩展CPT”；由于LoRA失败，没有继续生成CPT混合候选。原始CPT与LoRA对照已经完成，不受这个条件影响。

## 5. 产物与后续工作

远端统一入口：`/private/yp/jiuge-training/checks/20261031-numeric/`。

- `protocol.json`、`preregistration.json`：原始方案、阈值、源码及工具哈希。
- `cpt-comparison.json`、`lora-comparison.json`：原始后端全量比较。
- `rounding-probe.json`、`fusion-probe.json`、`forward-probe-summary.json`：隔离与真实激活诊断。
- `lora-comparison-strict-v2.json`、`lora-comparison-stats-v3.json`：两个独立候选的完整失败证据。
- `strict-v2/`、`stats-v3/`、`forward-probe/`：候选声明、命令、控制器和资源日志。
- `runs/*numeric-v1/`、`runs/*numeric-strict-v2/`、`runs/*numeric-stats-v3/`：逐步审计、完整梯度/主参数快照、原生最终checkpoint；所有大产物均在账户持久化目录。

本地代码镜像在 `experiments/jiuge_training/numeric_20261031/`，实际运行工具在远端 `tools/`。总证据JSON已逐字节同步到本地，包含配置、实际命令、工具SHA、逐张量指标、最终资源快照及产物清单。

后续先保持reference训练基线。修复方向是明确复现reference前向FP32统计量的归约语义，并核对反向链式计算顺序与每处BF16舍入；分别验证dx和适用于CPT的dw，再沿用本轮不变的整模型门槛。局部算子通过、loss接近或权重整体误差小均不能代替这一步。

本轮没有新增Profiler采集，也不作端到端性能或训练质量结论；正式测量规范仍待用户确定。结束时四卡0%/约68MiB，无GPU进程，容器保留；账户网盘约762G已用、263G可用。没有删除历史产物。
