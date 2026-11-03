# 九格四卡天数实验：统一配置与计划

更新：2026-10-31（北京时间）。本文记录本次用户约定、仓库核查和实际环境快照，作为后续新实验入口。旧7B报告保留原始条件，不作为新4B实验的验收结论。

## 1. 已确定的实验约定

| 项目 | 统一约定 |
|---|---|
| 训练工程 | [ZhouBencheng/jiuge-training](https://github.com/ZhouBencheng/jiuge-training)，每轮记录实际 commit 与本地补丁 |
| 硬件与模型 | 后续在四卡天数环境上统一使用九格 **4B**；四卡不自动等于 TP4 |
| 对照目标 | 对比 Megatron local spec 的原生算子后端；具体 reference 的实现边界见下一节 |
| 激活重计算 | **关闭**，启动时显式覆盖 `--set task.recompute=false` 并核对有效配置 |
| Trace | 直接使用 **PyTorch Profiler**；本轮不要求额外 ixSYS 采集 |
| 持久化空间 | 用户已明确：**以后所有DataThinking新实验统一在 `/private/yp/` 下开展**，本工程工作区为 `/private/yp/jiuge-training/`；不再使用此前的 `/private/cdx/` 记录目录开展新实验 |
| 正式端到端规范 | 用户后续补充；目前只制定准备、正确性检查及热点分析计划，不预设正式测速步数、序列长度或收益门槛 |
| 旧容器 | 先核查再决定复用；仅在不再需要、关键内容已保存且确认无人使用时才考虑删除 |

## 2. 仓库版本与必须处理的默认值差异

只读核查的 main 为 [`9fabd71df4897f08267df5431aa480ab9ef70a75`](https://github.com/ZhouBencheng/jiuge-training/commit/9fabd71df4897f08267df5431aa480ab9ef70a75)（2026-10-29）；固定 Megatron 源码 commit 为 `5be9626709af2722333bf54797c954c09edeada3`。后续运行若换版本，重新导出配置并检查差异。

1. **重计算需要覆盖。** 当前 [CPT](https://github.com/ZhouBencheng/jiuge-training/blob/9fabd71df4897f08267df5431aa480ab9ef70a75/configs/task/cpt.toml#L14) 和 [LoRA](https://github.com/ZhouBencheng/jiuge-training/blob/9fabd71df4897f08267df5431aa480ab9ef70a75/configs/task/lora.toml#L14) 均为 `recompute=true`。用户最新约定优先，必须改为 false；历史开启重计算的显存结果不能保证关闭后仍能运行。
2. **local 的边界要写清。** [config.py](https://github.com/ZhouBencheng/jiuge-training/blob/9fabd71df4897f08267df5431aa480ab9ef70a75/src/jiuge_training/config.py#L217) 传入 `--transformer-impl local`，但 [model.py](https://github.com/ZhouBencheng/jiuge-training/blob/9fabd71df4897f08267df5431aa480ab9ef70a75/src/jiuge_training/model.py#L311) 手工组装 ModuleSpec，包含九格 attention 和 RMSNorm 适配。因此现成 baseline 应标注为“固定 MCore local 实现 + 九格语义适配 reference”。若目标要求严格的 Megatron 原生 `get_gpt_layer_local_spec`，先建立独立对照并确认语义及数值一致。
3. **backend 开关范围有限。** 当前 `reference/triton` 只切换 RMSNorm 实现；attention 仍为工程的 PyTorch eager 路径，不能把 triton 选项解释为全模型 Triton 化。
4. **Profiler 使用外部包装器。** 原始 [train.py](https://github.com/ZhouBencheng/jiuge-training/blob/9fabd71df4897f08267df5431aa480ab9ef70a75/src/jiuge_training/train.py) 与 [CLI](https://github.com/ZhouBencheng/jiuge-training/blob/9fabd71df4897f08267df5431aa480ab9ef70a75/src/jiuge_training/cli.py) 没有内置采集入口；工作区现已用外部包装器采到四rank GPU trace及S512 shape，未修改固定源码。具体窗口和开关见测试报告。
5. **实际权重路径与元数据不同。** 仓库4B默认 `/private/models/FM9G-4B/9G4B` 在本次旧容器中不存在；原始权重在 `/private/models/9G4B`。原始JSON还省略了原配置类提供的三个默认值，直接启动会在适配器校验处失败。因此后续使用 `/private/yp/jiuge-training/models/9G4B-resolved`：仅显式补充 `model_type="fm9g"`、`tie_word_embeddings=true`、`rope_theta=10000.0`，原有字段和权重不改；原/新AutoConfig有效配置严格等价。目录内保存原配置、有效配置和 `METADATA_NORMALIZATION.json`。数据manifest绑定模型路径及config指纹，必须针对该目录准备数据。

## 3. 四种模型与当前默认参数

来源：[模型配置目录](https://github.com/ZhouBencheng/jiuge-training/tree/9fabd71df4897f08267df5431aa480ab9ef70a75/configs/model)、[配置说明](https://github.com/ZhouBencheng/jiuge-training/blob/9fabd71df4897f08267df5431aa480ab9ef70a75/docs/configuration.md)。以下区分仓库默认与本次正式约定；默认值尚不等于本轮测量规范。

| CLI 模型 | 类型 | 层数 | hidden | FFN | Q/KV heads | vocab |
|---|---|---:|---:|---:|---:|---:|
| **4b** | **FM9G-4B，GQA，本轮四卡统一模型** | **62** | **2560** | **6400** | **40/4** | **73448** |
| 7b | FM9G-7B，MHA | 28 | 3584 | 18944 | 28/28 | 73448 |
| 8b-mha | CPM-9G-8B-MHA | 32 | 4096 | 14336 | 32/32 | 119696 |
| 8b | CPM-9G-8B Thinking，GQA | 32 | 4096 | 16384 | 32/2 | 73448 |

| 参数 | CPT 默认 | LoRA 默认 |
|---|---|---|
| 训练范围 | 全部基座参数 | 并行线性层 adapter；rank=8、alpha=16、dropout=0 |
| dtype / seed | BF16 / 1234 | BF16 / 1234 |
| seq / micro batch / global batch | 512 / 1 / 16 | 512 / 1 / 16 |
| 迭代 / 保存间隔 | 10 / 5 | 10 / 5 |
| optimizer / LR / weight decay | Adam / 1e-5 / 0.01 | Adam / 1e-4 / 0.01 |
| distributed optimizer | true | false |
| LR schedule / clip-grad | constant / 1 | constant / 1 |
| attention / hidden dropout | 0 / 0 | 0 / 0 |
| 激活重计算 | 仓库true；**本轮覆盖false** | 仓库true；**本轮覆盖false** |

仓库关闭多种融合开关；CP=1、sequence parallel=false。不要沿用旧7B CPU AdamW offload或融合设置。四卡预设的 DP/TP/PP 为：dp4=4/1/1、dp2-tp2=2/2/1、dp2-pp2=2/1/2、tp4=1/4/1、tp2-pp2=1/2/2、pp4=1/1/4。4B PP4层划分为16/15/15/16。

默认数据为官方中文版 Alpaca 48,818条，去重后 seed1234、98/2切分，各模型单独分词。CPT监督所有非padding target；LoRA监督回答及EOS，不能与历史全参数SFT混称。修改seq必须准备相匹配的数据与manifest。GBS=MBS×DP×梯度累积次数，累积次数至少为PP。

## 4. Edge 与旧容器实际核查

2026-10-30约18:50（北京时间），从 Edge 已打开的 DataThinking 专家模式确认旧资源为运行中，通过已有SSH密钥及已存主机校验只读核查；未读取页面密码。

| 项目 | 本次观察 |
|---|---|
| 容器 | `ai-exp-sy2506311-jiuge7bbiv150tp2-pod-yxxt` |
| 镜像 | `vllm0.11.2-4.4.0-x86` |
| 设备 | 4张BI-V150，各32768MiB；容器名中的TP2不代表只有两张卡 |
| 空闲快照 | 四卡均0%利用率、各约68MiB占用；ixsmi显示No running processes；容器内只观察到sshd及本次检查进程 |
| 持久化 | `/private` 为可写NFS挂载，约1.0T总量、894G可用 |
| 4B模型 | `/private/models/9G4B`，有约9.23GB的pytorch_model.bin及tokenizer/config；元数据与4B配置主要尺寸一致，尚未实际加载或验证权重内容 |
| 软件 | torch 2.7.1+corex.4.4.0、triton 3.1.0+corex.4.4.0、transformers 4.57.6；IX-ML/driver 4.4.0，与仓库锁文件对应版本一致 |
| MCore | 未发现已安装的megatron-core发行包；后续确认固定源码目录及加载路径，不能据此断定源码不存在 |
| 工具调用 | ixsmi位于`/usr/local/corex/bin/ixsmi`；SSH默认环境缺共享库搜索路径，检查子进程需包含CoreX lib/lib64 |
| 决定 | **保留并优先复用**，无需为本次记录额外创建容器；训练启动前重新确认占用与可用额度 |

这是单次空闲快照，不保证后续独占。此次未启动训练、未停止/删除容器、未改其他人的目录或平台端口。旧资料已在`/private`下发现，但未完成逐项内容/恢复审计，本次没有宣布“已完成删除前备份”。

## 5. 实验目标与推进计划

总体目标：在统一四卡4B、关闭重计算的条件下，建立可复现的参考后端，找出真实训练热点，再验证单个算子替换的正确性与收益，为后续统一端到端规范提供输入。

| 阶段 | 工作 | 产出与进入下一阶段的依据 |
|---|---|---|
| P0 配置与环境 | 独立工作空间；固定工程/MCore版本；确认4B权重、数据、四卡通信、后端边界；导出关闭重计算后的完整配置 | 命令、有效配置、环境/数据/代码指纹；原生local spec对照的定义 |
| P1 短跑正确性与容量 | 建议先CPT PP4及LoRA TP2PP2做有界检查，从seq64、MBS1、GBS4、2步起步，再逐步检查seq512；这些是预检候选，尚非正式配置 | 无OOM、loss有限、关键输出/梯度/参数更新正确、保存恢复可用；关闭重计算时重新测峰值显存 |
| P2 reference trace | 在实际训练步接PyTorch profiler；CPU+CUDA activities，各rank独立trace；有限预热和采集，记录shape，内存/stack选项按需 | GEMM、attention、RMSNorm、通信和CPU等待的热点表；trace与step边界可对应 |
| P3 单算子对照 | 依据新4B trace选1–2个热点；优先评估现有RMSNorm reference/triton对照，attention等另立候选；每次只改变一个因素 | 前向/反向数值检查、同shape微基准、相同配置的A/B trace；结论限制在实际dtype、shape和布局 |
| P4 端到端验收 | 用户补齐正式测量规范后统一预热、步数、重复、计时边界、吞吐分母、精度与资源条件；正式测速关闭profiler | 可复现的整模型对照结论；目前不宣布正式端到端加速 |

固定对照条件包括初始权重、实际输入、seed/RNG、并行布局、dtype、batch、seq、优化器、数据监督语义和重计算开关。trace中的累计算子时间不等于step墙钟，微基准加速也不直接等于训练加速。旧7B热点、收益和三种子/120步协议均不自动迁入本轮。

## 6. 预检命令模板

下例在已准备好固定版本工程、环境与seq64数据后使用；`dry-run`仅打印启动命令，不验证设备、权重或数据，也不启动训练。

```bash
cd /private/yp/jiuge-training/source
source /private/yp/jiuge-training/env.sh
python3 -m jiuge_training.cli dry-run \
  --model 4b --task cpt --parallel pp4 --backend reference \
  --megatron-root /private/yp/jiuge-training/Megatron-LM \
  --run-dir /private/yp/jiuge-training/runs/4b-cpt-pp4-reference-no-recompute-seq64 \
  --set 'model.path="/private/yp/jiuge-training/models/9G4B-resolved"' \
  --set task.recompute=false \
  --set task.seq_length=64 \
  --set task.micro_batch_size=1 \
  --set task.global_batch_size=4 \
  --set task.train_iters=2 \
  --set task.save_interval=2 \
  --set task.eval_iters=0 \
  --set 'task.data_path="/private/yp/jiuge-training/data/4b-cpt-seq64-resolved"'
```

LoRA预检改为`--task lora --parallel tp2-pp2`，对应独立数据和run-dir。Triton对照只改backend及输出目录，其他条件固定。CLI的`validate`可显示合并配置，但它早于run-dir保存路径重写返回；最终保存路径以dry-run中的`--save`或显式`task.save`覆盖为准。

数据准备模板：

```bash
python3 -m jiuge_training.cli prepare-data \
  --source /path/to/alpaca_gpt4_data_zh.json \
  --output /private/yp/jiuge-training/data/4b-cpt-seq64-resolved \
  --model-path /private/yp/jiuge-training/models/9G4B-resolved \
  --task cpt --seq-length 64 --seed 1234 \
  --validation-fraction 0.02 --verify-alpaca
```

以上是预检模板，实际执行结果另存测试报告。用户已授权开始使用旧容器测试；所有新源码、依赖、缓存、数据、日志与checkpoint均置于 `/private/yp/`。

账户级存储已从DataThinking「资源仓库→我的网盘」核实：`/private` 是个人网盘在容器中的映射路径，持久化保存；该页提供SFTP入口。`/private/yp/` 因此位于账户网盘，并非容器临时文件层。

## 7. 基线验证（2026-10-30）

[S512验证报告](S512_VALIDATION_20261030.md)：真实4B、BF16、S512/MBS1/GBS16、重计算关闭，CPT与LoRA均以四卡TP2PP2完成10步。CPT的PP4在本配置OOM，未作为后续S512基线。恢复路径补上workerless迭代器的CPU RNG保护后，两个任务的step9→10与连续运行在输入、精确loss/梯度范数、RNG和最终全量状态上相符。RMSNorm单算子27组通过既有容差，S512/GBS4 shape trace四rank通过；整模型后端数值对齐及正式端到端性能仍待后续验收。

## 8. 整模型后端数值对照（2026-10-31）

[整模型数值报告](NUMERIC_ALIGNMENT_20261031.md)：沿用4B、四卡TP2PP2、BF16、S512、MBS1/GBS16、关闭重计算，CPT/LoRA两后端均完成10步。严格初始状态、输入、RNG和reference复现控制通过，但原始Triton的step1/10完整梯度相对L2分别为CPT21.13%/11.12%、LoRA6.58%/14.54%，累计更新差异26.59%/11.23%，未通过预先制定的工程门槛。

继续使用reference基线。RMSNorm的BF16反向中间舍入及前向FP32统计量是明确的检查方向；独立strict-v2和混合stats-v3候选均未通过LoRA整模型门槛，未替换原始后端。原始失败、候选、阈值和逐张量证据保存在 `/private/yp/jiuge-training/checks/20261031-numeric/`，本轮未作正式性能验收。

## 9. 反向图修复后的混合候选（2026-10-31）

[修复与整模型报告](NUMERIC_FIX_20261031.md)：发现合并RMSNorm反向会改变其与残差分支的BF16梯度累加顺序。保留`x`及`x.float()`两个梯度边的`split-v4`混合候选，在相同4B/TP2PP2/BF16/S512/MBS1/GBS16/关闭重计算配置下，通过CPT和LoRA十步原数值门槛。LoRA捕获梯度及主参数内容SHA一致；CPT第1/10步完整梯度相对L2为`4.94e-7`/`2.84e-7`，累计更新`3.32e-6`，最终模型数值一致。

该候选显式使用Torch前向统计量和反向行归约，其他相应算术使用Triton；原始纯Triton后端仍未验收。固定源码未改，实际使用外部`tools/numeric_training_split.py`，审计数据保存在 `/private/yp/jiuge-training/checks/20261031-numeric-fix/`。reference继续保留为基线；后续可逐项迁移原生归约并复测，再按正式规范测性能。

## 10. 统计量与行归约的完整替换（2026-10-31）

[Triton归约替换报告](TRITON_REDUCTIONS_20261031.md)：在同一4B/四卡TP2PP2/BF16/S512/MBS1/GBS16/关闭重计算条件下，前向统计量替换、完整行VJP替换两个阶段的CPT/LoRA四组10步全部通过原数值门槛。与混合split-v4相比，捕获梯度、主参数及最终模型参数内容SHA均一致；完整替换版的Torch目标统计量/行算术调用已由Profiler核验为0，两条反向边与外部cast/梯度累加保留。

实际入口为`REDUCTION_STAGE=full`配合`tools/compact_numeric_training.py`，实现位于`tools/reduction_patch.py`与`tools/native_order_reductions.py`，原始训练源码未改。数据保存在 `/private/yp/jiuge-training/checks/20261031-triton-reductions/`。本轮改用无损引用固定reference并完整保存变化张量，重建后仍全元素审查；未重复生成optimizer checkpoint，须保留引用链。reference作对照，完整替换版可作为后续数值已验收候选；正式性能和候选原生恢复尚未验收。

## 11. 三版探索性性能对比（2026-10-31）

[性能报告](PERFORMANCE_COMPARISON_20261031.md)：同配置reference、split-v4混合版、reductions-full-v1完整Triton归约版分别测LoRA/CPT。每任务每版独立三轮，5步预热后计时10步，轮换版本顺序；另做5步预热后采1步的四rank PyTorch Profiler。共18组计时+6组trace。外部测速入口`tools/performance_training.py`不带全量张量审计，实际命令移除checkpoint写盘。

三版平均步时（reference/mixed/triton）为LoRA 6.0065/5.9840/6.0254秒、CPT 5.3278/5.2243/5.1826秒。使用四rank同一步同步耗时最大值后取均值；覆盖train_step内前反向、通信、optimizer，不包含加载、step外日志和保存。重复范围重叠，LoRA没有稳定加速证据，CPT候选均值改善仍应保留波动边界；两候选峰值显存下降。trace有探针开销，只用于解释执行分布，不能算加速比。正式端到端验收仍等待用户规范，候选恢复和长训练仍未验收。

## 12. trace归因后的优化顺序（2026-10-31）

[归因报告](TRACE_DIAGNOSIS_20261031.md)：原24份trace做区间并集与主机发射关联，另完成reference/full的LoRA/CPT四组无profiler阶段补测。LoRA的GEMM和TP通信数量未被norm替换减少，约为CPT两倍；full norm前向host调用增加约60–76ms/步。CPT改善出现在前反向而非optimizer，但原候选三轮波动量级足以掩盖小收益，具体抖动原因尚不能唯一归因。

下一候选先将norm前向统计+输出、反向行归约+双输出分别融合一次发射，减少未消费的中间张量；保留外部cast、两条反向边与原舍入/数值门槛。LoRA更大优化空间在低秩线性层、scale/add及小张量TP通信。PP每步17次同步有固定MCore批量P2P竞态兼容依据，单列验证，不在本轮删除。后续以相邻配对、跨启动重复及阶段遥测核实收益，正式端到端验收仍待规范。


## 13. RMSNorm双分支融合候选（2026-10-31）

[融合与配对报告](FUSED_RMSNORM_20261031.md)：fused-split-v1把前向统计+输出、反向行VJP+双梯度分别合为一次Triton发射。仍使用外部FP32 cast输入边，direct为BF16、variance为FP32；dw原三kernel不变。同一kernel内借用R缓冲区写mean，经barrier/volatile读取后计算r覆盖R，保留FP32物化界线，消除已观测的统计量差异。失败prototype和IR保留。

133组局部样本逐元素对齐当前版；LoRA/CPT整模型10步通过原protocol，捕获梯度/主参数和最终模型SHA均与reductions-full-v1一致。Profiler核验独立norm前反向kernel数LoRA7→5、CPT10→8。每任务四对AB/BA交错启动、5步预热+10步测量，共16组；每对训练标量一致且融合更快。LoRA均值5.8626→5.7025秒、CPT5.0539→4.9251秒，吞吐+2.81%/+2.62%；norm前向host约219→170ms、243→194ms，显存峰值不变。

产物位于/private/yp/jiuge-training/checks/20261031-fused-norm/；数值入口为tools/fused_numeric_training.py，配对入口tools/fused_performance_training.py需显式PERF_ARM=triton,PERF_FUSED=1。这是固定条件下的探索性收益，正式端到端、长训练和候选原生恢复未验收；原始源码和PP同步未修改。

## 14. 融合版整模型trace与LoRA主机路径（2026-11-01）

[FUSED_FULLTRACE_20261101.md](FUSED_FULLTRACE_20261101.md)记录融合norm版两任务四rank trace。LoRA rank0 A/B前反向GPU时间和约403ms，scale/add前向约45.7ms，8KiB逻辑低秩TP约47.2ms；各项可重叠，不等同可去除的整步时间。异步通信须用GPU注释关联，不把未匹配launch当作零通信。

[LORA_CPP_AUTOGRAD_20261101.md](LORA_CPP_AUTOGRAD_20261101.md)：LoRA A/B保持原生BF16 GEMM及梯度求解顺序，用释放GIL的C++ autograd包装压缩节点。LoRA/CPT十步原门槛通过，捕获内容与融合norm版SHA一致。LoRA四对均更快，平均5.8850→5.7431秒，吞吐+2.47%；CPT算法未改，一对仅作负对照。入口tools/lora_cpp_numeric.py与tools/lora_cpp_perf.py，后者PERF_LORA_CPP=1启用候选，0为融合norm+native linear。原CLI不会自动启用。

## 15. 慢步与后续筛查（2026-11-01）

[GC_MEASUREMENT_20261101.md](GC_MEASUREMENT_20261101.md)：30步只观测发现LoRA两次约1秒慢步对应单rank gen2 GC；CPT本轮未复现，不能泛化到历史全部抖动。统一两臂预热末GC且自动GC仍开启的20步窗口，两对LoRA平均5.8457→5.6495秒（+3.47%吞吐）；GC成本另计，不作生产优化收益，也不覆盖原四对结果。两种测量协议不能混合求均值。CPU绑核隔离筛查未达门槛，不进入整模型。

scale/add融合的Python Triton入口逐元通过但隔离更慢；C++/CoreX设备入口通过局部、边界及LoRA/CPT十步原判据整模型数值验证，捕获SHA与此前融合版一致。两对LoRA配对方向相反，均值吞吐约-0.18%，故不启用。阶段checks/20261101-scale-add/，本地代码镜像scale_add_20261101/。正式端到端规范和长训练仍未完成。

## 16. 已选基线的原生恢复与新trace（2026-11-01）

[原生恢复](CANDIDATE_NATIVE_RESTORE_20261101.md)：当前C++ A/B加fused-split-v1 norm、不启用scale/add；两任务在第9步保存，第10步新进程加载，四rank模型、FP32主参数、实际Adam moments、scheduler及输入/RNG/loss/grad_norm/计数均精确一致。只保存一份第9步原生checkpoint，保留全部引用链。适用范围限当前workerless固定数据与10步短程。

[更新trace](CANDIDATE_TRACE_20261101.md)：两臂LoRA每rank差1984个kernel，但融合epilogue的设备时间和没有稳定改善；A/B GEMM、低秩TP次数不变。采样CPU区间不能当纯计算时间，collective GPU时间和不等于通信链路耗时或关键路径。原trace不变，v2分析补充无External id的C++ launch关联。

## 17. CPT norm products融合（2026-11-02继续）

[候选报告](NORM_DW_20261101.md)：只对CPT把独立products写入并入现有反向行kernel，保留BF16物化和后续两级dw归约；LoRA实际路径保持原norm。133组逐元一致，两任务十步原数值判据通过，step9/10完整状态与已选基线一致。四对CPT探索性配对均同向，步时平均5.037026→4.997180秒，吞吐约+0.80%；一对LoRA负对照不变算法。新候选的原生step9只读加载→step10验证已通过，四rank完整状态和训练审计精确一致，未新写大型checkpoint；适用范围限当前固定workerless数据与短程配置。正式端到端测量规范、长训练仍另行验收。

## 18. LoRA A 与低秩TP主机路径（2026-11-02）

[LoRA A+TP候选](LORA_TP_20261102.md)：仅压缩C++ A与原TP映射的autograd/主机路径，仍使用原group、SUM同步、原生BF16 GEMM和相同通信。四rank两个TP2组共192例逐元一致，column/row隔离前后向慢端成本分别327.288→263.781µs、332.711→292.301µs。原十步LoRA/CPT整模型数值及step9/10完整状态指纹均通过；四对LoRA整模型配对同向，平均5.688279→5.564454秒，吞吐约+2.23%，满足预注册选择条件。LoRA从旧原生step9只读加载执行step10也已精确通过；CPT一对未变算法负对照有明显波动，不能据此称CPT性能回退。新候选整模型trace正在采集；正式端到端协议和长训练仍未验收。

