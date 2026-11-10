# 复现CPT端到端提升

本仓库提供两种核验：普通电脑离线复算已发布证据，以及在同型号四卡环境重新执行完整实验。**16.38%是相对适配local基线、已准备数据缓存的50步完整任务提升**；吞吐为409600/任务耗时，提升为基线均值/候选均值−1，耗时下降为1−候选均值/基线均值。

## 一、离线复算：无需GPU

安装Python3.11或更高版本，从仓库根目录执行：

```bash
git clone https://github.com/yangpu-333/jiuge-training-experiments.git
cd jiuge-training-experiments
python scripts/verify_package_20261110.py
python scripts/check_links.py
```

程序先核对证据包SHA256、安全解包并核对842项文件哈希，再复查3600条rank-step记录、3600组成对标量、144份CRC回执、27个作业exit0、保存恢复状态指纹，独立计算六轮均值及20000次配对bootstrap。预期输出`passed: true`、吞吐`+16.3827%`、耗时`-14.0766%`、`target=True`。离线核验检查的是原GPU审计回执，不会在本机重新验证未交付的GPU张量或大checkpoint。

## 二、重新跑四卡实验

### 1. 硬件、环境与固定版本

四张Iluvatar BI-V150（每卡32GiB）、Linux x86_64，使用提供CoreX4.4的厂商镜像。建议至少128GiB主机内存；输入准备完后保留至少80GiB可用持久化空间，测量期间始终保留10GiB余量。一次完整checkpoint约56GiB，程序只轮换自己的新scratch槽位。不同存储、CPU负载、驱动、主机配置会改变绝对耗时和比例，不能保证另一台机器精确得到16.38%。

| 依赖 | 固定条件 |
|---|---|
| 九格训练配置 | `ZhouBencheng/jiuge-training@9fabd71df4897f08267df5431aa480ab9ef70a75` |
| Megatron-LM | `NVIDIA/Megatron-LM@5be9626709af2722333bf54797c954c09edeada3` |
| Python | 3.12.3（GPU实测环境） |
| PyTorch / Triton | `2.7.1+corex.4.4.0` / `3.1.0+corex.4.4.0` |
| Transformers / pybind11 | 4.57.6 / 2.13.6 |

完整版本见[实测环境清单](reproduction/20261110/environment/environment.lock.json)。这是厂商镜像清单，不是可用普通PyPI重建的完整lock。须已有可用的CoreX编译器、Apex和通信库；`ninja`、C++编译器及构建头文件也须可用。本文不重新安装或替换厂商torch。

保持原固定路径，使用**没有旧tools和旧运行配置的新项目目录**。以下命令在GPU服务器执行；`PUB`指向本仓库的绝对路径。已存在的工作目录不覆盖，先保留原实验，换用干净的环境或挂载点。

```bash
export PUB=/absolute/path/to/jiuge-training-experiments
export B=/private/yp/jiuge-training
mkdir -p "$B"
git clone https://github.com/ZhouBencheng/jiuge-training.git "$B/source"
git -C "$B/source" checkout --detach 9fabd71df4897f08267df5431aa480ab9ef70a75
git clone https://github.com/NVIDIA/Megatron-LM.git "$B/Megatron-LM"
git -C "$B/Megatron-LM" checkout --detach 5be9626709af2722333bf54797c954c09edeada3
source "$PUB/reproduction/20261110/environment/env.sh"
bash "$B/source/scripts/bootstrap.sh"
python "$PUB/reproduction/20261110/prepare.py" --install --run-name replication-20261110
export PYTHONPATH="$B/tools:$B/source/src:$B/Megatron-LM:$B/source/.deps${PYTHONPATH:+:$PYTHONPATH}"
```

[安装器](reproduction/20261110/prepare.py)核对两套上游commit、全部实测训练源码SHA和[28份工具源码](reproduction/20261110/TOOLS_SHA256.json)。所有worker和内核保持原字节；只生成一个新控制器，移除旧历史归档等待和已过期日历截止时间，变更保存为`reproduction-controller.diff`及`reproduction-adaptation.json`。仍保留每任务1200秒超时、原数值门槛、完整保存及恢复检查。不会创建、续费或删除云平台资源。

### 2. 取得原模型并校验

使用原生PyTorch版**FM9G-4B / 9G4B.tar**，不要换成同名HF转存版。公开发布页见[九格项目](https://www.osredm.com/jiuyuan/CPM-9G-8B#20250225fm9g4b)，归档来源及全部成员哈希保存在[MODEL_SOURCE.json](reproduction/20261110/inputs/MODEL_SOURCE.json)。在遵循上游获取条件后，将原始解压目录传给准备器；本仓库不转发权重。

```bash
mkdir -p "$B/models/original"
curl -fL https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/9G4B.tar -o "$B/models/original/9G4B.tar"
echo '28ff0bad05c04b97433c57de1ed2aef0ccd233536e260773a4a9e7482b6db89c  '"$B/models/original/9G4B.tar" | sha256sum -c -
tar -xf "$B/models/original/9G4B.tar" -C "$B/models/original"
python "$PUB/reproduction/20261110/prepare_inputs.py" --model-source "$B/models/original/9G4B"
```

程序逐文件验证原件，包括权重SHA`f74ea28d869a6467995ace692cef2e62cb801732691ca087eed3e6a3ab16826c`；然后新建`models/9G4B-resolved`，仅在配置中显式写出`model_type=fm9g`、`tie_word_embeddings=true`、`rope_theta=10000.0`，权重保持原字节。模型配置及处理说明见[config.resolved.json](reproduction/20261110/inputs/config.resolved.json)、[归一化记录](reproduction/20261110/inputs/METADATA_NORMALIZATION.json)。

### 3. 准备相同数据

语料为固定版Alpaca GPT-4中文48,818条记录，用于流程复现，不用于论证领域训练质量。数据遵循上游CC BY-NC4.0条件，代码与数据许可不同；来源见[固定语料](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/38b93b75a80b019cc3aad362ca6f5bb0c3908959/data/alpaca_gpt4_data_zh.json)和[上游说明](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)。

```bash
mkdir -p "$B/data/raw"
curl -fL https://raw.githubusercontent.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/38b93b75a80b019cc3aad362ca6f5bb0c3908959/data/alpaca_gpt4_data_zh.json -o "$B/data/raw/alpaca_gpt4_data_zh.json"
python -m jiuge_training.cli prepare-data \
  --source "$B/data/raw/alpaca_gpt4_data_zh.json" \
  --output "$B/data/4b-cpt-seq512-resolved" \
  --model-path "$B/models/9G4B-resolved" \
  --task cpt --seq-length 512 --seed 1234 \
  --validation-fraction 0.02 --verify-alpaca
python "$PUB/reproduction/20261110/prepare_inputs.py" --check-data
```

必须匹配[原数据manifest](reproduction/20261110/inputs/data-manifest.original.json)：源文件SHA`6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903`，train/validation分别15,995/338个样本，S512。若tokenizer、模型config、transformers版本或生成数据SHA不一致，应停止并排查，不能换用合成数据宣称同一结果。

### 4. 环境检查、预热和正式运行

```bash
python "$B/source/scripts/doctor.py"
torchrun --standalone --nproc_per_node=4 "$B/source/scripts/collective_smoke.py"
python -c 'import torch,triton,transformers; print(torch.__version__,triton.__version__,transformers.__version__); assert torch.cuda.device_count()==4'
cd "$B"
python -u tools/run_reproduction_20261110.py > reproduction-controller.log 2>&1
```

请预留约4小时资源，实际时间受完整读回速度影响。正式测速前控制器先完成三臂共6次真实save10→连续11/原生load10→11预检，构建扩展并预热实际路径，预检耗时不进入正式统计。所有GPU任务串行，CRC/SHA和完整状态解码在训练进程退出后执行，不与测速并行。

正式部分固定四卡4B/BF16/DP1/TP2/PP2/S512/MBS1/GBS16/seed1234、关闭重计算；三臂顺序采用全部6种排列，共18次50步任务，末轮再各运行一次原生50→51恢复。三臂使用同一个同步原生ZIP64/CRC32、fsync及原子发布封装，禁止跳过optimizer保存、关闭GC或仅计算kernel时间。

结果写入`$B/checks/20261109-e2e-coalesced/replication-20261110/`。`complete.json`记录新测量的均值、区间和目标，`status.json`、逐rank记录、完整保存与恢复回执及证据包均保存。任何数值/CRC/恢复失败都会终止，不能把部分运行或损坏checkpoint当作有效加速。

## 三、验收与解释

- [原数值门槛](reproduction/20261110/inputs/numerical-gate.json)：loss `atol=0.001, rtol=0.001`；grad_norm `atol=1e-12, rtol=0.01`。输入、恢复后的完整状态另做精确指纹比较，不能只看loss。
- 新分块候选：六轮相对direct均更快，汇总吞吐至少+0.5%；两者全部持久化状态和原生恢复精确一致。
- CPT最终目标：相对适配baseline汇总吞吐≥+15%、配对95%区间下界也≥+15%，耗时下降≥12%。保留全部慢样本，固定6轮和bootstrap seed1234，不因结果不足而增加或筛选样本。
- 最新增量主要节省一次最终保存前的状态准备。训练越长且只保存一次，这部分比例可能被摊薄；LoRA目标+17%/−15%和长训练结论不能由本轮代替。

可浏览[所有原始配置和日志](results/e2e-20261110/runs)，以及[源码/结果导航](REPOSITORY_CONTENTS.md)。本次发布验证了证据复算、源码闭包、生成控制器及配置；没有重新租用GPU再跑一轮，因此新环境复现比例以其自身完整测量为准。
