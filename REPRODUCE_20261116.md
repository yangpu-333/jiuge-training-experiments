# 八卡CoreX5复现指南

本指南复现2026年11月16日的三种并行配置。旧四卡CoreX4.4资料保留在[历史指南](REPRODUCE.md)。复算已记录证据与重新获得GPU性能是两件事；本次发布完成CPU全量核验、源码闭包及控制器静态检查，没有为发布再次租GPU重跑。新机器的绝对耗时和百分比会受CPU、存储、驱动及并行布局影响。

## 1. 普通电脑：复算公开证据

Python3.12或更高版本，仅需标准库：

```bash
python reproduction/20261116/verify.py --full
python scripts/check_links.py
```

默认安全解包到新的`.verified-20261116/`。覆盖原目录会被拒绝，可用`--output /new/path`选择新目录。程序核对[公开包SHA](results/20261116/PACKAGES.json)、逐成员SHA，独立检查84次正式任务涉及的19,200组配对逐步标量、14项十步数值、12组原生恢复、初末状态与梯度指纹、配置回执及20,000次bootstrap。预期输出`PASS public packages/member hashes and full independent replay`。

包内PUBLICATION.json逐文件记录原始与公开SHA。内部账号、pod名称、SSH主机被脱敏，受影响文件的引用SHA随之重建；原始张量指纹、测量数值和全部慢样本保留。公开包SHA与私有原包不同，不能交叉使用。

## 2. GPU环境与来源

八张Iluvatar BI-V150，每卡32GiB；Linux x86_64、Python3.12；厂商CoreX5.0训练镜像`bi150-5.0.0-x86-ubuntu24.04-py3.12-llm-training-codeserver:5.0.0_llm_training`，driver5.0、torch2.10.0+corex.5.0.0、triton3.2.0+corex.5.0.0。厂商Apex、通信库、编译器和头文件必须可用，不用PyPI版torch替代厂商包。单个完整CPT checkpoint约56GiB，全六组保留全部轮次需要约3TiB可用本地scratch；持久化启动至少22GiB并保留10GiB余量。建议主机内存至少256GiB，以完整CPT状态审计峰值为准。

完整84次正式任务及审计建议预留八卡12小时（96 GPU卡时），可按`--cases`分组。脚本不会开卡、续费、删除旧实验或自动延长资源。

在干净Linux项目固定路径执行；`PUB`为本仓库绝对路径。安装器拒绝覆盖旧tools和旧环境：

```bash
export PUB=/absolute/path/to/jiuge-training-experiments
export B=/private/yp/jiuge-training
mkdir -p "$B"
git clone https://github.com/ZhouBencheng/jiuge-training.git "$B/source"
git -C "$B/source" checkout --detach 9fabd71df4897f08267df5431aa480ab9ef70a75
git clone https://github.com/NVIDIA/Megatron-LM.git "$B/Megatron-LM"
git -C "$B/Megatron-LM" checkout --detach 5be9626709af2722333bf54797c954c09edeada3
python "$PUB/reproduction/20261116/prepare.py" --install
python -m pip install --target "$B/checks/20261113-eight-gpu-corex5/deps" --no-deps transformers==4.57.6 tokenizers==0.22.1 huggingface-hub==0.36.0 pybind11==2.13.6
export PYTHONPATH="$B/checks/20261113-eight-gpu-corex5/deps:$B/checks/20261113-eight-gpu-corex5/overlay:$B/tools:$B/source/src:$B/source/.deps:$B/Megatron-LM"
export USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
```

安装器逐文件检查[源码清单](reproduction/20261116/FILES_SHA256.json)，生成pinned Megatron namespace overlay和vendor libcuda链接，保留[实测环境变量](reproduction/20261116/environment.json)。图像已有的numpy、safetensors、PyYAML、requests、packaging、tqdm、ninja、tensorboard等通用依赖须由doctor检查；不将该环境清单宣称为所有厂商二进制的可移植pip lock。

## 3. 同一模型、语料和输入校验

模型为原生PyTorch版FM9G-4B原始9G4B.tar，非同名HF转存。固定来源及所有成员SHA见[MODEL_SOURCE](reproduction/20261110/inputs/MODEL_SOURCE.json)，归一化只显式写入config默认项，权重字节不变。

```bash
mkdir -p "$B/models/original" "$B/data/raw"
curl -fL https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/9G4B.tar -o "$B/models/original/9G4B.tar"
echo '28ff0bad05c04b97433c57de1ed2aef0ccd233536e260773a4a9e7482b6db89c  '"$B/models/original/9G4B.tar" | sha256sum -c -
tar -xf "$B/models/original/9G4B.tar" -C "$B/models/original"
python "$PUB/reproduction/20261110/prepare_inputs.py" --model-source "$B/models/original/9G4B"
curl -fL https://raw.githubusercontent.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/38b93b75a80b019cc3aad362ca6f5bb0c3908959/data/alpaca_gpt4_data_zh.json -o "$B/data/raw/alpaca_gpt4_data_zh.json"
for TASK in cpt lora; do
 python "$B/tools/eight_gpu_20261113/corex5_exec.py" python -m jiuge_training.cli prepare-data --source "$B/data/raw/alpaca_gpt4_data_zh.json" --output "$B/data/4b-$TASK-seq512-resolved" --model-path "$B/models/9G4B-resolved" --task "$TASK" --seq-length 512 --seed 1234 --validation-fraction 0.02 --verify-alpaca
done
python "$PUB/reproduction/20261110/prepare_inputs.py" --check-data
```

语料源SHA为`6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903`，固定版48,818条Alpaca GPT-4中文记录；数据许可沿用上游CC BY-NC4.0。权重和语料正文不随仓库转发。原tokenizer及CPT训练/验证split校验见[manifest](reproduction/20261110/inputs/data-manifest.original.json)。LoRA单独由同一pinned CLI生成。

六组配置与原始初始模型/首步微批和RNG指纹分别在[configs](reproduction/20261116/configs)与[golden](reproduction/20261116/golden)。重跑时每组基线必须先匹配原始输入及初始权重；错误tokenizer、数据、LoRA随机初始化不会静默成为新的“等效实验”。

## 4. 预检、审计和正式重跑

```bash
python "$B/tools/eight_gpu_20261113/corex5_exec.py" python "$B/source/scripts/doctor.py"
python "$B/tools/eight_gpu_20261113/corex5_exec.py" torchrun --standalone --nproc_per_node=8 "$B/source/scripts/collective_smoke.py"
python "$B/tools/eight_gpu_20261113/corex5_exec.py" python -u "$B/tools/run_public_e2e_20261116.py" --name replication-001 --until 2026-12-01T23:00:00+08:00
```

`--until`必须换成实际获准的预约结束时间，不使用示例日期直接启动。也可先运行单组，例如追加`--cases pp2-tp4-cpt`。名称必须唯一，不覆盖旧证据；每任务最多1800秒，并受资源截止限制。

新控制器复用冻结worker、数学实现、原数值门槛、原生保存与完整校验；只将旧历史参考改为同轮新基线、R/T改为新目录、使用显式资源截止。历史控制器仍在snapshot中供审查，不直接执行其过期日期或旧complete.json依赖。

每组先新跑十步基线并检查golden，候选对同轮基线比较完整模型/master/Adam、梯度第1/10步和输入/RNG；字节差异停止调查，不放宽门槛。再原生save9/continue10对新进程load9/10，最终状态一致后才正式测速。TP2/PP4为六轮三臂全排列；TP4/PP2及TP8为六对AB/BA两臂。记录所有50步标量，完整checkpoint、文件和目录fsync、退出均计时，CRC/SHA在计时外。新结果在`$B/checks/public-replay/<name>/`。

## 5. 验收和边界

CPT须同时吞吐+15%、耗时−12%；LoRA须+17%、−15%。分别报告均值与配对bootstrap95%区间下界是否通过。不要删除慢样本、反复跑到通过或将不同轮基线拼接。完整数字见[结果JSON](results/20261116/stage2-all-layouts-e2e-summary.json)和[详细报告](REPORT_20261116.md)。

已发布数值与原生恢复范围是10步、9→10；50步正式任务另做逐步标量和完整保存校验，没有新50→51或长训练质量结论。模型和大checkpoint正文不在证据包，回执不能代替权重备份。
