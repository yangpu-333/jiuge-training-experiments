# 当前候选原生恢复验证（2026-11-01）

当前采用`lora-linear-cpp-v2 + fused-split-v1`，不启用未取得一致整模型收益的scale/add融合。固定四卡BI-V150、4B、DP1/TP2/PP2、BF16、S512、MBS1/GBS16、seed1234、关闭重计算。

LoRA/CPT均通过原生保存及恢复对照：连续训练10步，只保存第9步；新进程原生加载该checkpoint执行第10步。四rank逐项精确一致：

- 第9步保存前与加载后的模型、FP32主参数、Adam moments/step、scheduler全元素哈希。
- 第10步结束的上述完整状态、最终训练/验证样本计数。
- 第10步的实际输入、Python/NumPy/Torch CPU/CUDA/MCore RNG、完整精度loss、grad_norm、无跳步。

LoRA每rank覆盖248个FP32主参数张量、每套5618688元素，两套Adam moments各覆盖相同元素。CPT rank0/1覆盖187个主参数张量、每套1079505920元素；rank2/3覆盖188个、1079508480元素，两套moments分别等量覆盖。这是各rank实际状态存量，不能当作去重后的逻辑模型参数数。

LoRA实际为Float16OptimizerWithFloat16Params，CPT为DistributedOptimizer。后者外层state_dict不包含moments，因此同时检查实际inner optimizer state及FP32参数组。汇总器v1曾把LoRA wrapper与inner中的moments重复计数，v2修复统计口径；原始恢复比对结果及训练代码未变，失败汇总源文件/日志保留。

原生checkpoint保存在：

```
/private/yp/jiuge-training/runs/4b-lora-tp2pp2-s512-candidate-native-control-20261101-v1/checkpoints/
/private/yp/jiuge-training/runs/4b-cpt-tp2pp2-s512-candidate-native-control-20261101-v1/checkpoints/
```

两处latest均为9，只有iteration9实际checkpoint；第10步完整状态另存为指纹，没有重复保存大型checkpoint。所有checkpoint、reference和引用链均保留。

恢复沿用已验证的`AUDIT_PRESERVE_RESUME_DATALOADER_RNG=1`修复，仅限num_workers=0、single迭代器、eval_iters=0与当前确定性样本。此结果只验收本配置短程native save/load，不泛化到不同TP/PP、数据管线、长期训练或正式端到端性能。

证据：`checks/20261101-candidate-restore/summary.json`、`delivery-v2.json`；`state-evidence-v2.zip`包含配置、源码、每rank状态指纹、逐步输入/RNG审计和复核，SHA256 `249dde750344db8f4204783f46862e8df59a6eb9c76c4fc1760cd3edd3d68c0c`。本地索引`results/candidate-native-restore-20261101.json`，代码镜像`candidate_restore_20261101/`。
