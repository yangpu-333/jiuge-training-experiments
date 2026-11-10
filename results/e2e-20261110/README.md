# 50步CPT完整任务：原始结果与证据

- [结果JSON](complete.json)：六轮三臂的每次耗时、吞吐、bootstrap区间及验收结论。
- [CSV](cycles.csv)：可直接导入表格的逐轮数据。
- [实测协议](protocol.original.json)、[27个作业状态](status.original.json)：原记录不改写，包含当时的调度时限。
- [18次配置、命令、源码manifest和日志](runs)：展开即可查看；计时由父进程的完整任务wall clock决定。
- [分项诊断](component-analysis.json)：用于定位状态准备收益，各分项边界不同，不可机械相加。
- [独立复核](independent-local-review.json)、[原归档回执](receipt.json)。
- [完整关键证据包](key-evidence-e2e-coalesced-20261109-continued.tar.gz)：5,809,584字节；SHA256：`b474c80e8d9fb9a4d85b0b73218a686e64ecfb5bef638b92bfdb23b61108afb9`。

证据包保留843个成员，其中842项文件有逐项SHA256。包括所有rank的逐步指标、完整状态指纹、文件CRC/SHA回执、实际启动命令、恢复审计及控制器源码。历史归档与调度续接记录作为来源链保留，**不需要运行其中的历史压缩、去重或续接工具来复现**。

原控制器在02:00的启动新作业时限处退出；续接器核对已完成数据后，仅执行原序列剩余任务，沿用全部已有耗时。未重跑挑样，最终27个GPU作业全exit0。原时限错误和续接记录保留在证据包的`scheduling-continuation-v3/`中。

正式训练144份checkpoint文件已在原环境完整读回并验证；仓库保存其哈希和状态指纹，**没有上传模型权重或checkpoint本体**。候选大文件在验证及原生恢复后按单槽位协议回收，最后保留槽位为baseline，不能把它作为候选模型交付。新复现会从公开上游原权重重新产生checkpoint。

离线重新计算全部数字和核对证据：

```bash
python scripts/verify_package_20261110.py
```

详见[完整复现指南](../../REPRODUCE.md)。
