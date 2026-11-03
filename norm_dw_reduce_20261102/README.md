# CPT dw 两级归约合并筛查

原实现先对每32行求FP32 partial，再对16组求和并输出BF16。本草案以三维Triton张量在单kernel内依次做两次`tl.sum`，目标省一次发射及partial缓冲；沿用已验收的products融合和双输入边。

133组实际捕获及合成输入隔离检查未通过：输出、direct、variance、残差dx和冻结weight路径逐元一致，dw共335001个元素不同，最大relative L2约9.14。差异远超可忽略范围；本实验停止推广，未做模型数值或测速。三维归约布局与跨线程汇总如何产生差异尚未定位，不能仅凭结果断言编译器原因。失败日志、完整逐例数据、预注册源码哈希保存在`/private/yp/jiuge-training/checks/20261102-norm-dw-reduce/`，当前CPT已验收实现不变。

诊断显示行号输入的第一级partial逐元一致，但第二级输出错误；随机输入第一级也存在少量差异。v2在同一kernel内显式写出partial，barrier后volatile重读，再做第二级归约。133组实际捕获与合成样本的输出、dw、两条输入梯度及残差图均逐元一致。隔离Profiler中kernel数7→6；四轮交错前后向计时均值1275.7→977.5µs，但首轮波动较大，只作为筛查。原十步LoRA/CPT整模型数值验证已启动，未完成前不推广。v2证据位于`checks/20261102-norm-dw-reduce-v2/`。

数值入口复制自LoRA A+TP验证，`numeric-launcher.json`中的candidate文本遗留旧标签`lora-a-tp-v1`；实际绑定与预注册源码哈希、run目录后缀为`norm-dw-reduce-v2`。为保留预注册一致性，运行中不改脚本；复核以实际源码和绑定为准。

暂停时原控制器和测速等待进程收到停止信号，但LoRA训练子进程处于独立进程组，继续完成了输出；恢复后确认四rank捕获及step9/10状态文件齐全，使用独立continuation控制器复用这些文件、核对预注册哈希并执行原审查，再依次运行CPT及审查。原中断日志不覆盖；配对测速仅在两任务门槛通过后重启。训练在暂停期间未完全停止，不能把该段称为暂停期间无GPU工作。

原十步LoRA/CPT数值门槛均通过。CPT相对原reference，第1/10步梯度relative L2分别4.93875e-7/2.84130e-7，累计主参数更新3.32212e-6，最终BF16模型逐元一致。但额外诊断发现CPT四rank捕获SHA和step9/10状态指纹与已选基线不同；最终状态顶层差异位于optimizer，模型及scheduler指纹一致。上述relative L2是相对原reference，并非新旧候选直接差值；具体optimizer张量和差异原因尚未定位。因该问题，已在放行前停止配对测速等待器，保留所有结果，暂停新GPU工作。证据`numerical/continuation-status.json`、`cpt-comparison.json`、各run的状态指纹及预注册文件。

恢复后直接比较v2与已选products候选：捕获中变化的张量均为RMSNorm权重，rank0的step1梯度1个元素不同（最大2.98e-8）、最终master一处1个元素不同（1.19e-7）；rank2的step1有6个权重各1元素不同，最大5.96e-8，step10有1个权重1元素不同（2.33e-10）。rank1/3对应TP副本；optimizer指纹还存在多处FP32 moment差异。直接比较脚本对每个所取张量核验内容SHA，完整路径和数值在`checks/20261102-norm-dw-reduce-v2/{capture-diff-names-v1,capture-direct-differences-v1,optimizer-diff-paths-v1}.json`。

v3保持原第一级32×64及第二级16×32归约块，在同一CTA写partial、barrier后重读，完成133组前反向逐元对照；额外100组随机BF16 products的partial和dw均逐元一致。相同随机输入中v2 partial共1213处不同而dw均相同，说明v2第一级存在细微求和差异；与模型中差异的具体因果仍未单独证明。v3隔离kernel 7→6、四轮均值576.49→533.68µs（约8%），只作筛查。原十步LoRA/CPT数值与四rank完整状态验证已启动；配对测速只在原门槛和捕获、状态SHA均通过后放行。新证据在`checks/20261102-norm-dw-reduce-v3/`。

v3整模型LoRA/CPT原十步审查均通过，四rank捕获SHA及step9/10完整状态与已选基线一致。LoRA负对照一对、CPT四对的交错测速正在执行；此时尚无完整v3整模型性能结论。

四对CPT已完成：平均步时4.908267→4.877919秒，吞吐比1.006221；四对分别1.023472、0.997501、1.002933、1.000682，未满足预设四对均快的选择条件。LoRA未变算法单对负对照0.975172，说明这批次存在明显波动；不能把单对变化归因为候选，也不能认定约0.6%的平均差为稳定收益。保留已选products版，v3不推广，native恢复选择器按预设条件未触发。全部两任务配对25步标量一致、输入哈希未变、实际安装核验通过；测量窗口无gen2 GC，自动GC保留。完整摘要`performance/performance-summary.json`，SHA256 `db2b44b1f091d9dc2ccca39d660917c12968e141da6ed6473ac4db0ca27e85ed`，本地镜像`results/norm-dw-reduce-v3-performance-20261102.json`。
