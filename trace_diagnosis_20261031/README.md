# 2026-10-31 trace归因

固定四卡BI-V150、4B、DP1/TP2/PP2、BF16、S512、MBS1、GBS16、seed1234、关闭重计算；所有远端产物位于`/private/yp/jiuge-training/checks/20261031-trace-diagnosis/`。

`analyze_timeline.py`只读分析上一轮24份原始trace，并校验原SHA。设备活跃时间使用区间并集；compute/communication单独计算重叠，不能把kernel累计时长当wall time。CPU self耗时按线程嵌套关系减去子区间；跨线程结果可重叠。GPU空档到下一次launch的关联只给出部分证据，不能将全部空档认定为CPU瓶颈。原始trace受profiler扰动。

`phase_training.py`复用已记录的lean训练入口，补测reference/full两个端点的LoRA/CPT。5步预热+10步计时，只增加主机计数器和阶段包装，不添加设备同步、tensor检查、额外通信或checkpoint。forward_backward、optimizer_step、prepare_grads可能嵌套，不能相加。process CPU time汇总一个rank进程的所有线程，不是主线程耗时或机器整体CPU占用。阶段结束未额外同步，host时间包含发射及内部等待，不能当GPU独占时间。

首次阶段包装遗漏调度器关键字参数，训练步前退出，失败源码和日志保留；v2透传全部参数，以独立run目录重做。补测期间暂停本次CPU解析进程，四组结束后恢复，以减少资源争用。

`summarize_diagnosis.py`检查补测有效配置与历史同臂配置一致、15步训练标量一致，并合并四rank结果。它不替代既有全量数值判据；本轮不修改模型算法或PP同步设置。后续融合必须保留外部cast、两条反向分支和既定数值门槛。
