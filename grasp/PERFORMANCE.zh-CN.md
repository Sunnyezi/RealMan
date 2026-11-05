根因是单进程多线程的控制/物理阶段未有效并行，不是内存容量不足。保持一个MuJoCo阵列窗口，计算侧已改为16个独立工作进程，统一汇总16份真实状态显示。

同一策略、16环境的分段测试，每轮耗时：

| 配置 | CPU控制/物理 | 观察与奖励等收尾 | 其中RGB-D | 网络推理 | 每秒整批步数 |
|---|---:|---:|---:|---:|---:|
| 共享进程、8线程 | 1009 ms | 61 ms | 30 ms | 9 ms | 0.93 |
| 共享进程、8线程、仅运动学IK残差 | 964 ms | 64 ms | 32 ms | 4 ms | 0.97 |
| 共享进程、1线程、仅运动学IK残差 | 265 ms | 62 ms | 31 ms | 4 ms | 3.00 |
| 共享进程、4线程、仅运动学IK残差 | 859 ms | 63 ms | 32 ms | 3 ms | 1.08 |

RGB-D列是收尾列的子集。测试表明：增加Python线程反而明显变慢；仅将IK误差计算改为运动学更新帮助有限。最终修复采用真实多进程并行，未把试验性的IK替代应用到运行代码。

此前实际回放数据刷新约1.8次/秒。修改后16环境整批推进约13.6次/秒（会随接触和独立重置波动），没有物理警告。显示和计算已解耦；一个窗口不要求全部仿真计算挤在同一个Python进程中。

随后还修复了重复限频导致的状态丢帧：每次批量推进都发布真实状态。最终10秒实测收到145次状态更新，约14.5次/秒，相比修复前约提升8倍。此时可用内存约102 GiB，进程组RSS约19.5 GiB，物理警告为0。

当前默认：

- `train.py`和`grid_play.py`默认`--backend subproc`，由`parallel_vec.py`的`ParallelGridVec`汇总批量状态。
- 仍然只有一个4×4 Viewer窗口。每格是真实独立环境，不是动作复制。
- 独立工作进程上限16；当前进程组RSS约20 GiB，可用内存约95 GiB。守护进程在可用内存低于64 GiB、进程组RSS超过32 GiB或swap增长超过512 MiB时停止本次回放。
- 低内存共享后端仍可显式使用`--backend shared --physics-threads 1`，不再默认启用8线程。
- 计算和窗口状态发布目标均为20 Hz；当前并未稳定达到20 Hz，回合重置时仍可能有停顿。
- 当前运行的是策略回放，网络没有继续学习；原权重和已完成的训练测试仍保留。

物理与IK仍在CPU，GPU负责策略计算及EGL RGB-D。这次诊断中GPU利用率有接近99%的读数，但网络推理只占几毫秒，不能仅用GPU/显存读数解释卡顿。

训练与Viewer命令（服务器仓库根目录）：

```bash
bash grasp/grasp.sh live --envs 16 --stream runs/parallel_train/live_state.npz
bash grasp/grasp.sh train --backend subproc --envs 16 --device cuda:0 \
  --randomize --warmstart --steps 300000 --run runs/parallel_train
```

当前运行日志和监控数据在`/home/szy2025/Desktop/RealMan/work/shared_grid/`，分别见`test.log`、`parallel_speed.json`、`status.json`、`grid_status.json`。
