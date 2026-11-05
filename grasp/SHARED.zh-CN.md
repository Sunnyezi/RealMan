> 当前默认已改为subproc计算后端，仍使用同一个阵列窗口。本文为共享后端测试记录；最新结果见[卡顿诊断与修复](PERFORMANCE.zh-CN.md)。

已改为一个MuJoCo窗口显示4×4训练环境阵列，并测试单进程共享模型架构。

每批环境共享一个MjModel和一个128×128 RGB-D渲染器，各自拥有独立MjData、随机物体位置、IK状态、触觉、奖励、终止和重置。物理与IK通过CPU线程推进，RGB-D在主线程顺序渲染，策略前向和PPO优化使用CUDA。窗口把各份真实状态平移到不同格子显示；它们不是同一动作的视觉复制，也不会跨环境碰撞。

这实现了同屏多环境和资源共享，物理计算仍在CPU，并非上千环境的GPU物理仿真。

验证结果：

- 4环境×32步与修改前的实现比较，观察、奖励和qpos一致；单独重置一个环境不改变其余环境。
- 原有场景检查、17个触觉区域探测及随机脚本抓取10/10通过。
- 共享向量环境内完整脚本抓取4/4成功。
- 使用既有策略测试4、8、16环境，采样分别约27.4、22.8、24.0条状态转移/秒，没有物理警告。
- CUDA PPO测试完成4096步采样及5次优化更新，最大参数变化约0.00017187，确认不是仅运行推理。
- 测试进程峰值RSS约2.6 GiB，可用内存保持在约104 GiB。此次工作负载为已训练策略的抓取过程，与上次未训练策略的并行数基准不同，不能直接按吞吐量比值比较。
- 阵列窗口确认接收到16份不同状态；每格显示独立编号和回合仿真时间，独立重置会使不同格子的时间不同。

当前状态：训练测试已结束，VNC窗口持续显示16环境策略回放，标注`POLICY PLAYBACK (no learning)`。没有自动开始额外30万步长程训练。新测试权重位于服务器：

`/home/szy2025/Desktop/RealMan/grasp/runs/shared_grid_test_20261105/policy.zip`

此前16子进程的30万步训练已经正常结束，最终30回合验证成功率为30%，未达到80%验收目标。原权重保留在`runs/floor_cuda_guarded_16env_20261105_115244/`。新旧实现的等价性证据和原环境指纹作为来源记录保存在测试权重的run.json中；没有覆盖旧权重。

主要代码：`grasp/shared_vec.py`负责共享模型、分阶段线程物理和批量状态发布；`grasp/grid_live.py`负责一个窗口中的阵列；`grasp/grid_play.py`提供阵列策略回放入口。`train.py`新增`--backend shared|subproc`与`--physics-threads`，默认 subproc。共享后端不接受会逐环境修改模型的domain profile；这种随机化仍需subproc后端。

从服务器仓库`/home/szy2025/Desktop/RealMan`根目录运行：

```bash
# 终端A：阵列Viewer，先于训练启动
bash grasp/grasp.sh live --envs 16 --stream runs/shared_train/live_state.npz

# 终端B：启动新的GPU训练
bash grasp/grasp.sh train --backend shared --physics-threads 1 \
  --device cuda:0 --envs 16 --randomize --warmstart \
  --steps 300000 --run runs/shared_train
```

已有策略的阵列回放：

```bash
# 两个终端分别运行，相对路径以grasp目录解析
bash grasp/grasp.sh live --envs 16 \
  --stream runs/shared_grid_test_20261105/live_state.npz
bash grasp/grasp.sh grid-eval runs/shared_grid_test_20261105/policy.zip \
  --envs 16 --device cuda:0
```

当前自动启动的回放使用`work/shared_grid/live_state.npz`，并由独立监控进程检查内存、swap和Viewer存活状态。新训练若手动执行上述命令，需要沿用资源监控流程；本次没有再次尝试32或48环境。

MuJoCo接口依据：[官方Python接口与Viewer文档](https://mujoco.readthedocs.io/en/stable/python.html)。性能判断依据为本次实测，不承诺随环境数增大而加速。
