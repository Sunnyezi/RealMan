# RM65＋RH56 的 π0.5 微调

服务器仓库：`/home/szy2025/Desktop/RealMan`。本目录复用 `grasp/scene.xml` 的真实接触模型，并独立保存 π0.5 数据和权重。没有真实机器人驱动。

## 当前任务与接口

当前任务为随机位置的蓝色方块抓起并稳定保持：抬升至少 10 cm、速度满足既有判据、至少两个手部接触、无地面接触，并持续 2 秒。每回合最多 12 秒。这与“抓起后放入目标区域”是两个不同任务；当前成功率不能用于声明已完成搬运放置。

- 输入：外部 `overview` 和臂载 `rgbd` 两路 224×224 RGB，6 个机械臂＋6 个手部主动关节的实际位置，以及任务文本。
- 输出：同一顺序的 12 维绝对关节目标，单位为弧度。相机旋转执行器和被动耦合关节不作为策略输出。
- 手部示教沿用现有固定抓型，六路通道是相关动作；这不证明手指独立操作或手内操作能力。
- 执行：20 Hz，物理步长 2 ms，预测 10 步、执行 2 步后重新观测。目标按源模型执行器限位裁剪，力矩由现有模型限制。
- 模型：`pi05_base`，JAX，VLM 与动作专家使用 LoRA，EMA 关闭，内部状态/动作补齐至 32 维。
- 策略输入不含物体真实位姿、视觉几何估计、规划器目标或触觉。旧视觉规划器仅用于教师和初始场景检查，策略执行直接使用绝对关节目标。

现有源模型的机械臂为 RM65-6FB-V 替代资产，手部为 RH56DFTP 右手，尚未核验 RH56DFTP-2R 的精确修订与安装参数。目标硬件仍为 RM65-6F-V＋RH56DFTP-2R。仿真训练不构成 sim2real 验证。

## 环境与数据

以下命令在仓库根目录执行，仿真沿用已有 Conda 环境；OpenPI 单独使用 `pi05/vendor/openpi/.venv`，不升级现有 PPO 环境。

```bash
bash pi05/setup.sh
~/.conda/envs/realman-mujoco/bin/python pi05/fetch_weights.py

MUJOCO_GL=egl OMP_NUM_THREADS=1 ~/.conda/envs/realman-mujoco/bin/python \
  pi05/pipeline.py check --count 100 --output pi05/outputs/teacher_gate

MUJOCO_GL=egl OMP_NUM_THREADS=1 ~/.conda/envs/realman-mujoco/bin/python \
  pi05/pipeline.py collect --count 50 --max-attempts 100 \
  --start-index 100 --output pi05/data/pilot50

MUJOCO_GL=egl ~/.conda/envs/realman-mujoco/bin/python \
  pi05/pipeline.py replay pi05/data/pilot50/episode_000000
```

超过 50 条的采集需要 `--gate` 指向同一代码/场景指纹下至少 100 次、成功率不低于 90% 的教师检查。保存所有尝试摘要，仅成功轨迹写入图像、执行前状态、控制目标和时间戳。train、validation、test 的种子起点分别为 10000、100000、200000，每个区间最多 90000 个，测试数据禁止转换为训练数据。

本阶段不改变物体类别、尺寸或朝向，不能报告未见物体泛化。后续扩充场景时必须重新验证教师。

## 转换与试训

`prepare_dataset.py` 每回合只解压一次图像数组，并同时逐回合更新上游 RunningStats 计算归一化；动作统计包含 10 步预测窗口和回合末尾重复填充，与固定 LeRobot 的动作窗口一致；分位数采用上游直方图近似，统计批次按回合组织。

```bash
JAX_PLATFORMS=cpu pi05/vendor/openpi/.venv/bin/python pi05/prepare_dataset.py \
  --raw pi05/data/pilot50 --repo-id local/rm65_rh56_pi05_pilot50

JAX_PLATFORMS=cpu pi05/vendor/openpi/.venv/bin/python pi05/openpi_run.py check-data \
  --repo-id local/rm65_rh56_pi05_pilot50

# 先训练一步验证真实权重加载和反向传播，再做 100 步试跑。
bash pi05/train_visible.sh local/rm65_rh56_pi05_pilot50 probe1 pi05/data/pilot50 1 1
bash pi05/train_visible.sh local/rm65_rh56_pi05_pilot50 smoke100 pi05/data/pilot50 100 1
```

启动器先在 VNC `:15` 建立窗口，然后播放本次数据集的真实动作回放；窗口明确标注 `supervised training / demonstration replay`。监督训练进行参数更新时没有实时策略控制轨迹，回放不代表模型表现。策略闭环效果需要下节独立评估。缺少活跃 Viewer 时训练入口拒绝启动。

当前服务器的 GPU 报告约 48 GiB 显存，另有任务运行，启动器默认将 JAX 分配限制为 55%；显存占用变化时需重新实测 batch。不会停止其他任务。训练参数和显存可行性以实际日志为准。

OpenPI 固定为 `215abfb217dbac7d5f1273282331b9b1866c0479`，LeRobot 固定为 `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`。安装脚本从固定锁文件导出带哈希的依赖（可用 `PYPI_INDEX_URL` 指定镜像），仅将同一 LeRobot 提交改为本地源码路径，以绕过服务器 Git TLS 故障。基础权重从官方 GCS 下载，逐文件检查 MD5/CRC32C 后才放入 OpenPI 缓存。

## 持久训练与状态

完整数据集已采集并转换：训练 1,000 条、153,395 帧；验证 100 条、15,384 帧。batch 1/2/4 显存探测均通过，正式训练采用 batch 4。完整训练集的 100 步试跑和策略检查完成后，可用以下入口启动初轮训练：

```bash
~/.conda/envs/realman-mujoco/bin/python pi05/launch_training.py \
  --repo-id local/rm65_rh56_pi05_train1000 \
  --experiment baseline_20261117 --raw pi05/data/train1000 \
  --steps 30000 --batch-size 4 --validation-count 100

cat pi05/runs/baseline_20261117/job_status.json
tail -n 20 pi05/runs/baseline_20261117/train.log
tmux ls
```

入口在独立 tmux 会话运行，SSH 断线不会结束训练。运行目录已经存在时会拒绝覆盖。阶段由 `job_status.json` 记录：`training` → `validation` → `trained_and_validated`；异常时写入 `failed`、具体阶段和错误。模型性能是否合格仍由验证成功率判断。

训练结束后自动评估预先指定的最终检查点，并保存 100 个验证回合、Wilson 95% 区间、失败类别和推理延迟。该入口不声称最终检查点是所有检查点中最优的；测试集留待完成模型选择后使用。

## 独立评估

选择实际存在的检查点目录；100 步试跑的末次保存通常为 `99`。

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 pi05/vendor/openpi/.venv/bin/python pi05/openpi_run.py serve \
  --repo-id local/rm65_rh56_pi05_pilot50 \
  --checkpoint pi05/checkpoints/pi05_realman_absolute12/smoke100/99

# 另一终端：使用固定客户端源码和单独目录中的 msgpack，不降级原仿真环境的 NumPy。
bash pi05/evaluate.sh --split validation --count 30 \
  --output pi05/outputs/smoke100_validation
```

按 validation 闭环成功率选择后续模型；测试集只在选定模型后使用。100 步试跑只证明训练链路，不预设抓取成功率。扩大训练集、验证集和训练步数需要保留独立种子、真实推理延迟与失败记录。

## 检查

```bash
cd pi05
~/.conda/envs/realman-mujoco/bin/python -m unittest -v test_contracts
```

训练输入、旧 7 维动作拒绝、12 维输出解填充、非有限值拒绝由契约测试覆盖；真实接触与时序一致性由教师检查及绝对动作回放验证。各阶段实测结果见 `IMPLEMENTATION_REPORT.zh-CN.md`。
