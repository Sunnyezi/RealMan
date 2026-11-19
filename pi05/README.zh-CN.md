# RM65＋RH56 的 π0.5 微调

服务器仓库：`/home/szy2025/Desktop/RealMan`。本目录复用 `grasp/scene.xml` 的真实接触模型，并独立保存 π0.5 数据和权重。没有真实机器人驱动。

## 当前任务与接口

当前任务为随机位置的蓝色方块抓起并稳定保持：抬升至少 10 cm、速度满足既有判据、至少两个手部接触、无地面接触，并持续 2 秒。每回合最多 12 秒。这与“抓起后放入目标区域”是两个不同任务；当前成功率不能用于声明已完成搬运放置。

- 输入：外部 `overview` 和臂载 `rgbd` 两路 224×224 RGB，6 个机械臂＋6 个手部主动关节的实际位置，以及任务文本。
- 输出：同一顺序的 12 维绝对关节目标，单位为弧度。相机旋转执行器和被动耦合关节不作为策略输出。
- 手部示教沿用现有固定抓型，六路通道是相关动作；这不证明手指独立操作或手内操作能力。
- 执行：20 Hz，物理步长 2 ms。接近阶段预测 10 步、执行 2 步后重新观测，经过平滑、关节速率限制和手掌姿态 IK 投影；模型请求闭合手指时切换至 RGB-D 抓取位置校正，对准前保持张手。夹紧后连续确认拇指与食指接触，才抬升并保持。力矩仍由源模型限制。
- 模型：`pi05_base`，JAX，VLM 与动作专家使用 LoRA，EMA 关闭，内部状态/动作补齐至 32 维。
- π0.5 模型输入不含物体真实位姿、视觉几何估计、规划器目标或触觉。当前执行器另读机器人关节、初始 RGB-D 视觉估计与拇指、食指触觉，结果标记为 **π0.5＋RGB-D 视觉/触觉控制器**，不代表纯模型能力或重新训练后的模型。物体真值与成功判据仅由评估器记录。

现有源模型的机械臂为 RM65-6FB-V 替代资产，手部为 RH56DFTP 右手，尚未核验 RH56DFTP-2R 的精确修订与安装参数。目标硬件仍为 RM65-6F-V＋RH56DFTP-2R。仿真训练不构成 sim2real 验证。

2026-11-18，当前混合控制器在固定方块的100回合验证中成功100/100，12项测试通过；模型权重仍为原检查点29999。详见实施记录中的v4验收。

2026-11-19，**纯 π0.5 策略**（无任何控制器接管）采用时间集成执行后，检查点29999在 test 种子 200000–200199 上成功200/200，Wilson 95% 区间 98.1%–100%；原“执行2步”方式在验证区间 130000–130099 上为 78/100。推荐使用 `evaluate_checkpoint.py --execution policy --ensemble-decay 0.01`。详见[纯策略记录](PURE_POLICY_20261119.zh-CN.md)。

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

## 稳定执行与独立评估

`evaluate.sh` 默认运行 `evaluate_stabilized.py`；训练完成后的自动验证也使用此入口。旧版 `pipeline.py evaluate` 的原始执行代码保留，用于复现历史记录。

稳定器 `stabilizer.py` 分为接近、视觉对准、夹紧、抬升、保持五阶段。π0.5 负责至多 2 秒的初始接近，手指闭合预测可提前触发接管；接近阶段手掌朝向固定，平移命令不超过 0.06 m/s。模型请求食指闭合超过 0.04 rad，或初始接近达到 2 秒时，控制器保持张手，按既有 RGB-D 估计对准抓取位置；对准路径每轴不超过 0.05 m/s。误差小于 1 mm 后夹紧至少 0.5 秒，三帧拇指与食指接触且关节达到闭合阈值、手部命令达到固定抓型后，才以 0.05 m/s 将手掌向上推进 0.13 m 并保持。2 秒内未确认夹紧则交回模型并重新获取动作块；抬升或保持阶段尚无掉落后重新抓取逻辑。全阶段臂关节命令不超过 1.2 rad/s，手部不超过 0.4 rad/s。

RGB-D 估计复用源模型的相机标定、蓝色分割和已知 4 cm 方块的单面深度修正；初始化时从相机图像与深度估计物体中心，再加固定手掌抓取偏移。它没有直接读取仿真物体真实位置，但依赖已知颜色、大小、朝向和当前场景标定。这一控制器接管了精确对准、闭合和保持，因此最终成功率不能用来评价纯 π0.5 策略的独立能力。

位姿增量以**上一条命令的正运动学**为基准，避免重力引起的静态跟踪误差逐步累积成向下命令。控制器不读取物体真实位置、真实高度、物体接触身份或成功标记；触觉只按手部区域汇总，存在碰到环境而触发的局限。以上阈值为当前固定抓型的仿真配置，尚未在真实机器人上标定。

每回合 JSON 保存阶段转换、阶段帧数、IK 拒绝与原有成功指标；同名 NPZ 保存逐步原始模型输出、实际命令、实际关节与速度、手掌位姿、17 区域触觉。接管后原始模型输出记为 NaN，并用 `has_policy_action` 区分。物体位置和成功指标在控制器更新后由评估器独立写入。汇总保存执行器名称、源代码指纹与环境指纹。

推荐用自动启动服务和 VNC 窗口的入口，输出目录须未使用：

```bash
~/.conda/envs/realman-mujoco/bin/python pi05/evaluate_checkpoint.py \
  --repo-id local/rm65_rh56_pi05_train1000 \
  --checkpoint pi05/checkpoints/pi05_realman_absolute12/baseline_20261117/29999 \
  --output pi05/outputs/stabilized_acceptance --count 100
```

也可手动启动服务与评估客户端：

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
~/.conda/envs/realman-mujoco/bin/python -m unittest -v test_contracts test_stabilizer
```

训练输入、旧 7 维动作拒绝、12 维输出解填充、非有限值拒绝由契约测试覆盖；稳定器测试覆盖慢策略接近超时、速率限制、接触确认、阶段切换、夹紧未确认超时与静态跟踪误差回归；真实接触与时序一致性由教师检查及绝对动作回放验证。各阶段实测结果见 `IMPLEMENTATION_REPORT.zh-CN.md`。

## 纠正示教与继续微调

按计划直接采集模型访问状态下的成功恢复后缀，混合原示教，从29999继续微调12000步。使用`launch_recovery.py`持久执行采集、数据校验、warm-start试跑、训练和新模型完整任务验收；不重跑旧模型对照。具体数据隔离、原归一化保留、VNC和验收方式见[纠正训练记录](RECOVERY_TRAINING.zh-CN.md)。

2026-11-18 02:09 CST，纠正示教混合数据400条/53196帧转换和warm-start实跑通过，`recovery_20261118`正式12000步微调已启动。训练进行中，最终模型表现以该任务后续的无接管完整策略验收为准。

两轮纠正微调的纯策略验收分别为 31/100（`recovery_20261118/11999`）和 66/100（`grasp_closure_v2_20261118/11999`），均低于基线。

## 纯策略执行与新示教（2026-11-19）

失败轨迹分析表明，原执行方式的掉落主要来自动作块切换处的跳变和过快的抬升。`evaluate_policy.py` 新增 `--ensemble-decay`（ACT 式时间集成，只平均模型自己的重叠预测）和 `--execute-steps`。29999 加时间集成后：验证区间 99/100，测试区间 200/200。

`teacher.py`、`collect_teacher.py`、`collect_teacher_parallel.py` 和 `launch_teacher_v2.py` 提供了抬升更慢、带停顿、并含 DART 式执行噪声的新示教，以及从29999继续微调的流程（1200条/229,164帧，16,000步）。在同一验证区间 130000–130099 上，新模型原方式 56/100、时间集成 90/100，均低于29999的78/100、99/100，因此不采用，最终方案仍是29999加时间集成。数据、教师扰动实验和全部数字见[纯策略记录](PURE_POLICY_20261119.zh-CN.md)。

```bash
cd pi05 && MUJOCO_GL=egl ~/.conda/envs/realman-mujoco/bin/python test_teacher.py
```
