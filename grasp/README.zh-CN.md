# RGB-D 方块抓取：使用说明

服务器目录：`/home/szy2025/Desktop/RealMan/grasp`。Conda 环境：`realman-mujoco`。

当前训练模型保留 RM65-6FB-V、RH56DFTP 右手、零偏移安装和原始手指联动。训练用 `scene.xml` 与原展示模型分开。

## 三个入口

```bash
cd /home/szy2025/Desktop/RealMan/grasp

# 场景、视觉、环境接口与 10 次脚本抓取验证
bash grasp.sh check

# 四环境 PPO，从头训练
bash grasp.sh train --steps 300000 --envs 4 --run runs/my_run

# 可选：先用成功脚本轨迹和状态反馈专家预热，再进行 PPO
bash grasp.sh train --warmstart --steps 300000 --envs 4 --run runs/my_warm_run

# 恢复训练；steps 指追加的采样步数
bash grasp.sh train --resume runs/my_run/policy.zip --steps 100000 --envs 4 --run runs/my_resume

# 独立种子 100 回合评估；viewer 会发布同一回合状态到 VNC
bash grasp.sh eval runs/my_run/policy.zip --episodes 100 --seed 10000 --viewer
```

VNC 可见窗口：

```bash
bash grasp.sh live
```

已有实时窗口正在运行时不要重复启动。窗口标题为 `RGB-D grasp training`。
- `SCRIPTED CHECK (not PPO)`：仅在明确使用 `live --scripted` 时执行脚本演示；不是已学会的策略。
- `LIVE WORKER STATE`：显示训练第一个环境或 `eval --viewer` 发布的状态。`WORKER PAUSED / LAST FRAME` 表示暂时没有新状态，停留在最后一帧，不自动切回脚本。
- PPO 采样比真实时间快，训练时动作可能显得较快；`eval --viewer` 按 20 Hz 实时回放。
- 本机 TigerVNC 连接 `127.0.0.1:15900`，对应服务器 `:15`；SSH 隧道断开时重连 `ssh -N uestc-vnc`。

## 任务与控制

蓝色方块边长 4 cm、质量 50 g，自由刚体，没有绑定到手、焊接约束或关闭重力。成功标准为比初始静止高度高至少 10 cm，连续 2 秒速度稳定、有至少两个手指组接触、与桌面脱离且机器人未撞桌。每回合最多 10 秒。

动作 9 维，范围 [-1,1]：前三维为末端 XYZ 增量，每步上限 5 mm；后六维为拇指两个主动关节、食指、中指、无名指、小指目标角增量，每步上限 0.06 rad。机械臂 IK 保持固定抓取朝向；求解失败时保留上一有效机械臂目标并计数。手指仍由位置控制和原 URDF mimic 联动驱动。

策略观察包含机器人关节位置/速度、末端位置/朝向/速度、视觉估计的方块相对位置、可见性和估计年龄、上一动作，以及控制器当前末端/手指目标。共 79 维。不包含物体真实位置、物体速度、物体 ID、接触 ID 或脚本阶段。物体真值仅用于奖励、终止与评估。

物理 500 Hz，控制 20 Hz，相机 10 Hz。RGB-D 相机 128×128、65° 视场，固定在摄像头模型上，外参是用于本仿真的设定，不是实机标定。相机转动关节保持零位位置目标，策略不控制它。视觉使用蓝色分割、深度反投影和已知 4 cm 方块尺寸修正表面偏差；暂不支持任意颜色、任意形状物体识别。遮挡时保留最后估计并标记过期。

碰撞按机器人/手/桌/物体分组，关闭手内部和机器人内部的碰撞，启用机器人与桌/物体的碰撞。掌部采用 box，分段手指采用凸网格，视觉网格独立。当前摩擦、控制增益、重力补偿与力限为仿真参数，尚未做实机标定。

## 验证与结果

- `checks/gate.json`：接口检查、视觉误差、遮挡/过期、无真值泄漏、逆解失败保持、桌面稳定，以及 10 次脚本抓取记录。
- `scripted_env_result.json`：同一 Gymnasium 环境、同一 9 维动作接口的脚本抓取检查。
- `runs/<名称>/`：配置、模型、观察归一化、监控 CSV、metrics.jsonl、TensorBoard、评估报告。
- `policy.zip` 必须配套使用同目录的 `vecnormalize.pkl`。周期 checkpoint 同样需要配套归一化文件。
- 训练入口会拒绝缺失、失败或与当前环境不一致的验收门槛；改动物理/环境后先重新运行 `check`。
- 固定位置成功不等于随机位置成功。首版先训练固定位置。随机阶段通过 `train --randomize` 或 `eval --randomize` 启用 XY 各 ±5 cm；两个评估报告分别保存。
- 80% 是策略验收目标，须以实际 100 回合评估结果为准，不能用脚本 10/10 成功代替。

## 日志与恢复

训练每 10,000 步保存 checkpoint 与对应归一化，正常结束或 Python 异常退出时保存最终 `policy.zip`。进程被强制终止时使用最近 checkpoint。TensorBoard 日志位于 run 目录的 `tensorboard/`，可用 `tensorboard --logdir runs --host 127.0.0.1 --port 6010` 查看。

依赖记录于 `requirements-lock.txt`；原模型源码和展示环境未被训练场景覆盖。不连接真实硬件。

## 本次实际验收结果

| 项目 | 结果 |
|---|---|
| 固定位置脚本抓取 | 10/10 成功，最高约 10.5 cm，保持约 4.1 秒 |
| 示范预热＋PPO 微调策略，固定位置 | 100/100 成功；抬高至少 10 cm 并保持 2 秒 |
| 随机位置，短程续训 1,024 步后 | 4/100，尚未达到 80% 目标 |
| 20 个随机位置的 RGB-D 定位 | 全部可见，最大误差约 2.2 mm |
| 环境与依赖 | Gymnasium/SB3 检查、遮挡过期、无物体真值泄漏、IK 失败保持、pip check 均通过 |
| 训练接口 | 4 环境 PPO、保存/恢复、最佳策略保存、实时 VNC 播放均已测试 |

固定位置评估使用种子 10000—10099、确定性策略；改变种子不会改变固定位置场景，因此 100/100 不表示随机位置泛化。随机位置评估使用种子 30000—30099，XY 各 ±5 cm。两种报告分别保存。

已验证模型：`runs/grasp_verified/policy.zip`，配套 `runs/grasp_verified/vecnormalize.pkl`。当前 VNC 正在循环播放这个策略，关闭 MuJoCo 窗口后，循环播放会退出。

```bash
# 对已验证模型做 100 回合固定位置评估
bash grasp.sh eval runs/grasp_verified/policy.zip --episodes 100 --seed 10000

# 在已打开的 VNC viewer 中循环播放真实策略；不是脚本示范
bash grasp.sh eval runs/grasp_verified/policy.zip --viewer --loop

# 从已验证模型继续随机位置训练，明确设置探索幅度与学习率
# 当前已有策略循环播放时，先停止相应播放进程，避免多个进程竞争实时画面。
bash grasp.sh train --resume runs/grasp_verified/policy.zip --randomize --steps 300000 --envs 4 --learning-rate 0.0003 --action-std 0.15 --run runs/random_train

# 最佳策略评估；最佳策略会配套保存归一化文件
bash grasp.sh eval runs/random_train/best_policy.zip --randomize --episodes 100 --seed 30000
```

训练每 10,000 步以及训练开始/结束时做小规模独立验证，保留 `best_policy.zip` 和对应归一化，避免只留下退化的最后一个模型。这不能代替正式的 100 回合验收。

当前限制：随机位置还需继续训练；工作空间边界附近会出现 IK 请求不可达，控制器会保持上一有效目标并记录次数。固定位置策略也有此类计数，尚未完成减少无效动作的优化。初版视觉仅支持已知大小、已知颜色的方块，摄像头和动力学参数尚未按实物标定。