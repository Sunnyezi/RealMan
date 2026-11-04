# 平地抓取训练和真机验证

当前基线使用一个 4 cm 方块、XY 各 ±5 cm、固定朝向和 128×128 MLP。原平台权重属于历史场景。仿真检查、验证、独立测试分别使用种子 0–19、71001–71030、81001–81200。朝向压力测试复用验证位置，按种子给方块分配 −45° 到 +45° 的偏航，单独报告。蓝色 RGB-D 分割和现有抓取姿态仍按固定朝向设计。

## 训练和比较

在服务器仓库根目录执行。运行目录按训练种子分开，复现实验时为 `root` 选择新路径。保留每次运行的配置、最佳模型及与模型成对保存的归一化文件。

```bash
bash grasp/grasp.sh check
root=runs/floor_pilot
for seed in 42 43 44; do
  bash grasp/grasp.sh train --randomize --warmstart --steps 20000 --envs 4 \
    --seed "$seed" --run "$root/$seed" --no-live
done
bash grasp/grasp.sh benchmark gate --root "$root" --output "$root/expansion_gate.json"
bash grasp/grasp.sh benchmark compare --root "$root" --output "$root/heldout.json" --workers 8
bash grasp/grasp.sh benchmark script --split yaw --output "$root/yaw_script.json"
bash grasp/grasp.sh benchmark pose --output "$root/yaw_pose_candidates.json"
```

`warmstart.zip` 是模仿候选，`best_policy.zip` 是 30 个验证场景上选出的 PPO 候选。三组训练中 PPO 都有正收益，且按共用验证场景配对重采样得到的增益区间下界大于零时，才扩到约 30 万步。独立测试每个候选使用同一组 200 个场景。`metrics.jsonl` 记录端到端吞吐、累计成功回合和物理警告。偏航候选检查用深度拟合方块顶面的方向，并检查调整后的抓取轨迹是否可达。该检查只评估几何与 IK。训练默认把帧写到各自运行目录，可用 `bash grasp/grasp.sh live --stream runs/floor_pilot/42/live_state.npz` 查看，`--no-live` 关闭发布。

## 录制、校准和随机化

真机目标是 **RM65-6F-V + RH56DFTP-2R**。目前仿真机械臂使用 **RM65-6FB-V**，手使用 RH56DFTP 右手模型。搭建后先核对控制器上报型号、实际法兰至手掌的刚体变换、各轴零位与限位，再用低速测得的 TCP 轨迹核对运动学。相机型号和外参以最终安装为准。厂商分别列出了 RM65-6F 和 RM65-6FB-V 的末端参数，应用前需取得与铭牌及固件匹配的模型或实测标定。RH56DFTP 系列的触觉区域数随硬件配置而异，需读取设备映射并逐区验证。 [睿尔曼 RM65 本体参数](https://develop.realman-robotics.com/robot/robotParameter/RM65OntologyParameters/) · [因时 RH56DFTP 系列手册](https://en.inspire-robots.com/wp-content/uploads/2025/01/INSPIRE-ROBOTS-The-Dexterous-Hand-RH56DFTP-User-Manual-V1.0.0.pdf)

先确认机械臂和手的型号及固件，标定机器人底座、手、相机的坐标变换，并按 `sensor_map.json` 建立真实触觉通道到仿真 17 个区域的映射。同步记录原始 RGB-D、关节、触觉、指令和响应。把标定后每个策略时刻的记录写为 JSONL，每行字段如下。

| 字段 | 含义 |
| --- | --- |
| `timestamp_ns` | 共用单调时钟上的策略时刻，单位 ns |
| `streams` | `joints`、`rgbd`、`touch` 各含 `timestamp_ns` 和设备 `sequence`，原始 RGB-D 可另存并记录帧文件索引 |
| `observation` | `deployment.py` 的 `FIELDS`，顺序与 `GraspEnv._obs` 对齐。位置单位 m、角度 rad、触觉 N，末端量在机器人底座坐标系，速度顺序为角速度、线速度 |
| `commanded_arm_q_rad`、`command_sent_ns`、`command_ack_ns`、`no_contact` | 可选。提供已对齐的 6 轴关节指令、匹配的发送与确认时刻以及无接触标签 |
| `estop`、`controller_fault`、`recorded_proposal` | 复核已录制的动作时提供。动作含 `arm_q_rad`、`hand_q_rad`、`target_xyz_m`、`duration_s` 和经独立工位检查得出的 `collision_free`、`min_ground_clearance_m` |

`bash grasp/grasp.sh deploy calibrate RECORDING.jsonl --output CALIBRATION.json` 输出延迟、跟踪误差、采样间隔和无接触读数的分位数。序号缺口是录制覆盖率诊断，需区分降采样和设备丢帧。由标定数据和台架测量生成一个范围 JSON，给 `train --domain-profile PROFILE.json` 或 `benchmark --profile PROFILE.json`。字段是闭区间 `[min,max]`。支持 `mass_scale`、`friction_scale`、`actuator_scale`、`light_scale`、`yaw_deg`、`camera_offset_m`、`vision_error_m`、`vision_drop_probability`、`touch_noise_N` 、`joint_position_noise_rad`、`joint_velocity_noise_rad_s`、`camera_rotation_deg` 和 `action_delay_steps`。缺省字段采用标称值。相机外参偏差在一回合内固定，观测噪声逐次采样，视觉丢帧从初始化后开始模拟。动作延迟以 50 ms 步为单位。标定与最终验收应使用不同的录制。

## 回放和验收

- 用 `bash grasp/grasp.sh deploy replay RECORDING.jsonl --checkpoint POLICY.zip --limits LIMITS.json --output REPLAY.json` 验证观测、归一化、策略推理时间和期望目标。限位文件填写已验证的 `arm_min/max`、`hand_min/max`、`xyz_min/max`（各自拆成两个键），`arm_max_speed_rad_s`、`hand_max_speed_rad_s`、`min_ground_clearance_m`，以及三路输入的 `max_age_s`。回放中的限位检查针对录制动作。新策略动作在连接设备前仍需由工位规划器独立检查 IK、轨迹碰撞、地面余量、限位和速度，并接入控制器故障、数据过期和急停。
- 用 `bash grasp/grasp.sh deploy scenarios --center-xy X Y --radius 0.05 --lighting LABEL_A LABEL_B --output SCENARIOS.json` 固定并打乱 100 个位置、偏航与光照组合。中心和光照标签取自已标定工位。先低速执行脚本基线，再在相同场景测策略。安全层拒绝后停止该回合并记录原因。
- 每次试验记录 `candidate`、`scenario_id`、`held_height_m`、`hold_s`、`cycle_s`、`drop`、`protective_stop`。保持高度是持续保持期间的最小离地高度。用 `bash grasp/grasp.sh deploy accept TRIALS.jsonl --scenarios SCENARIOS.json --output REPORT.json` 检查两组使用同一份场景表，并报告成功数、Wilson 95% 区间、周期时间、掉落和保护停机。首轮门槛为 90/100，成功要求抬升至少 10 cm 并持续 2 s。

当前命令覆盖离线回放和验收。真实设备的采集、标定、运动发送与急停应接入现场驱动，使用实际设备与工位验证。
