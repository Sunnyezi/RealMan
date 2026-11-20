# π0.5 搬运放置：任务定义、示教与纯策略验收

2026-11-20 记录。结论只适用于当前仿真：同一种 4 cm 蓝色方块（±5 cm 随机位置）、固定提示词、单一绿色目标区（±4 cm 随机位置）。不涉及未见物体、多目标或实机。

## 结论

检查点为 `pi05/checkpoints/pi05_realman_absolute12/place_20261119/29999`，由抓起保持基线 `baseline_20261117/29999` 热启动，在放置示教上微调 30,000 步得到。下表全程由 π0.5 输出 12 维绝对关节目标（`execution=pi05_policy_absolute12_no_takeover`），没有几何、RGB-D 或触觉控制器接管，也不读取物体或目标区真值。

| 执行方式 | 种子 | 成功 | Wilson 95% 区间 | 失败 |
| --- | --- | ---: | --- | --- |
| 原方式（每次执行 2 步） | validation 140000–140099 | 40/100 | 31.0%–49.8% | 未松手 25，未抬起 16，放在区外 7，松手后超时 7，未抓住 5 |
| 时间集成（decay 0.01） | validation 140000–140099 | 93/100 | 86.3%–96.6% | 放在区外 5，未松手 1，松手后超时 1 |
| 时间集成（decay 0.01） | **test 210000–210199** | **182/200** | **86.2%–94.2%** | 松手后超时 8，放在区外 6，未松手 4 |

成功率的点估计 91% 达到了 ≥90% 的目标，但 Wilson 区间下限是 86.2%，不能在统计意义上排除真实成功率略低于 90%。

验证区间上只比较了这两种执行方式，选定时间集成后才首次使用 test 区间，且只跑一次。test 区间没有物理异常；推理延迟中位数 142.5 ms，P95 为 174.8 ms，推理期间仿真暂停，所以不能据此声称实机 20 Hz 实时能力。

## 任务定义（`pi05/place_task.py`）

- 场景：由 `grasp/scene.xml` 程序化生成 `pi05/place/scene.xml` 并入库；`python place_task.py --check` 和测试会检查重新生成的结果与入库文件一致。新增一个 mocap 刚体 `place_target`：8×8 cm、厚 1 mm 的绿色薄垫，`contype=conaffinity=0`，没有自由度，qpos 布局与原场景相同。
- 目标区：中心为方块名义位置 + (0, −0.15) m，每个种子在 x/y 上各随机 ±4 cm（独立随机数生成器，种子 +9,000,000）。方块位置仍是原来的 ±5 cm 随机，两者不重叠。−y 侧是事先用 IK 探测过的可达侧。
- 提示词：`"Pick up the blue cube and place it on the green square."`；每回合最多 400 步（20 s），物理与 `RGBEnv.step` 相同（每步 25 个 2 ms 子步）。
- 成功（以下条件连续满足 1 s）：
  - 方块曾被抬起 ≥3 cm；
  - 方块中心与目标中心的 xy 距离 ≤2.5 cm；
  - 静置在地面：高度 <5 mm，线速度 <0.02 m/s，角速度 <0.3 rad/s；
  - 手与方块无接触，手掌已比抓取时高出 ≥4 cm；
  - 手和臂都不碰地；松手瞬间方块高度 ≤3 cm（不允许抛落）。
- 失败分类：`physics`、`placed_outside`（放在区外：抬起后在区外静置 1 s，提前终止）、`no_grasp`（未抓住）、`no_lift`（未抬起）、`not_released`（未松手）、`released_not_settled`（松手后超时：松过手，但 400 步内没有满足成功条件）。
- 里程碑：抓住、抬起 3 cm、在目标上方（抬起 ≥3 cm 时 xy 误差 ≤2.5 cm）、松手、放置成功。策略常在下降途中仍横向修正，所以“在目标上方”可低于成功率。
- `place_fingerprint()` 对抓起任务的 `fingerprint()`、`place_task.py` 和放置场景一起做 SHA256，值为 `f2f04afc…`。抓起任务的指纹 `8bcb9f6c…` 和 `adapter.py` 的 SHA 不变，`baseline_20261117/29999` 仍可按原方式服务；`--task place` 会拒绝抓起任务的检查点，反之亦然。

## 示教老师（`pi05/place_teacher.py`）

阶段：接近 → 闭合 → 抬升 → 平移 → 下放 → 张开 → 抬离 → 保持。每次阶段切换只取决于可观测进度（臂和手的目标是否到位、方块高度），没有隐藏计时器或停顿，避免同一观测对应不同动作。接近与闭合复用原老师的规划器抓取位姿、在线 RGB-D 重规划和 +0.12 rad 握持目标，抓取点上移 3 mm；老师读取方块和目标区的真实位置，策略只看到图像和关节。

| 参数 | 值 |
| --- | --- |
| 抬升 | 抓取高度 +5 cm；前 3 cm 每步 1.25 mm，之后每步 2.5 mm |
| 平移 | 每步 2 mm |
| 下放 | 每步 2 mm，方块高度 <2 cm 后每步 1 mm，方块高度 ≤3 mm 时张手 |
| 张开 | 手指每步 0.02 rad，回到复位时的张开目标 |
| 抬离 | 6 cm |

门槛与调参：
- 第一版（抬升 7 cm、平移每步 3 mm）在训练种子 70000–70099 上 95/100：区外掉落 3、松手后未静止 2，都发生在目标区远端（x≈0.59–0.60）的平移途中。
- 在训练种子 70000–70199 上扫描 6 组参数（仅分析，不进入训练数据）后，改为抬升 5 cm、平移每步 2 mm，200/200。
- 定版门槛：在未参与调参的训练种子 70300–70399 上 100/100（`pi05/outputs/place_gate_20261119_b`）。

## 数据与训练

- 采集：`collect_place.py --mode collect`，16 个进程，训练种子 71000 起、每进程间隔 1000 个，共尝试 1008 次全部成功，保存 1000 条（种子 71000–86054）。每回合 224–310 步，平均 266 步。标签就是执行的 float32 绝对目标，不加执行噪声。
- 审核：5 条轨迹按绝对动作重放后全部成功，最大关节误差 1.18e-7–1.19e-7 rad。
- 转换：266,381 帧，batch 4 数据检查通过。新任务的动作分布不同，归一化由这份数据重新计算（SHA256 `4794f8d7…`）。
- 训练：`launch_place.py` 在 tmux `pi05-place_20261119` 中依次执行数据校验、转换、数据检查、1 步热启动试跑和正式训练，状态写入 `pi05/runs/place_20261119/job_status.json`。
  - 从 `baseline_20261117/29999` 热启动，batch 4，30,000 步，峰值学习率 2e-5，余弦衰减到 2e-6。
  - 2026-11-20 00:02 CST 开始，08:02 保存最终检查点 29999，约 0.96 s/步。第 0 步 loss 0.038，最后 1000 步 0.0003–0.0013。
  - 全程在 VNC `:15` 播放带标签的放置示教回放（`supervised training / demonstration replay`）。
- 训练结束后自动在验证区间评估最终检查点的两种执行方式；结果位于 `pi05/runs/place_20261119_eval/`（不入库）。

## 验证结果分析

- 时间集成仍是关键。原方式在每个阶段都有损失：抓住 95、抬起 79、松手 54、成功 40；时间集成为抓住 100、抬起 100、松手 99、成功 93。原方式最多的失败是未松手（25）和未抬起（16），与抓起任务中“换块处动作跳变”的结论一致。
- 时间集成下的 7 次失败里 5 次是放在区外，xy 误差 2.8–4.1 cm，只有一次 16 cm；其余为一次未松手（误差 8.5 mm，到 400 步仍未张手）和一次松手后超时。
- 验证区间的失败偏向远端目标：目标区相对名义位置 dy < −2 cm 时 20/24，dy > +2 cm 时 23/23，中间 50/53。这和第一版老师在远端平移途中掉块的位置相同。
- 成功回合的放置误差中位数 5.4 mm，P90 为 10.7 mm，3 回合超过 15 mm；平均 270 步完成（197–371）。

## 测试结果分析

- 200 回合全部抓住并抬起 3 cm，196 回合松手，182 回合成功。成功回合平均 266 步（201–361），放置误差中位数 5.3 mm，P90 为 10.3 mm，5 回合超过 15 mm。
- 18 次失败中 12 次到 400 步超时（未松手 4、松手后超时 8）。按最后一步看：
  - 9 次手仍碰着方块、手掌没有抬起，其中 4 次手或臂同时碰地；
  - 2 次方块已放好且手已离开，但手掌只比抓取时高出 2.6、3.3 cm，没到 4 cm；
  - 1 次（210155）已满足全部条件，1 s 保持只进行了 0.2 s 就到了 400 步。
- 6 次放在区外：3 次在方块离地 2.0–2.8 cm 时就松手，方块落地后弹开或滚开，误差 5.5–11.2 cm；另 3 次松手较低（0–1.5 cm），误差 2.9–3.7 cm，刚好出界。
- 目标区位置的影响比验证区间弱：dy < −2 cm 时 39/44，中间 98/109，dy > +2 cm 时 45/47。
- 和验证区间相比，测试区间的主要失败从“放在区外”变成了“松手和撤离太慢，没在 400 步内完成”。验收以 test 区间的 182/200 为准。

## 复现

```bash
cd ~/Desktop/RealMan/pi05
S=~/.conda/envs/realman-mujoco/bin/python
# 门槛（训练种子 70300–70399）、采集和重放审核
MUJOCO_GL=egl $S collect_place.py --mode gate --count 100 --workers 16 --start-index 60300 \
  --output outputs/place_gate_<date>
MUJOCO_GL=egl $S collect_place.py --mode collect --count 1000 --workers 16 --start-index 61000 \
  --gate outputs/place_gate_<date>/summary.json --output data/place_<date>
MUJOCO_GL=egl $S collect_place.py --mode replay data/place_<date>/episode_000000
# 转换、数据检查、试跑、训练、验证评估（tmux，VNC :15 可见）
$S launch_place.py --experiment place_<date> --raw data/place_<date>
# 测试区间（选定执行方式后只跑一次）
$S evaluate_checkpoint.py --task place --execution policy --ensemble-decay 0.01 \
  --repo-id local/rm65_rh56_pi05_place_20261119 \
  --checkpoint checkpoints/pi05_realman_absolute12/place_20261119/29999 \
  --split test --start-index 10000 --count 200 --port 8096 --output runs/<new_dir>
```

`openpi_run.py`、`evaluate_policy.py`、`evaluate_checkpoint.py`、`view_data.py` 新增 `--task {hold,place}`，`train_visible.sh` 读环境变量 `PI05_TASK`，默认都是 `hold`，原有命令行为不变。`grasp/grid_live.py` 新增 `--scene`，并在流中有 `place_target` 时同步目标区位置。

## 检查

```bash
cd pi05
~/.conda/envs/realman-mujoco/bin/python -m unittest test_contracts test_stabilizer   # 原有 12 项
MUJOCO_GL=egl ~/.conda/envs/realman-mujoco/bin/python test_place.py                     # 放置任务 6 项
~/.conda/envs/realman-mujoco/bin/python place_task.py --check                           # 入库场景未过期
```

`test_place.py` 检查：抓起任务指纹和 `adapter.py` 的 SHA 不变；放置场景可重复生成且状态布局一致；放置环境与原环境在相同动作下物理一致；老师在固定种子上依次走完各阶段并成功；未抬起、放在区外、未松手、手未抬离、抛落、碰地等情况都判失败。

## 局限

- 只有一种物体和一种目标区，提示词固定，不能说明语言条件或未见物体的泛化。
- 老师和成功判据都依赖仿真真值；接触、摩擦和静置阈值没有在真实硬件上标定，仿真结果不构成 sim2real 验证。
- 剩余失败主要是末段松手和撤离太慢（测试 12 次），其次是松手偏高或略微出界（测试 6 次）。可以考虑让老师在更低处松手、示教中加入放置后的纠正片段，或者加长回合，本轮都没有做。
