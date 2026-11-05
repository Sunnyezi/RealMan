# AGENTS.md

## 硬件配置

- 机械臂：RM65-6F-V
- 灵巧手：RH56DFTP-2R

## 训练时的可见仿真窗口（view + VNC）

开始训练时必须使用 view + VNC 配置，让 MuJoCo 仿真窗口在服务器 VNC 桌面 `:15` 上可见，禁止在无可见窗口的情况下静默训练。

- 训练进程本身使用 `MUJOCO_GL=egl` 做无头物理与渲染，由 0 号 worker 把状态写入 `<run>/live_state.npz`。
- 可见窗口由独立 Viewer 进程提供：`DISPLAY=:15`、`MUJOCO_GL=glfw`、`LIBGL_ALWAYS_SOFTWARE=1`，读取同一个 `live_state.npz` 实时显示。
- VNC 监听 `localhost:5915`；Windows 本地先执行 `ssh -N -L 15900:127.0.0.1:5915 uestc-server`，再用 VNC 客户端连接 `127.0.0.1:15900`。

启动步骤（均在仓库根目录）：

```bash
# 终端 A：先启动可见 Viewer，--stream 指向本次训练 run 的状态文件
bash grasp/grasp.sh live --stream runs/<run>/live_state.npz

# 终端 B：启动训练；不要加 --no-live，否则 Viewer 收不到画面
bash grasp/grasp.sh train --device cuda:0 --randomize --warmstart \
  --steps 300000 --envs 4 --run runs/<run>
```

注意：

- `--no-live` 会关闭训练画面发布，训练时禁止使用。
- Viewer 的 `--stream` 与训练的 `--run` 必须对应同一路径（相对于 `grasp/` 目录解析，例如 `runs/floor/live_state.npz`）。
- Viewer 可先于训练启动，无数据时显示 `WAITING / LAST FRAME`。
- 仅查看模型本体时可使用根目录 `bash launch_view.sh`（同样走 `:15` VNC + glfw）。

## 提交规范

- Git 提交信息（commit message）使用中文。
