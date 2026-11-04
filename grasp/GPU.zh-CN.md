训练默认使用 `cuda:0`，在服务器 Conda 环境 `realman-mujoco` 中运行。MuJoCo 物理仍在 CPU，RGB-D 使用 EGL，VNC 单独显示实时场景。PPO的小型MLP和四个环境不一定比CPU训练更快，但网络前向、反向和优化器更新会使用GPU。

在仓库 `/home/szy2025/Desktop/RealMan` 根目录运行：

```bash
# 安装锁定的CUDA版PyTorch和依赖
source /opt/conda/etc/profile.d/conda.sh
conda activate realman-mujoco
python -m pip install -r grasp/requirements-lock.txt

# 物理限制修改后的训练前检查
bash grasp/grasp.sh check --trials 10

# GPU预训练和PPO训练（相对run路径以grasp目录为基准）
bash grasp/grasp.sh train --device cuda:0 --randomize --warmstart \
  --steps 300000 --envs 4 --run runs/floor_cuda

# 恢复同一环境指纹的策略，使用新的输出目录
bash grasp/grasp.sh train --device cuda:0 --randomize \
  --resume runs/floor_cuda/policy.zip --steps 300000 --envs 4 \
  --run runs/floor_cuda_resume

# VNC窗口与GPU策略循环回放分别启动
bash grasp/grasp.sh live
bash grasp/grasp.sh eval runs/floor_cuda/best_policy.zip \
  --device cuda:0 --randomize --viewer --loop
```

评估/回放默认使用CPU，也可通过 `--device cuda:0` 使用GPU。训练默认GPU，CUDA不可用时会直接报错；显式 `--device cpu` 可切回CPU。`run.json`记录实际设备、GPU型号、PyTorch和CUDA运行时版本。旧场景或旧力矩限制下的权重仍会因环境指纹不符而拒绝续训。

安装仅修改指定Conda环境中的Python包和随wheel提供的CUDA库，不更改系统NVIDIA驱动。安装依据为[PyTorch官方安装文档](https://pytorch.org/get-started/locally/)，软件包来自官方CUDA 13.0 wheel源。

本次验证环境：RTX 4090、驱动580.178.04、PyTorch 2.14.1+cu130、CUDA运行时13.0。官方cu130源当时提供的最新版本为2.14.1，替换了原2.15.0+cpu；依赖检查及锁定文件安装解析均通过。

验证：新力矩限制下完整场景检查及随机脚本抓取10/10通过；四个并行环境完成1024步CUDA PPO，随后完成30回合评估并保存策略和归一化参数。检查点恢复后再更新32步，参数和优化器状态均在GPU；预训练梯度路径、CPU/GPU推理一致性及CUDA不可用时报错均通过。报告位于仓库 `docs/cuda_training_check.json`。未进行预训练的短程PPO模型评估为0/30成功，本次结果只验证GPU训练链路，未达到抓取性能验收目标。正式长程训练尚未启动。
