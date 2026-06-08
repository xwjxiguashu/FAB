"""Launcher: 在 late_hi 实例上训练 multihead SAS 策略 (报告4 方向一)。

为什么单独一个启动脚本:
  train_phase2_sas_ppo.py 的 __main__ 把参数硬编码成 pressure 实例, 且 _run_cli()
  被注释掉 —— 所以 `python train_phase2_sas_ppo.py --instance late_hi` 的 CLI 参数
  会被忽略 (这正是之前误覆盖 pressure_mh_hard.pt 的原因)。本脚本直接以 late_hi 参数
  调用 main(); 用独立文件 (而非 python -c) 是为了让基于 spawn 的并行 rollout 子进程
  能干净地重新 import __main__。

配置与 pressure 训练对齐 (同 worker 数 / 同更新次数 / 不开 Lagrangian, 固定 w_qtime),
以便 late_hi 与 pressure 两个 checkpoint 可比。
"""

import os

from train_phase2_sas_ppo import main

_HERE = os.path.dirname(os.path.abspath(__file__))

# 与 train_phase2_sas_ppo.py __main__ 同口径: worker = 逻辑核 - 2, 固定 ~50 次 PPO 更新。
_WORKERS = max(1, (os.cpu_count() or 4) - 2)
_ITERS = 50

if __name__ == "__main__":
    main(
        mode="multihead",
        instance="late_hi",
        num_episodes=_WORKERS * _ITERS,   # 例: 14 worker → 700 episode → ~50 次更新
        parallel=_WORKERS,                # 启动时各 worker import torch 会静默几十秒
        save_every=_WORKERS * 5,          # 约每 5 次迭代存一次 checkpoint
        save_path=os.path.join(_HERE, "late_hi_mh.pt"),
        device="auto",                    # 瓶颈在 CPU 仿真, GPU 无加速
    )
