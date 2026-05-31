"""Phase 2 SAS 策略网络 checkpoint 保存/加载。

checkpoint 是一个 dict，torch.save 到 .pt 文件：
    {
        "policy_type": "single" | "multihead",
        "candidate_dim": int,
        "global_dim": int,
        "hidden_dim": int,
        "channels": tuple | None,   # 多头才有意义，单头存 None
        "state_dict": policy.state_dict(),
        "metadata": dict,           # 可选训练配置 (top_k / episodes 等)
    }

save_policy_checkpoint: 保存权重 + 重建元数据。
load_policy_checkpoint: 重建对应策略网络并载入权重，返回 (policy, checkpoint_dict)。
"""

import torch

from phase2_sas_policy import Phase2SASActorCritic, Phase2SASMultiHeadActorCritic


def save_policy_checkpoint(policy, path, *, candidate_dim, global_dim, hidden_dim,
                           policy_type, channels=None, metadata=None):
    """保存策略网络权重 + 重建所需元数据到 path(.pt)。

    Args:
        policy: Phase2SASActorCritic 或 Phase2SASMultiHeadActorCritic 实例。
        path: 目标 .pt 文件路径。
        candidate_dim / global_dim / hidden_dim: 重建网络所需维度。
        policy_type: "single" 或 "multihead"。
        channels: 多头通道元组 (单头传 None)。
        metadata: 可选训练配置 dict。
    """
    if policy_type not in ("single", "multihead"):
        raise ValueError("policy_type must be 'single' or 'multihead'")
    checkpoint = {
        "policy_type": policy_type,
        "candidate_dim": int(candidate_dim),
        "global_dim": int(global_dim),
        "hidden_dim": int(hidden_dim),
        "channels": tuple(channels) if channels is not None else None,
        "state_dict": policy.state_dict(),
        "metadata": dict(metadata) if metadata is not None else {},
    }
    torch.save(checkpoint, path)
    return path


def load_policy_checkpoint(path, map_location="cpu"):
    """从 .pt 加载，重建对应策略网络并载入权重。

    根据 policy_type 选择 Phase2SASActorCritic 或 Phase2SASMultiHeadActorCritic 重建，
    用保存的 candidate_dim/global_dim/hidden_dim/channels 构造，再 load_state_dict。
    policy.eval() 后返回。

    Returns:
        (policy, checkpoint_dict)
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    policy_type = checkpoint["policy_type"]
    candidate_dim = int(checkpoint["candidate_dim"])
    global_dim = int(checkpoint["global_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])

    if policy_type == "single":
        policy = Phase2SASActorCritic(
            candidate_dim=candidate_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
        )
    elif policy_type == "multihead":
        channels = checkpoint["channels"]
        if channels is None:
            policy = Phase2SASMultiHeadActorCritic(
                candidate_dim=candidate_dim,
                global_dim=global_dim,
                hidden_dim=hidden_dim,
            )
        else:
            policy = Phase2SASMultiHeadActorCritic(
                candidate_dim=candidate_dim,
                global_dim=global_dim,
                hidden_dim=hidden_dim,
                channels=tuple(channels),
            )
    else:
        raise ValueError(f"unknown policy_type: {policy_type!r}")

    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy, checkpoint
