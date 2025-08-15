import torch
import pandas as pd
import numpy as np
import os

# === 模擬設定 ===
TARGET_RTP = 0.985
MAX_MULTIPLIER = 1_000_000
batch_size = 50_000
min_step = 1
max_step = 25
rounds = 10_000
sims_per_round = 10_000
output_dir = "森林茶會_爆炸倍率風險_98.5"
os.makedirs(output_dir, exist_ok=True)

# GPU 裝置
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 倍率池（25 顆）
full_pool_np = np.array(
    [1.25] * 9 + [1.5] * 6 + [2.0] * 4 + [3.0] * 3 + [5.0] * 3,
    dtype=np.float32
)

# 機率計算
def compute_probs(shuffled_tensor):
    a = (shuffled_tensor == 1.25).sum(dim=1)
    b = (shuffled_tensor == 1.5).sum(dim=1)
    c = (shuffled_tensor == 2.0).sum(dim=1)
    d = (shuffled_tensor == 3.0).sum(dim=1)
    e = (shuffled_tensor == 5.0).sum(dim=1)
    return TARGET_RTP * (1 / 1.25)**a * (1 / 1.5)**b * (1 / 2.0)**c * (1 / 3.0)**d * (1 / 5.0)**e

# 主迴圈
for step in range(min_step, max_step + 1):
    print(f"\n🚀 開始模擬 踩{step}（共 {rounds * sims_per_round:,} 局）")
    rtp_list = []
    all_rewards = []

    for r in range(rounds):
        total_reward = 0.0
        total_sims = 0

        for i in range(0, sims_per_round, batch_size):
            sims = min(batch_size, sims_per_round - i)
            sampled_np = np.array([
                np.random.choice(full_pool_np, size=step, replace=False)
                for _ in range(sims)
            ], dtype=np.float32)
            sampled = torch.tensor(sampled_np, device=device)
            probs = compute_probs(sampled)
            rand_vals = torch.rand(sims, device=device)
            is_win = rand_vals <= probs
            multipliers = torch.prod(sampled * is_win[:, None], dim=1)
            multipliers = torch.clamp(multipliers, max=MAX_MULTIPLIER)
            total_reward += multipliers.sum().item()
            total_sims += sims
            rewards_np = multipliers[multipliers > 0].cpu().numpy()
            all_rewards.extend(rewards_np)

        rtp = total_reward / total_sims
        rtp_list.append(rtp)
        if (r + 1) % 1000 == 0:
            print(f"  ✅ 完成 {r + 1:,} / {rounds:,} 輪，當前 RTP：{rtp:.6f}")

    # 統計
    rewards_np = np.array(all_rewards)
    mean_multiplier = rewards_np.mean() if len(rewards_np) > 0 else 0.0
    std_multiplier = rewards_np.std() if len(rewards_np) > 0 else 0.0
    percentiles = np.percentile(rewards_np, q=range(1, 100)) if len(rewards_np) > 0 else np.zeros(99)
    rtp_np = np.array(rtp_list)
    summary = {
        "踩步數": step,
        "模擬局數": rounds * sims_per_round,
        "RTP 平均值": rtp_np.mean(),
        "RTP 標準差": rtp_np.std(),
        "最小 RTP": rtp_np.min(),
        "最大 RTP": rtp_np.max(),
        "成功局平均倍數": mean_multiplier,
        "成功局標準差": std_multiplier,
        **{f"{p}% 倍數": val for p, val in zip(range(1, 100), percentiles)}
    }

    df_step = pd.DataFrame([summary])
    file_path = os.path.join(output_dir, f"森林茶會_踩{step}_爆炸倍率風險_98.5.xlsx")
    df_step.to_excel(file_path, index=False)
    print(f"📄 踩{step} 統計報表已儲存：{file_path}")

    # 清除記憶體
    del all_rewards, rewards_np, rtp_np
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
