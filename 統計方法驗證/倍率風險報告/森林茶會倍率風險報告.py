import torch
import pandas as pd
import numpy as np

# === 模擬設定 ===
TARGET_RTP = 0.93
batch_size = 50_000
min_step = 5
max_step = 10
simulations = 100_000_000  # 每步模擬局數（1000 億）

# 設定 GPU 裝置
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === 真實倍率池（25 顆）===
full_pool_np = np.array(
    [1.25] * 9 + [1.5] * 6 + [2.0] * 4 + [3.0] * 3 + [5.0] * 3,
    dtype=np.float32
)

# === 過關機率計算公式 ===
def compute_probs(shuffled_tensor):
    a = (shuffled_tensor == 1.25).sum(dim=1)
    b = (shuffled_tensor == 1.5).sum(dim=1)
    c = (shuffled_tensor == 2.0).sum(dim=1)
    d = (shuffled_tensor == 3.0).sum(dim=1)
    e = (shuffled_tensor == 5.0).sum(dim=1)
    return TARGET_RTP * (1 / 1.25)**a * (1 / 1.5)**b * (1 / 2.0)**c * (1 / 3.0)**d * (1 / 5.0)**e

# === 開始模擬 ===
for step in range(min_step, max_step + 1):
    total_simulations = simulations
    print(f"\n🚀 開始模擬 踩{step}，共 {total_simulations:,} 局")

    total_reward = 0.0
    total_success = 0
    all_rewards = []

    for i in range(0, total_simulations, batch_size):
        sims = min(batch_size, total_simulations - i)

        # 抽樣 & GPU tensor
        sampled_np = np.array([
            np.random.choice(full_pool_np, size=step, replace=False)
            for _ in range(sims)
        ], dtype=np.float32)
        sampled = torch.tensor(sampled_np, device=device)

        # 機率與中獎判定
        probs = compute_probs(sampled)
        rand_vals = torch.rand(sims, device=device)
        is_win = rand_vals <= probs

        # 成功局倍率
        multipliers = torch.prod(sampled * is_win[:, None], dim=1)
        total_reward += multipliers.sum().item()
        total_success += is_win.sum().item()

        # 成功局儲存倍率（轉 numpy）
        rewards_np = multipliers[multipliers > 0].cpu().numpy()
        all_rewards.extend(rewards_np)

        # 即時進度顯示
        if (i + sims) % 10_000_000 == 0 or (i + sims) == total_simulations:
            print(f"  ✅ 已完成 {i + sims:,} 局，當前 RTP = {total_reward / (i + sims):.6f}")

    # === 單一步模擬完成：計算統計、輸出單獨 Excel ===
    rewards_np = np.array(all_rewards)
    mean_multiplier = rewards_np.mean() if len(rewards_np) > 0 else 0.0
    std_multiplier = rewards_np.std() if len(rewards_np) > 0 else 0.0
    percentiles = np.percentile(rewards_np, q=range(1, 100)) if len(rewards_np) > 0 else np.zeros(99)

    step_row = {
        "踩步數": step,
        "模擬局數": total_simulations,
        "成功局數": total_success,
        "中獎率": total_success / total_simulations,
        "實際 RTP": total_reward / total_simulations,
        "成功局平均倍數": mean_multiplier,
        "成功局標準差": std_multiplier,
        **{f"{p}% 倍數": val for p, val in zip(range(1, 100), percentiles)}
    }

    df_step = pd.DataFrame([step_row])
    file_name = f"森林茶會_踩{step}_爆炸倍率風險_93.xlsx"
    df_step.to_excel(file_name, index=False)
    print(f"📄 踩{step} 統計報表已儲存：{file_name}")

    # 釋放記憶體
    del all_rewards, sampled, sampled_np, rewards_np, multipliers, probs, rand_vals, is_win
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
