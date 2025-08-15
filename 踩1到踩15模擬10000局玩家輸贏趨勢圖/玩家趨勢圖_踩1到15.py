import torch
import numpy as np
import matplotlib.pyplot as plt
import gc
import os

# 設定 matplotlib 字體以支援中文
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# === 模擬設定 ===
TARGET_RTP = 0.965
players = 20
games_per_player = 10_000
min_step = 1
max_step = 15
output_dir = "player_trends_0.965"
os.makedirs(output_dir, exist_ok=True)

# === 裝置設定（可用則使用 MPS，否則 CPU）===
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === 倍率池（25 顆，依據實際數量）===
full_pool = np.array(
    [1.25] * 9 + [1.5] * 6 + [2.0] * 4 + [3.0] * 3 + [5.0] * 3,
    dtype=np.float32
)

# === 計算成功機率公式 ===
def compute_probs(tensor):
    a = (tensor == 1.25).sum(dim=1)
    b = (tensor == 1.5).sum(dim=1)
    c = (tensor == 2.0).sum(dim=1)
    d = (tensor == 3.0).sum(dim=1)
    e = (tensor == 5.0).sum(dim=1)
    return TARGET_RTP * (1/1.25)**a * (1/1.5)**b * (1/2.0)**c * (1/3.0)**d * (1/5.0)**e

# === 主模擬迴圈 ===
for step in range(min_step, max_step + 1):
    player_trends = []

    for p in range(players):
        sampled_np = np.array([
            np.random.choice(full_pool, size=step, replace=False)
            for _ in range(games_per_player)
        ], dtype=np.float32)
        sampled = torch.tensor(sampled_np, device=device)

        probs = compute_probs(sampled)
        rand_vals = torch.rand(games_per_player, device=device)
        is_win = rand_vals <= probs

        rtp = torch.prod(sampled * is_win[:, None], dim=1)
        profit = (rtp - 1).cpu().numpy()

        cumulative = np.cumsum(profit)
        player_trends.append(cumulative)

        del sampled, probs, rand_vals, is_win, rtp, profit
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    # === 畫圖 ===
    plt.figure(figsize=(12, 6))
    for trend in player_trends:
        plt.plot(trend, alpha=0.7)
    plt.title(f"踩{step} 玩家趨勢圖（每人 {games_per_player} 局）")
    plt.xlabel("遊戲局數")
    plt.ylabel("累積獲利")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/踩{step}_趨勢圖.png")
    plt.close()

    print(f"✅ 踩{step} 模擬與趨勢圖儲存完成")

print("🎯 所有趨勢圖已完成！")

