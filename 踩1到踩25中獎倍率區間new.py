import torch
import pandas as pd
import numpy as np

# === 模擬設定 ===
TARGET_RTP = 0.965
total_simulations = 10_000_000
max_steps = 25

# === 設定 GPU 裝置 ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === 真實倍率池（固定 25 顆）===
full_pool = np.array(
    [1.25] * 9 + [1.5] * 6 + [2.0] * 4 + [3.0] * 3 + [5.0] * 3,
    dtype=np.float32
)

# === 計算過關機率公式 ===
def compute_probs(shuffled_tensor):
    a = (shuffled_tensor == 1.25).sum(dim=1)
    b = (shuffled_tensor == 1.5).sum(dim=1)
    c = (shuffled_tensor == 2.0).sum(dim=1)
    d = (shuffled_tensor == 3.0).sum(dim=1)
    e = (shuffled_tensor == 5.0).sum(dim=1)
    return TARGET_RTP * (1 / 1.25)**a * (1 / 1.5)**b * (1 / 2.0)**c * (1 / 3.0)**d * (1 / 5.0)**e

# === 開始模擬每一踩 ===
step_results = []

for step in range(1, max_steps + 1):
    # 使用 NumPy 向量化產生不重複樣本（每列是一局遊戲的 step 個倍率）
    sampled_np = np.array([
        np.random.choice(full_pool, size=step, replace=False)
        for _ in range(total_simulations)
    ], dtype=np.float32)

    # 轉成 Tensor 上 GPU
    sampled = torch.tensor(sampled_np, device=device)

    # 過關機率與成功局
    probs = compute_probs(sampled)
    rand_vals = torch.rand(total_simulations, device=device)
    is_win = rand_vals <= probs

    # 計算成功倍率
    rtp = torch.prod(sampled * is_win[:, None], dim=1)
    rtp_nonzero = rtp[rtp > 0]

    if rtp_nonzero.numel() > 0:
        min_rtp = rtp_nonzero.min().item()
        max_rtp = rtp_nonzero.max().item()
    else:
        min_rtp = 0.0
        max_rtp = 0.0

    step_results.append((step, min_rtp, max_rtp))
    print(f"✅ 踩{step}: 最小成功倍率 = {min_rtp:.2f}, 最大成功倍率 = {max_rtp:.2f}")

# === 輸出結果 ===
df = pd.DataFrame(step_results, columns=["踩到第幾格", "最小成功倍率", "最大成功倍率"])
df.to_excel("踩1到25_中獎倍率區間_修正版.xlsx", index=False)
print("✅ 模擬完成，結果已輸出至 Excel")
