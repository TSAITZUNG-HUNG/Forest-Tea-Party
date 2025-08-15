import torch
import pandas as pd
import numpy as np

# === 模擬設定 ===
TARGET_RTP = 0.965
total_simulations = 400_000_000
batch_size = 1_000_000
max_steps = 25

# === 設定 GPU 裝置 ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === 倍率池 ===
multiplier_pool = torch.tensor(
    [1.25] * 9 + [1.5] * 6 + [2.0] * 4 + [3.0] * 3 + [5.0] * 3,
    dtype=torch.float32,
    device=device
)
pool_size = len(multiplier_pool)

# === 計算每組過關機率 ===
def compute_probs(shuffled):
    a = (shuffled == 1.25).sum(dim=1)
    b = (shuffled == 1.5).sum(dim=1)
    c = (shuffled == 2.0).sum(dim=1)
    d = (shuffled == 3.0).sum(dim=1)
    e = (shuffled == 5.0).sum(dim=1)
    return TARGET_RTP * (1/1.25)**a * (1/1.5)**b * (1/2.0)**c * (1/3.0)**d * (1/5.0)**e

# === 模擬每個 step 並記錄 RTP ===
step_results = []

for step in range(max_steps):
    total_rtp = 0
    total_success = 0

    for i in range(0, total_simulations, batch_size):
        sims = min(batch_size, total_simulations - i)
        idx = torch.randint(0, pool_size, (sims, step + 1), device=device)
        shuffled = multiplier_pool[idx]

        probs = compute_probs(shuffled)
        rand_vals = torch.rand(sims, device=device)
        is_win = rand_vals <= probs
        rtp = torch.prod(shuffled * is_win[:, None], dim=1)

        total_rtp += rtp.sum().item()
        total_success += is_win.sum().item()

    avg_rtp = total_rtp / total_simulations
    step_results.append((step + 1, avg_rtp))
    print(f"✅ 踩{step+1}: 平均RTP = {avg_rtp:.6f}")

# === 儲存結果 ===
df = pd.DataFrame(step_results, columns=["踩到第幾格", "平均RTP"])
df.to_excel("踩1到25_每階段平均RTP.xlsx", index=False)
print("✅ 全部模擬完成，已儲存至 Excel")

