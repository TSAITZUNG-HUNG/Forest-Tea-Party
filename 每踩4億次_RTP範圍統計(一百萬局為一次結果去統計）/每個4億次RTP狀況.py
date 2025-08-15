import torch
import pandas as pd
import numpy as np

# === 模擬設定 ===
TARGET_RTP = 0.93
batch_size = 1_000_000 #設定每個批次數量
step_simulations = 400_000_000  # 每格模擬次數
min_step = 1
max_step = 25

# === 設定 GPU 裝置 ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === 真實倍率池（25 顆）===
full_pool_np = np.array(
    [1.25] * 9 + [1.5] * 6 + [2.0] * 4 + [3.0] * 3 + [5.0] * 3,
    dtype=np.float32
)

# === 過關機率公式 ===
def compute_probs(tensor):
    a = (tensor == 1.25).sum(dim=1)
    b = (tensor == 1.5).sum(dim=1)
    c = (tensor == 2.0).sum(dim=1)
    d = (tensor == 3.0).sum(dim=1)
    e = (tensor == 5.0).sum(dim=1)
    return TARGET_RTP * (1/1.25)**a * (1/1.5)**b * (1/2.0)**c * (1/3.0)**d * (1/5.0)**e

# === 模擬執行 ===
results = []

for step in range(min_step, max_step + 1):
    above_count = 0
    below_count = 0
    min_rtp = float("inf")
    max_rtp = 0.0
    print(f"\n🚀 踩{step}：開始模擬 {step_simulations:,} 次")

    for i in range(0, step_simulations, batch_size):
        sims = min(batch_size, step_simulations - i)

        sampled_np = np.array([
            np.random.choice(full_pool_np, size=step, replace=False)
            for _ in range(sims)
        ], dtype=np.float32)
        sampled = torch.tensor(sampled_np, device=device)

        probs = compute_probs(sampled)
        rand_vals = torch.rand(sims, device=device)
        is_win = rand_vals <= probs
        rtp = torch.prod(sampled * is_win[:, None], dim=1)

        batch_rtp = rtp.sum().item() / sims
        if batch_rtp >= TARGET_RTP:
            above_count += 1
        else:
            below_count += 1
        min_rtp = min(min_rtp, batch_rtp)
        max_rtp = max(max_rtp, batch_rtp)

        print(f"  ✅ 第{i//batch_size + 1}批 RTP = {batch_rtp:.6f}（↑{above_count} ↓{below_count}）")

    results.append((step, above_count, below_count, min_rtp, max_rtp))
    print(f"🎯 踩{step} 統計：高於RTP={above_count} 批，低於RTP={below_count} 批，最小RTP={min_rtp:.6f}，最大RTP={max_rtp:.6f}")

# === 儲存結果 ===
df = pd.DataFrame(results, columns=["踩到第幾格", "高於RTP的批數", "低於RTP的批數", "最小RTP", "最大RTP"])
df.to_excel("每踩4億次_RTP統計.xlsx", index=False)
print("\n✅ 模擬完成，結果已儲存至 Excel")
