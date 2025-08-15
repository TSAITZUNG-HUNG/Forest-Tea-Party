import torch
import pandas as pd
import numpy as np

# === 模擬設定 ===
TARGET_RTP = 0.93
batch_size = 1_000_000
min_step = 17
max_step = 25
simulations = 10_000_000_000

# 每一個踩階段模擬局數
step_simulations = {step: simulations for step in range(min_step, max_step + 1)}

# === 設定 GPU 裝置 ===
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

# === 執行每一踩模擬 ===
step_results = []

for step in range(min_step, max_step + 1):
    total_simulations = step_simulations[step]
    total_rtp = 0.0
    total_success = 0
    sim_count = 0
    print(f"\n🚀 開始模擬 踩{step}，共 {total_simulations:,} 局")

    for i in range(0, total_simulations, batch_size):
        sims = min(batch_size, total_simulations - i)
        sim_count += sims

        # numpy 抽樣後轉 tensor
        sampled_np = np.array([
            np.random.choice(full_pool_np, size=step, replace=False)
            for _ in range(sims)
        ], dtype=np.float32)
        sampled = torch.tensor(sampled_np, device=device)

        # 計算成功機率與 RTP
        probs = compute_probs(sampled)
        rand_vals = torch.rand(sims, device=device)
        is_win = rand_vals <= probs
        rtp = torch.prod(sampled * is_win[:, None], dim=1)

        total_rtp += rtp.sum().item()
        total_success += is_win.sum().item()

        # 每 1,000 萬次列印目前 RTP 狀況
        if sim_count % 10_000_000 == 0 or sim_count == total_simulations:
            current_rtp = total_rtp / sim_count
            print(f"  ✅ 已模擬 {sim_count:,} 局... 成功局 {total_success:,}，目前累積獎金 = {total_rtp:.2f}，平均RTP = {current_rtp:.6f}")

    actual_rtp = total_rtp / total_simulations
    step_results.append((step, actual_rtp))
    print(f"✅ 踩{step} 最終實際RTP = {actual_rtp:.6f}")

# === 輸出結果 ===
df = pd.DataFrame(step_results, columns=["踩到第幾格", "實際RTP"])
df.to_excel("踩17到25_實際RTP_含即時回報.xlsx", index=False)
print("\n🎯 模擬完成，結果已輸出至 Excel")



