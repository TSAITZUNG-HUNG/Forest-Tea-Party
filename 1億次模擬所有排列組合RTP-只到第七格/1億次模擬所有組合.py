import torch
import pandas as pd
import numpy as np
from collections import defaultdict

# === 模擬設定 ===
TARGET_RTP = 0.93
total_simulations = 100_000_000
batch_size = 1_000_000
step = 6  # 踩第幾格（0 = 踩1）

# === 設定 GPU 裝置 ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === 倍率池 ===
multiplier_pool = torch.tensor(
    [1.25] * 9 + [1.5] * 6 + [2] * 4 + [3] * 3 + [5] * 3,
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
    return TARGET_RTP * (1/1.25)**a * (1/1.5)**b * (1/2)**c * (1/3)**d * (1/5)**e

# === 儲存累計結果用 ===
combo_stats = defaultdict(lambda: [0, 0, 0])  # {combo_tuple: [成功次數, RTP加總, 出現次數]}

# === 分批模擬執行 ===
for i in range(0, total_simulations, batch_size):
    sims = min(batch_size, total_simulations - i)
    idx = torch.randint(0, pool_size, (sims, step + 1), device=device)
    shuffled = multiplier_pool[idx]

    probs = compute_probs(shuffled)
    rand_vals = torch.rand(sims, device=device)
    is_win = rand_vals <= probs
    rtp = torch.prod(shuffled * is_win[:, None], dim=1)

    # 統計資訊搬到 CPU 處理
    shuffled_np = shuffled.cpu().numpy()
    is_win_np = is_win.cpu().numpy()
    rtp_np = rtp.cpu().numpy()

    for combo, win, val in zip(shuffled_np, is_win_np, rtp_np):
        combo_key = tuple(combo)
        combo_stats[combo_key][2] += 1  # 出現次數
        if win:
            combo_stats[combo_key][0] += 1  # 成功次數
            combo_stats[combo_key][1] += val  # 成功RTP加總

    print(f"✅ 已完成 {i + sims:,} / {total_simulations:,} 模擬")

# === 結果轉換為 DataFrame ===
result_rows = []
for combo, (succ_count, rtp_sum, appear_count) in combo_stats.items():
    row = list(combo) + [succ_count, rtp_sum, appear_count]
    result_rows.append(row)

columns = [f"踩{i+1}" for i in range(step + 1)] + ["成功次數", "成功RTP加總", "組合出現次數"]
result_df = pd.DataFrame(result_rows, columns=columns)

# === 輸出 Excel ===
output_file = f"踩{step+1}_成功組合統計_含出現次數.xlsx"
result_df.to_excel(output_file, index=False)
print("✅ 模擬完成，已輸出：", output_file)
