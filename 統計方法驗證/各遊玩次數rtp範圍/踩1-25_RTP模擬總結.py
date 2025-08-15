import torch
import numpy as np
import pandas as pd
import os

# 模擬設定
TARGET_RTP = 0.95
BET_AMOUNT = 1.0
TRIALS_PER_PLAYER = 10000
PLAYER_COUNTS = [1000]
STEPS = list(range(1, 26))
OUTPUT_DIR = "森林茶會_踩1到25_1000位玩家各玩10000次_RTP範圍數據_95"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 倍率池
FULL_POOL = torch.tensor([1.25]*9 + [1.5]*6 + [2.0]*4 + [3.0]*3 + [5.0]*3, dtype=torch.float32)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# 計算成功機率
def compute_probs(samples):
    a = (samples == 1.25).sum(dim=1)
    b = (samples == 1.5).sum(dim=1)
    c = (samples == 2.0).sum(dim=1)
    d = (samples == 3.0).sum(dim=1)
    e = (samples == 5.0).sum(dim=1)
    return TARGET_RTP * (1/1.25)**a * (1/1.5)**b * (1/2.0)**c * (1/3.0)**d * (1/5.0)**e

# 主模擬迴圈
result_summary = []

for player_count in PLAYER_COUNTS:
    for step in STEPS:
        rtp_list = []

        batch_size = 100
        num_batches = (player_count + batch_size - 1) // batch_size

        for batch_id in range(num_batches):
            current_batch_size = min(batch_size, player_count - batch_id * batch_size)

            total_bet = torch.full((current_batch_size,), TRIALS_PER_PLAYER * BET_AMOUNT, dtype=torch.float32, device=device)
            total_payout = torch.zeros(current_batch_size, dtype=torch.float32, device=device)

            for _ in range(TRIALS_PER_PLAYER):
                idx = torch.stack([torch.randperm(len(FULL_POOL))[:step] for _ in range(current_batch_size)])
                samples = FULL_POOL[idx].to(device)

                win_prob = compute_probs(samples)
                wins = torch.rand(current_batch_size, device=device) < win_prob
                rewards = torch.prod(samples, dim=1) * BET_AMOUNT
                total_payout += rewards * wins

            rtp = (total_payout / total_bet).cpu().numpy()
            rtp_list.extend(rtp.tolist())

        # 儲存當前踩步數結果
        df = pd.DataFrame({
            "玩家": list(range(1, player_count + 1)),
            "RTP": rtp_list
        })
        df.to_excel(f"{OUTPUT_DIR}/RTP模擬_{player_count}人_踩{step:02d}.xlsx", index=False)

        result_summary.append({
            "模擬玩家數": player_count,
            "踩步數": step,
            "平均RTP": np.mean(rtp_list),
            "標準差": np.std(rtp_list),
            "下界":np.mean(rtp_list)-np.std(rtp_list),
            "上界":np.mean(rtp_list)+np.std(rtp_list)
        })

# 儲存總結表
summary_df = pd.DataFrame(result_summary)
summary_df.to_excel(f"{OUTPUT_DIR}/踩1-25_RTP模擬總結.xlsx", index=False)
print("✅ 所有模擬完成，結果已儲存。")
