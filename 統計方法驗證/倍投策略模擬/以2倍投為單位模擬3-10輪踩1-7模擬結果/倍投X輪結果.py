import torch
import numpy as np
import pandas as pd
import os
from collections import defaultdict

# 模擬設定
TARGET_RTP = 0.965
TRIALS = 10_000
STEPS = range(1, 8)
BET_SEQUENCE = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]
INITIAL_BET = 1.0

output_dir = "踩1-7倍投策略模擬_96.5"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
full_pool_np = np.array([1.25]*9 + [1.5]*6 + [2.0]*4 + [3.0]*3 + [5.0]*3, dtype=np.float32)

# 中獎機率計算函數
def compute_probs(sampled_tensor):
    a = (sampled_tensor == 1.25).sum(dim=1)
    b = (sampled_tensor == 1.5).sum(dim=1)
    c = (sampled_tensor == 2.0).sum(dim=1)
    d = (sampled_tensor == 3.0).sum(dim=1)
    e = (sampled_tensor == 5.0).sum(dim=1)
    return TARGET_RTP * (1 / 1.25)**a * (1 / 1.5)**b * (1 / 2.0)**c * (1 / 3.0)**d * (1 / 5.0)**e

results = []
details_all = []

for STEP in STEPS:
    print(f"🚀 模擬踩 {STEP} 步...")

    total_bet = 0.0
    total_win = 0.0
    balance = 0.0
    bet_results = defaultdict(lambda: {"win": 0, "lose": 0})

    for trial in range(TRIALS):
        for bet_amount in BET_SEQUENCE:
            sampled_np = np.random.choice(full_pool_np, size=STEP, replace=False).reshape(1, STEP)
            sampled_tensor = torch.tensor(sampled_np, device=device)
            prob = compute_probs(sampled_tensor)[0]
            is_win = torch.rand(1, device=device)[0] <= prob

            total_bet += bet_amount
            if is_win:
                reward = bet_amount * torch.prod(sampled_tensor).item()
                total_win += reward
                balance += reward - bet_amount
                bet_results[bet_amount]["win"] += 1
                break
            else:
                balance -= bet_amount
                bet_results[bet_amount]["lose"] += 1
                if bet_amount == BET_SEQUENCE[-1]:
                    break

    # 儲存結果
    results.append({
        "踩步數": STEP,
        "總下注額": total_bet,
        "總中獎金額": total_win,
        "最終資金": balance,
        "實際 RTP": total_win / total_bet if total_bet > 0 else 0.0,
    })

    for bet, stat in sorted(bet_results.items()):
        details_all.append({
            "踩步數": STEP,
            "下注金額": bet,
            "贏次數": stat["win"],
            "輸次數": stat["lose"]
        })

# 輸出為 DataFrame
summary_df = pd.DataFrame(results)
details_df = pd.DataFrame(details_all)

# 儲存表格
summary_df.to_excel(f"{output_dir}/倍投統計_踩{STEP:02d}.xlsx", index=False)
details_df.to_excel(f"{output_dir}/資金曲線_踩{STEP:02d}.xlsx", index=False)

