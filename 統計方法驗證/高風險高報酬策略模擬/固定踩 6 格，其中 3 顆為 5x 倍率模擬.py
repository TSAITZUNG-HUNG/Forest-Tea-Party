import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 支援中文顯示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 模擬參數
num_simulations = 100000
bet_amount = 1.0
target_rtp = 0.93

# 固定踩 6 格，其中 3 顆為 5x 倍率
fixed_high_multipliers = np.array([5.0, 5.0, 5.0])
low_multipliers_pool = [1.25, 1.5, 2.0, 3.0]

# 隨機種子
np.random.seed(42)

# 儲存模擬結果
results = []
details = []

# 計算中獎機率的函數
def compute_win_prob(multipliers):
    a = np.sum(multipliers == 1.25)
    b = np.sum(multipliers == 1.5)
    c = np.sum(multipliers == 2.0)
    d = np.sum(multipliers == 3.0)
    e = np.sum(multipliers == 5.0)
    return target_rtp * (1 / 1.25)**a * (1 / 1.5)**b * (1 / 2.0)**c * (1 / 3.0)**d * (1 / 5.0)**e

# 進行模擬
for sim in range(1, num_simulations + 1):
    total_cost = 0.0
    rounds = 0
    while True:
        sampled_low = np.random.choice(low_multipliers_pool, size=3, replace=True)
        multipliers = np.concatenate([fixed_high_multipliers, sampled_low])
        np.random.shuffle(multipliers)

        win_prob = compute_win_prob(multipliers)
        win = np.random.rand() < win_prob

        rounds += 1
        total_cost += bet_amount

        if win:
            total_payout = np.prod(multipliers) * bet_amount
            rtp = total_payout / total_cost  # ➜ RTP
            results.append(rtp)
            details.append({
                "模擬編號": sim,
                "中獎前局數": rounds,
                "總下注": total_cost,
                "總賠付": total_payout,
                "總損益": total_payout - total_cost,
                "RTP": rtp
            })
            break

# 統計資料
results_np = np.array(results)
avg_rtp = np.mean(results_np)
std_rtp = np.std(results_np)
above_1 = np.mean(results_np > 1.0)
below_1 = np.mean(results_np < 1.0)

# 輸出表格
summary_df = pd.DataFrame({
    "平均 RTP": [avg_rtp],
    "RTP 標準差": [std_rtp],
    "RTP > 1.0 比例": [above_1],
    "RTP < 1.0 比例": [below_1]
})
details_df = pd.DataFrame(details)

# 儲存資料
output_dir = "踩6_三個5x_直到中獎模擬_RTP版"
os.makedirs(output_dir, exist_ok=True)
summary_df.to_excel(f"{output_dir}/統計總表.xlsx", index=False)
details_df.to_excel(f"{output_dir}/模擬明細.xlsx", index=False)

# 畫圖：以 RTP 為 y 軸，紅線標記 Y=1
plt.figure(figsize=(10, 5))
plt.scatter(range(1, len(results_np) + 1), results_np, color='green', alpha=0.6)
plt.axhline(y=1, color='red', linestyle='--', label="Y = 1")
plt.title("踩 6（含三個5x）直到中獎的 RTP 散布圖")
plt.xlabel("模擬編號")
plt.ylabel("RTP（總賠付 ÷ 總下注）")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/RTP散布圖.png", dpi=150)
plt.show()
