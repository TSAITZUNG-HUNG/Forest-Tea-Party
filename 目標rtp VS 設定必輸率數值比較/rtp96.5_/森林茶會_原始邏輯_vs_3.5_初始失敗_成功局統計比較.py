import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 設定 matplotlib 字體支援中文
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# === 模擬設定 ===
TARGET_RTP = 0.965
MAX_MULTIPLIER = 1_000_000
BATCH_SIZE = 1000
REPEATS = 20
MIN_STEP = 1
MAX_STEP = 25
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 倍率池（25 顆）
full_pool_np = np.array([1.25]*9 + [1.5]*6 + [2.0]*4 + [3.0]*3 + [5.0]*3, dtype=np.float32)

# 過關機率計算公式
def compute_probs(sampled_tensor):
    a = (sampled_tensor == 1.25).sum(dim=1)
    b = (sampled_tensor == 1.5).sum(dim=1)
    c = (sampled_tensor == 2.0).sum(dim=1)
    d = (sampled_tensor == 3.0).sum(dim=1)
    e = (sampled_tensor == 5.0).sum(dim=1)
    return TARGET_RTP * (1/1.25)**a * (1/1.5)**b * (1/2.0)**c * (1/3.0)**d * (1/5.0)**e

# 模擬主函數
def simulate_with_logic(logic_name, early_fail_rate=0.0):
    results = []
    for step in range(MIN_STEP, MAX_STEP + 1):
        all_success = 0
        all_reward = 0.0
        mean_list, std_list = [], []

        for _ in range(REPEATS):
            sampled_np = np.array([
                np.random.choice(full_pool_np, size=step, replace=False)
                for _ in range(BATCH_SIZE)
            ], dtype=np.float32)
            sampled = torch.tensor(sampled_np, device=device)
            probs = compute_probs(sampled)

            if early_fail_rate > 0:
                gate = torch.rand(BATCH_SIZE, device=device)
                probs *= (gate > early_fail_rate)

            rand_vals = torch.rand(BATCH_SIZE, device=device)
            is_win = rand_vals <= probs

            multipliers = torch.prod(sampled * is_win[:, None], dim=1)
            multipliers = torch.clamp(multipliers, max=MAX_MULTIPLIER)

            rewards = multipliers[multipliers > 0].cpu().numpy()
            mean_list.append(rewards.mean() if len(rewards) > 0 else 0.0)
            std_list.append(rewards.std() if len(rewards) > 0 else 0.0)

            all_success += is_win.sum().item()
            all_reward += multipliers.sum().item()

        avg_rtp = all_reward / (BATCH_SIZE * REPEATS)
        results.append({
            "踩步數": step,
            "平均倍數": np.mean(mean_list),
            "標準差": np.mean(std_list),
            "實際RTP": avg_rtp,
            "邏輯": logic_name
        })
    return results

# 執行模擬
results_orig = simulate_with_logic("原始邏輯", early_fail_rate=0.0)
results_7 = simulate_with_logic("3.5%初始失敗", early_fail_rate=1 - TARGET_RTP)

# 整合結果與輸出 Excel
df_all = pd.DataFrame(results_orig + results_7)
excel_path = "森林茶會_原始邏輯_vs_3.5%初始失敗_成功局統計比較.xlsx"
df_all.to_excel(excel_path, index=False)
print(f"📄 已儲存：{excel_path}")

# 雙軸圖：RTP（左軸）與標準差（右軸）
fig, ax1 = plt.subplots(figsize=(14, 6))
colors = {'原始邏輯': 'tab:blue', '3.5%初始失敗': 'tab:green'}

# 左軸：RTP
for logic in df_all["邏輯"].unique():
    df_sub = df_all[df_all["邏輯"] == logic]
    ax1.plot(df_sub["踩步數"], df_sub["實際RTP"], marker='o', color=colors[logic], label=f"{logic} - RTP")
ax1.axhline(TARGET_RTP, color='red', linestyle='--', label=f"理論 RTP = {TARGET_RTP}")
ax1.set_xlabel("踩步數")
ax1.set_ylabel("實際 RTP", color='black')
ax1.tick_params(axis='y', labelcolor='black')

# 右軸：標準差
ax2 = ax1.twinx()
for logic in df_all["邏輯"].unique():
    df_sub = df_all[df_all["邏輯"] == logic]
    ax2.plot(df_sub["踩步數"], df_sub["標準差"], marker='s', linestyle='--', color=colors[logic], label=f"{logic} - 標準差")
ax2.set_ylabel("成功局倍數標準差", color='black')
ax2.tick_params(axis='y', labelcolor='black')

# 圖例整合
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

plt.title("每踩步數的實際 RTP 與成功局倍數標準差（雙軸比較）")
plt.grid(True)
plt.tight_layout()
plt.show()
