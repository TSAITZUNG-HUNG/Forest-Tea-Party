import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 設定 matplotlib 字體以支援中文
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# === 通用模擬設定 ===
TARGET_RTP = 0.95
TRIALS = 100_000
NUM_PLAYERS = 10
INITIAL_BET = 1.0
MAX_LOSS_STREAK = 10
STOP_PROFIT = 1000
STOP_LOSS = -1000

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 倍率池
full_pool_np = np.array([1.25]*9 + [1.5]*6 + [2.0]*4 + [3.0]*3 + [5.0]*3, dtype=np.float32)

# 儲存圖表資料夾
output_dir = "每踩倍投損益圖_贏輸1000倍就跑_0.95"
os.makedirs(output_dir, exist_ok=True)

# 計算中獎機率
def compute_probs(sampled_tensor):
    a = (sampled_tensor == 1.25).sum(dim=1)
    b = (sampled_tensor == 1.5).sum(dim=1)
    c = (sampled_tensor == 2.0).sum(dim=1)
    d = (sampled_tensor == 3.0).sum(dim=1)
    e = (sampled_tensor == 5.0).sum(dim=1)
    return TARGET_RTP * (1 / 1.25)**a * (1 / 1.5)**b * (1 / 2.0)**c * (1 / 3.0)**d * (1 / 5.0)**e

# 主模擬迴圈：踩 1~25
for STEP in range(1, 26):
    print(f"🚀 開始模擬踩 {STEP}...")

    all_player_records = []
    summary_stats = []
    bust_points = []

    for pid in range(NUM_PLAYERS):
        np.random.seed(42 + pid)
        bet_amount = INITIAL_BET
        loss_streak = 0
        total_bet = 0.0
        total_win = 0.0
        max_bet_reached = 0.0
        balance = 0.0
        balances = []
        loss_streaks = []
        bust_count = 0
        bust_frame = None
        recover_frame = None
        reached_profit = False
        reached_loss = False

        for trial in range(1, TRIALS + 1):
            sampled_np = np.random.choice(full_pool_np, size=STEP, replace=False).reshape(1, STEP)
            sampled_tensor = torch.tensor(sampled_np, device=device)
            prob = compute_probs(sampled_tensor)[0]
            is_win = torch.rand(1, device=device)[0] <= prob

            if is_win:
                reward = bet_amount * torch.prod(sampled_tensor).item()
                total_win += reward
                balance += reward - bet_amount
                bet_amount = INITIAL_BET
                loss_streak = 0
            else:
                reward = 0.0
                balance -= bet_amount
                loss_streak += 1
                if loss_streak >= MAX_LOSS_STREAK:
                    bust_count += 1
                    if bust_frame is None:
                        bust_frame = trial
                    bet_amount = INITIAL_BET
                    loss_streak = 0
                else:
                    bet_amount *= 2

            total_bet += bet_amount
            max_bet_reached = max(max_bet_reached, bet_amount)
            balances.append(balance)
            loss_streaks.append(loss_streak)

            if not reached_profit and balance >= STOP_PROFIT:
                reached_profit = True
                recover_frame = trial

            if not reached_loss and balance <= STOP_LOSS:
                reached_loss = True

            if reached_profit or reached_loss:
                break

        player_data = pd.DataFrame({
            "局數": list(range(1, len(balances) + 1)),
            "資金變化": balances,
            "連輸次數": loss_streaks,
            "玩家": f"玩家 {pid+1}"
        })
        all_player_records.append(player_data)

        if bust_frame is not None:
            bust_points.append({
                "玩家": f"玩家 {pid+1}",
                "局數": bust_frame,
                "資金": balances[bust_frame - 1]
            })

        summary_stats.append({
            "玩家": f"玩家 {pid+1}",
            "模擬局數": len(balances),
            "總投注": total_bet,
            "總回收": total_win,
            "最終資金": balances[-1],
            "實際 RTP": total_win / total_bet if total_bet > 0 else 0,
            "最大下注金額": max_bet_reached,
            "爆倉次數": bust_count,
            "首次爆倉局數": bust_frame,
            "翻本局數": recover_frame
        })

    summary_df = pd.DataFrame(summary_stats)
    combined_df = pd.concat(all_player_records, ignore_index=True)
    bust_df = pd.DataFrame(bust_points)

    # 儲存表格
    summary_df.to_excel(f"{output_dir}/倍投統計_踩{STEP:02d}.xlsx", index=False)
    combined_df.to_excel(f"{output_dir}/資金曲線_踩{STEP:02d}.xlsx", index=False)

    # 繪製圖
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=combined_df, x="局數", y="資金變化", hue="玩家", linewidth=0.8, legend=False)
    if not bust_df.empty:
        plt.scatter(bust_df["局數"], bust_df["資金"], color="red", label="爆倉點", s=30)
    plt.title(f"森林茶會倍投策略 - 踩 {STEP} 資金曲線（含爆倉點）")
    plt.xlabel("局數")
    plt.ylabel("累積資金")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/資金曲線_踩{STEP:02d}.png", dpi=150)
    plt.close()
