import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 設定 matplotlib 字體以支援中文
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# === 固定倍率組合 ===
fixed_multipliers = np.array([5, 1.5, 1.25, 1.25, 5, 5, 3], dtype=np.float32)

# === 參數設定 ===
TARGET_RTP = 0.93
num_players = 100
games_per_player = 10000

# === 成功機率計算 ===
def compute_success_prob(mults):
    a = np.sum(mults == 1.25)
    b = np.sum(mults == 1.5)
    c = np.sum(mults == 2.0)
    d = np.sum(mults == 3.0)
    e = np.sum(mults == 5.0)
    prob = TARGET_RTP * (1 / 1.25)**a * (1 / 1.5)**b * (1 / 2.0)**c * (1 / 3.0)**d * (1 / 5.0)**e
    return prob

# === 模擬玩家輸贏趨勢 ===
def simulate_profit_trends(fixed_multipliers, num_players, games_per_player):
    win_prob = compute_success_prob(fixed_multipliers)
    win_amount = np.prod(fixed_multipliers)
    cost_per_game = 1.0

    all_trends = []
    for _ in range(num_players):
        rand_vals = np.random.rand(games_per_player)
        wins = rand_vals <= win_prob
        profits = np.where(wins, win_amount - cost_per_game, -cost_per_game)
        cum_profit = np.cumsum(profits)
        all_trends.append(cum_profit)

    return all_trends

# === 執行模擬與繪圖 ===
profit_trends = simulate_profit_trends(fixed_multipliers, num_players, games_per_player)

plt.figure(figsize=(12, 6))
for trend in profit_trends:
    plt.plot(trend, alpha=0.4)
plt.title("玩家輸贏趨勢圖（100位玩家，每人10000局）")
plt.xlabel("局數")
plt.ylabel("累積輸贏")
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.grid(True)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.tight_layout()
plt.savefig("玩家輸贏趨勢圖_固定倍率組合.png")
plt.show()
