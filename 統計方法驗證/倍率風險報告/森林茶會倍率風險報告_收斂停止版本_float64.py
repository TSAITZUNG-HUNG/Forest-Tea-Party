import torch
import pandas as pd
import numpy as np
import os

# === 模擬設定 ===
TARGET_RTP = 0.95
BATCH_SIZE = 1_000_000
MAX_SIMULATIONS = 100_000_000_000  # ✅ 最大模擬局數限制
MIN_STEP = 23
MAX_STEP = 23
STOP_THRESHOLD = 0.005
STOP_COUNT = 10
OUTPUT_DIR = "森林茶會_爆炸倍率模擬_0.95"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 想要雙精度
USE_FLOAT64 = True

# === 裝置與精度設定 ===
if USE_FLOAT64:
    # MPS 不支援 float64，強制使用 CPU
    device = torch.device("cpu")
    torch.set_default_dtype(torch.float64)
    dtype = torch.float64
    print("⚠️ 使用 float64，已自動切換到 CPU（MPS 不支援 float64）")
else:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    dtype = torch.float32
    print(f"使用 {device} 與 {dtype}")

# 倍率池（改成 float64）
full_pool_np = np.array([1.25]*9 + [1.5]*6 + [2.0]*4 + [3.0]*3 + [5.0]*3,
                        dtype=np.float64 if USE_FLOAT64 else np.float32)

def compute_probs(multipliers: torch.Tensor):
    a = (multipliers == 1.25).sum(dim=1)
    b = (multipliers == 1.5).sum(dim=1)
    c = (multipliers == 2.0).sum(dim=1)
    d = (multipliers == 3.0).sum(dim=1)
    e = (multipliers == 5.0).sum(dim=1)
    return (TARGET_RTP
            * (1/1.25)**a
            * (1/1.5)**b
            * (1/2.0)**c
            * (1/3.0)**d
            * (1/5.0)**e)

for step in range(MIN_STEP, MAX_STEP + 1):
    print(f"\n🚀 開始模擬 踩{step}（{dtype} on {device.type}）...")

    total_simulations = 0
    total_success = 0
    total_reward = 0.0
    success_multipliers = []

    stable_count = 0
    iteration = 0

    while True:
        iteration += 1
        sims = BATCH_SIZE
        if total_simulations + sims > MAX_SIMULATIONS:
            sims = MAX_SIMULATIONS - total_simulations
        total_simulations += sims

        # 產生樣本（使用 numpy，dtype 依設定）
        sampled_np = np.array([
            np.random.choice(full_pool_np, size=step, replace=False)
            for _ in range(sims)
        ], dtype=full_pool_np.dtype)

        # 放到裝置 & 設定 dtype
        sampled_tensor = torch.tensor(sampled_np, device=device, dtype=dtype)

        probs = compute_probs(sampled_tensor)
        rand_vals = torch.rand(sims, device=device, dtype=dtype)
        is_win = rand_vals <= probs

        multipliers = torch.prod(sampled_tensor * is_win[:, None], dim=1)
        win_values = multipliers[multipliers > 0].cpu().numpy()

        total_success += len(win_values)
        total_reward += win_values.sum()
        success_multipliers.extend(win_values)

        current_rtp = total_reward / total_simulations
        print(f"  第 {iteration} 批 ➜ 模擬 {total_simulations:,} 局，RTP = {current_rtp:.6f}")

        if abs(current_rtp - TARGET_RTP) <= STOP_THRESHOLD:
            stable_count += 1
            print(f"    ✅ 第 {stable_count}/{STOP_COUNT} 次符合停止條件")
        else:
            stable_count = 0

        # 停止條件判斷
        if stable_count >= STOP_COUNT:
            print("🎯 已連續達標，提前停止模擬")
            break
        if total_simulations >= MAX_SIMULATIONS:
            print("⚠️ 已達最大模擬局數，強制結束")
            break

    # === 統計成功局倍數資料（依 dtype 設定）===
    rewards_np = np.array(success_multipliers, dtype=full_pool_np.dtype)
    mean_multiplier = rewards_np.mean() if len(rewards_np) > 0 else 0.0
    std_multiplier = rewards_np.std() if len(rewards_np) > 0 else 0.0
    percentiles = (np.percentile(rewards_np, q=range(1, 100))
                   if len(rewards_np) > 0 else np.zeros(99, dtype=full_pool_np.dtype))

    # === 組成統計報表資料 ===
    step_row = {
        "踩步數": step,
        "模擬局數": total_simulations,
        "成功局數": total_success,
        "中獎率": total_success / total_simulations,
        "實際 RTP": current_rtp,
        "成功局平均倍數": float(mean_multiplier),
        "成功局標準差": float(std_multiplier),
        **{f"{p}% 倍數": float(val) for p, val in zip(range(1, 100), percentiles)}
    }

    df_step = pd.DataFrame([step_row])
    file_name = f"{OUTPUT_DIR}/踩{step}_爆炸倍率風險統計_含百分位_{device.type}_{dtype}.xlsx"
    df_step.to_excel(file_name, index=False)
    print(f"📄 已儲存：{file_name}")

    # 清除記憶體
    del success_multipliers, sampled_tensor, sampled_np, rewards_np, multipliers, probs, rand_vals, is_win
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
