import numpy as np
from scipy.stats import norm

def simulate_forest_tea_sequence_numpy(step=10, num_trials=100_000, seed=42, target_rtp=0.93):
    """
    使用 NumPy 模擬森林茶會遊戲成功/失敗的序列，適用於隨機性檢定。
    
    Args:
        step (int): 玩家每局踩幾步
        num_trials (int): 模擬局數
        seed (int): 隨機種子
        target_rtp (float): 設定 RTP

    Returns:
        List[int]: 成功為 1，失敗為 0 的序列
    """
    np.random.seed(seed)
    full_pool = np.array(
        [1.25] * 9 + [1.5] * 6 + [2.0] * 4 + [3.0] * 3 + [5.0] * 3,
        dtype=np.float32
    )

    sampled = np.array([
        np.random.choice(full_pool, size=step, replace=False)
        for _ in range(num_trials)
    ], dtype=np.float32)

    def compute_probs(batch):
        a = np.sum(batch == 1.25, axis=1)
        b = np.sum(batch == 1.5, axis=1)
        c = np.sum(batch == 2.0, axis=1)
        d = np.sum(batch == 3.0, axis=1)
        e = np.sum(batch == 5.0, axis=1)
        return target_rtp * (1 / 1.25)**a * (1 / 1.5)**b * (1 / 2.0)**c * (1 / 3.0)**d * (1 / 5.0)**e

    probs = compute_probs(sampled)
    rand_vals = np.random.rand(num_trials)
    is_win = (rand_vals <= probs).astype(int)
    return is_win.tolist()

def runs_test(sequence, value_for_success=1):
    seq = np.array(sequence)
    n = len(seq)
    if n < 2:
        raise ValueError("序列太短無法進行檢定")
    n1 = np.sum(seq == value_for_success)
    n2 = n - n1
    if n1 == 0 or n2 == 0:
        raise ValueError("序列中只有一種結果，無法進行 runs test")
    runs = 1 + np.sum(seq[1:] != seq[:-1])
    expected_runs = 2 * n1 * n2 / (n1 + n2) + 1

    # 避免 overflow 的安全計算方式
    n1 = np.float64(n1)
    n2 = np.float64(n2)
    numerator = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)
    denominator = (n1 + n2)**2 * (n1 + n2 - 1)
    std_runs = np.sqrt(numerator / denominator) if denominator != 0 else 0

    z = (runs - expected_runs) / std_runs if std_runs != 0 else 0
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return {
        "total_runs": int(runs),
        "expected_runs": expected_runs,
        "std_dev": std_runs,
        "z_score": z,
        "p_value": p_value,
        "is_random": p_value > 0.05
    }


# === 測試執行 ===
if __name__ == "__main__":
    step = 25
    trials = 10_000_000
    seq = simulate_forest_tea_sequence_numpy(step=step, num_trials=trials)
    result = runs_test(seq)

    print(f"\n🎲 Forest Tea Game Runs Test（踩 {step}, 模擬 {trials:,} 局）")
    print("-----------------------------------------------------")
    print(f"🧮 實際連段數（runs）    : {result['total_runs']}")
    print(f"📈 預期連段數          : {result['expected_runs']:.2f}")
    print(f"📏 標準差              : {result['std_dev']:.2f}")
    print(f"🧪 Z 分數              : {result['z_score']:.2f}")
    print(f"🎯 p 值                : {result['p_value']:.4f}")
    print(f"✅ 判斷是否隨機        : {'✅ 是' if result['is_random'] else '❌ 否'}")
