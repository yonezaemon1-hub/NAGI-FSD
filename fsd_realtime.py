import torch
import time
import statistics
import json
from datetime import datetime

device = torch.device("cuda")
MATRIX_SIZE = 2048
BLOCK_SIZE = 10
N_BLOCKS = 30  # 300ステップ分

print("=== FSD Realtime Monitor ===")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Monitoring {N_BLOCKS} blocks ({N_BLOCKS * BLOCK_SIZE} steps total)")
print()
print("Block | PI    | Status          | Alert")
print("-" * 55)

pi_history = []
alert_log = []

for block_num in range(N_BLOCKS):
    step_times = []

    for step in range(BLOCK_SIZE):
        A = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
        B = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
        t_start = time.perf_counter()
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        step_times.append(t_end - t_start)
        del A, B, C
        torch.cuda.empty_cache()

    sorted_indices = sorted(range(len(step_times)), key=lambda j: step_times[j])
    rank_changes = sum(1 for a, b in zip(sorted_indices, range(len(step_times))) if a != b)
    pi = rank_changes / len(step_times)
    pi_history.append(pi)

    # ステータス判定
    if pi >= 0.9:
        status = "COLLAPSE      "
    elif pi >= 0.7:
        status = "DEGRADED      "
    else:
        status = "STABLE        "

    # アラート判定（3ブロック以上連続で高PI）
    alert = ""
    if len(pi_history) >= 3:
        last3 = pi_history[-3:]
        if all(p >= 0.9 for p in last3):
            alert = "⚠ SUSTAINED COLLAPSE"
            alert_log.append(block_num + 1)
        elif sum(1 for p in last3 if p >= 0.9) >= 2:
            alert = "! REPEATED COLLAPSE"

    # PIトレンド
    if len(pi_history) >= 3:
        trend = pi_history[-1] - pi_history[-3]
        if trend > 0.1:
            trend_str = "↑"
        elif trend < -0.1:
            trend_str = "↓"
        else:
            trend_str = "→"
    else:
        trend_str = " "

    print(f"  {block_num+1:2d}  | {pi:.2f}  | {status} {trend_str} | {alert}")

print()
print("=== Summary ===")
print(f"PI平均: {statistics.mean(pi_history):.3f}")
print(f"PI最大: {max(pi_history):.3f}")
print(f"PI最小: {min(pi_history):.3f}")
print(f"持続崩壊アラート発生: {len(alert_log)}回 (Block: {alert_log})")

collapse_rate = sum(1 for p in pi_history if p >= 0.9) / len(pi_history) * 100
print(f"崩壊率: {collapse_rate:.1f}%")

if collapse_rate >= 70:
    final_status = "CRITICAL - システムは構造的崩壊状態"
elif collapse_rate >= 40:
    final_status = "WARNING - 崩壊が頻発している"
else:
    final_status = "NORMAL - 許容範囲内"

print(f"最終判定: {final_status}")

output = {
    "timestamp": datetime.now().isoformat(),
    "device": torch.cuda.get_device_name(0),
    "pi_history": pi_history,
    "mean_pi": statistics.mean(pi_history),
    "collapse_rate": collapse_rate,
    "sustained_collapse_alerts": len(alert_log),
    "final_status": final_status
}

with open("fsd_realtime.json", "w") as f:
    json.dump(output, f, indent=2)

print()
print("結果を fsd_realtime.json に保存しました")