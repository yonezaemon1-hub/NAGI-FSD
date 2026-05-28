import torch
import time
import statistics
import json
from datetime import datetime

device = torch.device("cuda")
MATRIX_SIZE = 2048
N_STEPS = 200
BLOCK_SIZE = 10

print("=== Combined Predictor Analysis ===")
print(f"Device: {torch.cuda.get_device_name(0)}")
print()

step_times = []
cpu_launch_times = []
sync_drift_times = []

for step in range(N_STEPS):
    A = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
    B = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
    t_cpu_start = time.perf_counter()
    C = torch.matmul(A, B)
    t_cpu_launch = time.perf_counter()
    torch.cuda.synchronize()
    t_sync_done = time.perf_counter()
    step_times.append(t_sync_done - t_cpu_start)
    cpu_launch_times.append(t_cpu_launch - t_cpu_start)
    sync_drift_times.append(t_sync_done - t_cpu_launch)
    del A, B, C
    torch.cuda.empty_cache()

pi_blocks = []
cpu_blocks = []
sync_blocks = []

for i in range(0, N_STEPS, BLOCK_SIZE):
    block_times = step_times[i:i+BLOCK_SIZE]
    block_cpu = cpu_launch_times[i:i+BLOCK_SIZE]
    block_sync = sync_drift_times[i:i+BLOCK_SIZE]
    sorted_indices = sorted(range(len(block_times)), key=lambda j: block_times[j])
    rank_changes = sum(1 for a, b in zip(sorted_indices, range(len(block_times))) if a != b)
    pi = rank_changes / len(block_times)
    pi_blocks.append(pi)
    cpu_blocks.append(statistics.mean(block_cpu) * 1000)
    sync_blocks.append(statistics.mean(block_sync) * 1000)

# Block1除外（ウォームアップ）
pi_blocks = pi_blocks[1:]
cpu_blocks = cpu_blocks[1:]
sync_blocks = sync_blocks[1:]

cpu_threshold = statistics.mean(cpu_blocks)
sync_threshold = statistics.mean(sync_blocks)

print(f"CPU Launch閾値: {cpu_threshold:.4f}ms")
print(f"SyncDrift閾値: {sync_threshold:.4f}ms")
print()

# 3つの予測モデルを比較
results = {"cpu_only": 0, "sync_only": 0, "combined": 0, "total": 0}

print("Block | PI | CPU予測 | Sync予測 | 複合予測 | 実際")
print("-" * 60)

for i in range(1, len(pi_blocks)):
    prev_cpu = cpu_blocks[i-1]
    prev_sync = sync_blocks[i-1]
    curr_pi = pi_blocks[i]
    actual = "崩壊" if curr_pi >= 0.9 else "安定"

    cpu_pred = "崩壊" if prev_cpu > cpu_threshold else "安定"
    sync_pred = "崩壊" if prev_sync > sync_threshold else "安定"
    
    # 複合予測：どちらか一方でも崩壊予測なら崩壊
    combined_pred = "崩壊" if (prev_cpu > cpu_threshold or prev_sync > sync_threshold) else "安定"

    cpu_hit = "✓" if cpu_pred == actual else "✗"
    sync_hit = "✓" if sync_pred == actual else "✗"
    combined_hit = "✓" if combined_pred == actual else "✗"

    if cpu_pred == actual:
        results["cpu_only"] += 1
    if sync_pred == actual:
        results["sync_only"] += 1
    if combined_pred == actual:
        results["combined"] += 1
    results["total"] += 1

    print(f"  {i+1:2d}  | {curr_pi:.2f} | {cpu_pred}({cpu_hit}) | {sync_pred}({sync_hit}) | {combined_pred}({combined_hit}) | {actual}")

print()
total = results["total"]
print(f"CPU単独精度:　　{results['cpu_only']}/{total} = {results['cpu_only']/total*100:.1f}%")
print(f"SyncDrift単独:　{results['sync_only']}/{total} = {results['sync_only']/total*100:.1f}%")
print(f"複合予測精度:　　{results['combined']}/{total} = {results['combined']/total*100:.1f}%")

best = max(results['cpu_only'], results['sync_only'], results['combined'])
if results['combined'] == best:
    print(">>> 複合予測が最も精度が高い")
elif results['cpu_only'] == best:
    print(">>> CPU単独が最も精度が高い")
else:
    print(">>> SyncDrift単独が最も精度が高い")

output = {
    "timestamp": datetime.now().isoformat(),
    "device": torch.cuda.get_device_name(0),
    "cpu_only_accuracy": results['cpu_only']/total*100,
    "sync_only_accuracy": results['sync_only']/total*100,
    "combined_accuracy": results['combined']/total*100,
}

with open("pi_combined.json", "w") as f:
    json.dump(output, f, indent=2)

print()
print("結果を pi_combined.json に保存しました")