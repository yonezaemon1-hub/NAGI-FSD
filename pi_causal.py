import torch
import time
import statistics
import json
from datetime import datetime

device = torch.device("cuda")
MATRIX_SIZE = 2048
N_STEPS = 200

print("=== PI Causal Analysis ===")
print(f"Device: {torch.cuda.get_device_name(0)}")
print()

step_times = []
sync_drifts = []

for step in range(N_STEPS):
    A = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
    B = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
    t_start = time.perf_counter()
    C = torch.matmul(A, B)
    t_before_sync = time.perf_counter()
    torch.cuda.synchronize()
    t_after_sync = time.perf_counter()
    t_end = time.perf_counter()
    step_times.append(t_end - t_start)
    sync_drifts.append(t_after_sync - t_before_sync)
    del A, B, C
    torch.cuda.empty_cache()

# ブロックごとにPI計算
BLOCK_SIZE = 10
pi_blocks = []
sync_blocks = []

for i in range(0, N_STEPS, BLOCK_SIZE):
    block_times = step_times[i:i+BLOCK_SIZE]
    block_sync = sync_drifts[i:i+BLOCK_SIZE]
    sorted_indices = sorted(range(len(block_times)), key=lambda j: block_times[j])
    rank_changes = sum(1 for a, b in zip(sorted_indices, range(len(block_times))) if a != b)
    pi = rank_changes / len(block_times)
    pi_blocks.append(pi)
    sync_blocks.append(statistics.mean(block_sync) * 1000)

# SyncDriftがPIより先行しているか確認
# 1ブロック前のSyncDriftと現在のPIの相関を見る
print("=== 1ブロック先行分析 ===")
print("前ブロックSyncDrift → 現ブロックPI")
print()
print("前Block SyncDrift(ms) | 現Block PI | 予測")
print("-" * 50)

correct = 0
total = 0

for i in range(1, len(pi_blocks)):
    prev_sync = sync_blocks[i-1]
    curr_pi = pi_blocks[i]
    threshold = statistics.mean(sync_blocks)
    
    prediction = "崩壊予測" if prev_sync > threshold else "安定予測"
    actual = "崩壊" if curr_pi >= 0.9 else "安定"
    hit = "✓" if (prediction == "崩壊予測" and actual == "崩壊") or \
                 (prediction == "安定予測" and actual == "安定") else "✗"
    
    if hit == "✓":
        correct += 1
    total += 1
    
    print(f"Block {i:2d}: {prev_sync:.3f}ms ({prediction}) → PI:{curr_pi:.2f} ({actual}) {hit}")

print()
accuracy = correct / total * 100
print(f"予測精度: {correct}/{total} = {accuracy:.1f}%")

if accuracy >= 60:
    print(">>> SyncDriftはPIの先行指標になっている")
else:
    print(">>> SyncDriftとPIの先行関係は弱い")

output = {
    "timestamp": datetime.now().isoformat(),
    "device": torch.cuda.get_device_name(0),
    "prediction_accuracy": accuracy,
    "pi_blocks": pi_blocks,
    "sync_blocks": sync_blocks
}

with open("pi_causal.json", "w") as f:
    json.dump(output, f, indent=2)

print()
print("結果を pi_causal.json に保存しました")