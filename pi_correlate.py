import torch
import time
import statistics
import json
from datetime import datetime

device = torch.device("cuda")
MATRIX_SIZE = 2048
N_STEPS = 200
BLOCK_SIZE = 10

print("=== PI Correlation Analysis ===")
print(f"Device: {torch.cuda.get_device_name(0)}")
print()

step_times = []
sync_drifts = []
vram_list = []

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
    vram_list.append(torch.cuda.memory_allocated(0) / 1024**2)
    del A, B, C
    torch.cuda.empty_cache()

blocks = []
print("Block | Steps    | PI    | SyncDrift(ms) | VRAM(MB) | 判定")
print("-" * 70)

for i in range(0, N_STEPS, BLOCK_SIZE):
    block_times = step_times[i:i+BLOCK_SIZE]
    block_sync = sync_drifts[i:i+BLOCK_SIZE]
    block_vram = vram_list[i:i+BLOCK_SIZE]

    sorted_indices = sorted(range(len(block_times)), key=lambda j: block_times[j])
    rank_changes = sum(1 for a, b in zip(sorted_indices, range(len(block_times))) if a != b)
    pi = rank_changes / len(block_times)

    sync_mean = statistics.mean(block_sync) * 1000
    vram_mean = statistics.mean(block_vram)
    block_num = i // BLOCK_SIZE + 1

    if pi <= 0.7:
        status = "<<回復"
    elif pi >= 1.0:
        status = ">>崩壊"
    else:
        status = "  中間"

    print(f"  {block_num:2d}  | {i+1:3d}-{i+BLOCK_SIZE:3d}  | {pi:.2f}  | {sync_mean:12.3f}  | {vram_mean:7.1f}  | {status}")

    blocks.append({
        "block": block_num,
        "pi": pi,
        "sync_drift_mean_ms": sync_mean,
        "vram_mean_mb": vram_mean
    })

print()
pi_values = [b["pi"] for b in blocks]
sync_values = [b["sync_drift_mean_ms"] for b in blocks]

high_pi_sync = [s for p, s in zip(pi_values, sync_values) if p >= 0.9]
low_pi_sync = [s for p, s in zip(pi_values, sync_values) if p <= 0.7]

if high_pi_sync and low_pi_sync:
    print(f"PI高い時のSyncDrift平均: {statistics.mean(high_pi_sync):.3f}ms")
    print(f"PI低い時のSyncDrift平均: {statistics.mean(low_pi_sync):.3f}ms")
    diff = statistics.mean(high_pi_sync) - statistics.mean(low_pi_sync)
    if abs(diff) > 0.1:
        print(f">>> 差あり: {diff:+.3f}ms → SyncDriftがPIと相関している")
    else:
        print(f">>> 差なし → PIはSyncDriftと独立した現象")

output = {
    "timestamp": datetime.now().isoformat(),
    "device": torch.cuda.get_device_name(0),
    "blocks": blocks
}

with open("pi_correlate.json", "w") as f:
    json.dump(output, f, indent=2)

print()
print("結果を pi_correlate.json に保存しました")