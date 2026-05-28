import torch
import time
import statistics
import json
from datetime import datetime

print("=== FSD Measurement Script (GTX1070 / Discrete VRAM) ===")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print()

device = torch.device("cuda")

N_STEPS = 50
MATRIX_SIZE = 2048

print(f"Running {N_STEPS} steps with {MATRIX_SIZE}x{MATRIX_SIZE} matrices...")
print()

step_times = []
sync_drifts = []
vram_usage = []

for step in range(N_STEPS):
    mem_before = torch.cuda.memory_allocated(0)
    t_start = time.perf_counter()
    A = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
    B = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
    C = torch.matmul(A, B)
    t_before_sync = time.perf_counter()
    torch.cuda.synchronize()
    t_after_sync = time.perf_counter()
    t_end = time.perf_counter()
    step_time = t_end - t_start
    sync_drift = t_after_sync - t_before_sync
    mem_after = torch.cuda.memory_allocated(0)
    step_times.append(step_time)
    sync_drifts.append(sync_drift)
    vram_usage.append(mem_after / 1024**2)
    del A, B, C
    torch.cuda.empty_cache()
    if step % 10 == 0:
        print(f"Step {step:3d} | StepTime: {step_time*1000:.2f}ms | SyncDrift: {sync_drift*1000:.3f}ms | VRAM: {mem_after/1024**2:.1f}MB")

print()
print("=== Results ===")
print(f"Step Time  - mean: {statistics.mean(step_times)*1000:.2f}ms  stdev: {statistics.stdev(step_times)*1000:.3f}ms")
print(f"Sync Drift - mean: {statistics.mean(sync_drifts)*1000:.3f}ms  stdev: {statistics.stdev(sync_drifts)*1000:.4f}ms")
print(f"VRAM Usage - mean: {statistics.mean(vram_usage):.1f}MB  stdev: {statistics.stdev(vram_usage):.2f}MB")

sorted_indices = sorted(range(len(step_times)), key=lambda i: step_times[i])
rank_changes = sum(1 for i, j in zip(sorted_indices, range(N_STEPS)) if i != j)
perm_instability = rank_changes / N_STEPS

print()
print(f"Permutation Instability: {perm_instability:.3f}")
if perm_instability > 0.8:
    print(">>> STATUS: DEGRADED")
elif perm_instability > 0.5:
    print(">>> STATUS: FAILURE APPROACHING")
else:
    print(">>> STATUS: NORMAL")

output = {
    "timestamp": datetime.now().isoformat(),
    "device": torch.cuda.get_device_name(0),
    "step_time_mean_ms": statistics.mean(step_times)*1000,
    "step_time_stdev_ms": statistics.stdev(step_times)*1000,
    "sync_drift_mean_ms": statistics.mean(sync_drifts)*1000,
    "sync_drift_stdev_ms": statistics.stdev(sync_drifts)*1000,
    "vram_mean_mb": statistics.mean(vram_usage),
    "permutation_instability": perm_instability,
}

with open("fsd_result.json", "w") as f:
    json.dump(output, f, indent=2)

print()
print("結果を fsd_result.json に保存しました")