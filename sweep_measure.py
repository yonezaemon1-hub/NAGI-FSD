import torch
import time
import statistics
import json
from datetime import datetime

device = torch.device("cuda")
SIZES = [512, 1024, 2048, 4096]
N_STEPS = 50

print("=== Workload Size Sweep ===")
print(f"Device: {torch.cuda.get_device_name(0)}")
print()

all_results = []

for MATRIX_SIZE in SIZES:
    print(f"--- Matrix Size: {MATRIX_SIZE}x{MATRIX_SIZE} ---")
    step_times = []

    for step in range(N_STEPS):
        A = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
        B = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
        t_start = time.perf_counter()
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        step_times.append(t_end - t_start)
        del A, B, C
        torch.cuda.empty_cache()

    sorted_indices = sorted(range(len(step_times)), key=lambda i: step_times[i])
    rank_changes = sum(1 for i, j in zip(sorted_indices, range(N_STEPS)) if i != j)
    pi = rank_changes / N_STEPS

    mean_ms = statistics.mean(step_times) * 1000
    stdev_ms = statistics.stdev(step_times) * 1000

    print(f"Mean: {mean_ms:.2f}ms  Stdev: {stdev_ms:.3f}ms  PI: {pi:.3f}")
    if pi > 0.8:
        print(f">>> STATUS: DEGRADED")
    elif pi > 0.5:
        print(f">>> STATUS: FAILURE APPROACHING")
    else:
        print(f">>> STATUS: NORMAL")
    print()

    all_results.append({
        "matrix_size": MATRIX_SIZE,
        "mean_ms": mean_ms,
        "stdev_ms": stdev_ms,
        "permutation_instability": pi
    })

output = {
    "timestamp": datetime.now().isoformat(),
    "device": torch.cuda.get_device_name(0),
    "results": all_results
}

with open("sweep_result.json", "w") as f:
    json.dump(output, f, indent=2)

print("結果を sweep_result.json に保存しました")