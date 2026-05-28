import torch
import time
import statistics
import json
from datetime import datetime

device = torch.device("cuda")
MATRIX_SIZE = 2048
N_STEPS = 200
BLOCK_SIZE = 10

print("=== CPU-GPU Timing Analysis ===")
print(f"Device: {torch.cuda.get_device_name(0)}")
print()

step_times = []
cpu_launch_times = []
gpu_wait_times = []

for step in range(N_STEPS):
    A = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
    B = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)

    # CPU側の処理開始
    t_cpu_start = time.perf_counter()

    # カーネル投入（非同期）
    C = torch.matmul(A, B)

    # CPU側の投入完了時刻
    t_cpu_launch = time.perf_counter()

    # GPU完了待ち
    torch.cuda.synchronize()
    t_gpu_done = time.perf_counter()

    cpu_launch = t_cpu_launch - t_cpu_start
    gpu_wait = t_gpu_done - t_cpu_launch
    total = t_gpu_done - t_cpu_start

    step_times.append(total)
    cpu_launch_times.append(cpu_launch)
    gpu_wait_times.append(gpu_wait)

    del A, B, C
    torch.cuda.empty_cache()

# ブロックごとに分析
blocks = []
print("Block | Steps    | PI    | CPU Launch(ms) | GPU Wait(ms) | 比率CPU/GPU")
print("-" * 75)

for i in range(0, N_STEPS, BLOCK_SIZE):
    block_times = step_times[i:i+BLOCK_SIZE]
    block_cpu = cpu_launch_times[i:i+BLOCK_SIZE]
    block_gpu = gpu_wait_times[i:i+BLOCK_SIZE]

    sorted_indices = sorted(range(len(block_times)), key=lambda j: block_times[j])
    rank_changes = sum(1 for a, b in zip(sorted_indices, range(len(block_times))) if a != b)
    pi = rank_changes / len(block_times)

    cpu_mean = statistics.mean(block_cpu) * 1000
    gpu_mean = statistics.mean(block_gpu) * 1000
    ratio = cpu_mean / gpu_mean if gpu_mean > 0 else 0
    block_num = i // BLOCK_SIZE + 1

    print(f"  {block_num:2d}  | {i+1:3d}-{i+BLOCK_SIZE:3d}  | {pi:.2f}  | {cpu_mean:13.3f}  | {gpu_mean:11.3f}  | {ratio:.4f}")

    blocks.append({
        "block": block_num,
        "pi": pi,
        "cpu_launch_mean_ms": cpu_mean,
        "gpu_wait_mean_ms": gpu_mean,
        "ratio": ratio
    })

print()
pi_values = [b["pi"] for b in blocks]
cpu_values = [b["cpu_launch_mean_ms"] for b in blocks]
gpu_values = [b["gpu_wait_mean_ms"] for b in blocks]

high_pi_cpu = [c for p, c in zip(pi_values, cpu_values) if p >= 0.9]
low_pi_cpu = [c for p, c in zip(pi_values, cpu_values) if p <= 0.7]

print(f"PI高い時のCPU Launch平均: {statistics.mean(high_pi_cpu):.4f}ms")
if low_pi_cpu:
    print(f"PI低い時のCPU Launch平均: {statistics.mean(low_pi_cpu):.4f}ms")
    diff = statistics.mean(high_pi_cpu) - statistics.mean(low_pi_cpu)
    if abs(diff) > 0.01:
        print(f">>> 差あり: {diff:+.4f}ms → CPU LaunchタイミングがPIと相関している")
    else:
        print(f">>> 差なし → CPU LaunchはPIと独立")

output = {
    "timestamp": datetime.now().isoformat(),
    "device": torch.cuda.get_device_name(0),
    "blocks": blocks
}

with open("pi_cpu_gpu.json", "w") as f:
    json.dump(output, f, indent=2)

print()
print("結果を pi_cpu_gpu.json に保存しました")