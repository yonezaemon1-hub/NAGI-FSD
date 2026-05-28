import torch
import time
import statistics
import json
from datetime import datetime

device = torch.device("cuda")
MATRIX_SIZE = 2048
N_STEPS = 200
BLOCK_SIZE = 10

print("=== CPU Launch → PI Prediction Analysis ===")
print(f"Device: {torch.cuda.get_device_name(0)}")
print()

step_times = []
cpu_launch_times = []

for step in range(N_STEPS):
    A = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
    B = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
    t_cpu_start = time.perf_counter()
    C = torch.matmul(A, B)
    t_cpu_launch = time.perf_counter()
    torch.cuda.synchronize()
    t_gpu_done = time.perf_counter()
    step_times.append(t_gpu_done - t_cpu_start)
    cpu_launch_times.append(t_cpu_launch - t_cpu_start)
    del A, B, C
    torch.cuda.empty_cache()

# ブロックごとにPI計算
pi_blocks = []
cpu_blocks = []

for i in range(0, N_STEPS, BLOCK_SIZE):
    block_times = step_times[i:i+BLOCK_SIZE]
    block_cpu = cpu_launch_times[i:i+BLOCK_SIZE]
    sorted_indices = sorted(range(len(block_times)), key=lambda j: block_times[j])
    rank_changes = sum(1 for a, b in zip(sorted_indices, range(len(block_times))) if a != b)
    pi = rank_changes / len(block_times)
    pi_blocks.append(pi)
    cpu_blocks.append(statistics.mean(block_cpu) * 1000)

# Block1を除外（ウォームアップ）
pi_blocks = pi_blocks[1:]
cpu_blocks = cpu_blocks[1:]

threshold = statistics.mean(cpu_blocks)
print(f"CPU Launch閾値: {threshold:.4f}ms")
print()
print("前Block CPU Launch → 現Block PI 予測")
print("-" * 55)

correct = 0
total = 0

for i in range(1, len(pi_blocks)):
    prev_cpu = cpu_blocks[i-1]
    curr_pi = pi_blocks[i]

    prediction = "崩壊予測" if prev_cpu > threshold else "安定予測"
    actual = "崩壊" if curr_pi >= 0.9 else "安定"
    hit = "✓" if (prediction == "崩壊予測" and actual == "崩壊") or \
                 (prediction == "安定予測" and actual == "安定") else "✗"

    if hit == "✓":
        correct += 1
    total += 1

    print(f"Block {i+1:2d}: CPU={prev_cpu:.4f}ms ({prediction}) → PI:{curr_pi:.2f} ({actual}) {hit}")

print()
accuracy = correct / total * 100
print(f"予測精度: {correct}/{total} = {accuracy:.1f}%")

if accuracy >= 65:
    print(">>> CPU LaunchはPIの先行指標になっている　←重要")
else:
    print(">>> CPU LaunchとPIの先行関係は弱い")

output = {
    "timestamp": datetime.now().isoformat(),
    "device": torch.cuda.get_device_name(0),
    "threshold_ms": threshold,
    "prediction_accuracy": accuracy,
    "pi_blocks": pi_blocks,
    "cpu_blocks": cpu_blocks
}

with open("pi_cpu_predict.json", "w") as f:
    json.dump(output, f, indent=2)

print()
print("結果を pi_cpu_predict.json に保存しました")