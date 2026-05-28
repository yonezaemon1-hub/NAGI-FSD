import torch
import time
import statistics
import json
from datetime import datetime

device = torch.device("cuda")
MATRIX_SIZE = 2048
N_STEPS = 50
BLOCK_SIZE = 10

print("=== PI Timeline Analysis ===")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Matrix: {MATRIX_SIZE}x{MATRIX_SIZE}, {N_STEPS} steps, block size: {BLOCK_SIZE}")
print()

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

blocks = []
for i in range(0, N_STEPS, BLOCK_SIZE):
    block = step_times[i:i+BLOCK_SIZE]
    sorted_indices = sorted(range(len(block)), key=lambda j: block[j])
    rank_changes = sum(1 for a, b in zip(sorted_indices, range(len(block))) if a != b)
    pi = rank_changes / len(block)
    mean_ms = statistics.mean(block) * 1000
    stdev_ms = statistics.stdev(block) * 1000
    block_num = i // BLOCK_SIZE + 1
    print(f"Block {block_num} (step {i+1:2d}-{i+BLOCK_SIZE:2d}) | "
          f"Mean: {mean_ms:.2f}ms  Stdev: {stdev_ms:.3f}ms  PI: {pi:.3f}")
    blocks.append({
        "block": block_num,
        "step_start": i + 1,
        "step_end": i + BLOCK_SIZE,
        "mean_ms": mean_ms,
        "stdev_ms": stdev_ms,
        "pi": pi
    })

print()
pi_values = [b["pi"] for b in blocks]
print(f"PI推移: {' → '.join([f'{p:.2f}' for p in pi_values])}")

if max(pi_values) - min(pi_values) < 0.2:
    print(">>> 判定: PIは全域で安定して崩壊している（構造的現象）")
else:
    print(">>> 判定: PIに変化あり（時間依存の現象）")

output = {
    "timestamp": datetime.now().isoformat(),
    "device": torch.cuda.get_device_name(0),
    "matrix_size": MATRIX_SIZE,
    "blocks": blocks
}

with open("pi_timeline.json", "w") as f:
    json.dump(output, f, indent=2)

print()
print("結果を pi_timeline.json に保存しました")