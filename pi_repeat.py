import torch
import time
import statistics
import json
from datetime import datetime

device = torch.device("cuda")
MATRIX_SIZE = 2048
N_STEPS = 200
BLOCK_SIZE = 10
N_TRIALS = 5

print("=== Repeated Trial Analysis ===")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Trials: {N_TRIALS}")
print()

all_accuracies = []

for trial in range(N_TRIALS):
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

    # Block1除外
    pi_blocks = pi_blocks[1:]
    cpu_blocks = cpu_blocks[1:]

    threshold = statistics.mean(cpu_blocks)
    correct = 0
    total = 0

    for i in range(1, len(pi_blocks)):
        prev_cpu = cpu_blocks[i-1]
        curr_pi = pi_blocks[i]
        prediction = "崩壊" if prev_cpu > threshold else "安定"
        actual = "崩壊" if curr_pi >= 0.9 else "安定"
        if prediction == actual:
            correct += 1
        total += 1

    accuracy = correct / total * 100
    all_accuracies.append(accuracy)
    print(f"Trial {trial+1}: 予測精度 {accuracy:.1f}%")

print()
print(f"平均精度: {statistics.mean(all_accuracies):.1f}%")
print(f"最大精度: {max(all_accuracies):.1f}%")
print(f"最小精度: {min(all_accuracies):.1f}%")
print(f"標準偏差: {statistics.stdev(all_accuracies):.1f}%")

mean_acc = statistics.mean(all_accuracies)
if mean_acc >= 70:
    print(">>> CPU Launchは安定した先行指標として機能している")
elif mean_acc >= 60:
    print(">>> CPU Launchは弱い先行指標として機能している")
else:
    print(">>> CPU Launchの先行指標としての安定性は低い")

output = {
    "timestamp": datetime.now().isoformat(),
    "device": torch.cuda.get_device_name(0),
    "n_trials": N_TRIALS,
    "accuracies": all_accuracies,
    "mean_accuracy": statistics.mean(all_accuracies),
    "max_accuracy": max(all_accuracies),
    "min_accuracy": min(all_accuracies),
    "stdev_accuracy": statistics.stdev(all_accuracies)
}

with open("pi_repeat.json", "w") as f:
    json.dump(output, f, indent=2)

print()
print("結果を pi_repeat.json に保存しました")