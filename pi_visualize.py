import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# データ読み込み
with open("sweep_result.json") as f:
    sweep = json.load(f)

with open("pi_long.json") as f:
    long_run = json.load(f)

with open("pi_correlate.json") as f:
    correlate = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("NAGI-FSD: Permutation Instability Analysis\nGTX 1070 / Discrete VRAM", 
             fontsize=13, fontweight='bold')

# --- グラフ1: サイズスイープ ---
ax1 = axes[0]
sizes = [r["matrix_size"] for r in sweep["results"]]
pis = [r["permutation_instability"] for r in sweep["results"]]
colors = ["red" if p >= 0.9 else "orange" if p >= 0.5 else "green" for p in pis]
bars = ax1.bar([str(s) for s in sizes], pis, color=colors, edgecolor='black', linewidth=0.5)
ax1.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Collapse threshold')
ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Degraded threshold')
ax1.set_title("PI vs Workload Size", fontweight='bold')
ax1.set_xlabel("Matrix Size")
ax1.set_ylabel("Permutation Instability (PI)")
ax1.set_ylim(0, 1.1)
ax1.legend(fontsize=8)

# --- グラフ2: タイムライン ---
ax2 = axes[1]
blocks = long_run["blocks"]
block_nums = [b["block"] for b in blocks]
pi_vals = [b["pi"] for b in blocks]
colors2 = ["red" if p >= 0.9 else "orange" if p >= 0.7 else "green" for p in pi_vals]
ax2.bar(block_nums, pi_vals, color=colors2, edgecolor='black', linewidth=0.5)
ax2.plot(block_nums, pi_vals, 'k-', alpha=0.4, linewidth=1)
ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.7)
ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7)
ax2.set_title("PI Timeline (200 steps)", fontweight='bold')
ax2.set_xlabel("Block (10 steps each)")
ax2.set_ylabel("Permutation Instability (PI)")
ax2.set_ylim(0, 1.1)

# --- グラフ3: PI vs SyncDrift散布図 ---
ax3 = axes[2]
corr_blocks = correlate["blocks"]
pi_c = [b["pi"] for b in corr_blocks]
sync_c = [b["sync_drift_mean_ms"] for b in corr_blocks]
colors3 = ["red" if p >= 0.9 else "orange" if p >= 0.7 else "green" for p in pi_c]
ax3.scatter(sync_c, pi_c, c=colors3, edgecolor='black', linewidth=0.5, s=80, zorder=5)
z = np.polyfit(sync_c, pi_c, 1)
p = np.poly1d(z)
x_line = np.linspace(min(sync_c), max(sync_c), 100)
ax3.plot(x_line, p(x_line), "k--", alpha=0.5, linewidth=1, label='Trend')
ax3.set_title("PI vs Sync Drift", fontweight='bold')
ax3.set_xlabel("Sync Drift (ms)")
ax3.set_ylabel("Permutation Instability (PI)")
ax3.set_ylim(0, 1.1)
ax3.legend(fontsize=8)

# 凡例
red_patch = mpatches.Patch(color='red', label='Collapse (PI≥0.9)')
orange_patch = mpatches.Patch(color='orange', label='Degraded (PI≥0.7)')
green_patch = mpatches.Patch(color='green', label='Stable (PI<0.7)')
fig.legend(handles=[red_patch, orange_patch, green_patch], 
           loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("pi_analysis.png", dpi=150, bbox_inches='tight')
print("pi_analysis.png を保存しました")
plt.show()