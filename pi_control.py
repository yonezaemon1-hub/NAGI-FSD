"""
NAGI 4点結界 - 対照実験 (検証スクリプト)
==========================================
目的: V(t)のスパイクが「現象」か「アーティファクト」かを切り分ける。
現象を否定しにいく。否定できなければ本物に近づく。

検証A: 生のstep_timeをそのまま見る（V(t)を介さない）
検証B: 入力を変えて V(t) を比較
        B1: 元コードと同じ (step_time, sync_drift, cpu_launch) ← 3つの無関係な量
        B2: step_timeだけの遅延埋め込み (ジェシーへの投稿で説明した本来の手法)
検証C: time系列をシャッフルして V(t) を再計算（順序の構造が本質かを試す）
検証D: 複数seed/試行でスパイク位置が動くか（固定=決定的, ばらつく=ノイズ）

GTX1070 / PyTorch 2.7.1+cu118 を想定
"""
import numpy as np
import torch
import time
import json
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=== NAGI 対照実験 (現象 vs アーティファクト) ===\n")

MATRIX_SIZE = 2048
N_STEPS = 200
N_TRIALS = 5  # 検証D用

# ---------- 計測 (元コードと同じ計測方法を踏襲) ----------
def collect(seed):
    torch.manual_seed(seed)
    step_times, sync_drifts, cpu_launches = [], [], []
    for i in range(N_STEPS):
        cpu_start = time.perf_counter()
        A = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
        B = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
        cpu_end = time.perf_counter()
        cpu_launches.append((cpu_end - cpu_start) * 1000)

        t0 = time.perf_counter()
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_times.append((t1 - t0) * 1000)

        t2 = time.perf_counter()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        sync_drifts.append((t3 - t2) * 1000)
    return (np.array(step_times), np.array(sync_drifts), np.array(cpu_launches))

# ---------- 四面体体積 ----------
def tetra_volume_3col(data):
    """元コード方式: 3つの無関係な量を xyz にした空間の体積"""
    def vol(p0, p1, p2, p3):
        return abs(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0
    return np.array([vol(data[t], data[t+1], data[t+2], data[t+3])
                     for t in range(len(data) - 3)])

def tetra_volume_embed(series, tau=1):
    """本来の手法: 1本の系列を遅延埋め込みして体積"""
    # 各時刻tで [x(t), x(t+tau), x(t+2tau)] の3D点を作り、連続4点で四面体
    pts = np.array([[series[t], series[t+tau], series[t+2*tau]]
                    for t in range(len(series) - 2*tau)])
    def vol(p0, p1, p2, p3):
        return abs(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0
    return np.array([vol(pts[t], pts[t+1], pts[t+2], pts[t+3])
                     for t in range(len(pts) - 3)])

def spike_report(name, vols, baseline_end=100):
    base = vols[:baseline_end]
    bmean = base.mean()
    peak_idx = int(np.argmax(vols))
    peak_val = float(vols[peak_idx])
    ratio = peak_val / (bmean + 1e-12)
    print(f"  [{name}] baseline_mean={bmean:.6e}  peak@step{peak_idx}={peak_val:.6e}  ratio={ratio:.1f}x")
    return {"baseline_mean": bmean, "peak_idx": peak_idx,
            "peak_val": peak_val, "ratio": ratio}

# =======================================================
print("データ収集中 (seed=0)...")
step_times, sync_drifts, cpu_launches = collect(seed=0)
data3 = np.column_stack([step_times, sync_drifts, cpu_launches])

results = {"timestamp": datetime.now().isoformat()}

# ---- 検証A: 生信号 ----
print("\n=== 検証A: 生のstep_time (V(t)を介さない) ===")
st = step_times
print(f"  step_time: mean={st.mean():.4f}ms std={st.std():.4f}ms "
      f"min={st.min():.4f} max={st.max():.4f}")
# 125付近を生で出す
print("  step 120-132 の生step_time:")
for t in range(120, min(133, len(st))):
    bar = "#" * int((st[t] - st.min()) / (st.max() - st.min() + 1e-12) * 40)
    print(f"    step{t:3d}: {st[t]:7.4f}ms |{bar}")
results["A_raw_steptime"] = {"mean": float(st.mean()), "std": float(st.std()),
                              "max": float(st.max()), "argmax": int(np.argmax(st))}

# ---- 検証B: 入力の違いでV(t)を比較 ----
print("\n=== 検証B: V(t)の入力依存性 ===")
v_3col = tetra_volume_3col(data3)             # 元コード方式(3つの無関係な量)
v_embed = tetra_volume_embed(step_times, 1)   # 本来の手法(step_time埋め込み)
results["B1_3col"] = spike_report("B1: 3量混合(元コード)", v_3col)
results["B2_embed"] = spike_report("B2: step_time埋め込み(本来)", v_embed)

# cpu_launchだけ定数に固定して B1 を再計算 → スパイクがcpu由来か直接確認
data3_flatcpu = data3.copy()
data3_flatcpu[:, 2] = cpu_launches.mean()
v_flatcpu = tetra_volume_3col(data3_flatcpu)
results["B3_cpu_flattened"] = spike_report("B3: cpu_launchを定数化", v_flatcpu)
print("    -> B1で出たスパイクがB3で消えるなら、原因はcpu_launch(GPU無関係)")

# ---- 検証C: シャッフル ----
print("\n=== 検証C: 順序シャッフル (構造が本質か) ===")
rng = np.random.default_rng(0)
st_shuf = step_times.copy(); rng.shuffle(st_shuf)
v_shuf = tetra_volume_embed(st_shuf, 1)
results["C_shuffled"] = spike_report("C: step_timeシャッフル後", v_shuf)
print("    -> ピーク比がB2と大差ないなら、順序ではなく値の分布の裾が出てるだけ")

# ---- 検証D: 複数試行でスパイク位置が動くか ----
print("\n=== 検証D: スパイク位置の再現性 (固定=決定的 / 移動=ノイズ) ===")
peak_positions_3col = []
peak_positions_embed = []
for s in range(N_TRIALS):
    stt, syd, cpl = collect(seed=s)
    d3 = np.column_stack([stt, syd, cpl])
    p3 = int(np.argmax(tetra_volume_3col(d3)))
    pe = int(np.argmax(tetra_volume_embed(stt, 1)))
    peak_positions_3col.append(p3)
    peak_positions_embed.append(pe)
    print(f"  trial seed={s}: peak(3量混合)=step{p3:3d}   peak(埋め込み)=step{pe:3d}")
results["D_peak_positions_3col"] = peak_positions_3col
results["D_peak_positions_embed"] = peak_positions_embed
spread3 = np.std(peak_positions_3col)
print(f"\n  ピーク位置のばらつき(3量混合): std={spread3:.1f} step")
print(f"  -> stdが大きい(位置がバラバラ) = 125固定ではない = ノイズの裾")
print(f"  -> stdが0に近い(毎回同じstep)  = 決定的な原因(境界/バグ)を追う価値あり")

# ---- 結論ガイド ----
print("\n" + "=" * 55)
print("判定ガイド:")
print("  B3でスパイク消失      -> 原因はcpu_launch。現象ではない。")
print("  Cでピーク比が下がらない -> 順序構造ではなく分布の裾。")
print("  Dで位置がバラつく      -> step125固定ではない。非決定ノイズ。")
print("  上記を全て満たすと『平凡な説明』で閉じる。")
print("  逆に B3で残り/Cで消え/Dで固定 が揃えば、初めて本物候補。")
print("=" * 55)

with open("nagi_control_result.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\n結果を nagi_control_result.json に保存しました")
