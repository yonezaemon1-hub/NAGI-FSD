> ## ⚠️ Verification Update (2026-05-30): PI / V(t) did not survive controlled testing
>
> After the initial results below were posted, I ran controlled experiments to
> test whether the PI and 4-point volume V(t) signals were real or measurement
> artifacts. **The strong claim — that PI / V(t) is a direct detector of
> execution-order collapse or silent NCCL stalls — does not survive the controls.**
> I'm leaving the original write-up below as a record, with this correction on top.
>
> **What the controls showed (GTX 1070, matmul 2048², synchronize present):**
>
> - **Raw step_time is flat.** Around the reported "spike" region (steps 120–132)
>   the raw step_time is ~3.74ms throughout — no anomaly. The spike only appears
>   *after* the volume transform.
> - **V(t) was driven by CPU-side noise, not GPU execution.** The tetrahedron was
>   built from three unrelated quantities (step_time, sync_drift, cpu_launch) as
>   x/y/z axes — not the delay-embedding originally described. Holding cpu_launch
>   constant drops the V(t) spike to exactly **0.0×**. The spike was entirely
>   `torch.randn` CPU allocation jitter.
> - **The spike isn't fixed at step 125.** Across 5 seeds the V(t) peak location
>   moves to steps 168, 48, 196, 0, 1 (std ≈ 83). "Reproduced twice" was a noise-tail
>   coincidence.
> - **PI is rank noise, not order structure.** Normal PI = **0.915**; PI after fully
>   shuffling step_time = **0.925**; PI after replacing step_time with pure Gaussian
>   noise = **0.910**. The three are statistically identical. Shuffling the order
>   doesn't change PI, and replacing the data with pure noise doesn't change PI —
>   so PI is measuring the rank noise of ~10 nearly-identical values, not execution
>   ordering. "Cross-size constant PI" was just that artifact being size-independent.
>
> **Reproduce the falsification yourself:** see [`nagi_control.py`](nagi_control.py)
> (V(t) controls A/B/C/D) and [`pi_control.py`](pi_control.py) (PI shuffle / pure-noise
> controls).
>
> **What this does *not* retract:** silent NCCL stalls are a real problem, and local
> structural change in the synchronization path while latency still looks normal is
> still worth instrumenting. What these controls ruled out is *this particular
> observable* (wall-clock step_time + volume/rank transforms), not the question
> itself. Next direction: per-kernel timing via **CUDA events** instead of wall-clock
> step_time, measured on the collective path rather than a matmul proxy.
>
> — The findings below are preserved as the original (now falsified) hypothesis.

---

# NAGI-FSD
GPU Execution Order Collapse Detection - Failure Signature Detector
# NAGI-FSD: GPU Execution Order Collapse Detector

## What is this?

A measurement framework for detecting **Permutation Instability (PI)** in GPU execution timing — a structural phenomenon that conventional monitoring tools (Datadog, Prometheus) cannot observe.

## Key Finding

GPU execution step times appear normal by conventional metrics (latency, throughput, VRAM), yet their **ordering structure collapses** consistently.

This is not a performance degradation. It is a structural property of GPU execution ordering.

## Permutation Instability (PI)

PI measures how much the rank ordering of execution times deviates from stability:

- `PI = 0.0` → Stable ordering
- `PI = 0.5` → Degraded ordering  
- `PI = 0.9+` → Collapse of exe

## Analysis Results

![PI Analysis](pi_analysis.png)

## FSD Realtime Monitor

`fsd_realtime.py` implements a realtime collapse monitor:

- Tracks PI per block in real time
- Detects sustained collapse (3+ consecutive high-PI blocks)
- Outputs NORMAL / DEGRADED / COLLAPSE / SUSTAINED COLLAPSE

### Sample Output (GTX 1070)
- Mean PI: 0.937
- Collapse rate: 90.0%
- Sustained collapse alerts: 19/30 blocks
- Final status: CRITICAL
## Analysis Results

### Complete Experimental Summary
![PI Summary](pi_summary.png)

### Detailed Analysis
![PI Analysis](pi_analysis.png)
