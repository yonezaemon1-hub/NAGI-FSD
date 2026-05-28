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
