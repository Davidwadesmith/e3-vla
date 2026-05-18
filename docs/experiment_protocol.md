# E3-VLA Experiment Protocol

## Prerequisites

```bash
# Core deps
uv venv && source .venv/bin/activate && uv pip install -e .

# For benchmark with external baselines (optional)
uv pip install -e ".[benchmark]"

# For LIBERO environment
pip install libero

# Verify
python -c "import torch; assert torch.cuda.is_available()"
python -m pytest tests/ -v  # 26 tests
```

---

## Pipeline Overview

```
generate_cache.py ──→ teacher_cache/ ──→ train_drafter.py × 3 ──→ checkpoints/
                                                                       │
                                                                       ▼
                                                              run_benchmark.py
                                                                       │
                                                                       ▼
                                                              generate_report.py
                                                                       │
                                                          experiments/report/
```

---

## Stage 1: Generate Teacher Cache (with real VLA)

Collect full VLA rollout data from LIBERO.

```bash
# Full suite
uv run python scripts/generate_cache.py \
    env=libero \
    model=openpi \
    model.checkpoint=/path/to/pi0_checkpoint \
    output_dir=/root/autodl-tmp/e3vla/teacher_cache \
    max_episodes=100

# Single-task quick test
uv run python scripts/generate_cache.py \
    env.suite=libero_spatial \
    env.task=libero_spatial_pick_place \
    model=openpi \
    output_dir=/root/autodl-tmp/e3vla/teacher_cache \
    max_episodes=5
```

**What this does:**
1. Loads real OpenPI/π₀ model (or any BaseVLAAdapter)
2. Creates LIBERO environments
3. Runs full VLA rollouts, collecting per-timestep:
   - Action Expert features (ae_low, ae_mid, ae_high, ae_mixed)
   - Target action chunks
   - Robot states, ee poses
4. Writes safetensors shards + JSONL metadata

**Output:** `teacher_cache/` with `metadata/`, `features/`, `actions/`, `states/`

**Configuration:** `configs/cache/default.yaml`

---

## Stage 2: Train NoCachedAE Baseline (Gate G1)

FLASH-style baseline without cached AE features.

```bash
uv run python scripts/train_drafter.py \
    model=no_cached_ae \
    data.cache_dir=/root/autodl-tmp/e3vla/teacher_cache \
    training.epochs=100 \
    checkpoint.save_dir=/root/autodl-tmp/e3vla/checkpoints/no_cached_ae
```

**Expected:** `checkpoints/no_cached_ae/best.pt`

---

## Stage 3: Train CachedAE-NoOffset (Gate G1)

Cached AE features, no offset alignment.

```bash
uv run python scripts/train_drafter.py \
    model=cached_ae_no_offset \
    data.cache_dir=/root/autodl-tmp/e3vla/teacher_cache \
    training.epochs=100 \
    checkpoint.save_dir=/root/autodl-tmp/e3vla/checkpoints/no_offset
```

**Gate G1:** Compare `l_total` and accepted prefix length vs NoCachedAE.
Cached AE features must show measurable improvement.

---

## Stage 4: Train CachedAE-FullOffset (Ours, Gate G2)

Full method with offset alignment.

```bash
uv run python scripts/train_drafter.py \
    model=default \
    data.cache_dir=/root/autodl-tmp/e3vla/teacher_cache \
    training.epochs=100 \
    checkpoint.save_dir=/root/autodl-tmp/e3vla/checkpoints/full_offset
```

**Gate G2:** Offset alignment must further improve metrics vs NoOffset.

---

## Stage 5: Benchmark Evaluation

Run all methods through the same environment for fair comparison.

```bash
# Ablation study (E3-VLA variants only)
uv run python scripts/run_benchmark.py \
    benchmark=ablation \
    model.checkpoint=/path/to/pi0_checkpoint \
    checkpoint_dir=/root/autodl-tmp/e3vla/checkpoints

# Full comparison (all methods)
uv run python scripts/run_benchmark.py \
    benchmark=full_comparison \
    model.checkpoint=/path/to/pi0_checkpoint \
    checkpoint_dir=/root/autodl-tmp/e3vla/checkpoints
```

**Methods evaluated:**

| Method | What it tests |
|--------|---------------|
| Full VLA (π₀) | Upper bound — accuracy |
| Cached Full Action Reuse | Simple baseline — complexity justification |
| Cached Action Reuse + Offset | Simple baseline + correction |
| NoCachedAE | FLASH-style baseline — Gate G1 |
| CachedAE-NoOffset | AE features w/o alignment — Gate G1 |
| CachedAE-FullOffset (Ours) | Full method — Gate G2 |
| FLASH-style (optional) | External baseline |

**Metrics collected per method × task:**
- success_rate, avg_latency_ms, speedup_vs_full
- avg_accepted_prefix_len, fallback_rate, full_refresh_rate
- Per-component latency breakdown

**Output:** `experiments/results/benchmark_results.json`

---

## Stage 6: Generate Report

```bash
uv run python scripts/generate_report.py \
    --results experiments/results/benchmark_results.json \
    --output experiments/report/
```

**Output files:**
- `main_table.csv` — Primary comparison table
- `main_table.md` — Paper-ready markdown table
- `latency_breakdown.csv` — Per-component timing
- `metrics.json` — All metrics for downstream analysis

---

## Stage 7: Ablation Experiments (after G1/G2 pass)

### Feature source ablation
```
CachedVLMFeature:  Replace AE features with VLM features
AE-MultiLayer:     ae_low + ae_mid + ae_high
AE-FinalLayer:     ae_high only
```

### Offset component ablation
```
NoOffset → TimeOnly → PoseOffset → FullOffset
```

### Threshold sweep
```bash
# Vary tau_radius from 0.1 to 1.0
for tau in 0.1 0.2 0.3 0.5 0.7 1.0; do
    uv run python scripts/run_benchmark.py \
        verifier.tau_radius=$tau \
        checkpoint_dir=/root/autodl-tmp/e3vla/checkpoints \
        output_dir=/root/autodl-tmp/e3vla/experiments/sweep_tau_${tau}
done
```

### Cache age analysis
Group results by `cache_age` buckets (0-5, 5-10, 10-20, 20+) to analyze staleness impact.

---

## Success Criteria (Gates)

| Gate | Test | Criterion |
|------|------|-----------|
| **G1** | CachedAE-NoOffset vs NoCachedAE | `accepted_prefix_len` significantly longer |
| **G2** | CachedAE-FullOffset vs CachedAE-NoOffset | Further improvement in prefix or fallback rate |
| **G3** | Ours vs FLASH-style baseline | Positive shift in success-latency Pareto |
| **G4** | Ours vs CachedFullActionReuse | Significantly better (complexity justified) |

**If G1 fails:** Core hypothesis invalid — cached AE features do not help.
**If G1 passes, G2 fails:** Use NoOffset as main method. Offset alignment is decoration.
**If G3/G4 fail:** Method not worth the added complexity vs simpler alternatives.

---

## Quick Reference

| Stage | Script | Config | Output |
|-------|--------|--------|--------|
| Cache | `scripts/generate_cache.py` | `configs/cache/default.yaml` | `teacher_cache/` |
| Train | `scripts/train_drafter.py` | `configs/train/default.yaml` | `checkpoints/` |
| Eval | `scripts/run_benchmark.py` | `configs/benchmark/default.yaml` | `experiments/results/` |
| Report | `scripts/generate_report.py` | — (argparse) | `experiments/report/` |
