# E3-VLA Experiment Protocol

## Stage 1: Teacher Cache Generation

Generate offline training data from full VLA rollouts.

```bash
# Collect teacher cache from LIBERO rollouts
python scripts/generate_cache.py \
    env=libero_spatial \
    model=openpi \
    output_dir=./teacher_cache \
    max_episodes=100
```

**Expected output**: `teacher_cache/` with metadata, features, actions, states.

**Validation**: `test_cache_roundtrip` — verify read-back consistency.

## Stage 2: Train NoCachedAE Baseline

Train the FLASH-style baseline (no cached AE features).

```bash
uv run python scripts/train_drafter.py \
    model=no_cached_ae \
    data.cache_dir=./teacher_cache \
    training.epochs=100
```

**Expected output**: `checkpoints/no_cached_ae_best.pt`

**Metrics to track**: train/val loss curves, convergence speed.

## Stage 3: Train CachedAE-NoOffset

Add cached AE features without offset alignment.

```bash
uv run python scripts/train_drafter.py \
    model=cached_ae_no_offset \
    data.cache_dir=./teacher_cache \
    training.epochs=100
```

**Gate G1**: Compare `accepted_prefix_len` vs NoCachedAE. Must show significant improvement.

## Stage 4: Train CachedAE-FullOffset (Ours)

Full method with offset alignment.

```bash
uv run python scripts/train_drafter.py \
    model=default \
    data.cache_dir=./teacher_cache \
    training.epochs=100
```

**Gate G2**: Compare vs CachedAE-NoOffset. Offset alignment must further improve metrics.

## Stage 5: Benchmark Evaluation

Run all methods through the same BenchmarkRunner for fair comparison.

```bash
# Run full benchmark suite
uv run python scripts/run_benchmark.py \
    configs/eval/speculative.yaml \
    methods=[full_vla,reduced_step,action_reuse,flash,no_cached_ae,cached_ae_no_offset,ours]
```

### Method Checklist

- [ ] Full VLA (π₀) — OpenPI thin wrapper
- [ ] Reduced-step Flow (K=2,3,4) — Config change
- [ ] Open-loop Full Chunk Reuse — Action reuse baseline
- [ ] CachedFullActionReuse + Offset — Action reuse + correction
- [ ] FLASH-style Drafter + Verifier — FLASH thin wrapper
- [ ] NoCachedAE (our FLASH-style baseline) — Own drafter
- [ ] CachedAE-NoOffset — Own drafter
- [ ] CachedAE-FullOffset (Ours) — Full method
- [ ] Oracle Full L2 Verifier — Oracle only

### Primary Metrics

| Metric | Direction | Meaning |
|--------|-----------|---------|
| Success Rate | ↑ | Task completion rate |
| E2E Latency | ↓ | End-to-end per-step latency (ms) |
| Speedup vs Full VLA | ↑ | Multiplicative speedup |
| Accepted Prefix Length | ↑ | Average accepted draft steps |
| Fallback Rate | ↓ | Fraction of rounds requiring full refresh |

### Secondary Metrics

| Metric | Purpose |
|--------|---------|
| `accepted_prefix_len_by_cache_age` | How staleness affects quality |
| `fallback_rate_by_cache_age` | Staleness-induced fallback |
| `draft_error_vs_cache_age` | Error growth with age |
| `false_accept_vs_oracle` | Anchor verifier vs oracle disagreement |
| `false_accept_vs_rollout` | Anchor verifier acceptance → task failure |
| `gripper_phase_fallback_rate` | Gripper-related safety |
| `contact_phase_failure_rate` | Contact stage reliability |
| `success-latency Pareto AUC` | Aggregate quality metric |

### Latency Breakdown

| Component | Measurement |
|-----------|-------------|
| Full Refresh | Time for full VLA forward |
| Cache Read | RuntimeFeatureCache access |
| Offset Align | OffsetAlignAdapter forward |
| Draft | CachedAEDrafter forward |
| Verify | ActionExpertAnchorVerifier |
| Total E2E | Sum + environment step |

## Stage 6: Ablation Experiments

### Feature Source Ablation
- CachedVLMFeature: Replace AE features with VLM features
- CachedAE-MultiLayer vs CachedAE-FinalLayer

### Offset Component Ablation
- NoOffset → TimeOnly → PoseOffset → FullOffset

### Threshold Sensitivity
- Sweep `tau_radius` from 0.1 to 1.0
- Plot speedup vs success tradeoff curve

### Cache Age Analysis
- Group results by `cache_age` buckets (0-5, 5-10, 10-20, 20+)
- Analyze per-phase (approach / contact / grasp / place)

## Stage 7: Report Generation

```bash
uv run python scripts/generate_report.py \
    --results ./experiments/results/ \
    --output ./experiments/report/
```

Outputs:
- `main_table.csv` — Primary comparison table
- `latency_breakdown.csv` — Per-component timing
- `ablation.csv` — Ablation study results
- `threshold_sweep.png` — Speedup-success curve
- `cache_age_analysis.png` — Staleness impact
- `metrics.json` — All metrics for downstream analysis

## Success Criteria

The project is considered successful if:

1. **G1**: CachedAE-NoOffset > NoCachedAE on accepted_prefix_len
2. **G2**: CachedAE-FullOffset > CachedAE-NoOffset
3. **G3**: Ours > FLASH-style baseline on success-latency Pareto
4. **G4**: Ours > CachedFullActionReuse (method complexity justified)

If G1 fails, the core hypothesis is invalid — cached AE features don't help.
If G1 passes but G2 fails, use NoOffset as the main method.
