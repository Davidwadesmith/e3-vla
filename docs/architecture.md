# E3-VLA Architecture

## Overview

E3-VLA is a speculative inference framework for non-autoregressive Vision-Language-Action
models. It builds on FLASH-style speculative VLA inference with a key addition: **cached
Action Expert latent features** that serve as short-term control memory between full
VLA refresh rounds.

## System Layers

```
┌─────────────────────────────────────────────────┐
│              SpeculativeCachedPolicy             │
│                                                  │
│  ┌──────────────┐   ┌───────────────────────┐   │
│  │FullRefreshPath│   │ SpeculativeCachedPath │   │
│  │              │   │                       │   │
│  │ full VLA fwd │   │  cache → ae_mixed     │   │
│  │ extract AE   │   │  OffsetAlignAdapter   │   │
│  │ cache AE     │   │  CachedAEDrafter      │   │
│  │ reset state  │   │  AnchorVerifier       │   │
│  └──────┬───────┘   │  PrefixAcceptor       │   │
│         │           └───────────┬───────────┘   │
│         ▼                       ▼               │
│    RuntimeFeatureCache    accepted actions       │
└─────────────────────────────────────────────────┘
```

## Key Components

### 1. RuntimeFeatureCache (`e3vla/cache/`)

In-memory cache holding Action Expert features from the most recent full refresh round.
Provides O(1) read during speculative rounds. Invalidated by cache_age, episode reset,
or max_age exceeded.

### 2. OffsetAlignAdapter (`e3vla/align/`)

Gated alignment + cross-attention module that adapts stale cached AE features to the
current timestep. Three mechanisms:

- **Per-token gating**: Each cached AE token gets independent keep/discard weight based
  on distance to `cache_feature_cursor`.
- **Cross-attention**: Current K draft queries freely attend to gated cached tokens.
  No forced one-to-one temporal mapping.
- **Temporal bias**: Cached tokens far from cursor get lower attention weight.

### 3. CachedAEDrafter (`e3vla/drafter/`)

The core drafter that predicts action chunks conditioned on:
- Cached AE mixed features (from last full refresh)
- Offset embeddings (time, pose, gripper phase)
- Current robot state
- Action history
- Optional cheap image features

Architecture: AE Feature Mixer → OffsetAlignAdapter → Context Fusion →
Lightweight Transformer Decoder → Action + Uncertainty heads.

### 4. ActionExpertAnchorVerifier (`e3vla/verifier/`)

Low-cost verification using only the Action Expert's denoising step at a few flow
timesteps. **Does not** run vision encoder or VLM prefill.

Context: stale VLM/KV from last full refresh + fresh robot_state (semi-open-loop).

### 5. SpeculativeCachedPolicy (`e3vla/policy/`)

State machine orchestrating the full pipeline:

```
INIT → FULL_REFRESH → SPECULATIVE (loop) → FALLBACK → FULL_REFRESH
```

## Data Flow

### Full Refresh Round
```
obs → full VLA forward → action chunk [K, D_a]
    → extract AE features → RuntimeFeatureCache
    → execute first K_exec steps
    → reset cache_age, cursor
```

### Speculative Round (no full VLA)
```
obs → RuntimeFeatureCache.get_latest()
    → OffsetEncoder.compute(obs, cached, elapsed, cursor)
    → CachedAEDrafter(cached_ae, offset, cheap_features)
    → ActionExpertAnchorVerifier(draft, stale_vlm_context, fresh_robot_state)
    → PrefixAcceptor(errors, uncertainty, gripper_phase)
    → execute accepted prefix [0:m]
    → update elapsed, cursor, cache_age
```

### Fallback Trigger
- accepted_len == 0
- drafter output NaN/Inf
- gripper switch detected
- cache_age >= max_cache_age
- periodic_full_every_n rounds elapsed

## Speedup Mechanism

```
Full baseline:  cost_per_step = C_full / L_full
Speculative:    cost_per_step = (C_spec × N_spec + C_full × N_full) / total_steps

  C_spec = C_cache_read + C_offset_align + C_draft + C_verify
  C_spec << C_full required for speedup

  speedup ≈ 1 / (refresh_rate + (1 - refresh_rate) × C_spec / C_full)
```

Key: **Full L2 verifier does NOT provide speedup for non-AR action chunk policies.**
It is oracle/debug only.

## Ablation Variants

| Variant | Drafter | Purpose |
|---------|---------|---------|
| NoCachedAE | No AE features, state+history only | Baseline: value of AE features |
| CachedAE-NoOffset | AE features, no alignment | Value of offset alignment |
| CachedAE-TimeOnly | AE + time offsets only | Time offset contribution |
| CachedAE-PoseOffset | AE + pose offsets only | Pose offset contribution |
| CachedAE-FullOffset | AE + time + pose + gripper | **Full method** |
| CachedVLMFeature | VLM features instead of AE | AE vs VLM feature comparison |

## External Dependencies

- **OpenPI/π₀**: `ultra-robotics/openpi` — Full VLA baseline
- **FLASH**: `dexmal/realtime-vla-flash` — FLASH-style baseline
- Both are optional — `pip install e3vla[benchmark]` to include them
