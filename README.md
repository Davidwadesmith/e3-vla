# E3-VLA

FLASH-style Speculative VLA with Cached Action-Expert Latent Drafting.

## Quick Start

```bash
# Clone and install
git clone <repo-url> && cd e3-vla
uv venv && uv pip install -e .

# Install benchmark dependencies (optional)
uv pip install -e ".[benchmark]"

# Run tests
python -m pytest tests/ -v
```

## Core Idea

During full VLA refresh rounds, we cache Action Expert intermediate features as
short-term control memory. During speculative rounds, a lightweight drafter uses
cached AE features + temporal/pose offset alignment to predict draft action chunks,
verified by a low-cost Action Expert anchor verifier. The goal: longer accepted
prefixes, lower fallback rates, no full VLA in the speculative path.

## Architecture

```
obs → FullRefreshPath (occasional, ~every R steps)
    │  · full VLA forward
    │  · cache AE features → RuntimeFeatureCache
    │
    → SpeculativeCachedPath (most steps)
       · RuntimeFeatureCache → cached AE
       · OffsetAlignAdapter (gated + cross-attn)
       · CachedAEDrafter → draft chunk
       · ActionExpertAnchorVerifier → verify
       · PrefixAcceptor → accept prefix
       · execute accepted steps
       → if rejected → full refresh
```

## Key Components

| Module | Path | Purpose |
|--------|------|---------|
| RuntimeFeatureCache | `e3vla/cache/` | In-memory AE feature cache |
| OffsetAlignAdapter | `e3vla/align/` | Gated alignment + cross-attention |
| CachedAEDrafter | `e3vla/drafter/` | Draft action chunk prediction |
| ActionExpertAnchorVerifier | `e3vla/verifier/` | Low-cost flow-consistency check |
| PrefixAcceptor | `e3vla/verifier/` | Strict + gripper-aware acceptance |
| SpeculativeCachedPolicy | `e3vla/policy/` | Full pipeline orchestration |

## Training

```bash
# Generate teacher cache first, then:
uv run python scripts/train_drafter.py \
    model=default \
    data.cache_dir=./teacher_cache
```

## Evaluation

```bash
uv run python scripts/run_benchmark.py \
    configs/eval/speculative.yaml
```

## Project Structure

```
e3vla/
├── cache/        Runtime feature cache
├── align/        Offset alignment (gated + cross-attn)
├── drafter/      Drafter models + ablation variants
├── verifier/     Anchor verifier + oracle + prefix acceptor
├── policy/       Speculative cached policy
├── adapters/     VLA model adapters
├── data/         Teacher cache + dataset
├── benchmark/    Benchmark runner + wrappers
├── eval/         Rollout runner + report writer
└── utils/        Seed, device, checkpoint utilities

configs/          Hydra YAML configs
scripts/          Training + evaluation scripts
tests/            Unit + integration tests
docs/             Architecture + interface docs
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- Optional: OpenPI (for full VLA baseline), FLASH (for FLASH baseline)

## Citation

```
TBD
```
