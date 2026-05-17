# E3-VLA Interfaces

## Core Protocols

All swappable components implement abstract base classes defined in `e3vla/protocols.py`.

### BaseVLAAdapter

```python
class BaseVLAAdapter(ABC):
    def full_inference(self, obs: Observation) -> Tensor[K, D_a]
    def extract_ae_features(self, obs, action_chunk) -> AEFeatureBundle
    def action_expert_denoise_step(self, x_t, t, vlm_context, robot_state) -> Tensor
```

| Method | Called By | Frequency | Cost |
|--------|-----------|-----------|------|
| `full_inference` | FullRefreshPath | Every ~R steps | High (full VLA) |
| `extract_ae_features` | FullRefreshPath | Every ~R steps | Low (hook extraction) |
| `action_expert_denoise_step` | ActionExpertAnchorVerifier | Every speculative step | Low (single AE step) |

### BaseDrafter

```python
class BaseDrafter(ABC):
    def forward(self, cached_ae_features, offset_features, cheap_features) -> DraftOutput
    def compute_loss(self, batch) -> dict
```

### BaseVerifier

```python
class BaseVerifier(ABC):
    def verify(self, obs, draft_chunk, adapter, cached_vlm_context, verify_spec) -> VerificationResult
```

### BasePrefixAcceptor

```python
class BasePrefixAcceptor(ABC):
    def accept(self, errors_per_step, uncertainty, draft_chunk, gripper_phase) -> tuple[int, bool]
```

## Data Schemas

All schemas are defined in `e3vla/schema.py`.

### Observation → ActionCommand Pipeline

```
Observation
  ├── image: Tensor[B, C, H, W]
  ├── instruction: str
  ├── robot_state: Tensor[B, D_r]
  ├── history: Optional[Tensor]
  └── env_info: Optional[Dict]

        ↓ policy.act()

ActionCommand
  ├── actions: Tensor[execute_len, D_a]
  ├── execute_len: int
  ├── can_interrupt: bool
  ├── mode: "full_refresh" | "speculative" | "action_reuse" | "fallback"
  ├── prefix_length: int
  ├── confidence: float
  └── diagnostics: Dict
```

### AEFeatureBundle

```python
AEFeatureBundle:
  ae_low:    Tensor[K, D]  # shallow AE layer
  ae_mid:    Tensor[K, D]  # middle AE layer
  ae_high:   Tensor[K, D]  # deep AE layer
  ae_mixed:  Tensor[K, D]  # weighted mixture
```

### RuntimeCacheRecord

```python
RuntimeCacheRecord:
  full_round_id: int
  episode_id: str
  full_round_timestep: int
  ae_features: AEFeatureBundle
  full_action_chunk: Tensor[K, D_a]
  full_robot_state: Tensor[D_r]
  full_ee_pose: Tensor[D_ee]
  cache_age: int           # incremented each speculative step
  valid_until_step: int    # max age before forced refresh
  kv_cache_ref: Optional   # VLM KV cache from full round (stale)
```

### OffsetFeatures

```python
OffsetFeatures:
  cache_age: int                    # env steps since last full refresh
  elapsed_steps_since_full: int     # real env steps since full refresh
  cache_feature_cursor: int         # position in cached K-token (0..K-1)
  draft_step_index: int             # start in draft (usually 0)
  delta_robot_state: Tensor[D_r]    # state change
  delta_ee_pose: Tensor[D_ee]       # end-effector pose change
  gripper_phase: float              # -1 close, 0 trans, +1 open
```

### DraftOutput

```python
DraftOutput:
  action_chunk: Tensor[B, K, D_a]
  uncertainty: Tensor[B, K, 3]   # pos/rot/grip per-step
  hidden_states: Tensor[B, K, D]
```

### VerificationResult

```python
VerificationResult:
  accepted_prefix_length: int
  errors_per_step: Tensor[K]
  fallback_required: bool
  confidence_per_step: Tensor[K]
  verification_latency_ms: float
  reason: str  # "accepted" | "rejected_action_error" | "rejected_gripper_phase" | ...
```

## Adding a New VLA Backend

1. Subclass `BaseVLAAdapter` in `e3vla/adapters/`
2. Implement `full_inference`, `extract_ae_features`, `action_expert_denoise_step`
3. Register in Hydra config: `configs/model/<name>.yaml`

## Adding a New Drafter Variant

1. Subclass `BaseDrafter` (or subclass `nn.Module` + match the `forward` signature)
2. Implement `forward(cached_ae_features, offset_features, cheap_features) → DraftOutput`
3. Optional: implement `compute_loss(batch) → dict`
4. Register in Hydra config

## Adding a New Benchmark Method

1. Implement `BenchmarkPolicy` protocol:
   - `reset(task_info) → None`
   - `act(obs) → ActionCommand`
   - `method_name` property
   - `method_type` property
2. Place wrapper in `e3vla/benchmark/wrappers/`
3. Load external dependencies lazily (so core system runs without them)
