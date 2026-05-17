# E3-VLA Data Format

## Teacher Cache Format

The offline teacher cache stores full VLA rollout data for drafter training.

### Directory Structure

```
teacher_cache/
├── metadata/
│   └── records.jsonl           # One JSON object per line, one per timestep
├── features/
│   ├── shard_0000.safetensors   # AE features (ae_low, ae_mid, ae_high, ae_mixed)
│   ├── shard_0001.safetensors
│   └── ...
├── actions/
│   ├── shard_0000.safetensors   # Target action chunks [K, 7]
│   └── ...
└── states/
    ├── shard_0000.safetensors   # Robot states [D_r] + ee_pose [D_ee]
    └── ...
```

### Metadata Format (JSONL)

Each line is a JSON object:

```json
{
  "record_id": 0,
  "episode_id": "ep_001",
  "task_id": "libero_spatial_pick_place",
  "timestep": 42,
  "instruction": "pick up the black bowl and place it on the plate",
  "robot_state": [0.1, 0.2, ...],
  "ee_pose": [0.15, 0.22, 0.35, 0, 0, 0, 1],
  "shard": "shard_0000",
  "feature_idx": 42,
  "action_idx": 42,
  "chunk_len": 16,
  "timestamp": 1715900000.123
}
```

### Feature Tensor Format (safetensors)

Each shard file contains a flat dictionary of named tensors:

```
shard_0000.safetensors:
  ae_low_0:    Tensor[K, D]    # record 0, AE shallow layer features
  ae_mid_0:    Tensor[K, D]    # record 0, AE middle layer
  ae_high_0:   Tensor[K, D]    # record 0, AE deep layer
  ae_mixed_0:  Tensor[K, D]    # record 0, AE weighted mixture
  ae_low_1:    Tensor[K, D]    # record 1
  ...
```

### Action Tensor Format (safetensors)

```
shard_0000.safetensors:
  action_0: Tensor[K, 7]       # record 0 target action chunk
  action_1: Tensor[K, 7]       # record 1
  ...
```

Action dimension layout:

| Index | Meaning | Range |
|-------|---------|-------|
| 0 | delta_x | meters |
| 1 | delta_y | meters |
| 2 | delta_z | meters |
| 3 | delta_rot_x | radians or 6D |
| 4 | delta_rot_y | |
| 5 | delta_rot_z | |
| 6 | gripper | -1 (close) to +1 (open) |

### State Tensor Format (safetensors)

```
shard_0000.safetensors:
  state_0: Tensor[D_r]         # record 0 robot state
  state_1: Tensor[D_r]         # record 1
  ...
```

Robot state layout (example):

| Index | Meaning |
|-------|---------|
| 0:7 | joint positions |
| 7:10 | end-effector position (x, y, z) |
| 10:14 | end-effector rotation (quat or 6D) |
| 14 | gripper state (-1 to +1) |
| 15 | timestep |

## Training Sample Format

The `CrossTemporalDataset` produces training samples as `TrainingSample` dataclass instances:

```python
TrainingSample:
  # From timestep t0 (simulating full refresh cache)
  cached_full_round_features: AEFeatureBundle
  full_round_robot_state: Tensor[D_r]
  full_round_ee_pose: Tensor[D_ee]

  # From timestep t0 + Δ (current state)
  current_timestep: int
  current_robot_state: Tensor[D_r]
  current_ee_pose: Tensor[D_ee]
  action_history: Tensor[H, D_a]

  # Offset
  delta_t: int                    # cache_age = Δ
  delta_ee_pose: Tensor[D_ee]
  delta_robot_state: Tensor[D_r]
  delta_action_index: int
  gripper_phase: float

  # Target
  target_action_chunk: Tensor[K, D_a]
```

## Runtime Cache Format

The runtime cache holds data from the most recent full refresh round only (single record):

```python
RuntimeCacheRecord:
  full_round_id: int
  episode_id: str
  full_round_timestep: int
  ae_features: AEFeatureBundle      # [K, D] each
  full_action_chunk: Tensor[K, D_a]
  full_robot_state: Tensor[D_r]
  full_ee_pose: Tensor[D_ee]
  cache_age: int
  valid_until_step: int
  kv_cache_ref: Optional[Any]       # reference to VLM KV cache
```

## Environment Interface

Policies expect `Observation` objects and return `ActionCommand` objects.

For LIBERO integration, wrap raw environment output:

```python
Observation(
    image=torch.tensor(env_obs["image"]),
    instruction=env_obs["instruction"],
    robot_state=torch.tensor(env_obs["robot_state"]),
)
```

Environment step receives numpy actions:

```python
cmd = policy.act(obs)
for i in range(cmd.execute_len):
    obs, reward, done, info = env.step(cmd.actions[i].numpy())
```

## Episode Split Strategy

Train/validation split uses deterministic hashing at the episode level:

```python
# splitmix64-inspired: ensures same episode always goes to same split
h = md5(f"{episode_id}_{split_seed}".encode()).hexdigest()[:16]
if int(h, 16) % 100 < val_frac * 100:
    → validation
else:
    → training
```

This prevents data leakage across train/val when sampling cross-temporal pairs from
the same episode.
