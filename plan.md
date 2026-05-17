---

# 项目名称

**E3-VLA: EAGLE-3-style Speculative Inference for Non-Autoregressive VLA**

项目目标：

> 在非 AR VLA，尤其是 flow / diffusion action-chunk VLA 上，实现一个 EAGLE-3-style speculative inference 框架：利用 target VLA 的多层特征训练一个轻量 drafter，直接预测 action chunk / flow velocity / action latent，然后由 target action expert 做一致性验证，接受可靠 prefix，否则 fallback 到 full inference。

核心不是重新训练一个完整 VLA，而是在已有 VLA 推理流程外加一个 speculative path。

---

# 1. 项目仓库定义

建议仓库名：

```text
e3-vla
```

推荐目录结构：

```text
e3-vla/
├── README.md
├── pyproject.toml
├── configs/
│   ├── env/
│   │   ├── libero.yaml
│   │   └── sim_env.yaml
│   ├── model/
│   │   ├── smolvla.yaml
│   │   ├── openpi_pi0.yaml
│   │   └── mock_vla.yaml
│   ├── train/
│   │   ├── drafter_final_layer.yaml
│   │   ├── drafter_multilayer.yaml
│   │   └── drafter_training_time_test.yaml
│   └── eval/
│       ├── baseline.yaml
│       ├── speculative.yaml
│       └── ablation.yaml
│
├── e3vla/
│   ├── __init__.py
│   ├── adapters/
│   │   ├── base_vla_adapter.py
│   │   ├── smolvla_adapter.py
│   │   ├── openpi_adapter.py
│   │   └── mock_adapter.py
│   │
│   ├── features/
│   │   ├── feature_schema.py
│   │   ├── feature_collector.py
│   │   ├── feature_cache.py
│   │   └── layer_selectors.py
│   │
│   ├── drafter/
│   │   ├── drafter_schema.py
│   │   ├── base_drafter.py
│   │   ├── action_drafter.py
│   │   ├── multilayer_fusion_drafter.py
│   │   └── training_time_test.py
│   │
│   ├── verifier/
│   │   ├── verifier_schema.py
│   │   ├── base_verifier.py
│   │   ├── action_l2_verifier.py
│   │   ├── flow_consistency_verifier.py
│   │   └── prefix_acceptor.py
│   │
│   ├── policy/
│   │   ├── policy_schema.py
│   │   ├── full_vla_policy.py
│   │   ├── draft_only_policy.py
│   │   └── speculative_policy.py
│   │
│   ├── data/
│   │   ├── teacher_cache_schema.py
│   │   ├── cache_writer.py
│   │   ├── cache_reader.py
│   │   └── dataset_builder.py
│   │
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── latency_profiler.py
│   │   ├── rollout_runner.py
│   │   └── report_writer.py
│   │
│   └── utils/
│       ├── logging.py
│       ├── seed.py
│       ├── device.py
│       └── checkpoints.py
│
├── scripts/
│   ├── 00_check_env.sh
│   ├── 01_run_full_vla_baseline.sh
│   ├── 02_collect_teacher_cache.sh
│   ├── 03_train_drafter.sh
│   ├── 04_eval_draft_only.sh
│   ├── 05_eval_speculative.sh
│   └── 06_generate_report.sh
│
├── experiments/
│   ├── libero_spatial/
│   ├── libero_object/
│   ├── libero_goal/
│   └── libero_long/
│
├── docs/
│   ├── architecture.md
│   ├── interfaces.md
│   ├── data_format.md
│   ├── experiment_protocol.md
│   └── failure_cases.md
│
└── tests/
    ├── test_adapter_contract.py
    ├── test_feature_collection.py
    ├── test_drafter_shapes.py
    ├── test_verifier_prefix.py
    ├── test_policy_fallback.py
    └── test_cache_roundtrip.py
```

---

# 2. 系统总体架构

整体分为五层：

```text
External Environment
    ↓
VLA Adapter Layer
    ↓
Feature Collection Layer
    ↓
Drafter Layer
    ↓
Verifier + Prefix Acceptance Layer
    ↓
Policy Execution Layer
```

完整推理路径：

```text
observation
    ↓
target VLA full inference
    ↓
target action chunk
    ↓
execute
```

speculative 推理路径：

```text
observation
    ↓
target VLA lightweight forward / cached features
    ↓
multi-layer feature collector
    ↓
E3-style drafter
    ↓
draft action chunk / draft velocity / draft latent
    ↓
target action expert verification
    ↓
accept longest reliable prefix
    ↓
execute accepted prefix
    ↓
if rejected: fallback to full VLA inference
```

核心设计原则：

```text
target VLA 不被侵入式修改；
所有模型接入通过 Adapter；
drafter 和 verifier 可以独立替换；
policy 只依赖统一接口，不依赖具体 VLA 实现；
所有中间数据都可缓存、可复现、可离线分析。
```

---

# 3. 外部交互定义

## 3.1 外部依赖对象

系统外部主要有四类对象：

```text
1. VLA 模型
   例如 SmolVLA、OpenPI/π0、mock VLA

2. 仿真环境
   例如 LIBERO / MetaWorld / SimplerEnv

3. 数据缓存
   teacher action、hidden features、rollout logs

4. 评测系统
   success rate、latency、fallback rate、accepted prefix
```

## 3.2 外部输入

每一轮控制输入统一定义为：

```text
Observation:
    image: camera image or multi-view images
    instruction: language command
    robot_state: proprioceptive state
    history: optional previous actions / observations
    env_info: optional environment metadata
```

其中 `robot_state` 至少包含：

```text
end_effector_pose
joint_positions
gripper_state
timestep
```

## 3.3 外部输出

policy 对环境输出：

```text
ActionCommand:
    actions: accepted action prefix
    mode: full | draft_only | speculative | fallback
    prefix_length: accepted prefix length
    confidence: verifier confidence
    diagnostics: optional debug info
```

动作格式统一为：

```text
ActionChunk:
    shape: [chunk_len, action_dim]
    fields:
        delta_position
        delta_rotation
        gripper
        optional_joint_control
```

---

# 4. 内部接口定义

## 4.1 VLA Adapter 接口

文件：

```text
e3vla/adapters/base_vla_adapter.py
```

职责：

> 屏蔽不同 VLA 模型的差异，对上层暴露统一推理、特征提取和 action expert 验证接口。

接口设计：

```text
class BaseVLAAdapter:

    encode_observation(obs) -> EncodedContext
        将图像、语言、机器人状态编码为 VLA context。

    full_inference(obs) -> ActionChunk
        运行完整 target VLA，输出 target action chunk。

    collect_features(obs, layer_spec) -> MultiLayerFeatures
        提取 low / mid / high / action expert features。

    action_expert_forward(context, action_latent, timestep) -> ActionExpertOutput
        调用 target action expert 的指定 step。

    verify_candidate(obs, draft, verify_spec) -> VerificationTrace
        使用 target action expert 对 draft chunk 做验证。
```

适配器实现：

```text
SmolVLAAdapter:
    首选工程实现对象。

OpenPIAdapter:
    后续强实验对象。

MockVLAAdapter:
    单元测试和接口测试对象。
```

---

## 4.2 Feature Collector 接口

文件：

```text
e3vla/features/feature_collector.py
```

职责：

> 从 target VLA 中收集多层特征，并整理成 drafter 可用格式。

输入：

```text
obs
layer_spec
adapter
```

输出：

```text
MultiLayerFeatures:
    vision_low
    vlm_mid
    vlm_high
    action_mid
    pooled_context
    masks
    metadata
```

推荐 layer spec：

```text
LayerSpec:
    vision_low_layers: [l1, l2]
    vlm_mid_layers: [m1]
    vlm_high_layers: [h1]
    action_layers: [a1]
    pooling: mean | cls | action_query_attention
```

内部处理流程：

```text
raw features
    ↓
projection to common dimension
    ↓
normalization
    ↓
optional pooling
    ↓
feature package
```

---

## 4.3 Drafter 接口

文件：

```text
e3vla/drafter/base_drafter.py
```

职责：

> 给定多层特征和机器人状态，快速生成 candidate action chunk、velocity 或 latent。

接口：

```text
class BaseDrafter:

    forward(features, robot_state, draft_spec) -> DraftOutput

    loss(batch, teacher_output) -> DrafterLoss

    load_checkpoint(path)

    save_checkpoint(path)
```

输出：

```text
DraftOutput:
    draft_type: action | velocity | latent
    action_chunk: optional [K, action_dim]
    velocity_anchors: optional [N, K, action_dim]
    action_latent: optional tensor
    uncertainty: optional [K]
    diagnostics:
        feature_norms
        predicted_phase
        confidence
```

实现版本：

```text
ActionDrafter:
    普通 action-level drafter，用作 baseline。

FinalLayerDrafter:
    只用最后一层 feature，用作 ablation。

MultiLayerFusionDrafter:
    EAGLE-3-style 主方法。

MultiLayerFusionDrafter + TrainingTimeTest:
    完整版本。
```

---

## 4.4 Verifier 接口

文件：

```text
e3vla/verifier/base_verifier.py
```

职责：

> 用 target action expert 检查 draft chunk 是否可靠，并返回可接受 prefix。

接口：

```text
class BaseVerifier:

    verify(obs, draft_output, adapter, verify_spec) -> VerificationResult
```

输出：

```text
VerificationResult:
    accepted_prefix_length: int
    accepted_actions: ActionChunk
    fallback_required: bool
    errors_per_step: [K]
    confidence_per_step: [K]
    verification_latency_ms: float
    reason: accepted | rejected_action_error | rejected_velocity_error | rejected_gripper_phase | invalid_output
```

Verifier 版本：

```text
ActionL2Verifier:
    用 target action 和 draft action 的 L2 距离验收。

FlowConsistencyVerifier:
    用 target action expert 的 velocity / endpoint consistency 验收。

HybridVerifier:
    action distance + velocity consistency + gripper phase。

ClosedLoopVerifier:
    使用最新 observation 做额外检查，后续阶段实现。
```

---

## 4.5 Prefix Acceptor 接口

文件：

```text
e3vla/verifier/prefix_acceptor.py
```

职责：

> 根据每个 step 的误差和阈值，决定最长可执行 prefix。

输入：

```text
errors_per_step
confidence_per_step
thresholds
acceptance_policy
```

输出：

```text
accepted_prefix_length
fallback_required
```

验收策略：

```text
strict:
    第一个失败 step 之前全部接受。

tolerant:
    允许局部小误差，但连续失败则截断。

phase_aware:
    gripper close/open 阶段阈值更严格。

risk_aware:
    接触阶段、靠近物体阶段阈值更严格。
```

第一版建议实现：

```text
strict + gripper phase aware
```

---

## 4.6 Policy 接口

文件：

```text
e3vla/policy/speculative_policy.py
```

职责：

> 将 adapter、feature collector、drafter、verifier 组合成可被环境调用的机器人 policy。

接口：

```text
class SpeculativePolicy:

    reset(task_info)

    act(obs) -> ActionCommand

    get_metrics() -> PolicyMetrics
```

内部逻辑：

```text
act(obs):
    features = adapter.collect_features(obs)
    draft = drafter.forward(features, obs.robot_state)
    result = verifier.verify(obs, draft, adapter)

    if result.accepted_prefix_length > 0:
        return accepted prefix

    else:
        target_action = adapter.full_inference(obs)
        return fallback action
```

注意：

```text
policy 不直接访问模型内部；
所有模型内部访问都通过 adapter；
policy 只关心 draft、verify、fallback。
```

---

# 5. 数据流动设计

## 5.1 Teacher cache 生成流程

目的：

> 离线收集 target VLA 的 action chunk 和 hidden features，用于训练 drafter。

流程：

```text
LIBERO dataset / rollout observation
    ↓
BaseVLAAdapter.full_inference(obs)
    ↓
target action chunk
    ↓
BaseVLAAdapter.collect_features(obs)
    ↓
multi-layer features
    ↓
CacheWriter
    ↓
teacher cache on disk
```

缓存条目：

```text
TeacherCacheRecord:
    record_id
    task_id
    episode_id
    timestep
    instruction
    image_refs
    robot_state
    target_action_chunk
    target_action_latent optional
    target_velocity_anchors optional
    multi_layer_features
    metadata:
        model_name
        layer_spec
        chunk_len
        action_dim
        timestamp
```

建议存储格式：

```text
metadata: jsonl
large tensors: safetensors / npz / pt
images: compressed image files or dataset references
```

---

## 5.2 Drafter 训练数据流

```text
TeacherCacheReader
    ↓
batch:
    multi_layer_features
    robot_state
    target_action_chunk
    optional target_velocity
    optional target_latent
    ↓
Drafter.forward()
    ↓
loss:
    action loss
    velocity loss
    smoothness loss
    uncertainty calibration loss
    ↓
checkpoint
```

训练阶段分三版：

```text
Stage A:
    supervised distillation from target action chunk

Stage B:
    add multi-layer feature fusion

Stage C:
    training-time test / rollout perturbation
```

---

## 5.3 Speculative inference 数据流

```text
obs_t
    ↓
feature collection
    ↓
multi-layer drafter
    ↓
draft action chunk a_d[1:K]
    ↓
target action expert verification
    ↓
accepted prefix a_d[1:m]
    ↓
environment executes prefix
    ↓
obs_{t+m}
    ↓
next inference round
```

fallback 流程：

```text
if m == 0:
    target_action = full_vla_policy(obs_t)
    execute target_action prefix
```

记录日志：

```text
SpeculativeStepLog:
    task_id
    episode_id
    timestep
    mode
    draft_latency
    verification_latency
    full_inference_latency_if_fallback
    accepted_prefix_length
    fallback_required
    errors_per_step
    success_at_episode_end
```

---

# 6. 训练目标设计

第一版不做复杂理论，直接做实用 loss。

## 6.1 普通 action distillation loss

```text
L_action = mean || draft_action - target_action ||_1 or L2
```

## 6.2 temporal smoothness loss

```text
L_smooth = mean || a_{i+1} - a_i ||_2
```

## 6.3 gripper loss

```text
L_gripper = BCE or L1 on gripper dimension
```

## 6.4 velocity consistency loss

如果 target VLA 是 flow-based，可以加入：

```text
L_velocity = mean || draft_velocity - target_velocity ||_2
```

## 6.5 uncertainty calibration loss

drafter 输出每个 step 的 uncertainty：

```text
uncertainty_i ≈ expected verification error_i
```

用于 prefix acceptance。

---

# 7. Verification 设计

## 7.1 最小实现

第一版 verification 不追求优雅，先跑通：

```text
target_action = adapter.full_inference(obs)
err_i = || draft_action_i - target_action_i ||
accept prefix until err_i > threshold
```

缺点：

```text
full_inference 仍然很贵；
只能作为 correctness baseline；
不能体现真正 speculative speedup。
```

## 7.2 正式实现

正式版本要避免完整 full inference，只调用少量 target action expert anchor。

```text
draft action chunk
    ↓
construct noisy / latent state at selected flow timesteps
    ↓
target action expert predicts velocity / endpoint
    ↓
compare reconstructed endpoint with draft endpoint
    ↓
compute prefix errors
```

验证指标：

```text
action error
velocity error
endpoint reconstruction error
gripper phase mismatch
uncertainty confidence
```

---

# 8. 外部评测接口

## 8.1 Rollout Runner

文件：

```text
e3vla/eval/rollout_runner.py
```

职责：

> 在环境中执行 policy，收集完整 episode 结果。

接口：

```text
run_rollout(env, policy, task_spec, max_steps) -> EpisodeResult
```

输出：

```text
EpisodeResult:
    success
    total_steps
    total_wall_time
    average_latency
    control_frequency
    fallback_count
    average_accepted_prefix
    logs
```

---

## 8.2 Latency Profiler

文件：

```text
e3vla/eval/latency_profiler.py
```

需要 profile：

```text
feature collection latency
drafter latency
verification latency
fallback full inference latency
environment step latency
end-to-end act latency
```

输出：

```text
LatencyReport:
    p50
    p90
    p95
    mean
    std
```

---

## 8.3 Report Writer

文件：

```text
e3vla/eval/report_writer.py
```

输出：

```text
tables.csv
metrics.json
latency_breakdown.md
failure_cases.md
plots optional
```

---

# 9. 实验 baseline 设计

第一阶段必须实现：

```text
Full VLA
Reduced-step flow
Small action drafter
Final-layer drafter
Multi-layer fusion drafter
Multi-layer fusion drafter + verifier
```

第二阶段再加入：

```text
FLASH-style plain action drafter + verifier
ProbeFlow if target model is flow-based
EfficientVLA-style cache/pruning if实现成本可控
SaiVLA-style multi-layer action head without verifier
```

最重要的对比不是“是否最快”，而是：

```text
multi-layer fusion 是否比 plain action drafter 更容易被 target verifier 接受；
verification 是否比 draft-only 更稳定；
training-time test 是否降低 fallback rate；
方法是否与 ProbeFlow 互补。
```

---

# 10. 核心指标

必须记录：

```text
success rate
average accepted prefix length
fallback rate
end-to-end latency
action expert latency
drafter latency
verification latency
control frequency
GPU memory
```

建议主表：

```text
Method | Success ↑ | E2E Latency ↓ | Speedup ↑ | Accepted Prefix ↑ | Fallback Rate ↓
```

还要做 latency-success tradeoff：

```text
threshold 越宽：
    speedup 更高
    success 可能下降

threshold 越严：
    success 更稳
    fallback 更多
```

---

# 11. 错误处理策略

## 11.1 Drafter 输出非法

情况：

```text
NaN
Inf
action 超出范围
gripper 值非法
shape mismatch
```

策略：

```text
立即 fallback full VLA；
记录 invalid_draft；
不执行 draft action。
```

## 11.2 Verifier 超时

策略：

```text
超过 timeout_ms：
    fallback full VLA；
    记录 verifier_timeout。
```

## 11.3 Accepted prefix 为 0

策略：

```text
fallback full VLA；
记录 rejected_all。
```

## 11.4 环境 step 失败

策略：

```text
停止当前 episode；
保存完整日志；
标记 env_error。
```

## 11.5 Feature hook 失败

策略：

```text
如果训练阶段：
    直接报错，停止。

如果推理阶段：
    fallback full VLA。
```

---

# 12. 测试策略

## 12.1 单元测试

```text
test_adapter_contract:
    adapter 输出 shape 正确。

test_feature_collection:
    指定 layer 能正确返回 feature。

test_drafter_shapes:
    drafter 输出 action chunk shape 正确。

test_verifier_prefix:
    人工构造 errors，检查 prefix acceptance 正确。

test_policy_fallback:
    verifier 拒绝时，policy 必须调用 full inference。

test_cache_roundtrip:
    teacher cache 写入读取后内容一致。
```

---

## 12.2 集成测试

```text
mock VLA + mock env:
    跑完整 speculative policy，不依赖真实模型。

small VLA + small task:
    跑 5 episodes，检查无 crash。

latency smoke test:
    输出每个模块耗时。
```

---

## 12.3 回归测试

每次改动必须跑：

```text
1. cache roundtrip
2. drafter forward
3. verifier prefix
4. one-episode mock rollout
```

---

# 13. 实施里程碑

## Milestone 1：仓库和接口跑通

目标：

```text
建立项目结构；
实现 BaseVLAAdapter、BaseDrafter、BaseVerifier、SpeculativePolicy 的接口；
用 MockVLA 跑通完整数据流。
```

验收：

```text
mock rollout 可执行；
policy 能在 accepted / fallback 两种路径中切换；
所有 shape 和日志正确。
```

---

## Milestone 2：Full VLA baseline + teacher cache

目标：

```text
接入 SmolVLA 或 OpenPI/π0；
跑通 LIBERO baseline；
生成 teacher cache。
```

验收：

```text
能记录 target action chunk；
能记录多层 features；
能生成 latency profile。
```

---

## Milestone 3：普通 action drafter

目标：

```text
训练 action-level drafter；
评测 draft-only 和 draft+verify。
```

验收：

```text
draft-only 有速度优势；
draft+verify 成功率高于 draft-only；
获得 accepted prefix / fallback rate 统计。
```

---

## Milestone 4：EAGLE-3-style multi-layer fusion drafter

目标：

```text
接入 low / mid / high / action features；
训练 multi-layer fusion drafter；
和 final-layer-only、plain drafter 对比。
```

验收：

```text
multi-layer fusion 的 accepted prefix 更长；
fallback rate 更低；
同等 success 下 latency 更低。
```

---

## Milestone 5：Training-time test

目标：

```text
加入 perturbation 或 rollout-style training；
降低部署时误差累积。
```

验收：

```text
相比 supervised distillation：
    fallback rate 降低；
    accepted prefix 提高；
    success rate 更稳。
```

---

## Milestone 6：完整实验报告

目标：

```text
输出主表、消融表、latency breakdown、failure case。
```

验收：

```text
可以清楚回答：
    为什么不是普通 drafter？
    为什么不是 FLASH-style action drafter？
    为什么不是 ProbeFlow 这类 solver-level acceleration？
    multi-layer feature fusion 是否真的有用？
```

---

# 14. 最小可实现版本

如果工程资源有限，最小版本只做：

```text
SmolVLA / mock flow VLA
LIBERO 一个 task suite
teacher cache
plain action drafter
multi-layer fusion drafter
action L2 verifier
prefix acceptance
fallback full inference
```

先不做：

```text
真实机器人
复杂 closed-loop verifier
多个 VLA backbone
ProbeFlow 复现
EfficientVLA 复现
```

最小版本的成功标准：

```text
multi-layer fusion drafter
相比 plain action drafter：
    accepted prefix 更长；
    fallback rate 更低；
    success rate 不下降；
    latency-success tradeoff 更好。
```

---

# 15. 推荐工程顺序

```text
第 1 步：
    写 BaseVLAAdapter / BaseDrafter / BaseVerifier / SpeculativePolicy 接口。

第 2 步：
    用 MockVLA 跑通所有数据流。

第 3 步：
    接入真实 VLA，跑 full baseline。

第 4 步：
    缓存 teacher action 和 hidden features。

第 5 步：
    训练 plain action drafter。

第 6 步：
    实现 verifier 和 prefix acceptance。

第 7 步：
    实现 multi-layer fusion drafter。

第 8 步：
    加 training-time test。

第 9 步：
    做完整 ablation。
```

---

# 16. 工程交付物

最终工程应交付：

```text
1. 可运行仓库 e3-vla
2. 统一 VLA adapter 接口
3. teacher cache 生成工具
4. drafter 训练脚本
5. speculative policy 推理脚本
6. LIBERO rollout 评测脚本
7. latency profiler
8. ablation report
9. failure case report
10. docs/interfaces.md 和 docs/data_format.md
```

---

# 17. 这个计划的核心边界

本项目第一阶段**不做**：

```text
不训练完整 VLA；
不重写仿真环境；
不做真实机器人部署；
不承诺超过所有加速方法；
不做复杂理论证明。
```

第一阶段只验证一个核心命题：

> **EAGLE-3-style 多层特征融合 drafter，是否比普通 action drafter 更适合 non-AR VLA speculative inference。**

如果这个命题成立，再扩展到更多模型、更多任务、真实机器人和论文级实验。
