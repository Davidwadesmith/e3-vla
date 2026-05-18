# E3-VLA 超参数分析与调优建议

## 已应用的修改

| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| `policy.full_exec_len` | 2 | 6 | 减少 full VLA 成本浪费（87.5% → 62.5%） |
| `acceptor.alpha_uncert` | 1.0 | 20.0 | 让 uncertainty 机制真正影响 threshold |
| `loss.lambda_gripper` | 0.5 | 0.2 | 平衡 gripper loss，避免过度关注 1/7 维度 |

---

## 模型架构超参数

### hidden_dim: 512 ✅

50M 参数 drafter，512 维隐藏层。与 FLASH 的 ~256 相比偏大，但对于 PRO 6000 的 96GB 显存来说不是问题。如果后续需要极致轻量（如部署到 8GB 卡），可降到 256。

### num_layers: 2 ✅

轻量 transformer decoder 层数。2 层足以建模 K 个 action step 之间的时序依赖（16 个 step）。FLASH 用 1 层 Gemma block，我们用 2 层更轻的自定义 decoder，参数更少但层数更多。1-4 层均可接受。

### action_history_len: 5 ⚠️

5 × 16 × 7 = 560 维历史向量。LIBERO 动作平滑度高，2-3 步历史通常足够。当前值不影响正确性但增加参数量。可降到 2-3，但不是紧急项。

### use_image_feature: false ✅

MVP 阶段不需要图像特征。cached AE 特征已足够。后续 ablation 可打开验证。

### mixer_mode: weighted ✅

可学习标量权重池化三种 AE 层级特征。比简单 mean 多 3 个参数但有更好的表达能力。attention 模式更复杂但收益不确定，暂时保留 weighted。

---

## 训练超参数

### lr: 2.0e-3 ⚠️

50M transformer 的 AdamW 学习率偏高（标准范围 1e-4 到 1e-3）。FLASH 也使用 2e-3，但他们的模型架构不同（单层 Gemma block）。

**风险**：早期训练可能震荡，loss 曲线出现 spikes。

**监控方法**：跑 5 epoch 后检查 loss 曲线。如果出现：
- 突然跳高 → 降到 1e-3
- 平稳下降 → 保持 2e-3

500 步 warmup 提供了一定缓冲。

### warmup_steps: 500 ✅

对于 5750 样本 / batch_size 32 ≈ 180 步/epoch，500 步 ≈ 2.8 个 epoch。在 100 epoch 训练中，warmup 占 ~3%，合理。

### batch_size: 32 ⚠️

PRO 6000 96GB 上偏小。Drafter 只有 50M 参数，每个 sample 约 1MB（512-dim × 4 AE levels + action + state）。batch_size=32 时 GPU 显存只用了 ~4GB。

**建议**：PRO 6000 上可拉到 128-256。好处：
- 更稳定的梯度估计
- 更高的 GPU 利用率（当前 23% → 60-80%）
- 每 epoch 步数减少，训练更快

### num_workers: 4 ✅

基本够用。如果 IO 成为瓶颈（safetensors 读取慢），可升到 8。

### max_delta: 50 ❓

训练对中 cached feature 与 target action 的最大时间跨度。50 步意味着训练覆盖了 inference 中的最大 cache_age。

**与 inference 的对齐**：`periodic_full_every_n: 10`，假设平均 accept 5 步，10 × 5 = 50 步的 cache_age。与 max_delta 一致。

**风险**：大 delta 意味着 cached AE feature 与 target action 之间的相关性弱。如果 Δ=50 的对在训练数据中占很大比例，模型可能学不到有用的 cached AE→action 映射。

**建议**：真实数据上验证不同 max_delta (20 vs 35 vs 50) 对 val_loss 的影响。

### epochs: 100 ✅

5750 样本的 CrossTemporalDataset 会产生大量训练对（每个 episode 约 C(300,2) ≈ 45000 对）。100 epoch 足够。

---

## 损失权重

### lambda_smooth: 0.1 ✅

时序平滑作为正则化项，不应主导 loss。0.1 合理。

### lambda_gripper: 0.2（已修改） ✅

Gripper 是 7 维 action 中的 1 维。旧值 0.5 意味着 gripper loss 权重过高——会导致模型牺牲 6 维连续 action 精度来拟合 1 维二值 gripper。0.2 更平衡。

### lambda_uncert: 0.1 ✅

不确定性校准是辅助目标。0.1 合理。

### lambda_align: 0.0（MVP 禁用） ✅

对齐 loss 需要额外的监督信号（target AE feature at t+Δ），MVP 阶段不开。

---

## 数据增强（Staleness Training）

| 参数 | 值 | 评估 |
|------|-----|------|
| `cache_age_jitter: 2` | ±2 步 | ✅ 合理 |
| `pose_noise_std: 0.01` | 1cm | ✅ 对于 delta_ee_pose (m) 合理 |
| `feature_dropout: 0.1` | 10% | ✅ 合理 |
| `gripper_flip_prob: 0.0` | 禁用 | 后续可尝试 0.05 |

---

## Verifier 超参数

### t_list: [0.10, 0.05] ✅

与 FLASH 一致。near-x0 的 flow timestep，速度场对 endpoint 的信息量最大。两个 timestep 的并行验证成本低。

### tau_radius: 0.3 ❓

FLASH 的原始值。RMS 归一化后 L2 距离阈值。这个值高度依赖模型和任务——必须在真实 LIBERO 数据上扫参：
```bash
for tau in 0.1 0.2 0.3 0.5 0.7 1.0; do
    uv run python scripts/run_benchmark.py verifier.tau_radius=$tau
done
```
画 `speedup vs success` 曲线找最优 tradeoff 点。

### eval_h: 12 ✅

验证前 12 步（共 16 步）。后期 step 误差大、可靠性低，不验证是合理的。

---

## Policy 超参数

### full_exec_len: 6（已修改） ✅

| 值 | full VLA 成本浪费 | 说明 |
|----|-------------------|------|
| 2（旧） | 87.5%（16 步只用 2 步） | 过于激进 |
| 6（新） | 62.5% | 平衡 |
| 10 | 37.5% | 保守 |
| 16 | 0%（永不进入 speculative） | 无加速 |

选 6 是因为在不过度浪费 VLA 成本的前提下尽快进入 speculative path。

### max_cache_age: 50 ✅

缓存 50 步后强制 refresh。与 `periodic_full_every_n: 10 × avg_accept: 5 = 50` 对齐。

### periodic_full_every_n: 10 ⚠️

每 10 轮 speculative 强制一次 full refresh。与 max_cache_age 共同约束 refresh 频率。实际上这个参数主导了 refresh 行为——因为大多数情况下 10 轮会先于 50 步触发。

**建议**：真实数据上验证不同值（10 vs 20 vs 30）对 speedup 和 success 的影响。

### gripper_full_window: 3 ✅

夹爪切换后连续 3 轮 full refresh，保证接触/放置阶段的稳定性。

---

## Prefix Acceptor 超参数

### tau_pos: 0.01 / tau_rot: 0.05 / tau_grip: 0.1 ❓

这些阈值高度依赖模型和任务。必须从真实 teacher cache 中统计 target action 的 per-dim 方差来校准。

**校准方法**：
```python
# 从 teacher cache 统计
pos_std = target_actions[:, :, :3].std()
rot_std = target_actions[:, :, 3:6].std()
grip_std = target_actions[:, :, 6].std()

# 建议阈值: tau = c * std, c ∈ [0.1, 0.5]
```

### alpha_uncert: 20.0（已修改） ✅

旧值 1.0 时，uncertainty 的影响微乎其微。公式 `tau / (1 + alpha * u)`：
- alpha=1.0: `0.01 / (1 + 0.05) = 0.0095`（-5%）
- alpha=20: `0.01 / (1 + 20*0.05) = 0.01/2 = 0.005`（-50%）

20 倍让 uncertainty 对 threshold 产生显著影响。

### gripper_phase_tighten: 0.5 ✅

夹爪切换处 threshold 减半。合理。

---

## 需要真实数据校准的参数（优先级排序）

| 优先级 | 参数 | 校准方法 |
|--------|------|---------|
| P0 | `tau_radius` | 扫 0.1-1.0，画 speedup-success 曲线 |
| P0 | `tau_pos / tau_rot / tau_grip` | 从 teacher cache 统计 action std |
| P1 | `max_delta` | 不同值训练，比较 val_loss |
| P1 | `lr` | 看 loss 曲线是否平稳 |
| P1 | `periodic_full_every_n` | 不同值 benchmark，看 speedup |
| P2 | `batch_size` | PRO 6000 上拉到 128+ |
| P2 | `action_history_len` | 2 vs 3 vs 5 |
