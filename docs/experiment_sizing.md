# 实验规模评估：数据量与训练轮数

## 训练数据：30 episodes 够不够？

### 数据量推算

```
libero_spatial: 3 tasks × 10 episodes/task = 30 episodes
每 episode: ~250 timesteps
总 timesteps: 30 × 250 = 7,500

CrossTemporalDataset 对生成（max_delta=50）:
  每 episode ~11,500 个 (t0, t0+Δ) 对
  30 episodes = 345,000 对
  val_frac=0.1 → 训练 27 episodes = 310,000 对
  batch_size=32 → ~9,700 step/epoch
```

### 与竞品对比

| 项目 | 数据量 | 模型大小 | 样本:参数比 |
|------|--------|---------|------------|
| FLASH | ~100 episodes（估计） | 110M | ~100:1 |
| EAGLE-3 (LLM) | 数十万条文本 | ~200M | ~1000:1 |
| **E3-VLA (当前)** | **30 episodes, 27 train** | **3M** | **100:1** |

样本:参数比（310k : 3M ≈ 100:1）在合理范围。但关键不是总量——是**多样性**。

### 多样性不足

```
27 train episodes ÷ 3 tasks = 每 task 9 个 episode
9 个 episode = 9 种不同的物体初始位姿

Drafter 需要学习的是:
  cached AE feature (from t0) → action chunk (at t0+Δ)

问题: 9 种初始位姿够不够覆盖 LIBERO 的分布？
  - Pick-place: 物体在桌面不同位置 → 9 个位置勉强覆盖
  - Put-in: 物体+容器组合 → 9 种组合太少
  - 如果测试 episode 的物体位置不在训练分布内，drafter 会失效
```

**结论：30 episodes 不够。建议至少 100 episodes（每 task 30-35 集）。**

| | 30 episodes | 100 episodes |
|---|-------------|--------------|
| 每 task 训练集 | 9 集 | 30 集 |
| 训练对总量 | 310k | ~1M |
| 模型泛化风险 | 高（可能记住轨迹） | 中等 |
| Teacher cache 大小 | ~8 GB | ~25 GB |
| 生成时间（A100） | ~1 小时 | ~3 小时 |

---

## 训练轮数：100 epochs 够不够？

### 收敛分析

```
310k pairs / batch_size=32 = 9,700 step/epoch
100 epochs = 970,000 training steps

50M 参数的 drafter，学习率 2e-3，AdamW:
  - 小模型（3M 参数）在 310k 数据上收敛很快
  - 预期 30-50 epoch 内 val_loss 到达平台
  - early_stop_patience=5 会在平台后 5 epoch 自动停止
```

100 epoch **不是不够，是多了**。实际训练会在 30-50 epoch 被 early_stop 截断。

### 模拟实验验证（autodl 合成数据）

```
5750 timesteps / batch=32 ≈ 180 step/epoch
100 epoch 总步数: 18,000

3 个模型 5 epoch 训练结果:
  NoCachedAE:      val_loss 0.2829
  CachedAE-NoOffset: val_loss 0.2830
  CachedAEDrafter:   val_loss 0.2832

5 epoch 已完成大部分收敛（合成数据简单）
```

合成数据上 5 epoch 就收敛了，因为 mock AE feature 是 action chunk 的线性投影，任务太简单。真实数据上需要更多。

**结论：100 epoch 偏多但无害（early_stop 会截断）。如果调到 300+ episodes，可以设 200 epoch。**

---

## Benchmark 评测：50 episodes 够不够？

### 统计显著性

```
成功率为二项分布，n=50:
  80% success → 95% CI: [0.69, 0.91]（±11%）
  85% success → 95% CI: [0.75, 0.95]（±10%）

两个方法的 95% CI 大量重叠:
  Method A 85%, Method B 80% → 几乎无法区分
  Method A 85%, Method B 70% → 勉强可区分
```

### 实际可行性

| 场景 | 50 episode | 100 episode |
|------|-----------|-------------|
| 区分 5% success 差异 | ❌ 不可行 | ❌ 仍然难 |
| 区分 10% success 差异 | ⚠️ 勉强 | ✅ 可行 |
| 区分 15%+ 差异 | ✅ 可行 | ✅ 可靠 |
| 论文标准 | 可接受（多数 VLA 论文用 20-50） | 更稳妥 |
| 3 task × 50 ep × 3 method 耗时 | ~2-4 小时 | ~4-8 小时 |

### 关于 seeds

当前 `seeds: [42, 123, 456, 789, 1024]`，5 个种子，总共 50 episode = 每种子 10 集。

```
5 seeds × 10 episodes = 50
vs
10 seeds × 10 episodes = 100

种子太少的问题：
- 个别 outlier seed 会严重影响均值
- 无法可靠估计 seed-to-seed variance
- Reviewer 可能质疑结果的稳定性
```

**结论：50 episode 在论文中可接受。但要配合 ≥10 个 seeds 才能控制方差。当前 5 个不够。**

---

## 修正建议

| 参数 | 当前 | 建议 | 理由 |
|------|------|------|------|
| 训练 episodes | 30 | **≥100** | 每 task 只有 9 集，多样性不足 |
| 训练 epochs | 100 | **100（保持）** | early_stop 会截断，写 100 无害 |
| Benchmark episodes | 50 | **50（保持）** | 论文可接受 |
| Seeds | 5 | **≥10** | 控制 seed-to-seed 方差 |
| task suite | `libero_spatial` (3 tasks) | `libero_10` 或 `libero_spatial + object + goal` | 更多 task 证明泛化性 |
| Threshold sweep 范围 | 6 个 tau × 3 method × 50 ep | **1 个 method (Ours) × 8 个 tau × 20 ep** | 50h → 4h，低端加 0.05/0.15 |

### 推荐的实验配置

```bash
# 1. 收集更多数据（libero_10: 10 tasks × 10 episodes = 100）
uv run python scripts/generate_cache.py \
    env.suite=libero_10 \
    max_episodes=100 \
    episodes_per_task=10

# 2. 训练（不变）
uv run python scripts/train_drafter.py model=default training.epochs=100

# 3. Benchmark：更多 seeds
uv run python scripts/run_benchmark.py \
    benchmark=ablation \
    seeds="[42,123,456,789,1024,2048,4096,8192,16384,32768]" \
    num_episodes=50

# 4. Threshold sweep：只扫 Ours，减少 episode
for tau in 0.05 0.10 0.15 0.20 0.30 0.50 0.70 1.00; do
    uv run python scripts/run_benchmark.py \
        methods=[ours] \
        verifier.tau_radius=$tau \
        num_episodes=20
done
```
