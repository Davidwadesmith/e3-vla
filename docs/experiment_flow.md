# E3-VLA 实验流程说明

## 需要什么

### 硬件

| 项目 | 最低 | 推荐 |
|------|------|------|
| GPU | 24 GB VRAM | 40+ GB (A100) |
| RAM | 32 GB | 64 GB |
| 存储 | 50 GB | 100 GB |

### 数据

| 数据 | 来源 | 大小 |
|------|------|------|
| OpenPI/π₀ checkpoint | [ultra-robotics/openpi](https://github.com/ultra-robotics/openpi) | ~6 GB |
| LIBERO 环境 | `pip install libero` | ~2 GB |
| Teacher cache | 自己生成（脚本见下） | 15-30 GB / 100 episodes |

不需要外部标注数据，不需要人工采集。

### 软件

```bash
Python 3.10+ | CUDA 12.4+ | PyTorch 2.5+
uv (Python 包管理器)
```

## 怎么操作

### Step 0：环境初始化（一次性）

```bash
# 克隆仓库
git clone https://github.com/Davidwadesmith/e3-vla
cd e3-vla

# 创建虚拟环境
uv venv && source .venv/bin/activate

# 安装依赖
uv pip install -e .
pip install libero

# 如需 FLASH 竞品对比
uv pip install -e ".[benchmark]"

# 验证
python -m pytest tests/ -v   # 26 passed
python -c "import torch; assert torch.cuda.is_available()"
```

### Step 1：生成 Teacher Cache（2-4 小时）

```bash
uv run python scripts/generate_cache.py \
    env=libero \
    model=openpi \
    model.checkpoint=/path/to/pi0_base \
    output_dir=/root/autodl-fs/teacher_cache \
    max_episodes=100 \
    episodes_per_task=10
```

**这个过程做什么：**
1. 加载 π₀ 模型到 GPU
2. 逐个 episode 跑 LIBERO 环境
3. 每个 timestep：π₀ 完整推理 → 记录 action chunk + AE 中间层特征 → 写入磁盘
4. 输出 safetensors + JSONL 到 `teacher_cache/`

**产出：** `teacher_cache/metadata/records.jsonl` + `features/` + `actions/` + `states/`

### Step 2：训练 Drafter（每个 1-3 小时）

```bash
# Baseline（无 cached AE）
uv run python scripts/train_drafter.py \
    model=no_cached_ae \
    data.cache_dir=/root/autodl-fs/teacher_cache \
    training.epochs=100 \
    checkpoint.save_dir=/root/autodl-fs/checkpoints/no_cached_ae

# Ablation（cached AE，无 offset）
uv run python scripts/train_drafter.py \
    model=cached_ae_no_offset \
    data.cache_dir=/root/autodl-fs/teacher_cache \
    training.epochs=100 \
    checkpoint.save_dir=/root/autodl-fs/checkpoints/no_offset

# Ours（cached AE + offset alignment）
uv run python scripts/train_drafter.py \
    model=default \
    data.cache_dir=/root/autodl-fs/teacher_cache \
    training.epochs=100 \
    checkpoint.save_dir=/root/autodl-fs/checkpoints/full_offset
```

**这个过程做什么：**
1. 读 teacher cache 中的跨时间对（t0 的 AE feature → t0+Δ 的 target action）
2. 训练 50M 参数的轻量 drafter
3. Loss: Huber action + temporal smoothness + gripper BCE + uncertainty calibration

**产出：** 三个 `best.pt` checkpoint 到 `checkpoints/`

### Step 3：Benchmark 评测（1-3 小时）

```bash
uv run python scripts/run_benchmark.py \
    benchmark=ablation \
    model.checkpoint=/path/to/pi0_base \
    checkpoint_dir=/root/autodl-fs/checkpoints \
    num_episodes=50 \
    output_dir=/root/autodl-fs/experiments/results
```

**这个过程做什么：**
1. 加载 π₀ 模型 + 训练好的 drafter checkpoint
2. 在 LIBERO 环境上跑每个方法 50 个 episode
3. 记录 success rate、latency per step、accepted prefix length、fallback rate、per-component 延迟拆解

**产出：** `experiments/results/benchmark_results.json`

### Step 4：生成报告（< 1 分钟）

```bash
uv run python scripts/generate_report.py \
    --results experiments/results/benchmark_results.json \
    --output experiments/report/
```

**产出：**
- `main_table.csv` — 主对比表
- `main_table.md` — 论文用 markdown 表格
- `latency_breakdown.csv` — 组件级延迟拆解
- `metrics.json` — 完整指标 JSON

### 事后分析（Step 5-7，可选）

```bash
# Threshold sweep：扫 tau_radius 画 speedup-success 曲线
for tau in 0.1 0.2 0.3 0.5 0.7 1.0; do
    uv run python scripts/run_benchmark.py \
        verifier.tau_radius=$tau \
        checkpoint_dir=/root/autodl-fs/checkpoints \
        output_dir=/root/autodl-fs/experiments/sweep_${tau}
done

# FLASH 竞品对比（需要先安装 uv pip install -e ".[flash]"）
uv run python scripts/run_benchmark.py \
    benchmark=full_comparison \
    model.checkpoint=/path/to/pi0_base \
    checkpoint_dir=/root/autodl-fs/checkpoints
```

---

## 实验成功判定

| Gate | 验证 | 标准 |
|------|------|------|
| G1 | CachedAE-NoOffset vs NoCachedAE | `accepted_prefix_len` 更长 |
| G2 | CachedAE-FullOffset vs CachedAE-NoOffset | `accepted_prefix_len` 或 `fallback_rate` 进一步改善 |
| G3 | Ours vs FLASH-style baseline | success-latency Pareto 有正位移 |
| G4 | Ours vs CachedActionReuse | 显著超过简单 reuse baseline |

**G1 不通过 → 核心假设失败，换方向。**
**G1 通过但 G2 失败 → 只用 NoOffset 作为主方法。**

---

## 目录布局总结

```
/root/autodl-fs/           ← 数据盘（关机保留）
├── teacher_cache/         ← Step 1 产出
│   ├── metadata/records.jsonl
│   ├── features/*.safetensors
│   ├── actions/*.safetensors
│   └── states/*.safetensors
├── checkpoints/           ← Step 2 产出
│   ├── no_cached_ae/best.pt
│   ├── no_offset/best.pt
│   └── full_offset/best.pt
├── experiments/
│   ├── results/           ← Step 3 产出
│   │   └── benchmark_results.json
│   └── report/            ← Step 4 产出
│       ├── main_table.csv
│       ├── main_table.md
│       ├── latency_breakdown.csv
│       └── metrics.json
└── pi0_checkpoints/       ← π₀ 模型（手动下载）

/root/e3-vla/             ← 系统盘（关机清空）
    └── 代码（git clone）
```

---

## 快速命令速查

```bash
# 从头跑通（假设 π₀ checkpoint 在 /root/autodl-fs/pi0_checkpoints）
uv run python scripts/generate_cache.py model.checkpoint=/root/autodl-fs/pi0_checkpoints/pi0_base max_episodes=100

# 三个训练可以并行
uv run python scripts/train_drafter.py model=no_cached_ae training.epochs=100 &
uv run python scripts/train_drafter.py model=cached_ae_no_offset training.epochs=100 &
uv run python scripts/train_drafter.py model=default training.epochs=100 &

# benchmark（等训练结束后）
uv run python scripts/run_benchmark.py model.checkpoint=/root/autodl-fs/pi0_checkpoints/pi0_base

# 报告
uv run python scripts/generate_report.py --results .../benchmark_results.json
```
