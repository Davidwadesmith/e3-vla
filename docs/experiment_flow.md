# E3-VLA 实验流程说明

## AutoDL 环境配置

> 本节供 Agent 在 AutoDL 上自动化执行。如使用其他平台可跳过。

### 存储路径说明

AutoDL 实例有四类存储：

| 路径 | 类型 | 大小 | 特点 |
|------|------|------|------|
| `/` | 系统盘（本地 SSD） | 30 GB | 快，关机保留，可保存为镜像 |
| `/root/autodl-tmp` | 数据盘（本地 SSD） | 50 GB 起可扩容 | 快，关机保留，**实验产物放这里** |
| `/root/autodl-fs` | 文件存储（NAS 网络盘） | 20 GB 免费 | 同地区多实例共享，高可靠，慢 |
| `/root/autodl-pub` | 公共数据 | — | 只读，预置数据集和模型 |

**关键规则**：系统盘 30 GB 只放代码和系统依赖；teacher cache（~25 GB）和 checkpoints（~5 GB）必须放 `/root/autodl-tmp`。关机后所有数据保留，但连续关机 15 天会被自动释放。重要结果建议拷贝一份到 `/root/autodl-fs`。

### 租机参数

| 参数 | 选择 |
|------|------|
| GPU | A100 40G（推荐） / RTX PRO 6000 / A40 48G |
| 镜像 | `PyTorch 2.5+ / CUDA 12.4 / Python 3.12` |
| 数据盘 | 勾选「可扩容」，设 100 GB |
| 计费 | 按量计费（先跑通）→ 确认无误后转包天 |

### 开机后初始化脚本

```bash
# === AutoDL 环境初始化（Agent 从零配置） ===
# 预期：租机完成后，以 root 身份执行

set -e
echo "=== AutoDL E3-VLA 环境初始化 ==="
date

# 1. 验证基础环境
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE'; print(torch.__version__, torch.cuda.get_device_name(0))"
echo "CUDA OK"

# 2. 安装 uv（快速 Python 包管理器）
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi
echo "uv $(uv --version)"

# 3. 克隆项目
cd /root
if [ ! -d e3-vla ]; then
    git clone https://github.com/Davidwadesmith/e3-vla
fi
cd e3-vla

# 4. 创建虚拟环境 + 安装依赖
uv venv
source .venv/bin/activate
uv pip install -e .
pip install libero

# 5. 如需 FLASH 竞品对比（可选）
# uv pip install -e ".[benchmark]"

# 6. 创建数据盘目录（所有大文件放 /root/autodl-tmp）
mkdir -p /root/autodl-tmp/e3vla/{teacher_cache,checkpoints,experiments}

# 7. 软链到项目目录
ln -sf /root/autodl-tmp/e3vla/teacher_cache ./teacher_cache
ln -sf /root/autodl-tmp/e3vla/checkpoints  ./checkpoints
ln -sf /root/autodl-tmp/e3vla/experiments  ./experiments

# 8. 验证
python -m pytest tests/ -q

echo "=== 初始化完成 ==="
echo "数据盘: /root/autodl-tmp/e3vla/"
du -sh /root/autodl-tmp/e3vla/
```

### 下载 π₀ Checkpoint

AutoDL 的公共数据盘 `/root/autodl-pub` 可能已缓存 π₀。先检查：

```bash
# 优先检查公共数据盘
find /root/autodl-pub -name "*pi0*" -o -name "*openpi*" 2>/dev/null

# 如果没有，从 HuggingFace 下载到数据盘
pip install huggingface_hub
huggingface-cli download <pi0_repo> --local-dir /root/autodl-tmp/e3vla/pi0_checkpoints

# 如果下载慢，用学术资源加速
# export HF_ENDPOINT=https://hf-mirror.com
```

### 验证一切就绪

```bash
# 确认以下全部通过
nvidia-smi                         # GPU 可见
python -c "import torch; assert torch.cuda.is_available()"  # CUDA 可用
python -m pytest tests/ -q         # 26 passed
ls /root/autodl-tmp/e3vla/         # 数据盘可写
df -h /root/autodl-tmp             # 数据盘空间充足（≥ 50 GB 剩余）
```

---

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
# 推荐: libero_10（10 tasks × 10 episodes = 100）
uv run python scripts/generate_cache.py \
    env.suite=libero_10 \
    model=openpi \
    model.checkpoint=/path/to/pi0_base \
    output_dir=/root/autodl-tmp/e3vla/teacher_cache \
    max_episodes=100 \
    episodes_per_task=10

# 最小验证: libero_spatial（3 tasks × 10 episodes = 30）
uv run python scripts/generate_cache.py \
    env=libero \
    model=openpi \
    model.checkpoint=/path/to/pi0_base \
    output_dir=/root/autodl-tmp/e3vla/teacher_cache \
    max_episodes=30 \
    episodes_per_task=10
```

**数据量评估**：每 task 至少 10 个 episode（最少 30 总），推荐 100+。详细分析见 `docs/experiment_sizing.md`。

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
    data.cache_dir=/root/autodl-tmp/e3vla/teacher_cache \
    training.epochs=100 \
    checkpoint.save_dir=/root/autodl-tmp/e3vla/checkpoints/no_cached_ae

# Ablation（cached AE，无 offset）
uv run python scripts/train_drafter.py \
    model=cached_ae_no_offset \
    data.cache_dir=/root/autodl-tmp/e3vla/teacher_cache \
    training.epochs=100 \
    checkpoint.save_dir=/root/autodl-tmp/e3vla/checkpoints/no_offset

# Ours（cached AE + offset alignment）
uv run python scripts/train_drafter.py \
    model=default \
    data.cache_dir=/root/autodl-tmp/e3vla/teacher_cache \
    training.epochs=100 \
    checkpoint.save_dir=/root/autodl-tmp/e3vla/checkpoints/full_offset
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
    checkpoint_dir=/root/autodl-tmp/e3vla/checkpoints \
    num_episodes=50 \
    output_dir=/root/autodl-tmp/e3vla/experiments/results
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
# Threshold sweep：只扫 Ours，8 tau × 20 ep，约 2 小时
for tau in 0.05 0.10 0.15 0.20 0.30 0.50 0.70 1.00; do
    uv run python scripts/run_benchmark.py \
        methods=[ours] \
        verifier.tau_radius=$tau \
        num_episodes=20
done

# FLASH 竞品对比（需要先安装 uv pip install -e ".[flash]"）
uv run python scripts/run_benchmark.py \
    benchmark=full_comparison \
    model.checkpoint=/path/to/pi0_base \
    checkpoint_dir=/root/autodl-tmp/e3vla/checkpoints
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
/root/autodl-tmp/e3vla/    ← 数据盘（关机保留，> 50 GB）
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

/root/e3-vla/             ← 系统盘 30 GB（关机保留）
    └── 代码（git clone）
```

---

## 快速命令速查

```bash
# 从头跑通（libero_10, 100 episodes）
uv run python scripts/generate_cache.py \
    env.suite=libero_10 \
    model.checkpoint=/root/autodl-tmp/e3vla/pi0_checkpoints/pi0_base \
    max_episodes=100

# 三个训练并行
for m in no_cached_ae cached_ae_no_offset default; do
    uv run python scripts/train_drafter.py model=$m training.epochs=100 &
done

# benchmark（等训练完）
uv run python scripts/run_benchmark.py \
    model.checkpoint=/root/autodl-tmp/e3vla/pi0_checkpoints/pi0_base \
    seeds="[42,123,456,789,1024,2048,4096,8192,16384,32768]"

# 报告
uv run python scripts/generate_report.py --results .../benchmark_results.json
```

---

## 实验后分析（Agent 自动执行）

实验结束后，按以下顺序检查产物并判断是否成功。

### 1. 检查训练是否有效收敛

```bash
# 查看三个模型的 best val_loss
for m in no_cached_ae no_offset full_offset; do
    ckpt=/root/autodl-tmp/e3vla/checkpoints/$m/best.pt
    if [ -f $ckpt ]; then
        python -c "
import torch
c = torch.load('$ckpt', map_location='cpu', weights_only=False)
print('$m: val_loss =', c.get('val_loss', 'N/A'), 'epoch =', c.get('epoch', 'N/A'))
"
    else
        echo "$m: CHECKPOINT NOT FOUND"
    fi
done
```

**判定**：
- 三个 checkpoint 都存在 → 训练正常
- `val_loss` 在 0.01-0.5 范围 → 正常（取决于 teacher cache 规模）
- `val_loss > 1.0` → 训练未收敛，检查 teacher cache 是否为空或数据是否错误
- checkpoint 缺失 → 训练失败或 early_stop 未触发

### 2. 检查 Benchmark 结果文件

```bash
cat /root/autodl-tmp/e3vla/experiments/results/benchmark_results.json | python -c "
import json, sys
data = json.load(sys.stdin)
results = data.get('results', data if isinstance(data, list) else [])
print(f'Total result entries: {len(results)}')
methods = set(r['method'] for r in results)
print(f'Methods: {methods}')
for m in sorted(methods):
    m_results = [r for r in results if r['method'] == m]
    avg_success = sum(r['success_rate'] for r in m_results) / len(m_results)
    avg_latency = sum(r['avg_latency_ms'] for r in m_results) / len(m_results)
    avg_prefix = sum(r['avg_accepted_prefix'] for r in m_results) / len(m_results)
    avg_fallback = sum(r['fallback_rate'] for r in m_results) / len(m_results)
    print(f'  {m}: success={avg_success:.3f}, latency={avg_latency:.1f}ms, '
          f'prefix={avg_prefix:.1f}, fallback={avg_fallback:.3f}')
"
```

**重要**：如果方法数量 < 3，说明部分 benchmark 失败。检查对应日志。

### 3. Gate 验证（自动判定）

```bash
python -c "
import json, sys

results_path = '/root/autodl-tmp/e3vla/experiments/results/benchmark_results.json'
try:
    with open(results_path) as f:
        data = json.load(f)
    results = data.get('results', data if isinstance(data, list) else [])
except FileNotFoundError:
    print(f'ERROR: {results_path} not found — benchmark may have failed')
    sys.exit(1)

if not results:
    print('ERROR: benchmark results are empty')
    sys.exit(1)

# Group by method
by_method = {}
for r in results:
    by_method.setdefault(r['method'], []).append(r)

def avg(key, rlist):
    return sum(r[key] for r in rlist) / len(rlist) if rlist else 0

# Initialize all diff variables to safe defaults
g1_pass, g2_pass, g4_pass = False, False, False
g1_diff, g2_pdiff, g2_fdiff, g4_pdiff, g4_sdiff = 0, 0, 0, 0, 0
g1_ok, g2_ok, g4_ok = False, False, False

# Find methods
no_ae   = by_method.get('NoCachedAE (FLASH baseline)', [])
no_off  = by_method.get('CachedAE-NoOffset', [])
ours    = by_method.get('CachedAE-FullOffset (Ours)', [])
reuse   = by_method.get('Cached Full Action Reuse', [])

print('=== Gate Validation ===')
print()

# G1: CachedAE-NoOffset vs NoCachedAE
if no_ae and no_off:
    g1_ok = True
    g1_diff = avg('avg_accepted_prefix', no_off) - avg('avg_accepted_prefix', no_ae)
    g1_pass = g1_diff > 0.5
    print(f'G1: CachedAE-NoOffset vs NoCachedAE')
    print(f'    NoCachedAE prefix: {avg(\"avg_accepted_prefix\", no_ae):.1f}')
    print(f'    NoOffset prefix:   {avg(\"avg_accepted_prefix\", no_off):.1f}')
    print(f'    diff: {g1_diff:+.1f}  [threshold: >0.5]')
    print(f'    -> {\"PASS\" if g1_pass else \"FAIL\"}' + 
          (' — cached AE features improve drafting' if g1_pass else ' — CORE HYPOTHESIS REJECTED'))
else:
    print(f'G1: SKIP — missing NoCachedAE ({len(no_ae)} results) or CachedAE-NoOffset ({len(no_off)} results)')
print()

# G2: CachedAE-FullOffset vs CachedAE-NoOffset
if no_off and ours:
    g2_ok = True
    g2_pdiff = avg('avg_accepted_prefix', ours) - avg('avg_accepted_prefix', no_off)
    g2_fdiff = avg('fallback_rate', no_off) - avg('fallback_rate', ours)
    g2_pass = g2_pdiff > 0.5 or g2_fdiff > 0.02
    print(f'G2: CachedAE-FullOffset vs CachedAE-NoOffset')
    print(f'    NoOffset prefix: {avg(\"avg_accepted_prefix\", no_off):.1f}, fallback: {avg(\"fallback_rate\", no_off):.3f}')
    print(f'    Ours prefix:     {avg(\"avg_accepted_prefix\", ours):.1f}, fallback: {avg(\"fallback_rate\", ours):.3f}')
    print(f'    prefix diff: {g2_pdiff:+.1f}, fallback diff: {g2_fdiff:+.3f}')
    print(f'    -> {\"PASS\" if g2_pass else \"FAIL\"}' +
          (' — offset alignment adds value' if g2_pass else ' — offset alignment has no effect'))
else:
    print(f'G2: SKIP — missing NoOffset ({len(no_off)} results) or Ours ({len(ours)} results)')
print()

# G4: Ours vs CachedFullActionReuse
if ours and reuse:
    g4_ok = True
    g4_pdiff = avg('avg_accepted_prefix', ours) - avg('avg_accepted_prefix', reuse)
    g4_sdiff = avg('success_rate', ours) - avg('success_rate', reuse)
    g4_pass = g4_pdiff > 1.0 or g4_sdiff > 0.03
    print(f'G4: Ours vs CachedFullActionReuse')
    print(f'    Reuse prefix: {avg(\"avg_accepted_prefix\", reuse):.1f}, success: {avg(\"success_rate\", reuse):.3f}')
    print(f'    Ours prefix:  {avg(\"avg_accepted_prefix\", ours):.1f}, success: {avg(\"success_rate\", ours):.3f}')
    print(f'    prefix diff: {g4_pdiff:+.1f}, success diff: {g4_sdiff:+.3f}')
    print(f'    -> {\"PASS\" if g4_pass else \"FAIL\"}' +
          (' — method complexity justified' if g4_pass else ' — not better than simple action reuse'))
else:
    print(f'G4: SKIP — missing Ours ({len(ours)} results) or Reuse ({len(reuse)} results)')

# Summary and next steps
print()
print('=== Summary ===')
all_evaluated = g1_ok and g2_ok and g4_ok
print(f'Gates evaluable: G1={g1_ok}, G2={g2_ok}, G4={g4_ok}')
if all_evaluated:
    print(f'Gates passed:    G1={g1_pass}, G2={g2_pass}, G4={g4_pass}')
    print()
    if not g1_pass:
        print('CRITICAL: Core hypothesis rejected.')
        print('  cached AE features do not improve draft quality.')
        print('  Next: try CachedVLMFeature, or reconsider approach.')
    elif not g2_pass:
        print('Offset alignment does not add value.')
        print('  Use CachedAE-NoOffset as the main method.')
        print('  Paper contribution: cached AE features improve speculative VLA drafting.')
    elif not g4_pass:
        print('Method not better than simple action reuse.')
        print('  Paper contribution needs to be carefully scoped.')
        print('  Consider: only claim benefit in high-variance or long-horizon tasks.')
    else:
        print('All gates passed. Full steam ahead.')
        print('  Run full_comparison benchmark and start writing.')
else:
    missing = []
    if not g1_ok: missing.append('NoCachedAE or CachedAE-NoOffset')
    if not g2_ok: missing.append('Ours or CachedAE-NoOffset')
    if not g4_ok: missing.append('CachedFullActionReuse or Ours')
    print(f'Cannot fully evaluate — missing results for:')
    for m in missing: print(f'  - {m}')
    print('Re-run benchmark with the missing methods.')
"
```

### 4. 检查训练 loss 曲线（如果有 wandb）

```bash
# 如果使用了 wandb online 模式
python -c "
import wandb
api = wandb.Api()
runs = api.runs('e3vla')
for run in runs:
    print(f'{run.name}: {run.state}, val_loss={run.summary.get(\"val/l_total\", \"N/A\")}')
"
```

如果没有 wandb，从 checkpoint 直接读取：

```bash
for m in no_cached_ae no_offset full_offset; do
    echo "=== $m ==="
    python -c "
import torch
c = torch.load('/root/autodl-tmp/e3vla/checkpoints/$m/best.pt', map_location='cpu', weights_only=False)
for k, v in c.items():
    if isinstance(v, (int, float)):
        print(f'  {k}: {v}')
"
done
```

### 5. 结果文件完整性检查

```bash
echo "=== Output Inventory ==="
echo "Teacher cache:"
du -sh /root/autodl-tmp/e3vla/teacher_cache/
ls /root/autodl-tmp/e3vla/teacher_cache/metadata/records.jsonl && echo "  metadata OK" || echo "  metadata MISSING"
ls /root/autodl-tmp/e3vla/teacher_cache/features/shard_0000.safetensors && echo "  features OK" || echo "  features MISSING"

echo ""
echo "Checkpoints:"
for m in no_cached_ae no_offset full_offset; do
    ckpt=/root/autodl-tmp/e3vla/checkpoints/$m/best.pt
    [ -f $ckpt ] && echo "  $m: $(du -h $ckpt | cut -f1)" || echo "  $m: MISSING"
done

echo ""
echo "Results:"
ls /root/autodl-tmp/e3vla/experiments/results/benchmark_results.json && echo "  results OK" || echo "  results MISSING"

echo ""
echo "Report:"
ls /root/autodl-tmp/e3vla/experiments/report/main_table.csv && echo "  main_table OK" || echo "  report not yet generated"
```

### 6. 常见异常及处理

| 现象 | 可能原因 | 处理 |
|------|---------|------|
| `val_loss > 1.0` | 训练未收敛 | 增加 epochs，检查 lr |
| 三个模型 loss 几乎相同 | Teacher cache 特征无区分力 | 检查 extract_ae_features 是否正确 hook |
| `accepted_prefix` 始终 < 3 | tau 太严或 drafter 质量差 | 放宽 tau，检查 drafter loss |
| `fallback_rate > 0.5` | drafter 频繁被拒绝 | 放宽 tau_radius，降低 periodic_full_every_n |
| `success_rate` 在所有方法都很低 | 环境或模型问题 | 先跑 full_vla baseline 确认基础成功率 |
| benchmark 只有 1-2 个方法的结果 | 部分 policy 构建失败 | 检查 checkpoint 路径，看 stderr |
| results 文件为空或不存在 | benchmark 脚本崩溃 | 手动跑单 method: `methods=[ours]` |
| `speedup < 1.0` | speculative 比 full VLA 还慢 | 增加 full_exec_len，减少 verifier 开销 |

