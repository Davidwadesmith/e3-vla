# AutoDL 实验环境搭建指南

## 1. 租机配置

| 选项 | 推荐 | 备选 |
|------|------|------|
| GPU | A100 40G | A40 48G / A100 80G |
| 镜像 | PyTorch 2.5+ / CUDA 12.4 / Python 3.12 | 社区镜像 > torch2.0 |
| 数据盘 | 50 GB 扩容盘 | 100 GB（多 task suite） |
| 时长 | 按量计费，先租几小时验证环境 | 确认无误后包天 |

> RTX 4090（24G）不够同时加载 π₀ + FLASH + Ours，不推荐。

## 2. 开机后初始化

```bash
# 1. 验证 GPU 和 CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.__version__, torch.cuda.get_device_name(0))"
# 输出必须: True 2.x NVIDIA A100-SXM4-40GB

# 2. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# 3. Clone 项目
cd /root
git clone https://github.com/Davidwadesmith/e3-vla
cd e3-vla

# 4. 创建 venv + 安装依赖
uv venv
source .venv/bin/activate
uv pip install -e .

# 如需 bench 对比，加装竞品依赖
uv pip install -e ".[benchmark]"

# 5. 确认核心依赖版本
python -c "import torch; import safetensors; import hydra; print('OK')"
```

## 3. 数据盘挂载与目录布局

```bash
# autodl 数据盘默认路径为 /root/autodl-fs
# 将大文件目录软链到数据盘

mkdir -p /root/autodl-fs/teacher_cache
mkdir -p /root/autodl-fs/checkpoints
mkdir -p /root/autodl-fs/experiments
mkdir -p /root/autodl-fs/wandb

ln -s /root/autodl-fs/teacher_cache ./teacher_cache
ln -s /root/autodl-fs/checkpoints  ./checkpoints
ln -s /root/autodl-fs/experiments  ./experiments
```

系统盘 (`/root/e3-vla/`) 只放代码；所有产出（cache、checkpoint、结果）全部落在数据盘。关机后系统盘清空，数据盘保留。

## 4. 上传 Teacher Cache

如果本地已有 teacher cache，上传到数据盘：

```bash
# 方式 A: scp（本机执行）
scp -rP <ssh_port> ./teacher_cache root@<autodl_ip>:/root/autodl-fs/

# 方式 B: autodl 控制台的文件传输功能
# 方式 C: 先 tar 再传
tar -czf teacher_cache.tar.gz teacher_cache/
scp -P <ssh_port> teacher_cache.tar.gz root@<ip>:/root/autodl-fs/
# 服务器上解压
cd /root/autodl-fs && tar -xzf teacher_cache.tar.gz
```

如果没预生成 cache，在服务器上跑 `scripts/generate_cache.py` 先采集：

```bash
python scripts/generate_cache.py \
    env=libero \
    model=openpi \
    output_dir=/root/autodl-fs/teacher_cache \
    max_episodes=200
```

## 5. 验证环境可运行

```bash
# 单元测试
python -m pytest tests/ -v

# GPU 内存压力测试（确保 VRAM 够）
python -c "
import torch
x = torch.randn(1024, 1024, 64, device='cuda')
print(f'Allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB')
print(f'Max: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

## 6. 跑实验

```bash
# Step 1: NoCachedAE baseline（验证 Gate G1）
uv run python scripts/train_drafter.py \
    model=no_cached_ae \
    data.cache_dir=/root/autodl-fs/teacher_cache \
    training.epochs=100 \
    checkpoint.save_dir=/root/autodl-fs/checkpoints/no_cached_ae

# Step 2: CachedAE-NoOffset（验证 cached AE 是否有增益）
uv run python scripts/train_drafter.py \
    model=cached_ae_no_offset \
    data.cache_dir=/root/autodl-fs/teacher_cache \
    training.epochs=100 \
    checkpoint.save_dir=/root/autodl-fs/checkpoints/cached_ae_no_offset

# Step 3: CachedAE-FullOffset（Ours，验证 Gate G2）
uv run python scripts/train_drafter.py \
    model=default \
    data.cache_dir=/root/autodl-fs/teacher_cache \
    training.epochs=100 \
    checkpoint.save_dir=/root/autodl-fs/checkpoints/ours
```

## 7. 注意事项

### 7.1 关机前检查

```bash
# 确认所有产出在数据盘，不在系统盘
ls /root/autodl-fs/checkpoints/
ls /root/autodl-fs/experiments/
ls /root/autodl-fs/wandb/
```

### 7.2 不要用 pip 覆盖 autodl 自带的 PyTorch

```bash
# 错误：可能装到 CPU 版
# pip install torch  ❌

# 正确：让 uv pip 自动 resolve 已安装的 CUDA 版 torch
uv pip install -e .   # 不会重装已有的 torch
```

### 7.3 无外网时 wandb 离线模式

```bash
export WANDB_MODE=offline
# 日志存本地，之后可导出
```

### 7.4 后台运行长时间任务

```bash
# 用 nohup 或 tmux，避免 SSH 断开后进程被杀
tmux new -s train
uv run python scripts/train_drafter.py ...
# Ctrl+B D 退出 tmux
# tmux attach -t train 回到会话
```

### 7.5 监控 GPU 使用

```bash
# 另开终端实时监控
watch -n 2 nvidia-smi

# 或
nvitop  # pip install nvitop
```

## 8. autodl 常用操作速查

```bash
# 关机（保留数据盘）
sudo shutdown now

# 查看磁盘用量
df -h /root/autodl-fs

# 查看当前实例 ID 和到期时间
cat /root/.autodl/instance_id 2>/dev/null
```
