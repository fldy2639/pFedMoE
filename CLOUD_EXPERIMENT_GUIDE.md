# 矩池云快速实验指南（单卡 A16）

本指南面向“尽快看到趋势”的实验目标，保持本地版本不变，新增云端配置独立运行。

## 1. 推荐使用方式

- 代码来源：直接从 GitHub 导入仓库
- GPU：单卡 A16
- 首选配置：`configs/cloud_quick.yaml`
- 一键脚本：`scripts/run_cloud_quick.sh`

## 2. 服务器初始化

```bash
git clone https://github.com/fldy2639/pFedMoE.git
cd pFedMoE
pip install -r requirements.txt
```

可选检查：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. 数据放置（手动）

将 `cifar-10-python.tar.gz` 放到：

```text
data/raw/cifar-10-python.tar.gz
```

说明：

- 当前代码使用 `torchvision CIFAR10(..., download=True)`。
- 若本地已存在压缩包，会直接校验并解压，不重复下载。

## 4. 快速趋势实验（推荐）

```bash
bash scripts/run_cloud_quick.sh
```

默认配置特点（A16 友好）：

- `device: cuda`
- `rounds: 30`
- `local_epochs: 2`
- `batch_size: 256`

输出目录：

```text
outputs/cloud_quick/
```

## 5. 稳定版实验（较慢）

```bash
bash scripts/run_cloud_full.sh
```

默认配置：

- `rounds: 100`
- `local_epochs: 5`
- `batch_size: 256`

输出目录：

```text
outputs/cloud_full/
```

## 6. 后台运行建议

若你关闭终端，建议使用 `nohup`：

```bash
nohup bash scripts/run_cloud_quick.sh > cloud_quick.log 2>&1 &
tail -f cloud_quick.log
```

## 7. 常见问题

### 7.1 CUDA 不可用

- 检查镜像是否含 GPU 版 PyTorch
- 检查 `nvidia-smi` 是否正常
- 若不可用，可临时改 `device: cpu`（速度会显著下降）

### 7.2 训练过慢

优先顺序：

1. 先用 `cloud_quick.yaml`
2. `rounds` 再降到 `20`
3. `local_epochs` 降到 `1`

### 7.3 无法画图

确认 `metrics.jsonl` 已生成，再执行 `analyze`。
