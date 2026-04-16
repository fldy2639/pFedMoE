# pFedMoE Phase-0 复现说明（低资源优先）

本仓库提供一个可直接运行的 Phase-0 复现实验链路，目标是先验证 pFedMoE 的核心机制正确性，再逐步扩展到论文完整规模。

## 1. 当前已实现能力

- 数据集：`MNIST`、`CIFAR-10`、`CIFAR-100`
- 数据划分：
  - `pathological non-IID`
  - `practical Dirichlet non-IID`
  - 固定 seed 可复现，并缓存到 `data/partitions/`
- 方法：
  - `standalone`
  - `fedgh`（仓库内轻量实现）
  - `pfedmoe`
- 联邦流程：
  - `prepare_partition -> train -> analyze`
- 回放元数据：
  - `partition_hash`
  - `config_hash`
  - `sampled_clients_by_round`
  - `global_seed / sampler_seed`

## 2. 环境准备

```bash
pip install -r requirements.txt
```

默认配置在 `configs/default.yaml`，关键字段：

- 数据集与目录：`dataset.name`、`dataset.data_dir`
- 划分方式：`dataset.partition_type`
- 联邦超参：`federated.rounds`、`local_epochs`、`batch_size`
- 评估目标：`eval.target_accuracy`

## 3. CIFAR-10 本地数据说明（手动放置）

如果你已经把 `cifar-10-python.tar.gz` 放到 `data/raw/`，直接运行即可。

- 代码中使用 `torchvision.datasets.CIFAR10(..., download=True)`
- 当本地已有 `tar.gz` 时，`torchvision` 会直接校验并解压，无需重新联网下载
- 首次运行后会出现 `data/raw/cifar-10-batches-py/`

## 4. 一键执行顺序（推荐）

```bash
python -m src.main --config configs/default.yaml --mode prepare_partition
python -m src.main --config configs/default.yaml --mode train --method standalone
python -m src.main --config configs/default.yaml --mode train --method fedgh
python -m src.main --config configs/default.yaml --mode train --method pfedmoe
python -m src.main --config configs/default.yaml --mode analyze
```

## 5. 输出目录说明

以 `outputs/phase0` 为例：

- `prepare_partition_meta.json`：分区文件和分区哈希
- `standalone|fedgh|pfedmoe/<dataset>/metrics.jsonl`：逐轮指标
- `standalone|fedgh|pfedmoe/<dataset>/run_meta.json`：可回放元信息
- `figures/convergence_<dataset>.png`：收敛曲线

`metrics.jsonl` 每轮至少包含：

- `round`
- `mean_accuracy`
- `std_accuracy`
- `sampled_clients`

## 6. 常见问题排查（重点）

### 6.1 mean_acc 长期在 10%~20%

优先检查分区是否异常（这是最常见原因）：

1. 删除旧分区缓存后重建：

```bash
python -m src.main --config configs/default.yaml --mode prepare_partition
```

2. 检查每客户端样本规模（CIFAR-10, N=10, pathological, train_ratio=0.8 时应接近 4000/1000）：

```bash
python -c "import json; p=json.load(open('data/partitions/cifar10_pathological_n10_s42.json','r',encoding='utf-8')); lens=[(len(v['train']),len(v['test'])) for v in p.values()]; print('train',min(x[0] for x in lens),max(x[0] for x in lens)); print('test',min(x[1] for x in lens),max(x[1] for x in lens))"
```

如果每客户端只有百级样本，模型通常会停留在低准确率。

### 6.2 训练时间过长

- Phase-0 快速调试可先把 `federated.rounds` 改到 `20~50`
- 确认 `device` 为可用设备（CPU 下 100 轮会慢很多）

### 6.3 画图为空

- 确认 `outputs/phase0/<method>/<dataset>/metrics.jsonl` 已生成
- 再执行 `analyze` 模式

## 7. 一个可参考的 Phase-0 快速配置

建议先做 Quick Debug（仅验证链路）：

- `num_clients=10`
- `participation_rate=1.0`
- `rounds=20~50`
- `local_epochs=1~5`
- `batch_size=64~128`

跑通并确认趋势后，再提升到更接近论文口径的设置。

## 8. 设计边界与后续计划

- 当前仓库优先保证：链路正确性、可复现性、低资源可运行
- 要复现论文主表规模，请按 `claude.md` 继续扩展：
  - 更完整网格搜索
  - 更多 baseline
  - 更全面评估与分析实验
