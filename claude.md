# CLAUDE.md

## 项目目标

复现论文 **pFedMoE: Data-Level Personalization With Mixture of Experts in Model-Heterogeneous Personalized Federated Learning** 的核心实验，并建立一个可扩展、可对比、可调试的研究代码库。

本仓库优先保证：

1. **主结果可复现**：先复现 MNIST / CIFAR-10 / CIFAR-100 的主表格与主曲线。
2. **结构可扩展**：后续可无痛加入真实医疗数据、Stack Overflow NLP、更多异构模型、更多基线。
3. **调试友好**：任何一个实验都应能拆成“数据划分 → 客户端本地训练 → 服务器聚合 → 评估”四个可独立检查的阶段。
4. **严格区分共享模块与本地模块**：这是 pFedMoE 的核心，不允许在实现中把本地异构模型误上传到服务器。

---

## 一句话理解 pFedMoE

每个客户端都维护一个本地完整模型，但本地训练时不是只用自己的模型，而是把：

- **全局共享的小型同构特征提取器**（global proxy extractor）
- **本地异构特征提取器**（local heterogeneous extractor）
- **本地门控网络**（local gating network）

组成一个两专家的 MoE，在**样本级别**动态决定更依赖全局知识还是本地知识；再把融合后的表征交给**本地预测头**完成最终分类。

服务器每轮只负责：

- 下发全局 proxy extractor 参数
- 聚合客户端上传回来的 proxy extractor 参数

本地异构模型和门控网络始终保留在客户端。

---

## 复现策略

### Phase 0：最低可运行版本

先只做：

- 数据集：MNIST、CIFAR-10、CIFAR-100
- 模型：5 个异构 CNN
- 方法：Standalone、FedGH、pFedMoE
- 指标：mean accuracy、individual accuracy、收敛曲线

目标：先打通论文最核心对比链路，而不是一开始就把 9 个 baseline 全部实现。

#### Phase 0（低资源）硬约束

在单机 CPU / 低显存环境下，Phase 0 必须优先保证“方向正确 + 口径一致”，并采用分层目标：

1. **链路正确性层（Quick Debug）**  
   - 仅用于检查代码链路，非论文结果对齐。  
   - 建议：`N=10, C=100%, T=20~50, E=1, B=64`。

2. **趋势验证层（Mini Repro）**  
   - 用于验证 `pFedMoE` 相比 `FedGH` 的收敛速度或精度趋势。  
   - 建议：`N=10, C=100%, T=100, E in {1,10}, B in {64,128}`。

3. **论文口径过渡层（Pre-Table）**  
   - 用于后续平滑升级到主表设置。  
   - 建议：在资源允许后再启用 `T=500` 与更完整网格。

### Phase 1：主实验完整复现

补齐：

- 9 个 baseline
- 三组 FL 设置：
  - N=10, C=100%
  - N=50, C=20%
  - N=100, C=10%
- 两类 non-IID 数据划分：
  - pathological
  - practical (Dirichlet)
- 论文主表格与曲线

### Phase 2：分析实验

补齐：

- client-wise accuracy 差值图
- T-SNE 表征可视化
- shared test set generalization
- communication/computation overhead
- robustness to pathological non-IIDness
- robustness to practical non-IIDness
- sensitivity to homogeneous extractor size
- dynamic vs static fusion ablation

### Phase 3：真实任务扩展

最后再做：

- Fed-Heart-Disease
- Stack Overflow next-word prediction
- higher model heterogeneity（RedNet/CNN/ViT）

---

## 仓库结构规划

```text
pfedmoe/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── default.yaml
│   ├── dataset/
│   │   ├── mnist.yaml
│   │   ├── cifar10.yaml
│   │   ├── cifar100.yaml
│   │   ├── fed_heart_disease.yaml
│   │   └── stackoverflow.yaml
│   ├── method/
│   │   ├── pfedmoe.yaml
│   │   ├── fedgh.yaml
│   │   ├── fedavg.yaml
│   │   ├── fml.yaml
│   │   ├── fedkd.yaml
│   │   ├── fedapen.yaml
│   │   ├── fd.yaml
│   │   ├── fedproto.yaml
│   │   └── fedtgp.yaml
│   ├── model/
│   │   ├── cnn_mnist.yaml
│   │   ├── cnn_cifar.yaml
│   │   ├── lstm_stackoverflow.yaml
│   │   └── high_hetero.yaml
│   └── experiment/
│       ├── main_table_mnist.yaml
│       ├── main_table_cifar10.yaml
│       ├── main_table_cifar100.yaml
│       ├── convergence.yaml
│       ├── robustness.yaml
│       ├── sensitivity.yaml
│       └── ablation.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── partitions/
├── src/
│   ├── main.py
│   ├── runner/
│   │   ├── train_federated.py
│   │   ├── evaluate.py
│   │   ├── analyze.py
│   │   └── launch_grid.py
│   ├── core/
│   │   ├── registry.py
│   │   ├── seed.py
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   ├── checkpoint.py
│   │   └── profiler.py
│   ├── data/
│   │   ├── datasets/
│   │   │   ├── mnist.py
│   │   │   ├── cifar10.py
│   │   │   ├── cifar100.py
│   │   │   ├── fed_heart_disease.py
│   │   │   └── stackoverflow.py
│   │   ├── partition/
│   │   │   ├── pathological.py
│   │   │   ├── practical_dirichlet.py
│   │   │   ├── partition_utils.py
│   │   │   └── cache.py
│   │   ├── client_dataset.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── backbones/
│   │   │   ├── cnn_mnist.py
│   │   │   ├── cnn_cifar.py
│   │   │   ├── rednet.py
│   │   │   ├── vit.py
│   │   │   └── lstm.py
│   │   ├── heads/
│   │   │   ├── linear_head.py
│   │   │   └── mlp_head.py
│   │   ├── gating/
│   │   │   ├── linear_gate.py
│   │   │   └── norms.py
│   │   ├── wrappers/
│   │   │   ├── split_model.py
│   │   │   ├── moe_client_model.py
│   │   │   └── heterogeneous_factory.py
│   │   └── utils.py
│   ├── federated/
│   │   ├── server/
│   │   │   ├── base_server.py
│   │   │   ├── pfedmoe_server.py
│   │   │   └── aggregation.py
│   │   ├── client/
│   │   │   ├── base_client.py
│   │   │   ├── pfedmoe_client.py
│   │   │   └── baseline_client.py
│   │   ├── samplers/
│   │   │   └── client_sampler.py
│   │   └── protocol.py
│   ├── methods/
│   │   ├── pfedmoe/
│   │   │   ├── method.py
│   │   │   ├── losses.py
│   │   │   ├── train_step.py
│   │   │   ├── communication.py
│   │   │   └── analysis.py
│   │   ├── baselines/
│   │   │   ├── standalone.py
│   │   │   ├── lg_fedavg.py
│   │   │   ├── fedgh.py
│   │   │   ├── fml.py
│   │   │   ├── fedkd.py
│   │   │   ├── fedapen.py
│   │   │   ├── fd.py
│   │   │   ├── fedproto.py
│   │   │   └── fedtgp.py
│   ├── evaluation/
│   │   ├── accuracy.py
│   │   ├── communication.py
│   │   ├── computation.py
│   │   ├── tsne.py
│   │   └── generalization.py
│   └── visualization/
│       ├── curves.py
│       ├── barplots.py
│       ├── scatter.py
│       └── tables.py
├── scripts/
│   ├── run_main_tables.sh
│   ├── run_convergence.sh
│   ├── run_robustness.sh
│   ├── run_sensitivity.sh
│   └── run_ablation.sh
├── tests/
│   ├── test_partition.py
│   ├── test_shapes.py
│   ├── test_aggregation.py
│   ├── test_gate.py
│   └── test_train_step.py
└── outputs/
    ├── logs/
    ├── checkpoints/
    ├── metrics/
    └── figures/
```

---

## 关键模块职责

### 1. `data/partition/`

负责严格复现实验划分。

必须支持两类划分：

#### Pathological non-IID
- MNIST / CIFAR-10：每个客户端分配 2 个类
- CIFAR-100：每个客户端分配 10 个类
- 同类样本量再通过 Dirichlet 做不均衡控制

#### Practical non-IID
- 每个客户端都可见所有类
- 类比例由 Dirichlet(γ) 控制
- 客户端内部 train:test = 8:2

**要求**：
- 同一个 seed 下可重复生成完全一致的分区
- 分区结果保存到 `data/partitions/*.json`
- 任何实验都优先读取缓存分区，避免“每次随机重划分”导致结果不稳定

### 2. `models/wrappers/split_model.py`

统一把任意本地模型拆成：

- `extractor`
- `head`

这是 pFedMoE 能工作的前提。

约束：
- 本地异构模型内部结构可以不同
- 但 `extractor` 的最后输出维度必须和全局 proxy extractor 的输出维度一致
- 图像主实验中，MNIST 统一到 50 维，CIFAR 统一到 500 维

### 3. `models/gating/linear_gate.py`

实现论文中的轻量门控网络：

- 输入：原始样本 `x`
- 图像场景下先 flatten
- 可选 switch normalization / batch norm
- 两层线性层
- 第一层激活可配置，默认 Sigmoid
- 第二层输出维度固定为 2
- Softmax 输出 `[alpha_global, alpha_local]`

返回：
- shape = `[batch_size, 2]`
- 每行和为 1

### 4. `methods/pfedmoe/train_step.py`

这里是整个方法实现的核心。

对每个 batch：

1. `global_repr = global_proxy_extractor(x)`
2. `local_repr = local_extractor(x)`
3. `gate_weight = gate(x)`
4. `fused_repr = alpha_g * global_repr + alpha_l * local_repr`
5. `logits = local_head(fused_repr)`
6. `loss = criterion(logits, y)`
7. 反向传播，同时更新：
   - proxy extractor 参数
   - local heterogeneous model 参数
   - gate 参数

注意：
- 本地完整模型虽然更新，但**只上传 proxy extractor 参数**
- 预测头、local extractor、gate 都保留在本地

### 5. `federated/server/pfedmoe_server.py`

每轮执行：

1. 随机采样 `K = C * N` 个客户端
2. 下发当前全局 proxy extractor 参数
3. 收集各客户端上传的本地 proxy extractor 参数
4. 按样本数加权平均
5. 更新全局 proxy extractor

禁止：
- 聚合本地异构模型参数
- 聚合本地预测头
- 聚合门控网络

### 6. `evaluation/`

#### 必做指标
- mean local test accuracy
- client-wise individual accuracy
- communication rounds to target accuracy
- communication cost
- computation overhead

#### 分析指标
- shared test set generalization
- gate weight 分布
- t-SNE 表征可视化
- dynamic vs static fusion ablation

---

## 模型实现约定

### 异构本地 CNN

主图像实验需要 5 个异构 CNN。

#### MNIST
- Conv1: 20 filters, kernel 5x5（所有模型一致）
- Conv2: 20 filters, kernel 5x5（所有模型一致）
- FC1: {300, 200, 150, 100, 50}
- FC2: 50
- FC3: 10

#### CIFAR-10 / CIFAR-100
- Conv1: filters {32, 32, 16, 16, 16}
- Conv2: filters {32, 16, 32, 32, 32}
- FC1: {2000, 2000, 1000, 800, 500}
- FC2: 500
- FC3: 10 或 100

**建议实现策略**：
- 把 FC3 之前定义为 extractor
- FC3 定义为 head
- 这样可天然保证所有异构模型输出接口对齐到 50 / 500 维

### 全局 proxy homogeneous extractor

主图像实验中统一使用**最小的 CNN-5** 的 extractor 作为共享 proxy extractor。

### 预测头

默认先实现 homogeneous linear head。

### 门控网络

输入原始图像 flatten 向量，输出 2 维权重。

---

## 配置设计原则

所有实验都必须通过配置驱动，不允许把关键超参数写死在代码里。

### 论文超参数口径（必须保留）

论文给出的网格搜索空间（用于最终可比复现）：

- `T in {100, 500}`
- `E in {1, 10}`
- `B in {64, 128, 256, 512}`
- `lr in {0.001, 0.01, 0.1, 1}`
- 优化器：SGD
- `pFedMoE` 默认约束：`eta_proxy = eta_local = 0.01`

### Phase 0 低资源子网格（优先执行）

为避免前期成本失控，在 Phase 0 先使用以下子网格：

- `T in {20, 50, 100}`（MNIST 可先 20，CIFAR 先 50/100）
- `E in {1, 5}`（不先跑 10，除非趋势不明显）
- `B in {64, 128}`
- `lr in {0.01, 0.1}`（SGD）
- `num_clients=10, participation_rate=1.0`

要求：
- 所有实验日志中必须记录“full-grid / phase0-subgrid”标签；
- 任何对比图必须标明是否属于子网格结果；
- 进入 Phase 1 前再切回论文完整网格。

推荐最小配置字段：

```yaml
seed: 42
method: pfedmoe
num_clients: 100
participation_rate: 0.1
rounds: 500
local_epochs: 10
batch_size: 128
optimizer: sgd
lr_proxy: 0.01
lr_local: 0.01
lr_gate: 0.01
weight_decay: 0.0
momentum: 0.9

model:
  family: cnn_cifar
  heterogeneous_pool: [cnn1, cnn2, cnn3, cnn4, cnn5]
  proxy_model: cnn5
  feature_dim: 500

partition:
  type: pathological
  num_classes_per_client: 2
  dirichlet_alpha: null
  train_ratio: 0.8

gate:
  hidden_dim: 128
  use_switch_norm: true
  use_batch_norm: true
  hidden_activation: sigmoid
  output_activation: softmax

eval:
  target_accuracy: 0.9
  save_client_metrics: true
  run_tsne: false
```

---

## 训练流程约定

### 单轮联邦训练

1. 服务器采样客户端
2. 服务器广播 proxy extractor 参数
3. 每个客户端本地构建：
   - global expert = received proxy extractor
   - local expert = local heterogeneous extractor
   - local gate
   - local head
4. 对本地数据做 E 个 epoch 训练
5. 客户端仅上传 proxy extractor 参数
6. 服务器聚合上传参数
7. 本轮结束后评估

### 评估约定

- 默认每轮做一次 mean accuracy 评估
- 每 10 轮保存一次 checkpoint
- 每个实验单独保存：
  - config 副本
  - metrics.jsonl
  - best checkpoint
  - final checkpoint
  - client_metrics.csv

### communication / computation 统计口径（强约束）

必须采用论文一致口径：

- `communication_cost = rounds_to_target_accuracy * comm_per_round`
- `computation_overhead = rounds_to_target_accuracy * flops_per_round`

其中 `rounds_to_target_accuracy` 为“首次达到目标 mean accuracy 的轮数”，不是最后一轮轮数。

### 目标精度分层（低资源到论文口径）

为兼顾低资源与可比性，先采用分层 target accuracy：

1. **工程验证目标（Phase 0）**
   - CIFAR-10：`target_accuracy in [0.70, 0.80]`
   - CIFAR-100：`target_accuracy in [0.35, 0.45]`

2. **论文口径目标（Phase 1/2）**
   - CIFAR-10：`target_accuracy = 0.90`
   - CIFAR-100：`target_accuracy = 0.50`

要求：
- 报告中必须明确标注 target accuracy 层级；
- 禁止将工程验证目标下的开销结果直接当作论文主结论。

---

## 必做单元测试

### `test_partition.py`
- pathological 划分是否满足类数约束
- practical 划分是否满足 Dirichlet 分布约束
- train/test 是否按 8:2 切开
- 相同 seed 是否完全复现

### `test_shapes.py`
- 所有异构模型的 extractor 输出维度是否一致
- gate 输出是否为 `[B, 2]`
- fused repr 维度是否正确
- head 输入输出维度是否正确

### `test_aggregation.py`
- 服务器是否只聚合 proxy extractor
- 加权平均是否与样本量一致
- 不应错误聚合 local extractor / head / gate

### `test_train_step.py`
- 一个 batch 的 loss 是否能 backward
- proxy/local/gate 三类参数是否都收到梯度
- 上传参数中是否只包含 proxy extractor

---

## 结果复现优先级

### 优先级 A：必须先复现
1. Table VI 主精度表
2. CIFAR-10 / CIFAR-100 收敛曲线
3. individual accuracy 差值图
4. communication / computation overhead

### 优先级 B：方法机制验证
5. T-SNE
6. shared test set generalization
7. robustness to pathological non-IIDness
8. robustness to practical non-IIDness
9. homogeneous extractor size sensitivity
10. dynamic vs static fusion ablation

### 优先级 C：真实任务扩展
11. Fed-Heart-Disease
12. Stack Overflow
13. high heterogeneity model zoo

---

## Baseline 实现顺序

不要一开始同时写 9 个 baseline。

推荐顺序：

1. `Standalone`
2. `FedGH`
3. `pFedMoE`
4. `FedAPEN`
5. `FedKD`
6. `FML`
7. `LG-FedAvg`
8. `FD`
9. `FedProto`
10. `FedTGP`

理由：
- `Standalone` 是 sanity check
- `FedGH` 是论文最重要对比对象
- `pFedMoE` 先做通，能先验证主结论
- 其余 baseline 再按论文重要性逐步补齐

---

## 复现中的高风险点

### 风险 1：模型拆分点选错
如果 extractor / head 切分点不一致，会直接导致：
- 接口维度不对齐
- gate 融合失效
- 与论文结果偏差极大

### 风险 2：把本地模型错误上传到服务器
pFedMoE 只共享 proxy extractor。任何额外上传：
- local extractor
- local head
- gate
都属于实现错误。

### 风险 3：分区不稳定
联邦论文最常见复现失败原因不是模型，而是数据划分。必须缓存 partition。

### 风险 4：把 gate 输入写成表征而不是原始输入
论文默认 gate 直接看原始样本（图像先 flatten），不是看融合后的中间特征。

### 风险 5：评价指标算错
communication cost 和 computation overhead 都是“达到目标精度所需轮数”乘以单轮代价，不是简单记录最后一轮的代价。

### 风险 6：把 proxy extractor 大小写死
后续 sensitivity 实验要求可替换不同大小的 homogeneous extractor。

---

## 日志与产物规范

每个实验目录至少包含：

```text
outputs/exp_name/
├── config.yaml
├── train.log
├── metrics.jsonl
├── best.ckpt
├── final.ckpt
├── client_metrics.csv
├── comm_cost.json
├── comp_cost.json
└── figures/
```

`metrics.jsonl` 每轮至少记录：
- round
- sampled_clients
- mean_accuracy
- std_accuracy
- best_accuracy
- train_loss_mean
- upload_params
- download_params
- flops_per_round
- elapsed_time

此外，每个实验目录必须新增以下可回放字段（建议 `run_meta.json`）：

- `partition_hash`：分区文件哈希（如 SHA256）
- `config_hash`：完整配置哈希
- `global_seed`：全局随机种子
- `sampler_seed`：客户端采样随机种子
- `sampled_clients_by_round`：每轮采样客户端 ID 列表
- `git_or_code_snapshot`：代码快照标识（无 git 时可用源码压缩包哈希）

目标：任意一次失败实验都可以做到“配置、分区、采样路径”完全回放。

---

## 建议的首批命令

```bash
# 1. 先生成固定分区
python -m src.main mode=prepare_partition experiment=main_table_cifar10

# 2. 跑 Standalone sanity check
python -m src.main mode=train method=standalone experiment=main_table_cifar10

# 3. 跑 FedGH
python -m src.main mode=train method=fedgh experiment=main_table_cifar10

# 4. 跑 pFedMoE
python -m src.main mode=train method=pfedmoe experiment=main_table_cifar10

# 5. 画对比曲线
python -m src.main mode=analyze experiment=convergence
```

### Phase 0 低资源执行顺序（必须按序）

1. **Prepare：固定分区与回放元数据**
   - 生成并缓存 pathological 与 practical 分区
   - 写入 `partition_hash` 与 `sampler_seed`

2. **MNIST Quick Debug**
   - 依次跑 `Standalone -> FedGH -> pFedMoE`
   - 仅检查链路正确性：loss 下降、shape 对齐、仅上传 proxy

3. **CIFAR-10 Mini Repro**
   - 同样顺序跑三方法
   - 检查：pFedMoE 相比 FedGH 是否有精度优势或更快收敛趋势

4. **CIFAR-100 Mini Repro**
   - 先跑 50~100 轮保证跑通
   - 产出基础收敛曲线与工程验证目标下的开销统计

5. **Phase 0 验收**
   - 三数据集三方法均有可复现日志和图；
   - 至少 1 次重复实验可复现主要结论趋势；
   - 明确下一步升级路径（扩 grid / 扩 baseline / 提升 target accuracy）。

---

## 当前实现决策

### 必须严格遵守
- 只传 proxy extractor 参数
- 本地异构模型与 gate 不上传
- gate 输出固定为两维
- global/local extractor 输出维度必须一致
- 训练采用 end-to-end 反向传播

### 允许先做简化版
- 先不用 switch normalization，后续补
- 先只实现 homogeneous head
- 先只做图像任务
- 先只做 `FedGH + pFedMoE` 主要对比

### 需要标记“待核验”的地方
- 论文开源代码与正式实现是否在 gate hidden dim、normalization 细节上有额外 trick
- 各 baseline 的最优超参数是否需要单独网格搜索
- 真实任务的具体 preprocessing 与 tokenizer / feature pipeline

---

## 给实现者的硬约束

### 任何 PR 都必须回答 5 个问题
1. 这个改动会不会改变分区可重复性？
2. 这个改动会不会改变 extractor/head 的切分点？
3. 这个改动会不会误把本地参数上传到服务器？
4. 这个改动会不会影响 communication / computation 统计口径？
5. 这个改动是否能通过最小单元测试？

### 任何新的 baseline 都必须提供
- 方法简述
- 共享参数范围
- 本地参数范围
- 单轮训练流程
- 与统一评估器兼容的输出接口

---

## 最后一句

先复现主结论，再追求全量覆盖。

对于这篇论文，**最重要的不是先把所有 baseline 写全，而是先把“共享 proxy extractor + 本地异构 extractor + gate + local head”这条核心链路写对。**

