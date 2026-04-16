# 模型结构说明（Phase-0）

本文档说明当前仓库中 `standalone`、`fedgh`、`pfedmoe` 三种方法所使用的模型结构与训练边界。

## 1. 模块总览

当前实现由四个核心模块组成：

1. `SimpleExtractor`：特征提取器（CNN）
2. `LocalHead`：分类头（线性层）
3. `LinearGate`：门控网络（仅 pFedMoE 使用）
4. `global_proxy`：服务器端聚合的共享提取器（仅 fedgh/pfedmoe 使用）

对应代码位置：

- `src/models/simple_models.py`
- `src/federated/engine.py`

## 2. 具体结构

### 2.1 SimpleExtractor

`SimpleExtractor(in_channels, feature_dim)` 结构：

- Conv2d: `in_channels -> 16`, kernel=3, padding=1
- ReLU + MaxPool2d(2)
- Conv2d: `16 -> 32`, kernel=3, padding=1
- ReLU + MaxPool2d(2)
- AdaptiveAvgPool2d((8,8))
- Flatten
- Linear: `32*8*8 -> feature_dim`

输出为统一特征向量，默认 `feature_dim=128`（由配置控制）。

### 2.2 LocalHead

`LocalHead(feature_dim, num_classes)`：

- Linear: `feature_dim -> num_classes`

用于客户端最终分类预测。

### 2.3 LinearGate（pFedMoE）

`LinearGate(input_shape, hidden_dim)`：

- 输入图像先 `flatten`
- Linear: `in_dim -> hidden_dim`
- `sigmoid`
- Linear: `hidden_dim -> 2`
- `softmax(dim=1)`

输出 `alpha_global, alpha_local`，每行和为 1。

## 3. 三种方法的前向逻辑

### 3.1 standalone

- 仅本地模型：`local_extractor + local_head`
- 前向：
  - `local_repr = local_extractor(x)`
  - `logits = local_head(local_repr)`

### 3.2 fedgh（当前仓库中的轻量实现）

- 本地模型 + 全局 proxy 特征提取器
- 前向：
  - `local_repr = local_extractor(x)`
  - `global_repr = global_proxy_local(x)`
  - `logits = local_head(0.5 * (local_repr + global_repr))`

### 3.3 pfedmoe

- 本地模型 + 全局 proxy + gate
- 前向：
  - `local_repr = local_extractor(x)`
  - `global_repr = global_proxy_local(x)`
  - `alpha = gate(x)`
  - `fused = alpha_g * global_repr + alpha_l * local_repr`
  - `logits = local_head(fused)`

## 4. 参数更新与上传边界

### 4.1 本地更新

- `standalone`：更新 `local_extractor + local_head`
- `fedgh`：更新 `local_extractor + local_head + global_proxy_local`
- `pfedmoe`：更新 `local_extractor + local_head + gate + global_proxy_local`

### 4.2 上传到服务器

- `standalone`：不上传模型参数
- `fedgh/pfedmoe`：只上传 `global_proxy_local` 参数

服务器端对上传的 proxy 参数按样本数加权平均，更新下一轮全局 `global_proxy`。

## 5. 训练状态持久化（跨轮）

为保证联邦训练可收敛，客户端会在轮次之间保留本地状态：

- 持久化：`local_extractor`、`local_head`
- `pfedmoe` 额外持久化：`gate`
- 每轮开始时，`global_proxy_local` 由服务器最新 `global_proxy` 初始化

这保证了“本地个性化持续学习 + 全局共享知识同步”同时成立。

## 6. 当前实现特点与限制

当前版本是 Phase-0 的可运行实现，重点是链路正确和可复现：

- 优点：结构清晰、计算负担可控、便于调试
- 限制：尚未实现论文中更完整的异构模型池、switch norm 等细节

后续可在保持接口不变的前提下，替换为更贴近论文的异构 backbone 与更完整训练策略。
