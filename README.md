
# HAR Research — 面向小样本与资源受限场景的人体活动识别

基于 UCI HAR 原始惯性信号（`[batch, 9, 128]`）的六分类研究，比较
ResNet1D-LSTM、ConvNeXt1D-LSTM 与 ResNet1D-TCN 等架构。详见 [DESIGN.md](DESIGN.md)。

当前进度：第 1–5 周已完成 —— 数据处理与统一实验框架、全部 7 个必做/扩展模型实现、
统一配置与正式训练结果；第 7 周附加实验（Transformer 对照）已实现并完成首次训练。

## 模型清单

| 配置                             | 模型            | 说明                                  |
| -------------------------------- | --------------- | ------------------------------------- |
| `configs/cnn1d.yaml`           | 1D-CNN          | 基础卷积基线                          |
| `configs/lstm.yaml`            | LSTM            | 基础时序建模基线                      |
| `configs/resnet1d.yaml`        | ResNet1D        | 残差卷积结构                          |
| `configs/cnn_lstm.yaml`        | CNN-LSTM        | 卷积 + LSTM                           |
| `configs/resnet1d_lstm.yaml`   | ResNet1D-LSTM   | **核心模型**                    |
| `configs/convnext1d_lstm.yaml` | ConvNeXt1D-LSTM | 现代 CNN 骨干（扩展）                 |
| `configs/resnet1d_tcn.yaml`    | ResNet1D-TCN    | 扩张卷积时序模块（扩展）              |
| `configs/transformer.yaml`     | Transformer     | 注意力时序建模对照（第 7 周附加实验） |

所有模型统一接口：输入 `[batch, 9, 128]` → 输出 `[batch, 6]` logits。
ResNet1D 的卷积骨干（`ResNet1DBackbone`）被核心模型与 TCN 模型复用，
保证「同一前端、不同时序后端」的对照条件。

Transformer 沿用 LSTM 基线的输入组织方式：`[9, 128]` → `[128, 9]`，
线性投影 + 可学习位置编码 + 可学习 `[CLS]` token，2 层 pre-norm
Transformer Encoder（GELU，4 头，`d_model=128`），`[CLS]` 输出接统一分类头。
数据划分、训练预算与优化器设置与其他模型保持一致。

## 环境

```bash
conda run -n mamba_env python train.py --config configs/cnn1d.yaml
```

## 目录结构

```
models/           各模型实现（统一接口：输入 [batch,9,128] -> 输出 [batch,6]）
src/data.py       原始信号加载 + 受试者分组划分
src/train.py      统一训练流程
src/evaluate.py   统一评估流程
src/metrics.py    指标计算（Accuracy / Macro-F1 / Recall / 混淆矩阵）
src/benchmark.py  参数量 / 模型大小 / CPU 推理延迟
configs/          各模型训练配置
results/          指标与结果输出（JSON + 混淆矩阵）
checkpoints/      最佳模型权重
train.py          入口
```

## 常用参数

```bash
python train.py --config configs/resnet1d_lstm.yaml \
    --seed 42 --epochs 30 --batch-size 64 --lr 0.001 \
    --device cpu
```

可用 `--model <name>` 覆盖配置中的模型，`--device auto` 在有 CUDA 时使用 GPU。
默认 `--device cpu`，与资源效率评估的 CPU 推理口径一致。

## 运行全部模型

```bash
for cfg in cnn1d lstm resnet1d cnn_lstm resnet1d_lstm convnext1d_lstm resnet1d_tcn transformer; do
    conda run -n mamba_env python train.py --config configs/$cfg.yaml --device cpu
done
```

输出文件（`<model>_seed<seed>_*`）：

- `_summary.json`          训练配置 + 验证/测试指标 + 资源效率
- `_history.json`          逐 epoch 训练曲线
- `_confusion_matrix.txt`  测试集混淆矩阵
- `checkpoints/..._best.pt` 验证 Macro-F1 最优的模型权重

## 第一版结果（seed 42, 30 epochs, CPU）

> 单种子结果，用于打通流程与初步架构对照；正式结论需第 6 周三种子 / 小样本实验。
> CPU 延迟为**单线程**、单样本前向、500 次测量均值，8 个模型在同一进程中统一重测
> （`python -m src.benchmark --threads 1 --num-runs 500 --output results/benchmark_single_thread.json`）。
> 延迟与机器负载相关，跨机器比较时请以同一进程重测结果为准。

| 模型                  | 最佳 epoch | Val Macro-F1 | Test Acc | Test Macro-F1 |    参数量 | 大小 (KB) | 延迟 (ms) |
| --------------------- | ---------: | -----------: | -------: | ------------: | --------: | --------: | --------: |
| 1D-CNN                |          3 |       0.9674 |   0.9311 |        0.9316 |   127,494 |     509.4 |     0.290 |
| LSTM                  |         29 |       0.9203 |   0.8809 |        0.8820 |   204,038 |     800.8 |     2.323 |
| ResNet1D              |          7 |       0.9555 |   0.9365 |        0.9377 |   959,174 |   3,795.9 |     1.936 |
| CNN-LSTM              |         17 |       0.9569 |   0.9345 |        0.9349 |   291,654 |   1,148.5 |     1.102 |
| ResNet1D-LSTM（核心） |         11 |       0.9622 |   0.9162 |        0.9162 | 1,288,134 |   5,083.9 |     2.916 |
| ConvNeXt1D-LSTM       |          9 |       0.9740 |   0.9050 |        0.9050 | 1,805,190 |   7,076.5 |     5.120 |
| ResNet1D-TCN          |         27 |       0.9574 |   0.9430 |        0.9445 | 1,435,718 |   5,682.7 |     3.243 |
| Transformer           |          4 |       0.9563 |   0.8975 |        0.8970 |   283,910 |   1,120.6 |     2.160 |

初步观察（待多种子验证）：

- 本轮 **ResNet1D-TCN** 测试 Macro-F1 最高；在相同 ResNet1D 前端下，
  TCN 比 LSTM 后端高约 2.8 个点。
- 核心模型 ResNet1D-LSTM 的验证集表现高于 ResNet1D，但测试集反而更低，
  存在验证集（4 位受试者）与官方测试集分布不一致的问题。
- ConvNeXt1D-LSTM 验证 Macro-F1 最高但测试集最低，参数与延迟也最大，
  说明当前训练预算下现代骨干更容易过拟合验证受试者。
- 主要混淆集中在 **坐 vs 站**，其次为 **上楼 vs 下楼**。

### Transformer 对照（第 7 周附加实验）

- **定位**：沿用 LSTM 基线的输入组织（128 个时间步 × 9 通道），用自注意力替代
  LSTM / TCN 的时序建模；数据划分、训练预算、优化器与分类头保持一致。
- **结果**：最佳 epoch 4（Val Macro-F1 0.9563），测试 Acc 0.8975 / Macro-F1 0.8970，
  仅略高于 LSTM 基线（0.8820），低于 1D-CNN、ResNet1D、CNN-LSTM 与 ResNet1D-TCN。
- **资源**：283,910 参数、1,120.6 KB、单线程延迟 2.160 ms，处于轻量基线
  （1D-CNN / CNN-LSTM）与 ResNet 系列之间。
- **分析**：第 4 个 epoch 后验证集开始波动、测试集比最佳验证低约 6 个点，
  说明在 5,800 个训练窗口的设置下，128 步全局自注意力缺少卷积的局部归纳偏置，
  更容易过拟合。该结论为单种子初步观察，需第 6 周多种子实验确认。

## 资源效率单独复测

训练时记录的延迟会受到系统负载影响；需要横向比较时，用同一进程、固定线程数重测：

```bash
conda run -n mamba_env python -m src.benchmark --threads 1 --num-runs 500 \
    --output results/benchmark_single_thread.json
```

## 复现约定

- 验证集按受试者分组（20% 受试者），同一受试者不会同时出现在训练集与验证集；
- 官方测试集固定，不参与训练或调参；
- 最佳模型按验证集 Macro-F1 选择；
- 每个结果文件记录随机种子、完整配置与受试者划分。
