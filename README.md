
# HAR Research — 面向小样本与资源受限场景的人体活动识别

基于 UCI HAR 原始惯性信号（`[batch, 9, 128]`）的六分类研究，比较
ResNet1D-LSTM、ConvNeXt1D-LSTM 与 ResNet1D-TCN 等架构。详见 [DESIGN.md](DESIGN.md)。

当前进度：第 1–6 周已完成 —— 数据处理与统一实验框架、全部 7 个必做/扩展模型实现、
统一配置与正式训练结果，以及小样本正式实验（10% / 25% / 50% / 100% × 3 个模型种子）；
第 7 周附加实验（Transformer 对照）已实现并完成首次训练。

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
src/small_sample.py 第 6 周小样本实验调度、均值/标准差汇总
src/plot.py       小样本学习曲线（matplotlib，默认 PNG 输出）
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

### 种子分离（第 6 周起）

模型随机种子与数据种子是分开的：

- `--split-seed`：受试者分组训练 / 验证划分种子（默认 42）；
- `--subset-seed`：小样本训练子集种子（默认等于 `split_seed`）；
- `--seed`：模型种子，只控制权重初始化、dropout 与 batch 顺序。

三个模型种子共享同一套验证受试者与同一套训练子集，因此种子间的标准差只反映
训练随机性，不会混入“不同种子用了不同验证受试者”的划分差异。训练入口也支持
`--train-fraction 0.1 --tag-suffix _frac0p1`，便于单独复现某一次小样本运行。

## 小样本实验（第 6 周）

对应 DESIGN §4 实验三：训练集使用 **10% / 25% / 50% / 100%**，官方测试集固定，
每个比例用 **3 个模型种子**重复。子集按类别分层且跨比例嵌套（10% ⊂ 25% ⊂ 50%），
避免“抽到哪些窗口”成为比例间的额外变量。

```bash
# 完整网格：7 个模型 × 3 seeds × 4 fractions = 84 次训练
conda run -n mamba_env python -m src.small_sample --jobs 5 --threads 4 --device cpu

# 只重新汇总已有结果 / 重绘曲线
conda run -n mamba_env python -m src.small_sample --aggregate-only

# 指定模型、比例与种子
conda run -n mamba_env python -m src.small_sample \
    --models resnet1d_lstm convnext1d_lstm resnet1d_tcn \
    --fractions 0.1 0.25 0.5 1.0 --seeds 42 43 44
```

已有 `summary.json` 的运行会自动跳过（`--force` 可强制重跑）。输出写在
`results/small_sample/`：

- `summary.json` / `summary.md`：各模型 × 比例的性能均值 ± 标准差；
- `runs.csv`：逐次运行记录（种子、训练窗口数、验证 / 测试指标、参数量 / 模型大小）。
  其中的 CPU 延迟是在多个训练任务并行时顺带测得的，只作完整性记录；横向比较请使用
  第 7 周固定线程数的统一复测。
- `macro_f1_curve.png`：Macro-F1 随训练比例变化的曲线（误差棒为 ±1 标准差）；
- `macro_f1_curve_zoom.png`：自动生成的放大视图；若某模型在小比例下明显受
  固定 epoch 步数预算限制，主图会保留它以保证完整，放大图则去掉该曲线，
  以免压扁其余模型的 y 轴。

逐次训练日志在 `logs/small_sample/`，权重在 `checkpoints/small_sample/`。

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
  更容易过拟合。该结论仍为单种子初步观察；第 6 周主实验未包含 Transformer，
  如需确认可后续补三种子重复。

## 小样本正式结果（第 6 周，3 seeds）

- **设置**：训练集 10% / 25% / 50% / 100%，官方测试集固定；每个比例 3 个模型种子
  （42 / 43 / 44），共 **84 次训练**。数据划分种子 `split_seed=42` 与小样本子集种子
  `subset_seed=42` 固定，因此表中的标准差只反映训练随机性。
- **子集**：按类别分层，且 10% ⊂ 25% ⊂ 50%（嵌套），避免“抽到哪些窗口”成为比例间的
  额外变量。
- **产物**：[summary.md](results/small_sample/summary.md)（含 Accuracy / 验证集表）、
  [summary.json](results/small_sample/summary.json)、[runs.csv](results/small_sample/runs.csv)、
  [主曲线](results/small_sample/macro_f1_curve.png)、
  [放大曲线](results/small_sample/macro_f1_curve_zoom.png)。

测试集 Macro-F1（均值 ± 标准差，n=3）：

| 模型 | 10% | 25% | 50% | 100% |
| --- | ---: | ---: | ---: | ---: |
| 1D-CNN | 0.9043 ± 0.0195 | 0.9215 ± 0.0109 | 0.9291 ± 0.0187 | 0.9220 ± 0.0290 |
| LSTM | 0.4853 ± 0.1309 | 0.6345 ± 0.0970 | 0.5558 ± 0.1017 | 0.8971 ± 0.0113 |
| ResNet1D | 0.9027 ± 0.0249 | 0.9110 ± 0.0234 | 0.9453 ± 0.0078 | 0.9316 ± 0.0066 |
| CNN-LSTM | 0.8801 ± 0.0100 | 0.9081 ± 0.0096 | 0.8995 ± 0.0452 | 0.9243 ± 0.0107 |
| ResNet1D-LSTM（核心） | 0.8505 ± 0.0465 | 0.9039 ± 0.0166 | 0.9302 ± 0.0065 | 0.9199 ± 0.0085 |
| ConvNeXt1D-LSTM | 0.8659 ± 0.0337 | 0.9019 ± 0.0060 | 0.9049 ± 0.0072 | 0.9038 ± 0.0062 |
| ResNet1D-TCN | 0.8684 ± 0.0164 | 0.9235 ± 0.0084 | 0.9323 ± 0.0104 | 0.9372 ± 0.0055 |

> 本表 100% 列为 3 个种子的均值，与上文第一版单种子结果不逐位相同；第 6 周所有任务统一
> 使用 4 线程并行训练，CPU 归约顺序与单任务 20 线程时不同，属正常运行波动。

主要观察：

1. **简单卷积最“省数据”**：1D-CNN 从 10% 到 100% 仅变化 1.8 个点
   （0.9043 → 0.9220；50% 略高于 100%，差异在种子波动范围内），ResNet1D 为 2.9 个点；
   两者在 10% 数据时仍保持 0.90 左右，是低数据预算下最稳的选择。
2. **深模型更依赖数据量**：ResNet1D-TCN 与 ResNet1D-LSTM 从 10% 到 100% 均提升约
   6.9 个点（0.868 → 0.937 / 0.851 → 0.920）；CNN-LSTM 提升 4.4 个点，ConvNeXt1D-LSTM
   提升 3.8 个点。10% 数据时，深度模型的优势消失，1D-CNN / ResNet1D 反而领先。
3. **100% 数据下 TCN 后端最好**：ResNet1D-TCN 均值 0.9372，高于 ResNet1D-LSTM（0.9199）
   和 ResNet1D（0.9316）；在相同 ResNet1D 前端下，TCN 再次优于 LSTM。50% 时 ResNet1D
   （0.9453）最高，但与 TCN / ResNet1D-LSTM 的差距在 1–2 个标准差内，不宜过度解读。
4. **ConvNeXt1D-LSTM 没有兑现骨干优势**：参数量最大（1.8M），但各比例测试 Macro-F1
   均在 0.90 左右；其 100% 验证集 Macro-F1 高达 0.9671、测试集仅 0.9038（差 6.3 个点），
   说明在 4 位验证受试者上过拟合，未转化为测试集泛化。
5. **LSTM 基线在小比例下是“训练不足”而非“数据不足”**：固定 30 epoch 时，10% /
   25% / 50% 分别只有约 300 / 690 / 1380 个优化步（100% 约 2730 步），50% 时训练
   accuracy 仍只有约 60%，因此曲线跌到 0.49–0.63。该列不应直接解释为 LSTM 的数据效率；
   若要单独比较数据量，建议在第 7 周补一个“步数对齐”的对照实验。
   放大曲线 [macro_f1_curve_zoom.png](results/small_sample/macro_f1_curve_zoom.png)
   在保持配色不变的前提下去掉该条曲线，便于观察其余模型。
6. **验证集偏乐观**：多数模型 10% 时的验证 Macro-F1 为 0.93–0.97，而测试集只有
   0.85–0.90；100% 时 TCN 的验证/测试差最小（约 1.4 个点），ConvNeXt 最大（6.3 个点）。
   4 位验证受试者与 9 位官方测试受试者的分布差异仍是本实验的主要不确定来源。

> 曲线并非严格单调：多个模型在 50% 达到峰值、100% 略回落。这与 4 位受试者的验证集
> 选择、以及种子波动（多数 ±0.01–0.02，个别 ±0.03–0.05）有关；应看整体趋势与标准差，
> 不宜把单个点的排序差异当作结构优劣。

## 资源效率单独复测

训练时记录的延迟会受到系统负载影响；需要横向比较时，用同一进程、固定线程数重测：

```bash
conda run -n mamba_env python -m src.benchmark --threads 1 --num-runs 500 \
    --output results/benchmark_single_thread.json
```

## 复现约定

- 验证集按受试者分组（20% 受试者），同一受试者不会同时出现在训练集与验证集；
- 数据划分种子（`split_seed`）与小样本子集种子（`subset_seed`）固定，
  与模型种子分离；多种子实验的差异只来自训练随机性；
- 官方测试集固定，不参与训练或调参；
- 最佳模型按验证集 Macro-F1 选择；
- 每个结果文件记录模型种子、数据种子、训练比例、完整配置与受试者划分。
