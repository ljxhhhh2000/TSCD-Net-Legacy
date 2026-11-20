# TSCD-Net Legacy 逐帖症状时序模型复现

TSCD-Net Legacy 从一名用户的微博历史预测其 SWDD 用户级标签。每条微博对应一个时间步，输入包括 **15 个症状相似度分数**和**相邻发帖间隔**。模型在同一时间步内学习症状之间的关系，再沿帖子顺序做时间注意力。这个项目保留逐帖结构，适合复现论文方法、做消融和查看症状轨迹。 [TSCD-Net 论文](https://doi.org/10.1109/ICCECE69169.2026.11399767)

**第一次使用**可以先跑 [测试和小样本 Demo](#测试和小样本-demo)。已有缓存时，可直接按 [完整实验](#完整实验) 训练。该流程在本项目内完成。

## 输入、模型和输出

```text
微博文本 ── 冻结 SBERT ── 与 15 个症状描述计算余弦相似度
发帖时间 ── 相邻时间差 ── log(1 + 间隔天数) ── Time2Vec
                            ↓
逐帖症状向量 → 帖内症状自注意力 → 帖间时间自注意力 → 用户级预测
```

特征提取把余弦相似度映射到 `[0, 1]`。每名用户的主实验序列最多保留最近 500 条有效微博。注意力和用户池化通过长度掩码排除补齐位置。分类训练按用户级标签做加权二元交叉熵，正类权重用当轮训练集的类别数计算。 [模型实现](src/tscd_net_legacy/models/tscdnet.py) · [特征提取](src/tscd_net_legacy/features.py)

## 安装

以下命令从本项目根目录运行。推荐 Python 3.10。固定版本的 `requirements.txt` 使用 CUDA 12.1 的 PyTorch，适合复用已验证的 GPU 环境。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m tscd_net_legacy --help
```

使用 CPU 或已有 PyTorch 时，可在环境中执行 `python -m pip install -e ".[cjk,test]"`。绘制症状轨迹时加装 `analysis` 扩展。Windows 下可以直接使用虚拟环境的 Python 执行示例。 [项目依赖声明](pyproject.toml)

## 准备数据和句向量模型

数据支持 SWDD 的原始 ZIP 或包含两个 JSONL 文件的目录。

```text
SWDD/
├── control.jsonl
└── depressed.jsonl
```

每行包含一个用户。数据读取使用 `label.depressed`、`user.screen_name` 和 `tweets` 里的文本与时间。可选的 `label.symptoms` 用于用户级症状对齐评估。主实验保留原始类别比例并纳入转发。 [SWDD 数据发布页](https://github.com/ethan-nicholas-tsai/SWDD) · [数据读取规则](src/tscd_net_legacy/data.py)

示例配置里的 SBERT 路径指向原实验机器。首次在自己的环境中运行时，复制配置并按下表填写，保留其他字段。

| 配置字段 | 怎样填写 |
|---|---|
| `data.dataset_zip` | SWDD ZIP 或解压目录的绝对路径 |
| `features.embedding_model` | 本地 SBERT 模型目录，或 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `features.embedding_device` | 特征提取设备，可设 `cpu`、`cuda` 或 `auto` |
| `model.device` | 分类模型设备，可设 `cpu`、`cuda` 或 `auto` |
| `experiment.artifact_dir` | `prepare` 写入、`train` 读取的特征目录 |
| `experiment.output_dir` | 当前实验的指标目录，建议每次改参数用不同名称 |

使用模型标识时，首次特征提取需要能获取该模型文件。使用已有缓存时，训练配置中的 `embedding_model` 要与缓存记录一致，训练会直接读取已算好的分数。

## 测试和小样本 Demo

### 无数据验证

```powershell
python -m pytest -q
```

测试使用小型输入检查症状集合、读取规则、模型形状和掩码。

### 从 SWDD 验证完整链路

复制 Demo 配置，按上表填写数据路径与句向量模型。用 CPU 时，建议在配置中将 `features.embedding_device` 设为 `cpu`。

```powershell
Copy-Item configs/demo.json configs/demo.local.json
```

下面的命令使用这份小样本配置训练一个 epoch。

```powershell
python -m tscd_net_legacy all --config configs/demo.local.json --device cpu --epochs 1 --artifact-dir artifacts/quickstart --output-dir runs/quickstart
Get-Content runs/quickstart/metrics_summary.csv
```

Demo 只扫描每个 JSONL 的前 50 行，每类最多 40 人，使用轻量模型和 120 步分类输入。这里生成真实 SBERT 特征，用于验证从数据读取到指标输出的链路。 [Demo 配置](configs/demo.json) · [早期验证记录](VALIDATION.md)

## 完整实验

从零开始时，复制 `configs/main.json` 为 `configs/main.local.json`，按上表设置数据和模型位置。独立使用本项目时，可以将特征目录设为 `artifacts/main`。随后先生成特征，再训练。

```powershell
python -m tscd_net_legacy prepare --config configs/main.local.json
python -m tscd_net_legacy train --config configs/main.local.json
```

首次特征提取会处理大量微博。中断后可按相同配置继续生成缓存。训练按用户分层划分 **70%／10%／20%**，默认重复五个种子，采用验证集正类 F1 选最佳权重，固定 0.5 阈值计算最终指标。正式模型的症状注意力和时间注意力各用 2 层、4 头，隐藏维度为 64。 [主配置](configs/main.json)

当前工作区的主配置已指向 `../../Dataset/TSCD-Net-Features/main`。要验证共用缓存，直接运行一个短实验即可。

```powershell
python -m tscd_net_legacy train --config configs/main.json --runs 1 --epochs 1 --output-dir runs/check-main-cache
```

### 消融与分析

下列内置消融配置使用当前工作区主特征缓存。自建缓存可用 `--artifact-dir` 指定位置，并在消融配置中设置与缓存相同的模型标识。

| 问题 | 配置或命令 |
|---|---|
| 时间间隔编码有何作用 | `train --config configs/ablation_no_temporal_encoding.json` |
| 帖内症状关联有何作用 | `train --config configs/ablation_no_symptom_attention.json` |
| 帖间注意力有何作用 | `train --config configs/ablation_no_temporal_attention.json` |
| 只看部分症状 | `train --config configs/main.local.json --symptom-keys sadness,anhedonia --output-dir runs/symptom-subset` |
| 查看代表用户的症状轨迹 | `analyze --config configs/main.local.json` |
| 检查症状相似度排序 | `align --config configs/main.local.json` |

执行表内命令时，在前面加 `python -m tscd_net_legacy`。`align` 使用的是本地 SWDD 出版数据提供的**用户级**症状标注，输出也是用户级对齐指标。

## 已有结果与输出

当前工作区的主实验测试集有 4,462 名用户，其中正类 717 名。五轮指标均值为 **Accuracy 90.60%、Precision 71.91%、Recall 69.29%、F1 70.30%**，F1 标准差为 2.72 个百分点。实验复用了公开的多语言 SBERT 和论文英文症状描述，并用本地用户划分运行。工作区归档为 `results/main/metrics_summary.csv`。

| 文件 | 用途 |
|---|---|
| 特征目录中的 `features.npz` | 症状分数、对数间隔、有效长度、用户标签和症状向量 |
| 特征目录中的 `features.json`、`dataset_manifest.json` | 缓存维度及采样情况 |
| 特征目录中的 `user_features/` | 逐用户提取文件，用于续跑特征生成 |
| `runs/<实验名>/run_config.json` | 运行时的完整配置 |
| `runs/<实验名>/metrics_by_run.jsonl` | 每轮指标 |
| `runs/<实验名>/metrics_summary.csv` | 模型指标均值与标准差 |
| `runs/<实验名>/analysis/` | 轨迹和用户症状统计 |
| `runs/<实验名>/symptom_alignment_user_level.json` | 用户级症状对齐结果 |

旧版在训练进程中保存最佳状态供当轮测试。历史实验目录保存指标和配置，新实验仍写入 `runs/`。

## 从哪里读代码

| 想了解什么 | 文件 |
|---|---|
| 命令行如何工作 | [cli.py](src/tscd_net_legacy/cli.py) |
| 清洗、特征缓存和对数时间 | [cleaning.py](src/tscd_net_legacy/cleaning.py) · [features.py](src/tscd_net_legacy/features.py) |
| 数据如何划分 | [splits.py](src/tscd_net_legacy/splits.py) |
| 症状和时间怎样进入模型 | [tscdnet.py](src/tscd_net_legacy/models/tscdnet.py) · [time2vec.py](src/tscd_net_legacy/models/time2vec.py) |
| 怎样训练并输出指标 | [torch_trainer.py](src/tscd_net_legacy/models/torch_trainer.py) · [experiment.py](src/tscd_net_legacy/experiment.py) |
