# 417 因子模型实验入口

本目录提供独立的 417 因子、14 模型训练流程。它读取项目根目录的 `data/features.parquet`，默认选择任务中的十只股票，并将审计表、指标、预测和模型工件写入根目录 `results/` 与 `saved_models/`。

本包保留项目根目录的 `main.py`、`models.py` 接口，避免影响现有数据下载流程和 Streamlit 页面。依赖 B 的 417 因子代码以及 `price_source.py`，因此应与包含这些文件的分支一起使用。

## 运行

在项目根目录执行：

```bash
python -m modeling_417.main --selection-only --horizons 1,5,30,60
python -m modeling_417.main --horizons 1 --neural --modern --model-device cuda --nn-seeds 3
```

后续周期可分批执行；结果会按周期合并，同周期重跑会替换旧结果：

```bash
FINTECH_CPU_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
python -m modeling_417.main --horizons 5,30,60 --neural --modern --model-device cuda --nn-seeds 3
```

默认股票代码为 `300750,300308,600519,601318,300502,601899,600036,000333,603259,688256`，可以用 `--codes` 覆盖。因子必须先在完整股票池上构建，之后再由模型入口筛选股票。

LSTM 和 Transformer 在 CPU 上按每只股票独立构造 20 日窗口，不保存网络权重。RealMLP 和 TabPFN 顺序运行，并使用 `--model-device` 指定设备；TabPFN 在训练行数超过 8,000 时抽样 4,000 行。默认最多使用一个 CPU 线程，`FINTECH_CPU_THREADS` 可调整树模型和 RealMLP 的线程数。

运行指标用于课程流程验证。十只股票无法支持十分位组合；每日截面 IC 和有限的调仓次数不构成投资结论。CUDA 设备配置下保存的 RealMLP/TabPFN 工件换机器使用前应检查设备兼容性，必要时在 CPU 上重训。
