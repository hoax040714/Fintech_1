# 股票收益预测：在线下载 → Parquet → 因子与预测

默认由 `data.py` 在线下载行情、行业、大盘、财务、宏观、日历和海外指数，合并生成根目录 `prices.parquet`。后续因子、筛选、降维、训练和展示仍统一读取Parquet，不直接读取下载缓存CSV。

当前随附快照：300只股票，471,813行、215列，2020-01-02至2026-10-13。在线更新后股票池、行数和日期范围以新文件为准；默认下载2020年以来的沪深300当前全部成分股。

## 因子

保留原源码已有的137个因子（100个量价因子、37个市场行业因子），新增280个因子，共417个。新因子覆盖基本面、估值、成长、宏观利率、GDP、日历、全球市场、信息质量、行业排名与交互。

- [新增280个因子的完整公式](新增因子完整公式.md)
- [训练前因子筛选与降维说明](因子筛选与降维说明.md)
- [在线下载与Parquet接口说明](在线下载与Parquet接口说明.md)

417个候选因子不会直接全部送入每个预测模型。训练前会做质量过滤、训练期IC/稳定性评分、高相关去冗余，再由验证集比较因子数量、PCA和模型参数。默认严格选择验证MAE最低者；保留清洗基准，不假设压缩一定更准确。

## 运行

在可用Python 3.10+环境安装依赖后执行：

```powershell
python -m pip install -r requirements.txt
python data.py                        # 只下载并生成prices.parquet
python main.py                        # 默认先更新数据，再完整训练/评价/预测
python main.py --offline --features-only
python main.py --offline --selection-only
streamlit run app.py
```

`--data-only` 只在线更新Parquet；`--features-only` 默认先在线更新，再生成因子与质量报告，保留旧模型和旧预测。完整 `python main.py` 会先更新数据，再重新训练、生成新预测、计算两类IC并回测。

`--selection-only` 在训练分区做筛选和PCA候选维数诊断，不训练最终模型，也不宣布哪个候选精度最好。`--mae-tolerance` 默认0；可显式设为0.005，允许验证MAE相对最优值上浮最多0.5%时优先更低维。

`--offline` 跳过下载、复用已有Parquet；`--refresh` 忽略原始缓存重新下载。在线模式 `--stocks N` 控制下载股票数并替换主Parquet，默认300；离线模式只筛选现有股票。`--end-date YYYYMMDD` 在线控制下载截止日、离线筛选已有日期。离线不能与 `--refresh/--data-only` 同时使用。

## 主要文件

| 文件 | 用途 |
|---|---|
| `price_source.py` | 唯一Parquet入口、主键/价格检查、源文件指纹 |
| `factors.py` | 保留原因子，接入增量因子，构建标签和派生表 |
| `extended_factors.py` | 新增280因子、公式/依赖注册、时间检查、字段用途 |
| `factor_selection.py` | 仅训练期拟合的质量筛选、稳定性评分、去冗余、数量候选和PCA |
| `main.py` | 默认串联在线下载和后续流程，支持离线及分阶段运行 |
| `models.py` | 原Lasso/Ridge/RandomForest，训练集填充与标准化 |
| `evaluate.py` | 横截面IC、共同因子时间序列IC、原回测 |
| `app.py` / `export_html.py` | 使用Parquet行情展示/导出，拒绝混用旧版本结果 |
| `data.py` | 在线下载、合并完整数据，校验后原子写入prices.parquet |

## 输出

- `prices.parquet`：唯一合并主表；所有下载和字段校验完成后才替换旧文件。
- `data/download_cache/`：下载层原始缓存及结构说明，独立于原有只读CSV；下游不直接读取。
- `data/features.parquet`：417因子及未来1/5/30/60个交易观察的收益标签，共425列。
- `data/factor_catalog_extended.csv`：417因子名称、类别、公式、原始依赖、是否原有、IC口径。旧只读 `factor_catalog.csv` 保留。
- `data/field_usage.csv`：215个原始字段逐个列出用途和缺失率。
- `data/factor_quality.csv`：各因子缺失率、唯一值数和定义。
- `results/factor_ic_daily.csv`、`factor_ic_summary.csv`：有横截面差异的因子IC。
- `results/factor_time_ic_summary.csv`：219个共同因子对股票池等权未来收益的时间序列IC，训练/验证/测试分开。
- `results/selection_audit/`：仅筛选审计命令生成的训练期去留原因与PCA候选维数；不代表验证赢家。
- 完整训练另生成 `model_selection_trials.csv`、`factor_selection.csv`、`preprocessing_summary.csv`、`pca_loadings.csv`，记录候选验证误差、每阶段因子去留和PCA载荷。
- 其他 `results/` 和 `models/` 文件继续由完整训练流程生成。

页面会核对模型的因子清单和输入文件SHA-256。旧CSV模型结果不能直接配合新的Parquet行情展示，需先完整重训。静态HTML旧文件保留，重训后运行 `python export_html.py` 才会生成新报告。

页面还检查预处理版本；此前未经筛选的417因子模型也需要重训。保存的joblib包含筛选列、缩尾边界、填充、缩放、PCA和模型，预测时不会重新选择因子。

## 计算口径

原137因子公式及4个未来收益标签公式不变。公共宏观/日历/海外数据按A股日期去重计算，财务、宏观和海外字段检查原表可用时间；新因子保留缺失，不用未来值回填。零成交量或全缺失不冒充有效信号。

原模型仍按日期约70%/15%/15%切分，在边界保留预测周期长度的标签隔离带。模型只在训练部分拟合填充和标准化。新增共同因子与个股因子的IC分开评价。

原因子含当日收盘信息，整体仍为收盘后预测。仅使用本地文件无法消除源表本身可能存在的前复权回溯、财务重述、当前成分股及行业快照偏差；新增特征不意味着预测能力已经改善。
