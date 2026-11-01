# 股票收益预测：417因子模型实验

当前 `main.py` 使用 `prices.parquet` 对应的全部股票及 `data/features.parquet` 的全部记录进行模型实验。当前两张表均有300只股票、471,813行；运行时按文件内容确定股票池，不再使用固定股票名单或 `--codes` 子集入口。它不下载行情，也不重新计算因子。

随附 `prices.parquet` 包含300只股票、471,813行、215列，日期为2020-01-02至2026-10-13；`data/features.parquet` 有471,813行、425列，包含417个因子和未来1/5/30/60个交易观察后的收益标签。

## 当前运行状态

- 统一使用上级环境 `D:\Pycharm gyh\.venv\Scripts\python.exe`（Python 3.13.2）。项目内原 `.venv` 保留并忽略。
- 上级环境中的LightGBM、Streamlit 1.66.1、Plotly 7.1.0已可用；在线下载和可选模型依赖按所用功能另行验证。
- `.runtime-deps`、`.download-deps` 中含Python 3.12专用组件，不能直接加入Python 3.13环境。
- 随附 `results/metadata.json` 属于旧的50只股票、100因子实验，最新行情日期为2026-09-30；417因子版本尚未完成正式重训。
- 网页和HTML已适配当前实验输出。当前尚无正式全量重训结果，旧结果会显示清楚的等待提示，不会冒充当前实验。

## 本机环境配置

在PyCharm的项目Python解释器设置中选择上级环境 `D:\Pycharm gyh\.venv\Scripts\python.exe`，运行配置使用项目默认解释器，工作目录设为 `D:\Pycharm gyh\Fintech_1`。

在PyCharm底部Terminal中使用PowerShell，显式调用上级环境即可检查入口，不需要创建新环境：

```powershell
cd 'D:\Pycharm gyh\Fintech_1'
& '..\.venv\Scripts\python.exe' main.py --help
```

下文命令均从项目根目录执行。

`requirements.txt` 包含默认训练所需的LightGBM、XGBoost，以及数据下载、因子处理和网页所需的基础依赖。上级环境是共享环境，缺失依赖需按实际功能补装，本次股票池修改不执行安装。

## 模型实验入口

先查看当前入口支持的参数：

```powershell
& '..\.venv\Scripts\python.exe' main.py --help
```

仅进行1日周期的因子筛选审计，不训练预测模型：

```powershell
& '..\.venv\Scripts\python.exe' main.py --selection-only --horizons 1
```

审计会写入 `results/selection_audit/`，包含训练期因子去留原因和PCA候选维数，不代表验证集最优方案，也不生成新预测。

单独训练1日周期，或运行全部四个周期：

```powershell
& '..\.venv\Scripts\python.exe' main.py --horizons 1
& '..\.venv\Scripts\python.exe' main.py
```

以上两条命令是不同的运行选择。只有实验标识相同的成功结果才能合并周期；同周期重跑会替换该周期，其他兼容周期保留。不同数据、股票池或模型配置的结果不会合并，详见下节。

所有股票都进入同一套时间切分、训练、验证、测试和生产拟合流程。启动时核对行情表与因子表的完整“日期+股票代码”主键；少了股票、少了日期或存在重复主键时直接报错，不会退回部分股票实验。

“全股票池”不表示把测试标签提前用于调参。仍保留日期隔离带和未成熟标签过滤；上市较晚或历史不足的股票可能在早期分区没有有效样本。模型指标中的 `n_train_stocks`、`n_validation_stocks`、`n_test_stocks` 记录各分区实际覆盖数。

| 参数 | 作用 |
|---|---|
| `--horizons` | 逗号分隔的预测周期，支持1、5、30、60；默认全部 |
| `--selection-only` | 仅生成筛选审计 |
| `--neural` | 在默认模型基础上增加LSTM和Transformer的测试评估 |
| `--modern` | 在默认模型基础上增加RealMLP和TabPFN |
| `--model-device` | 传给现代表格模型的设备参数，支持 `cpu`、`cuda`，默认 `cpu` |
| `--nn-seeds` | 神经网络重拟合使用的种子数量，默认3 |

例如，全部股票只运行5日周期：

```powershell
& '..\.venv\Scripts\python.exe' main.py --horizons 5
```

当前 `main.py` 不支持 `--codes`、`--offline`、`--features-only`、`--data-only`、`--stocks`、`--end-date`、`--refresh` 或 `--mae-tolerance`。不要继续使用旧文档中的这些主程序命令。

## 实验版本隔离与历史结果

实验标识由行情文件和因子文件的SHA-256、完整股票名单、因子清单、预处理版本、模型选项、训练配置、实现代码及Python/关键依赖版本共同确定。预测周期不参与实验标识，因此相同实验可以先运行1日，再补跑5日。

训练期间，CSV和模型只写入 `.experiment_staging/<本次运行编号>/`。当前 `results/`、`saved_models/` 保持原样。全部训练与评价完成后再次检查输入和代码指纹，再统一发布：

- 实验标识相同、上次状态完整且产物校验通过：合并兼容周期；本次周期的旧记录被替换，不重复追加。
- 实验标识不同，或旧版结果没有实验标识：从空结果开始；成功发布时把原结果和模型整套归档到 `experiment_history/<旧实验标识或legacy>/<归档编号>/`。
- 当前目录的文件缺失或内容与产物清单不符：拒绝自动合并，不静默接受被修改的文件。
- 训练或评价报错：保留当前成功结果，失败产物留在暂存目录并标记 `failed`。发布阶段出现普通文件操作错误时，尝试回滚目录替换。
- 同时启动另一份训练会被拒绝，避免两个进程交叉覆盖当前结果。进程退出后操作系统释放文件锁。

每份结果CSV增加 `experiment_id`、`run_id`。`metadata.json` 与 `run_metadata.json` 统一记录完整实验身份、已完成周期、运行历史、当前产物清单及文件指纹，不再无条件拼接旧元数据。

旧版 `models/` 在首次发布新实验时与旧结果一起保留在历史快照的 `legacy_models/` 中；新生产模型始终位于当前 `saved_models/`。原始 `prices.parquet` 与 `data/` 不参与归档。

例如：相同300股票数据先跑1日、再跑5日，当前结果包含同一实验的1日和5日；数据或配置变化后只跑1日，当前结果仅包含新实验的1日，旧实验留在历史目录。

结果保存检查只使用临时目录与假产物，不拟合真实模型：

```powershell
& '..\.venv\Scripts\python.exe' -B -m unittest discover -s tests -v
```

## 模型与可选依赖

默认模型键为 `ols`、`ols_huber`、`ols3`、`pls`、`lasso`、`enet`、`gbrt`、`random_forest`、`lightgbm`、`xgboost`。其中 `ols`、`ols3` 当前用小正则化系数的Ridge实现。

LSTM和Transformer需要额外安装 `torch`；RealMLP和TabPFN需要 `pytabkit`、`tabpfn` 及其依赖。这些可选包不包含在基础依赖文件中：

```powershell
# 按需安装神经网络依赖
& '..\.venv\Scripts\python.exe' -m pip install torch
# 按需安装现代表格模型依赖
& '..\.venv\Scripts\python.exe' -m pip install pytabkit tabpfn
```

可选依赖与Python 3.13、GPU的兼容性仍需实际验证。TabPFN在没有本地权重时可能下载模型权重。所有表格模型在各阶段使用完整可用训练分区，TabPFN也不再静默抽取4000行；全量TabPFN的资源消耗较大，完整运行能力尚未验证。当前神经网络分支输出测试指标和测试预测，未保存对应生产模型，也未生成神经网络的最新收益预测。

## 数据更新与因子生成

现有主表和因子表可以直接用于实验。只有需要更新数据时才单独执行以下步骤。

`data.py` 在线下载行情、行业、大盘、财务、宏观、日历和海外指数，合并、校验后替换根目录 `prices.parquet`。其参数为 `--stocks`（默认300）、`--end-date`（YYYYMMDD）和 `--refresh`（忽略缓存）：

```powershell
& '..\.venv\Scripts\python.exe' data.py --help
& '..\.venv\Scripts\python.exe' data.py --stocks 300
```

更新主表后，需要重新生成全股票池因子，才能在模型实验中使用新行情。以下命令会覆盖因子表及质量报告：

```powershell
& '..\.venv\Scripts\python.exe' -c "from price_source import load_prices; from factors import make_features; make_features(load_prices())"
```

因子始终在全股票池计算，模型也使用同一完整股票池，保留行业排名等横截面因子的计算口径。

## 主要文件和输出

| 文件或目录 | 用途 |
|---|---|
| `price_source.py` | Parquet路径、主键与价格检查、源文件指纹 |
| `data.py` | 在线下载并合并 `prices.parquet`，缓存位于 `data/download_cache/` |
| `factors.py`、`extended_factors.py` | 计算417因子与标签，输出因子目录、字段用途和质量报告 |
| `factor_selection.py` | 训练期质量筛选、稳定性评分、去冗余和PCA |
| `main.py` | 核对完整行情/因子主键，使用全部股票按周期运行实验 |
| `models.py` | 模型调参、测试评估、生产模型拟合、最新预测和组合评价 |
| `experiment_store.py` | 实验标识、兼容周期合并、暂存发布、历史快照与失败保护 |
| `backtest_metrics.py` | 当前及旧版回测共用的最大回撤计算，包含初始净值1 |
| `report_data.py` | 网页与HTML共用的成功实验校验、字段与模型标签、策略筛选 |
| `report_charts.py` | 共用的行情、模型、预测IC和组合净值图 |
| `evaluate.py` | 原有因子IC与回测函数，当前 `main.py` 不直接调用 |
| `app.py`、`export_html.py` | 展示或导出当前实验的实际模型、周期、预测、评价与组合策略 |
| `saved_models/` | 当前训练代码保存的生产模型；`models/` 中是旧模型 |
| `results/model_metrics.csv`、`test_predictions.csv` | 测试指标与逐条测试预测 |
| `results/latest_predictions.csv`、`feature_importance.csv` | 生产模型最新预测与测试评估模型的重要性 |
| `results/model_selection_trials.csv`、`tuning_trials.csv` | 参数和预处理候选的验证指标 |
| `results/factor_selection.csv`、`preprocessing_summary.csv`、`pca_loadings.csv` | 各阶段筛选记录、预处理摘要和PCA载荷 |
| `results/selected_features.csv` | 模型和周期使用的因子清单 |
| `results/ic_daily.csv`、`ic_summary.csv`、`per_stock_metrics.csv` | 预测值的截面IC、Rank IC及逐股误差；区别于原始因子IC |
| `results/backtest_nav.csv`、`backtest_summary.csv` | 新的5对5多空及前5名相对等权组合评价 |
| `results/factor_group_ablation.csv` | 1日、5日周期的因子组消融实验 |
| `results/metadata.json`、`run_metadata.json` | 输入指纹、股票名单、周期和运行记录 |
| `experiment_history/` | 成功发布时保留的上一套完整结果和模型 |
| `.experiment_staging/` | 本次训练的暂存输出与成功/失败状态；未成功输出不会作为当前结果 |

原有 `factor_ic_daily.csv`、`factor_ic_summary.csv` 以及 `evaluate.py` 可生成的 `factor_time_ic_summary.csv`，不会由当前 `main.py` 自动更新，不能把目录中的旧文件当成新一轮实验结果。

## 展示页面

回测最大回撤统一按 `min(NAV / 历史最高净值 - 1)` 计算，历史高点包含建仓前的初始净值1。指标沿用非正数表示，例如初始1、首期跌至0.799、之后不变，最大回撤为-20.1%，不会报告为0。无有效净值观察时返回NaN，不冒充零回撤。

当前组合评价和旧 `evaluate.py` 回测共用此公式，持仓、成本、累计收益和Sharpe的算法不变。已有历史CSV不自动改写；后续回测会生成修正指标，且公式代码已纳入实验版本标识，不能与旧公式结果混合。

使用上级环境启动网页：

```powershell
& '..\.venv\Scripts\python.exe' -m streamlit run app.py
```

网页和HTML共用以下读取规则：只展示 `status=complete` 的实验，核对实验标识、行情/因子指纹、产物文件指纹和每条记录的实验编号。结果发布或产物变化时网页缓存失效；未登记的历史CSV不会作为回退数据。

页面分为最新预测、模型评价、预测IC与逐股误差、组合回测、因子与预处理五部分。模型及周期由实际数据决定，支持当前各类模型，未知模型名也会原样显示；LSTM/Transformer只有测试评价时明确标注，不生成虚假的最新预测或零重要性解释。

回测按模型、周期、策略、费用四项选择，读取当前的 `nav`、`cumulative_return` 等字段。多空5对5与前5名相对全池等权分别解释；10/20 bp按实际输出展示，费用为费率乘换手率，不再沿用旧“每期固定扣0.1%”说明。相对收益曲线不会伪装成实际基准资产净值。

当前的 `ic_daily.csv`、`ic_summary.csv` 是模型预测IC/Rank IC，不是单因子IC。网页和HTML不会再读取旧 `factor_ic_*` 排行榜。没有生成的预测、重要性、回测或消融结果显示为空状态。

导出当前成功实验的单文件HTML：

```powershell
& '..\.venv\Scripts\python.exe' export_html.py
& '..\.venv\Scripts\python.exe' export_html.py --tables-only --output '当前实验_数据表.html'
```

默认新文件名为 `股票收益预测_当前实验报告.html`，保留已有历史 `股票收益预测项目_展示样板.html`。HTML包含本次所有模型和周期的数据表，可筛选、搜索、分页；图表可切换实际周期和组合，Plotly脚本内嵌，导出后无需联网加载绘图库。

`--result-dir` 可以指定成功实验的结果目录；查看历史实验时可配合 `--price-file`、`--feature-file` 指定对应快照。读取校验失败不会覆盖已存在的报告。`--tables-only` 不加载绘图库或历史行情，仅导出完整可交互数据表。

## 因子与计算口径

保留原137个因子（100个量价因子、37个市场行业因子），新增280个，共417个。新增因子覆盖基本面、估值、成长、宏观、日历、全球市场、信息质量、行业排名与交互。

- [新增280个因子的完整公式](新增因子完整公式.md)
- [训练前因子筛选与降维说明](因子筛选与降维说明.md)

417个候选因子先进行质量过滤、训练期IC/稳定性评分和高相关去冗余，再由验证集比较因子数量、PCA和模型参数。默认选择验证MAE最低的方案，保留质量过滤基准，不预设压缩一定改善精度。

因子筛选评分保留训练期均衡采样，以控制评分与相关矩阵的内存开销；这个评分样本不限制预测模型的训练股票池或训练行数。IC的最低有效股票数和回测的5对5持仓数也不代表训练股票数量。

收益标签为个股后续第h条交易观察的收盘价除以当前收盘价减1。按日期约70%/15%/15%切分，并使用隔离带及实际标签终点检查边界。测试集不参与调参；生产模型在测试评价结束后，另用全部成熟样本拟合。

公共宏观、日历和海外数据按A股日期去重计算；新增因子保留缺失，不用未来值回填。因子含当日收盘信息，整体为收盘后预测。源表仍可能存在前复权回溯、财务重述、当前成分股和行业快照偏差，增加因子本身不能证明预测效果改善。
