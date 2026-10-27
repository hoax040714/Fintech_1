# 沪深300成分股收益预测

这是一个尽量简单、可读的课程项目：使用 AkShare 获取2020年1月1日至最近收盘日的沪深300全部成分股前复权日频行情，匹配行业和大盘数据，构造137个因子，比较10个传统模型，并可选运行4个神经网络及预训练表格模型。Streamlit 展示预测、模型表现、分类因子 IC 和回测。

## 数据口径

- 每次运行从中证指数最新成分股名单获取沪深300全部300只股票，不用较旧的权重表决定样本。
- 行情从2020-01-01开始，截止日默认取运行当天。北京时间15:30之前排除当日日线，节假日及数据源延迟以实际返回的交易日为准。
- 每只股票从申万一级行业的当前成分股表反查所属行业，按“日期+行业”匹配行业日线；大盘统一使用沪深300，按日期匹配。
- 过期的个股、行业和大盘行情缓存会重新下载；读取后按指定日期范围筛选，避免混入2020年以前的数据。
- 固定使用当前成分股会有生存偏差；本项目适合课程演示，不适合据此下实盘结论。
- 预测目标为未来 1、5、30、60 个交易日的简单收益率。

## 137 个分类因子

因子分为价格类15个、成交量与流动性类18个、波动率类16个、动量类10个、反转类11个、Alpha组合类30个、市场与行业类37个。名称、类别和定义统一保存在 `data/factor_catalog.csv`，构建公式集中在 `factors.py`。市场与行业类使用匹配后的行业、沪深300价量字段，包括市场及行业状态、相对收益、Beta、相关性与量能共振；本次仅调整数据范围，保留所有现有公式。

## 模型

默认运行 Lasso、Ridge、Random Forest、OLS（弱Ridge）、OLS3（双因子）、PLS、ElasticNet、GBRT、LightGBM 和 XGBoost。Lasso、Ridge、Random Forest 保留本项目原参数，参考目录的同类模型不重复加入；其他传统模型沿用参考目录的固定参数。OLS名称沿用参考代码，但实际上是 `Ridge(alpha=0.0001)`；OLS3实际上只使用训练期相关性最高的2个因子，不能解释为三因子财务模型。

LSTM、Transformer、RealMLP、TabPFN通过参数选用，默认不运行。OLS3只使用2个因子，这4个可选模型使用训练期筛选的前30个因子，其余传统模型使用全部训练期可用因子。筛选、填补和标准化均不接触测试期标签。TabPFN每次拟合只抽样128条允许训练的记录，预测批次为32条，抽样数量会写进结果说明。

LSTM与Transformer使用每只股票自己的20日历史窗口。网络结构沿用参考目录，验证集选择训练轮数，再按选出的轮数重新训练；网络权重、预处理统计量和特征顺序都保存在模型文件中。窗口按批次展开，避免将300只股票的全部窗口一次性堆进内存。验证、测试分区各自形成窗口，因此每只股票分区开头19条记录不进入该模型的评价。

## 文件结构

```text
Fintech_1/
  data.py             # 股票、行业和大盘数据下载、清洗与匹配
  factors.py          # 137 个分类因子与未来收益标签
  models.py           # 模型定义、训练、测试指标与预测
  evaluate.py         # 因子 IC 与样本外回测
  main.py             # 主程序入口
  app.py              # Streamlit 四页展示
  requirements.txt
  requirements-neural.txt   # 可选 PyTorch
  requirements-modern.txt   # 可选 RealMLP / TabPFN
  data/               # 自动生成的个股、行业、大盘、因子与因子目录 CSV
  models/             # 自动生成的模型
  results/            # 指标、预测、IC、净值
```

## 运行

建议 Python 3.10 以上。

## GitHub与三人协作

本仓库只管理代码、说明文档及网页主题，不包含完整数据、因子CSV、模型权重、训练结果、本地环境或备份。当前电脑在GitHub Desktop中选择 **File → Add local repository → `D:\Fintech_1`**；其他成员克隆同一仓库到自己的文件夹。不要把 `data/` 当作仓库根目录，也不要再嵌套创建同名项目。

首次安装、数据包共享、推荐分工及“任务分支→提交→PR检查→合并main”的简单流程见[团队协作说明](团队协作说明.md)。完整字段说明见[合并大表数据目录](data/合并大表数据目录.md)。克隆代码不会自动带回原数据或预测页面，数据包和网站部署需要分别处理。

### 本地运行命令

```powershell
cd D:\Fintech_1
python -m pip install -r requirements.txt
python main.py --refresh
streamlit run app.py
```

第一次需要下载300只股票，耗时取决于数据接口。下载时每只股票单独缓存；若中途断开，再运行不带 `--refresh` 的命令可复用已更新的单股文件：

```powershell
python main.py
```

只更新股票、行业和大盘数据并完成匹配，不训练模型：

```powershell
python main.py --data-only
```

数据范围变更后，原有模型和结果不会自动变成300只股票的新结果，需要完整运行 `python main.py` 后再查看新的预测与评估。

快速检查代码也可以减少股票数（按股票代码排序截取），但少于10只将无法按当前口径计算横截面 IC：

```powershell
python main.py --stocks 12
```

按需运行模型或周期，模型名称用空格分隔：

```powershell
python main.py --models lasso ridge lgbm xgb --horizons 1 5
python -m pip install -r requirements-neural.txt
python main.py --models lstm transformer --horizons 1 --nn-epochs 8
python -m pip install -r requirements-modern.txt
python main.py --models realmlp --horizons 1
python main.py --neural --modern
```

`--neural` 增加LSTM和Transformer，`--modern` 只增加RealMLP，同时使用时共运行13个模型。默认运行CPU，具备CUDA环境时可传 `--device cuda`。按使用者选择，TabPFN暂不授权，只保留 `--models tabpfn` 显式入口，不随上述两个开关启动；以后解决官方许可或本地权重后再使用。若已有兼容权重，可在 `models.py` 的 `model_path` 指定路径。

每次训练生成的是当次所选模型及周期的一套CSV结果，不自动把不同实验的结果混在一起。默认10模型×4周期为40组结果；全部14模型×4周期为56组。少量股票试运行的结果只用于检查通路，不用于判断模型优劣。

## 方法说明

- 样本按交易日顺序切成约 70% 训练、15% 验证、15% 测试。
- 训练集和验证集尾部各留出与预测周期相同的隔离带，避免未来收益标签跨越分区。
- Lasso/Ridge 的缺失值填充与标准化都只在训练数据拟合；随机森林仅做缺失值填充。
- 超参数用验证集 MAE 选择，最终测试指标不参与调参。
- 因子 IC 是每日股票横截面的 Pearson 相关系数，再按日期等权平均。
- 回测只使用测试集预测；每隔预测周期调仓，等权持有预测收益最高的 10 只股票，并扣 0.1% 调仓成本。

## 主要输出

- `results/model_metrics.csv`：测试集 MAE、RMSE、R²、方向准确率、预测相关系数。
- `results/latest_predictions.csv`：每只股票的最新预测与参考目标价。
- `data/factor_catalog.csv`：137 个因子的类型与中文定义。
- `results/factor_ic_summary.csv`：137 个因子的平均 IC 排名。
- `results/feature_importance.csv`：系数、树模型内置重要性、置换重要性或输入层权重，`importance_type` 标明口径；序列网络输入层权重不代表预测重要性。
- `results/backtest_summary.csv`：策略与等权基准的回测摘要。

> 说明：前复权历史会随分红送转而变化。需要严格复现实验时，应保存当次下载的原始文件与股票池文件。

模型及参数参考 `D:\模型\模型\models.py`，没有加载或搬用该目录针对两只股票、不同因子顺序拟合的旧模型工件。缺少可选依赖时，Python直接报告原始导入错误；项目未增加 `try/except`、插件注册框架或复杂版本适配。

接入验证已完成10个传统模型、LSTM、Transformer、RealMLP，共13个模型的真实小样本训练、保存和重加载，以及Streamlit四页兼容检查。TabPFN已接入构造与训练入口，但实际下载权重要求用户登录并接受官方许可，目前未完成训练验证。正式300只股票全量训练尚未执行；验证结果单独保存，不覆盖正式结果。
