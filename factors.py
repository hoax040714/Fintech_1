"""六类共100个基础量价因子，以及未来收益标签。"""

import numpy as np
import pandas as pd

from price_source import DATA_DIR, FEATURE_FILE, FACTOR_CATALOG_FILE, prepare_folders
from extended_factors import (
    CATALOG, EXTRA_FEATURES, DAILY_FEATURES, STOCK_FEATURES,
    build_daily_factors, build_stock_factors, add_cross_factors, field_usage,
)


HORIZONS = (1, 5, 30, 60)

# 因子名称、类型和定义统一放在这里，便于训练、评价和网页共同使用。
PRICE_INFO = {
    "return_1d": "1日收盘收益率",
    "open_to_close": "开盘价/收盘价-1",
    "high_to_close": "最高价/收盘价-1",
    "low_to_close": "最低价/收盘价-1",
    "amplitude": "(最高价-最低价)/开盘价",
    "upper_shadow": "上影线长度/开盘价",
    "lower_shadow": "下影线长度/开盘价",
    "price_position": "收盘价在当日高低价区间中的位置",
    "log_price": "收盘价的自然对数",
    "price_change": "收盘价的1日绝对变化",
    "ma_gap_5d": "收盘价相对5日均线的偏离",
    "ma10_ratio": "收盘价相对10日均线的偏离",
    "ma_gap_20d": "收盘价相对20日均线的偏离",
    "ma60_ratio": "收盘价相对60日均线的偏离",
    "distance_52w_high": "收盘价相对252日最高价的距离",
}

VOLUME_INFO = {
    "volume": "当日成交量",
    "amount": "当日成交额",
    "volume_ratio_20d": "成交量/20日平均成交量",
    "turnover_rate": "当日换手率",
    "volume_change": "成交量1日变化率",
    "amount_change": "成交额1日变化率",
    "volume_price_corr": "20日成交量与收盘价相关系数",
    "obv": "按涨跌方向累计的成交量",
    "obv_change": "OBV的1日变化率",
    "volume_std": "20日成交量标准差",
    "large_order_ratio": "成交额/(成交量×收盘价)，不代表真实大单数据",
    "volume_price_divergence": "5日成交量变化率减5日价格变化率",
    "volume_trend": "5日平均成交量/20日平均成交量",
    "amount_trend": "5日平均成交额/20日平均成交额",
    "volume_breakout": "成交量/60日最大成交量",
    "volume_ratio_5d": "成交量相对5日均量的偏离",
    "turnover_5d": "5日平均换手率",
    "illiquidity_20d": "20日平均绝对收益/亿元成交额",
}

VOLATILITY_INFO = {
    "volatility_5d": "5日日收益率标准差",
    "volatility_10d": "10日日收益率标准差",
    "volatility_20d": "20日收益率标准差×sqrt(20)",
    "volatility_60d": "60日日收益率标准差",
    "vol_ratio_5_20": "5日波动率/20日原始波动率",
    "vol_ratio_20_60": "20日原始波动率/60日波动率",
    "atr": "14日平均真实波幅",
    "atr_ratio": "14日ATR/收盘价",
    "downside_volatility": "20日负收益标准差",
    "upside_volatility": "20日正收益标准差",
    "volatility_skew": "20日收益率偏度",
    "volatility_kurtosis": "20日收益率峰度",
    "return_variance": "20日收益率方差",
    "max_drawdown": "收盘价相对20日滚动最高价的回撤",
    "volatility_change": "20日原始波动率的1日变化率",
    "intraday_range": "(最高价-最低价)/前一日收盘价",
}

MOMENTUM_INFO = {
    "momentum_5d": "5日收盘价动量",
    "momentum_10d": "10日收盘价动量",
    "momentum_20d": "20日收盘价动量",
    "momentum_60d": "60日收盘价动量",
    "momentum_120d": "120日收盘价动量",
    "weighted_momentum": "5/10/20/60日动量的加权组合",
    "momentum_acceleration": "5日动量减20日动量",
    "rsi_14d": "14日相对强弱指标",
    "macd": "12日与26日EMA差的9日信号偏离",
    "momentum_ratio": "20日动量/60日动量",
}

REVERSAL_INFO = {
    "reversal_1d": "1日收益率的相反数",
    "reversal_3d": "3日收益率的相反数",
    "reversal_5d": "5日收益率的相反数",
    "reversal_10d": "10日收益率的相反数",
    "reversal_20d": "20日收益率的相反数",
    "intraday_reversal": "开盘至收盘收益率的相反数",
    "overnight_reversal": "前收盘至当日开盘收益率的相反数",
    "high_low_reversal": "当日最高价/最低价-1的相反数",
    "volume_weighted_reversal": "1日反转×成交量/20日均量",
    "extreme_reversal": "20日内绝对值最大单日收益的相反数",
    "reversal_strength": "1日反转与5日反转的乘积",
}

ALPHA_INFO = {
    "alpha_trend_strength": "5日平均收益/5日收益标准差",
    "alpha_breakout": "收盘价相对20日最高价的距离",
    "alpha_volume_breakout": "20日量比×20日均线偏离",
    "alpha_price_efficiency": "20日净价格变化/20日价格路径长度",
    "alpha_vol_adj_momentum": "20日动量/20日波动率",
}

_alpha_index = 6
for _momentum in ("momentum_5d", "momentum_10d", "momentum_20d", "momentum_60d", "momentum_120d"):
    for _volatility in ("volatility_5d", "volatility_10d", "volatility_20d", "volatility_60d"):
        ALPHA_INFO[f"alpha_{_alpha_index}"] = f"{_momentum}/({_volatility}+0.001)"
        _alpha_index += 1

ALPHA_INFO.update(
    {
        "alpha_26": "10日动量×20日量比",
        "alpha_27": "5日反转×10日波动率",
        "alpha_28": "14日RSI-50",
        "alpha_29": "MACD/(ATR+0.001)",
        "alpha_30": "当日换手率×5日动量",
    }
)

CONTEXT_INFO = {
    # 市场状态：10个
    "market_momentum_5d": "沪深300 5日动量",
    "market_momentum_20d": "沪深300 20日动量",
    "market_momentum_60d": "沪深300 60日动量",
    "market_ma_gap_20d": "沪深300收盘价相对20日均线偏离",
    "market_volatility_20d": "沪深300 20日收益波动率×sqrt(20)",
    "market_drawdown_60d": "沪深300相对60日高点回撤",
    "market_volume_ratio_20d": "沪深300成交量/20日平均成交量",
    "market_volume_change_5d": "沪深300成交量对数5日变化",
    "market_volume_zscore_20d": "沪深300成交量对数20日标准分",
    "market_price_volume_corr_20d": "沪深300收益与成交量变化20日相关性",

    # 行业状态：11个
    "industry_momentum_5d": "所属行业5日动量",
    "industry_momentum_20d": "所属行业20日动量",
    "industry_momentum_60d": "所属行业60日动量",
    "industry_ma_gap_20d": "所属行业收盘价相对20日均线偏离",
    "industry_volatility_20d": "所属行业20日收益波动率×sqrt(20)",
    "industry_drawdown_60d": "所属行业相对60日高点回撤",
    "industry_volume_ratio_20d": "行业成交量/行业20日平均成交量",
    "industry_amount_ratio_20d": "行业成交额/行业20日平均成交额",
    "industry_volume_change_5d": "行业成交量对数5日变化",
    "industry_amount_change_5d": "行业成交额对数5日变化",
    "industry_price_volume_corr_20d": "行业收益与成交量变化20日相关性",

    # 个股相对市场/行业：16个
    "market_excess_1d": "个股1日收益减沪深300收益",
    "market_excess_5d": "个股5日收益减沪深300 5日收益",
    "market_excess_20d": "个股20日收益减沪深300 20日收益",
    "market_excess_60d": "个股60日收益减沪深300 60日收益",
    "market_beta_60d": "个股对沪深300的60日滚动Beta",
    "market_corr_60d": "个股与沪深300的60日收益相关性",
    "residual_volatility_60d": "剔除市场影响后的60日特质波动率",
    "industry_excess_1d": "个股1日收益减所属行业收益",
    "industry_excess_5d": "个股5日收益减所属行业5日收益",
    "industry_excess_20d": "个股20日收益减所属行业20日收益",
    "industry_excess_60d": "个股60日收益减所属行业60日收益",
    "industry_beta_60d": "个股对所属行业的60日滚动Beta",
    "industry_corr_60d": "个股与所属行业的60日收益相关性",
    "stock_market_volume_sync_20d": "个股与市场成交量变化20日相关性",
    "stock_industry_volume_sync_20d": "个股与行业成交量变化20日相关性",
    "stock_industry_attention_gap_20d": "个股与行业成交额异常关注度之差",
}

FACTOR_INFO = {
    "价格类": PRICE_INFO,
    "成交量与流动性类": VOLUME_INFO,
    "波动率类": VOLATILITY_INFO,
    "动量类": MOMENTUM_INFO,
    "反转类": REVERSAL_INFO,
    "Alpha组合类": ALPHA_INFO,
    "市场与行业类": CONTEXT_INFO,
}
FACTOR_GROUPS = {group: list(items) for group, items in FACTOR_INFO.items()}
FACTOR_DEFINITIONS = {
    factor: definition
    for items in FACTOR_INFO.values()
    for factor, definition in items.items()
}
FEATURES = [factor for factors in FACTOR_GROUPS.values() for factor in factors]

# 以上137个原有因子的名称、公式、顺序和分类全部保留。新增因子只追加。
LEGACY_FEATURES = FEATURES.copy()
for _name, _spec in CATALOG.items():
    FACTOR_INFO.setdefault(_spec["factor_type"], {})[_name] = _spec["definition"]
FACTOR_GROUPS = {group: list(items) for group, items in FACTOR_INFO.items()}
FACTOR_DEFINITIONS.update({name: spec["definition"] for name, spec in CATALOG.items()})
FEATURES = LEGACY_FEATURES + EXTRA_FEATURES
CROSS_SECTION_FEATURES = [name for name in FEATURES if name not in set(DAILY_FEATURES)]


def factor_catalog() -> pd.DataFrame:
    """返回供结果文件和网页使用的因子分类目录。"""
    return pd.DataFrame(
        [
            {"factor": factor, "factor_type": factor_type, "definition": definition,
             "is_legacy": factor in LEGACY_FEATURES,
             "ic_scope": CATALOG.get(factor, {}).get("ic_scope", "cross_section"),
             "source_fields": CATALOG.get(factor, {}).get("source_fields", "原有量价/市场行业字段")}
            for factor_type, items in FACTOR_INFO.items()
            for factor, definition in items.items()
        ]
    ).set_index("factor").loc[FEATURES].reset_index()


def add_price_factors(stock: pd.DataFrame, close: pd.Series, returns: pd.Series) -> None:
    """价格类：收益、K线形态、均线偏离和长期价格位置。"""
    open_price = stock["open"].replace(0, np.nan)
    price_range = stock["high"] - stock["low"]

    stock["return_1d"] = returns
    stock["open_to_close"] = open_price / close - 1
    stock["high_to_close"] = stock["high"] / close - 1
    stock["low_to_close"] = stock["low"] / close - 1
    stock["amplitude"] = price_range / open_price
    stock["upper_shadow"] = (stock["high"] - np.maximum(close, open_price)) / open_price
    stock["lower_shadow"] = (np.minimum(close, open_price) - stock["low"]) / open_price
    stock["price_position"] = (close - stock["low"]) / price_range.replace(0, np.nan)
    stock["log_price"] = np.log(close)
    stock["price_change"] = close.diff()
    stock["ma_gap_5d"] = close / close.rolling(5, min_periods=5).mean() - 1
    stock["ma10_ratio"] = close / close.rolling(10, min_periods=10).mean() - 1
    stock["ma_gap_20d"] = close / close.rolling(20, min_periods=15).mean() - 1
    stock["ma60_ratio"] = close / close.rolling(60, min_periods=40).mean() - 1
    stock["distance_52w_high"] = close / close.rolling(252, min_periods=120).max() - 1


def add_volume_factors(stock: pd.DataFrame, close: pd.Series, returns: pd.Series) -> None:
    """成交量与流动性类：量能变化、量价关系和交易活跃度。"""
    volume = stock["volume"].replace(0, np.nan)
    amount = stock["amount"].replace(0, np.nan)
    volume_ma5 = volume.rolling(5, min_periods=5).mean()
    volume_ma20 = volume.rolling(20, min_periods=15).mean()
    amount_ma5 = amount.rolling(5, min_periods=5).mean()
    amount_ma20 = amount.rolling(20, min_periods=15).mean()

    stock["volume_ratio_20d"] = volume / volume_ma20
    stock["turnover_rate"] = stock["turnover"]
    stock["volume_change"] = volume.pct_change(fill_method=None)
    stock["amount_change"] = amount.pct_change(fill_method=None)
    stock["volume_price_corr"] = volume.rolling(20, min_periods=5).corr(close)
    stock["obv"] = (np.sign(close.diff().fillna(0)) * volume).cumsum()
    stock["obv_change"] = stock["obv"].pct_change(fill_method=None)
    stock["volume_std"] = volume.rolling(20, min_periods=5).std()
    stock["large_order_ratio"] = amount / (volume * close).replace(0, np.nan)
    stock["volume_price_divergence"] = (
        volume.pct_change(5, fill_method=None) - close.pct_change(5, fill_method=None)
    )
    stock["volume_trend"] = volume_ma5 / volume_ma20
    stock["amount_trend"] = amount_ma5 / amount_ma20
    stock["volume_breakout"] = volume / volume.rolling(60, min_periods=20).max()
    stock["volume_ratio_5d"] = volume / volume_ma5 - 1
    stock["turnover_5d"] = stock["turnover"].rolling(5, min_periods=5).mean()
    amount_100m = amount / 1e8
    stock["illiquidity_20d"] = (
        returns.abs() / amount_100m
    ).rolling(20, min_periods=15).mean()


def add_volatility_factors(stock: pd.DataFrame, close: pd.Series, returns: pd.Series) -> None:
    """波动率类：多周期波动、ATR、分布形态和下行风险。"""
    vol5 = returns.rolling(5, min_periods=3).std()
    vol10 = returns.rolling(10, min_periods=5).std()
    vol20 = returns.rolling(20, min_periods=15).std()
    vol60 = returns.rolling(60, min_periods=40).std()

    stock["volatility_5d"] = vol5
    stock["volatility_10d"] = vol10
    stock["volatility_20d"] = vol20 * np.sqrt(20)
    stock["volatility_60d"] = vol60
    stock["vol_ratio_5_20"] = vol5 / vol20.replace(0, np.nan)
    stock["vol_ratio_20_60"] = vol20 / vol60.replace(0, np.nan)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            stock["high"] - stock["low"],
            (stock["high"] - previous_close).abs(),
            (stock["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    stock["atr"] = true_range.rolling(14, min_periods=10).mean()
    stock["atr_ratio"] = stock["atr"] / close
    stock["downside_volatility"] = returns.where(returns < 0).rolling(20, min_periods=5).std()
    stock["upside_volatility"] = returns.where(returns > 0).rolling(20, min_periods=5).std()
    stock["volatility_skew"] = returns.rolling(20, min_periods=10).skew()
    stock["volatility_kurtosis"] = returns.rolling(20, min_periods=10).kurt()
    stock["return_variance"] = returns.rolling(20, min_periods=15).var()
    stock["max_drawdown"] = close / close.rolling(20, min_periods=10).max() - 1
    stock["volatility_change"] = vol20.pct_change(fill_method=None)
    stock["intraday_range"] = (stock["high"] - stock["low"]) / previous_close


def add_momentum_factors(stock: pd.DataFrame, close: pd.Series) -> None:
    """动量类：多周期趋势、RSI和MACD；不包含需行业数据的5个因子。"""
    for period in (5, 10, 20, 60, 120):
        stock[f"momentum_{period}d"] = close / close.shift(period) - 1

    stock["weighted_momentum"] = (
        0.4 * stock["momentum_5d"]
        + 0.3 * stock["momentum_10d"]
        + 0.2 * stock["momentum_20d"]
        + 0.1 * stock["momentum_60d"]
    )
    stock["momentum_acceleration"] = stock["momentum_5d"] - stock["momentum_20d"]

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = -delta.clip(upper=0).rolling(14, min_periods=14).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    stock["rsi_14d"] = 100 - 100 / (1 + relative_strength)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    stock["macd"] = macd_line - macd_line.ewm(span=9, adjust=False).mean()
    stock["momentum_ratio"] = stock["momentum_20d"] / stock["momentum_60d"].replace(0, np.nan)


def add_reversal_factors(stock: pd.DataFrame, close: pd.Series, returns: pd.Series) -> None:
    """反转类：收益反向、隔夜反向、量能加权反转和极端反转。"""
    for period in (1, 3, 5, 10, 20):
        stock[f"reversal_{period}d"] = -close.pct_change(period, fill_method=None)

    stock["intraday_reversal"] = -(close / stock["open"].replace(0, np.nan) - 1)
    stock["overnight_reversal"] = -(stock["open"] / close.shift(1) - 1)
    stock["high_low_reversal"] = -(stock["high"] / stock["low"].replace(0, np.nan) - 1)
    volume_ma20 = stock["volume"].rolling(20, min_periods=15).mean()
    stock["volume_weighted_reversal"] = (
        stock["reversal_1d"] * stock["volume"] / volume_ma20.replace(0, np.nan)
    )
    stock["extreme_reversal"] = -returns.rolling(20, min_periods=5).apply(
        lambda values: values[np.nanargmax(np.abs(values))], raw=True
    )
    stock["reversal_strength"] = stock["reversal_1d"] * stock["reversal_5d"]


def add_alpha_factors(stock: pd.DataFrame, close: pd.Series, returns: pd.Series) -> None:
    """Alpha组合类：趋势、突破、效率以及动量与波动率的简单组合。"""
    mean_return_5d = returns.rolling(5, min_periods=3).mean()
    std_return_5d = returns.rolling(5, min_periods=3).std()
    stock["alpha_trend_strength"] = mean_return_5d / std_return_5d.replace(0, np.nan)
    stock["alpha_breakout"] = close / close.rolling(20, min_periods=10).max() - 1

    volume_ma20 = stock["volume"].rolling(20, min_periods=15).mean()
    price_ma20 = close.rolling(20, min_periods=15).mean()
    stock["alpha_volume_breakout"] = (
        stock["volume"] / volume_ma20.replace(0, np.nan)
    ) * (close / price_ma20 - 1)

    net_change = (close - close.shift(20)).abs()
    path_length = close.diff().abs().rolling(20, min_periods=10).sum()
    stock["alpha_price_efficiency"] = net_change / path_length.replace(0, np.nan)
    stock["alpha_vol_adj_momentum"] = (
        stock["momentum_20d"] / stock["volatility_20d"].replace(0, np.nan)
    )

    alpha_index = 6
    for momentum in ("momentum_5d", "momentum_10d", "momentum_20d", "momentum_60d", "momentum_120d"):
        for volatility in ("volatility_5d", "volatility_10d", "volatility_20d", "volatility_60d"):
            stock[f"alpha_{alpha_index}"] = stock[momentum] / (stock[volatility] + 0.001)
            alpha_index += 1

    stock["alpha_26"] = stock["momentum_10d"] * stock["volume_ratio_20d"]
    stock["alpha_27"] = stock["reversal_5d"] * stock["volatility_10d"]
    stock["alpha_28"] = stock["rsi_14d"] - 50
    stock["alpha_29"] = stock["macd"] / (stock["atr"] + 0.001)
    stock["alpha_30"] = stock["turnover_rate"] * stock["momentum_5d"]

def _rolling_return(series: pd.Series, window: int) -> pd.Series:
    return series / series.shift(window) - 1


def _log_change(series: pd.Series, window: int = 1) -> pd.Series:
    return np.log(series.replace(0, np.nan)).diff(window)


def _rolling_log_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    values = np.log(series.replace(0, np.nan))
    mean = values.rolling(window, min_periods=15).mean()
    std = values.rolling(window, min_periods=15).std()
    return (values - mean) / std.replace(0, np.nan)


def add_market_industry_factors(
    stock: pd.DataFrame,
    close: pd.Series,
    returns: pd.Series,
) -> None:
    """市场状态、行业状态及个股相对表现因子。"""
    required = {
        "market_close",
        "market_volume",
        "industry_close",
        "industry_volume",
        "industry_amount",
    }
    missing = required - set(stock.columns)
    if missing:
        raise ValueError(f"缺少市场/行业字段：{sorted(missing)}")

    market_close = stock["market_close"].replace(0, np.nan)
    market_volume = stock["market_volume"].replace(0, np.nan)
    industry_close = stock["industry_close"].replace(0, np.nan)
    industry_volume = stock["industry_volume"].replace(0, np.nan)
    industry_amount = stock["industry_amount"].replace(0, np.nan)

    market_returns = market_close.pct_change(fill_method=None)
    industry_returns = industry_close.pct_change(fill_method=None)

    market_volume_change = _log_change(market_volume)
    industry_volume_change = _log_change(industry_volume)
    stock_volume_change = _log_change(stock["volume"])

    # 市场状态
    for period in (5, 20, 60):
        stock[f"market_momentum_{period}d"] = _rolling_return(
            market_close, period
        )

    stock["market_ma_gap_20d"] = (
        market_close / market_close.rolling(20, min_periods=15).mean() - 1
    )
    stock["market_volatility_20d"] = (
        market_returns.rolling(20, min_periods=15).std() * np.sqrt(20)
    )
    stock["market_drawdown_60d"] = (
        market_close / market_close.rolling(60, min_periods=40).max() - 1
    )
    stock["market_volume_ratio_20d"] = (
        market_volume / market_volume.rolling(20, min_periods=15).mean()
    )
    stock["market_volume_change_5d"] = _log_change(market_volume, 5)
    stock["market_volume_zscore_20d"] = _rolling_log_zscore(market_volume)
    stock["market_price_volume_corr_20d"] = (
        market_returns.rolling(20, min_periods=15).corr(market_volume_change)
    )

    # 行业状态
    for period in (5, 20, 60):
        stock[f"industry_momentum_{period}d"] = _rolling_return(
            industry_close, period
        )

    stock["industry_ma_gap_20d"] = (
        industry_close / industry_close.rolling(20, min_periods=15).mean() - 1
    )
    stock["industry_volatility_20d"] = (
        industry_returns.rolling(20, min_periods=15).std() * np.sqrt(20)
    )
    stock["industry_drawdown_60d"] = (
        industry_close / industry_close.rolling(60, min_periods=40).max() - 1
    )
    stock["industry_volume_ratio_20d"] = (
        industry_volume / industry_volume.rolling(20, min_periods=15).mean()
    )
    stock["industry_amount_ratio_20d"] = (
        industry_amount / industry_amount.rolling(20, min_periods=15).mean()
    )
    stock["industry_volume_change_5d"] = _log_change(industry_volume, 5)
    stock["industry_amount_change_5d"] = _log_change(industry_amount, 5)
    stock["industry_price_volume_corr_20d"] = (
        industry_returns.rolling(20, min_periods=15).corr(industry_volume_change)
    )

    # 个股相对市场、相对行业的收益
    for period in (1, 5, 20, 60):
        stock_return = _rolling_return(close, period)
        market_return = _rolling_return(market_close, period)
        industry_return = _rolling_return(industry_close, period)

        stock[f"market_excess_{period}d"] = stock_return - market_return
        stock[f"industry_excess_{period}d"] = stock_return - industry_return

    # 个股的市场/行业Beta、相关性与特质波动
    market_beta = (
        returns.rolling(60, min_periods=40).cov(market_returns)
        / market_returns.rolling(60, min_periods=40).var().replace(0, np.nan)
    )
    industry_beta = (
        returns.rolling(60, min_periods=40).cov(industry_returns)
        / industry_returns.rolling(60, min_periods=40).var().replace(0, np.nan)
    )

    stock["market_beta_60d"] = market_beta
    stock["market_corr_60d"] = (
        returns.rolling(60, min_periods=40).corr(market_returns)
    )
    stock["residual_volatility_60d"] = (
        (returns - market_beta * market_returns)
        .rolling(60, min_periods=20)
        .std()
    )

    stock["industry_beta_60d"] = industry_beta
    stock["industry_corr_60d"] = (
        returns.rolling(60, min_periods=40).corr(industry_returns)
    )

    # 量能与资金关注度的共振/偏离
    stock["stock_market_volume_sync_20d"] = (
        stock_volume_change.rolling(20, min_periods=15).corr(market_volume_change)
    )
    stock["stock_industry_volume_sync_20d"] = (
        stock_volume_change.rolling(20, min_periods=15).corr(industry_volume_change)
    )
    stock["stock_industry_attention_gap_20d"] = (
        _rolling_log_zscore(stock["amount"])
        - _rolling_log_zscore(industry_amount)
    )

def make_features(prices: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """原137个因子原样计算，追加新因子；标签仍为未来1/5/30/60日收益。"""
    prepare_folders()
    daily = build_daily_factors(prices).astype("float32")
    pieces = []
    required = set().union(*(set(spec["source_fields"].split(",")) for spec in CATALOG.values()))
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Parquet缺少新增因子所需字段：{sorted(missing)}")

    for _, stock in prices.groupby("code", sort=False):
        stock = stock.sort_values("date").copy()
        close = stock["close"].replace(0, np.nan)
        returns = close.pct_change(fill_method=None)

        add_price_factors(stock, close, returns)
        add_volume_factors(stock, close, returns)
        add_volatility_factors(stock, close, returns)
        add_momentum_factors(stock, close)
        add_reversal_factors(stock, close, returns)
        stock = stock.copy()
        add_alpha_factors(stock, close, returns)
        add_market_industry_factors(stock, close, returns)

        # 原函数结束后统一拼接，减少逐列添加导致的DataFrame碎片。
        stock = pd.concat([stock, build_stock_factors(stock).astype("float32")], axis=1)
        stock = stock.join(daily, on="date", validate="many_to_one")

        for horizon in HORIZONS:
            stock[f"future_return_{horizon}d"] = close.shift(-horizon) / close - 1

        stock["_raw_market_return"] = stock["market_return_1d"]
        stock["_raw_industry_return"] = stock["industry_return_1d"]
        # 不把215列原始宽表重复保存进因子文件；原始数据始终在prices.parquet。
        keep = list(dict.fromkeys([
            "date", "code", "name", "close", "industry", "_raw_market_return", "_raw_industry_return",
            *LEGACY_FEATURES, *STOCK_FEATURES, *DAILY_FEATURES,
            *[f"future_return_{h}d" for h in HORIZONS],
        ]))
        pieces.append(stock[keep].copy())
        if len(pieces) % 50 == 0:
            print(f"因子构建：{len(pieces)}/{prices['code'].nunique()}只股票", flush=True)

    features = pd.concat(pieces, ignore_index=True)
    pieces.clear()
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.sort_values(["date", "code"]).reset_index(drop=True)
    features = add_cross_factors(features, features["industry"])
    for name in EXTRA_FEATURES:
        features[name] = features[name].astype("float32")
    keep = list(dict.fromkeys(["date", "code", "name", "close", *FEATURES,
                              *[f"future_return_{h}d" for h in HORIZONS]]))
    features = features[keep]
    if save:
        features.to_parquet(FEATURE_FILE, index=False, engine="pyarrow")
        write_feature_reports(features, prices)
    return features


def write_feature_reports(features: pd.DataFrame, prices: pd.DataFrame) -> None:
    """独立输出目录、原始字段用途和质量表；保留只读的历史因子目录。"""
    prepare_folders()
    factor_catalog().to_csv(FACTOR_CATALOG_FILE, index=False, encoding="utf-8-sig")
    field_usage(prices).to_csv(DATA_DIR / "field_usage.csv", index=False, encoding="utf-8-sig")
    audit = pd.DataFrame([{"factor": name, "missing_pct": features[name].isna().mean() * 100,
                           "n_unique": features[name].nunique()} for name in FEATURES])
    audit = audit.merge(factor_catalog(), on="factor", validate="one_to_one")
    audit.to_csv(DATA_DIR / "factor_quality.csv", index=False, encoding="utf-8-sig")


def load_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURE_FILE, engine="pyarrow")
