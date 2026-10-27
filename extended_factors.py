"""仅使用 prices.parquet 已有字段的增量因子；不修改原137个因子。

日频公共变量先按A股日期计算一次，个股变量按股票计算，最后做当日行业内排名。
所有新因子以 ext_ 开头。CATALOG 同时记录公式、原始依赖与IC评估口径。
"""

import numpy as np
import pandas as pd


GLOBAL_NAMES = (
    "sp500", "nasdaq100", "russell2000", "semiconductor", "vix", "hangseng",
    "nikkei225", "kospi", "taiwan", "eurostoxx50", "dax", "ftse100",
)
CALENDAR_FIELDS = [
    "weekday", "month", "quarter", "day_of_month", "days_since_prev_trade",
    "days_to_next_trade", "is_month_first_trade", "is_month_last_trade",
    "is_after_long_break", "is_before_long_break", "closed_weekdays_before",
    "closed_weekdays_after", "is_after_holiday", "is_before_holiday",
]
RATE_FIELDS = [f"shibor_{tenor}_pct" for tenor in ("overnight", "1w", "1m", "3m", "1y")]
RATE_FIELDS += [f"{country}_yield_{year}y_pct" for country in ("cn", "us") for year in (2, 5, 10, 30)]
SPREAD_FIELDS = ["cn_term_spread_10y_2y_pp", "us_term_spread_10y_2y_pp", "cn_us_spread_10y_pp"]
CATALOG: dict[str, dict] = {}


def _register(name, group, definition, sources, scope="stock"):
    if name in CATALOG:
        raise ValueError(f"重复的新因子：{name}")
    CATALOG[name] = {
        "factor": name, "factor_type": group, "definition": definition,
        "source_fields": ",".join(sources),
        "ic_scope": "time_series" if scope == "daily" else "cross_section",
        "scope": scope,
    }


# 公式中的 pct/pp 为原始百分数/百分点，先除100转换为小数。
FUNDAMENTAL_LEVELS = {
    "ext_roe": ("roe_ytd_pct", "盈利质量类", "roe_ytd_pct/100"),
    "ext_roa": ("roa_ytd_pct", "盈利质量类", "roa_ytd_pct/100"),
    "ext_gross_margin": ("gross_margin_pct", "盈利质量类", "gross_margin_pct/100"),
    "ext_net_margin": ("net_margin_pct", "盈利质量类", "net_margin_pct/100"),
    "ext_leverage": ("debt_to_assets_pct", "盈利质量类", "debt_to_assets_pct/100"),
    "ext_revenue_growth": ("revenue_yoy_pct", "成长类", "revenue_yoy_pct/100"),
    "ext_profit_growth": ("net_profit_yoy_pct", "成长类", "net_profit_yoy_pct/100"),
    "ext_dividend_yield": ("dividend_yield_365d_pct", "估值与规模类", "dividend_yield_365d_pct/100"),
}
FIN_GUARDS = ["financial_available_at", "signal_time", "financial_currency"]
for name, (source, group, definition) in FUNDAMENTAL_LEVELS.items():
    _register(name, group, definition, [source, *FIN_GUARDS])

FUNDAMENTAL_SPECS = {
    "ext_earnings_yield": ("估值与规模类", "1/pe_static；仅正PE", ["pe_static", "annual_available_at"]),
    "ext_log_pb": ("估值与规模类", "log(pb)；仅正PB", ["pb"]),
    "ext_value_quality": ("估值与规模类", "book_to_market×roe_ytd_pct/100", ["book_to_market", "roe_ytd_pct"]),
    "ext_log_float_cap": ("估值与规模类", "log(unadjusted_close×float_shares_estimate)；估计流通市值，非总市值", ["unadjusted_close", "float_shares_estimate"]),
    "ext_log_float_shares": ("估值与规模类", "log(float_shares_estimate)", ["float_shares_estimate"]),
    "ext_eps_ytd_log": ("盈利质量类", "sign(eps_ytd)×log(1+abs(eps_ytd))；保留YTD口径", ["eps_ytd"]),
    "ext_annual_roe_proxy": ("盈利质量类", "eps_annual/book_value_per_share；静态年报盈利/当前已知每股净资产", ["eps_annual", "book_value_per_share", "annual_available_at"]),
    "ext_revenue_log": ("盈利质量类", "sign(revenue_ytd)×log(1+abs(revenue_ytd))", ["revenue_ytd"]),
    "ext_profit_log": ("盈利质量类", "sign(net_profit_ytd)×log(1+abs(net_profit_ytd))", ["net_profit_ytd"]),
    "ext_cashflow_yield_ytd": ("盈利质量类", "operating_cashflow_per_share_ytd/unadjusted_close；非年化", ["operating_cashflow_per_share_ytd", "unadjusted_close"]),
    "ext_cash_profit_ratio_ytd": ("盈利质量类", "operating_cashflow_per_share_ytd/eps_ytd；仅正EPS", ["operating_cashflow_per_share_ytd", "eps_ytd"]),
    "ext_accrual_proxy_ytd": ("盈利质量类", "(eps_ytd-operating_cashflow_per_share_ytd)/unadjusted_close", ["eps_ytd", "operating_cashflow_per_share_ytd", "unadjusted_close"]),
    "ext_margin_gap": ("盈利质量类", "(gross_margin_pct-net_margin_pct)/100", ["gross_margin_pct", "net_margin_pct"]),
    "ext_delevered_roe": ("盈利质量类", "roe_ytd_pct/100×(1-debt_to_assets_pct/100)", ["roe_ytd_pct", "debt_to_assets_pct"]),
    "ext_growth_gap": ("成长类", "(net_profit_yoy_pct-revenue_yoy_pct)/100", ["net_profit_yoy_pct", "revenue_yoy_pct"]),
    "ext_profit_growth_change_20d": ("成长类", "已知净利润同比增长率的20个个股交易观察值变化/100", ["net_profit_yoy_pct"]),
    "ext_revenue_growth_change_20d": ("成长类", "已知营收同比增长率的20个个股交易观察值变化/100", ["revenue_yoy_pct"]),
    "ext_dividend_coverage": ("估值与规模类", "cash_dividend_365d/eps_annual；仅正年报EPS，非支付率的严格会计口径", ["cash_dividend_365d", "eps_annual", "annual_available_at"]),
    "ext_report_age": ("信息时效与质量类", "financial_age_days；无负年龄", ["financial_age_days", "financial_report_date"]),
    "ext_report_known_days": ("信息时效与质量类", "(signal_time-financial_available_at)的天数", ["financial_available_at"]),
    "ext_notice_age": ("信息时效与质量类", "(signal_time-financial_notice_date)的天数", ["financial_notice_date"]),
    "ext_update_age": ("信息时效与质量类", "(signal_time-financial_update_date)的天数", ["financial_update_date"]),
    "ext_disclosure_delay": ("信息时效与质量类", "financial_notice_date-financial_report_date的天数；仅已知日期", ["financial_notice_date", "financial_report_date"]),
    "ext_annual_report_age": ("信息时效与质量类", "(signal_time-annual_report_date)的天数", ["annual_report_date", "annual_available_at"]),
    "ext_annual_known_days": ("信息时效与质量类", "(signal_time-annual_available_at)的天数", ["annual_available_at"]),
    "ext_report_quarter": ("日历效应类", "已知financial_report_date的季度；区分YTD报表季节性", ["financial_report_date"]),
    "ext_report_event": ("信息时效与质量类", "已知financial_available_at相比前一条个股记录发生变化为1；首次观察记0", ["financial_available_at"]),
    "ext_financial_missing": ("信息时效与质量类", "财报不可用、币种非CNY或盈利/营收/ROE任一缺失为1", ["eps_ytd", "revenue_ytd", "roe_ytd_pct"]),
    "ext_valuation_missing": ("信息时效与质量类", "PE或PB缺失/非正为1；负盈利不能伪装成低估值", ["pe_static", "pb"]),
    "ext_price_source_changed": ("信息时效与质量类", "未复权价格来源相比前一条个股记录变化为1；首次观察记0", ["unadjusted_price_source"]),
}
for name, (group, definition, sources) in FUNDAMENTAL_SPECS.items():
    _register(name, group, definition, list(dict.fromkeys([*sources, *FIN_GUARDS])))

for field in RATE_FIELDS:
    base = field.removesuffix("_pct")
    guard = "shibor" if field.startswith("shibor") else field.split("_yield")[0] + "_yield"
    for suffix, formula in (
        ("level", f"{field}/100"),
        ("change_5d_bp", f"({field}-{field}.shift(5))×100，单位基点"),
        ("change_20d_bp", f"({field}-{field}.shift(20))×100，单位基点"),
    ):
        _register(f"ext_{base}_{suffix}", "宏观利率类", formula,
                  [field, f"{guard}_available_at", "signal_time"], "daily")
for field in SPREAD_FIELDS:
    for suffix, formula in (("level", f"{field}/100"), ("change_20d_bp", f"({field}-{field}.shift(20))×100")):
        _register(f"ext_{field.removesuffix('_pp')}_{suffix}", "宏观利率类", formula,
                  [field, "cn_yield_available_at", "us_yield_available_at", "signal_time"], "daily")
CURVES = {
    "ext_shibor_curve_3m_on": ("shibor_3m_pct", "shibor_overnight_pct"),
    "ext_shibor_curve_1y_1m": ("shibor_1y_pct", "shibor_1m_pct"),
    **{f"ext_{country}_curve_{long}_{short}": (f"{country}_yield_{long}y_pct", f"{country}_yield_{short}y_pct")
       for country in ("cn", "us") for long, short in ((30, 10), (5, 2))},
}
for name, (long, short) in CURVES.items():
    _register(name, "宏观利率类", f"({long}-{short})/100", [long, short, "signal_time"], "daily")
for country in ("cn", "us"):
    sources = [f"{country}_yield_{year}y_pct" for year in (2, 5, 10)]
    _register(f"ext_{country}_curve_curvature", "宏观利率类", f"(2×{sources[1]}-{sources[0]}-{sources[2]})/100", sources, "daily")
for prefix in ("shibor", "cn_yield", "us_yield", "gdp"):
    date_field = "gdp_release_date" if prefix == "gdp" else f"{prefix}_date"
    _register(f"ext_{prefix}_source_age", "信息时效与质量类", f"signal_time-{date_field}的天数",
              [date_field, f"{prefix}_available_at", "signal_time"], "daily")
    _register(f"ext_{prefix}_known_age", "信息时效与质量类", f"signal_time-{prefix}_available_at的天数",
              [f"{prefix}_available_at", "signal_time"], "daily")
for name, formula, sources in (
    ("ext_gdp_growth", "cn_gdp_yoy_pct/100", ["cn_gdp_yoy_pct"]),
    ("ext_gdp_change_20d", "(cn_gdp_yoy_pct-cn_gdp_yoy_pct.shift(20))/100；非季度环比", ["cn_gdp_yoy_pct"]),
    ("ext_gdp_age_log", "log(1+gdp_age_days)", ["gdp_age_days"]),
    ("ext_gdp_release_event", "gdp_available_at相比前一个A股交易日变化为1；首次观察记0", ["gdp_available_at"]),
):
    _register(name, "宏观增长类", formula, [*sources, "gdp_available_at", "signal_time"], "daily")

for field, period in (("weekday", 5), ("month", 12), ("quarter", 4), ("day_of_month", 31)):
    for trig in ("sin", "cos"):
        _register(f"ext_{field}_{trig}", "日历效应类", f"{trig}(2π×({field}-1)/{period})", [field], "daily")
for field in CALENDAR_FIELDS[6:8] + CALENDAR_FIELDS[8:10] + CALENDAR_FIELDS[12:14]:
    _register(f"ext_{field}", "日历效应类", f"直接使用已保存的{field}标志", [field], "daily")
for field in ("days_since_prev_trade", "days_to_next_trade", "closed_weekdays_before", "closed_weekdays_after"):
    _register(f"ext_{field}_log", "日历效应类", f"log(1+{field})；不根据未来收益推断日历", [field], "daily")
for name, formula, sources in (
    ("ext_monday", "weekday==1", ["weekday"]), ("ext_friday", "weekday==5", ["weekday"]),
    ("ext_quarter_start", "month%3==1 且 is_month_first_trade==1", ["month", "is_month_first_trade"]),
    ("ext_quarter_end", "month%3==0 且 is_month_last_trade==1", ["month", "is_month_last_trade"]),
):
    _register(name, "日历效应类", formula, sources, "daily")

GLOBAL_FORMULAS = {
    "known_return": "本次已知close/前一个A股交易日已知close-1；非一律一日海外收益",
    "momentum_5d": "已知close/5个A股交易日前已知close-1",
    "momentum_20d": "已知close/20个A股交易日前已知close-1",
    "volatility_20d": "已知收益20日标准差×sqrt(20)，至少15个有效观察",
    "intraday": "已知close/open-1",
    "amplitude": "(已知high-low)/open",
    "open_gap": "已知open/前一个A股交易日已知close-1；无新海外交易日时缺失",
    "volume_ratio_20d": "已知volume/20日均量，至少15个正成交量；全零成交量保持缺失",
    "new_sessions": "原表new_sessions：本次A股日期新增海外交易日数量",
    "missing_flag": "原表missing_flag或收盘缺失或时间校验不通过为1",
    "staleness_days": "原表staleness_hours/24；校验available_at后使用",
}
for index in GLOBAL_NAMES:
    prefix = f"global_{index}_"
    guards = [prefix + "source_date", prefix + "available_at", "signal_time"]
    for suffix, formula in GLOBAL_FORMULAS.items():
        fields = {
            "known_return": ["close"], "momentum_5d": ["close"], "momentum_20d": ["close"],
            "volatility_20d": ["close"], "intraday": ["close", "open"],
            "amplitude": ["high", "low", "open"], "open_gap": ["open", "close", "new_sessions"],
            "volume_ratio_20d": ["volume"], "new_sessions": ["new_sessions"],
            "missing_flag": ["missing_flag", "close"], "staleness_days": ["staleness_hours"],
        }[suffix]
        _register(f"ext_global_{index}_{suffix}", "全球市场类", f"{index}：{formula}",
                  [*[prefix + f for f in fields], *guards, prefix + "missing_flag"], "daily")

for prefix in ("market", "industry"):
    for suffix, formula, fields in (
        ("intraday", "close/open-1", ["close", "open"]),
        ("amplitude", "(high-low)/open", ["high", "low", "open"]),
        ("open_gap", "open/前一条个股观察的指数close-1", ["open", "close"]),
    ):
        _register(f"ext_{prefix}_{suffix}", "市场形态扩展类", f"{prefix}：{formula}", [f"{prefix}_{f}" for f in fields])

RANK_FACTORS = ["ext_earnings_yield", "ext_log_float_cap", "ext_roe", "ext_gross_margin",
                "ext_profit_growth", "ext_dividend_yield", "ext_cashflow_yield_ytd", "momentum_20d"]
for factor in RANK_FACTORS:
    raw = CATALOG[factor]["source_fields"].split(",") if factor in CATALOG else ["close"]
    _register(f"ext_industry_rank_{factor.removeprefix('ext_')}", "行业相对基本面类",
              f"当日同行业内{factor}的百分位排名-0.5；至少3个有效样本", ["date", "industry", *raw], "cross")
INTERACTIONS = {
    "ext_rate_beta": ("market_beta_60d×ext_cn_yield_10y_change_20d_bp/100", ["close", "market_close", "cn_yield_10y_pct"]),
    "ext_growth_beta": ("market_beta_60d×ext_gdp_growth", ["close", "market_close", "cn_gdp_yoy_pct"]),
    "ext_sp500_beta": ("market_beta_60d×ext_global_sp500_known_return", ["close", "market_close", "global_sp500_close"]),
    "ext_vix_beta": ("market_beta_60d×ext_global_vix_known_return", ["close", "market_close", "global_vix_close"]),
    "ext_monthend_momentum": ("ext_is_month_last_trade×momentum_5d", ["is_month_last_trade", "close"]),
    "ext_holiday_reversal": ("ext_is_after_holiday×reversal_1d", ["is_after_holiday", "close"]),
    "ext_market_volume_response": ("原表market_return_1d×个股volume_ratio_20d", ["market_return_1d", "volume"]),
    "ext_industry_volume_response": ("原表industry_return_1d×个股volume_ratio_20d", ["industry_return_1d", "volume"]),
    "ext_value_rate_gap": ("ext_earnings_yield-ext_cn_yield_10y_level；静态盈利收益率减国债利率", ["pe_static", "cn_yield_10y_pct"]),
}
for name, (formula, sources) in INTERACTIONS.items():
    _register(name, "跨类别交互类", formula, sources, "cross")

EXTRA_FEATURES = list(CATALOG)
DAILY_FEATURES = [name for name, spec in CATALOG.items() if spec["scope"] == "daily"]
STOCK_FEATURES = [name for name, spec in CATALOG.items() if spec["scope"] == "stock"]


def _num(frame, column):
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _date(frame, column):
    return pd.to_datetime(frame[column], errors="coerce")


def _ratio(numerator, denominator, positive=False):
    valid = denominator > 1e-12 if positive else denominator.abs() > 1e-12
    return numerator / denominator.where(valid)


def _log_positive(series):
    return np.log(series.where(series > 0))


def _age(cutoff, timestamp):
    age = (cutoff - timestamp).dt.total_seconds() / 86400
    return age.where(age >= 0)


def _event(timestamp):
    return (timestamp.notna() & timestamp.shift().notna() & timestamp.ne(timestamp.shift())).astype(float)


def _known(frame, prefix):
    cutoff = _date(frame, "signal_time")
    available = _date(frame, f"{prefix}_available_at")
    return available.notna() & available.le(cutoff)


def _frame(values, index):
    return pd.DataFrame(values, index=index).replace([np.inf, -np.inf], np.nan)


def build_daily_factors(prices):
    """每个A股日期只计算一次；公共字段同日不一致时拒绝任意取一只股票。"""
    fields = set(CALENDAR_FIELDS + RATE_FIELDS + SPREAD_FIELDS + ["signal_time", "cn_gdp_yoy_pct", "gdp_age_days"])
    for prefix in ("shibor", "cn_yield", "us_yield", "gdp"):
        fields.update([f"{prefix}_available_at", "gdp_release_date" if prefix == "gdp" else f"{prefix}_date"])
    for name in GLOBAL_NAMES:
        fields.update(f"global_{name}_{suffix}" for suffix in (
            "source_date", "available_at", "open", "high", "low", "close", "volume",
            "new_sessions", "missing_flag", "staleness_hours"))
    missing = fields - set(prices.columns)
    if missing:
        raise ValueError(f"Parquet缺少新增日频因子字段：{sorted(missing)}")
    inconsistent = prices.groupby("date", sort=False)[sorted(fields)].nunique(dropna=False).gt(1).any()
    if inconsistent.any():
        raise ValueError(f"同一日期的公共字段不一致：{list(inconsistent[inconsistent].index)}")
    day = prices.drop_duplicates("date").set_index("date")[sorted(fields)].sort_index()
    cutoff = _date(day, "signal_time")
    result = {}
    rates = {}
    for field in RATE_FIELDS:
        prefix = "shibor" if field.startswith("shibor") else field.split("_yield")[0] + "_yield"
        series = _num(day, field).where(_known(day, prefix))
        rates[field] = series
        base = "ext_" + field.removesuffix("_pct")
        result[base + "_level"] = series / 100
        result[base + "_change_5d_bp"] = series.diff(5) * 100
        result[base + "_change_20d_bp"] = series.diff(20) * 100
    for field in SPREAD_FIELDS:
        # 中美利差同时要求两国数据已可用；本国期限利差只要求本国数据。
        known = _known(day, "cn_yield") if field.startswith("cn_") else _known(day, "us_yield")
        if field == "cn_us_spread_10y_pp":
            known &= _known(day, "us_yield")
        series = _num(day, field).where(known)
        base = "ext_" + field.removesuffix("_pp")
        result[base + "_level"] = series / 100
        result[base + "_change_20d_bp"] = series.diff(20) * 100
    for name, (long, short) in CURVES.items():
        result[name] = (rates[long] - rates[short]) / 100
    for country in ("cn", "us"):
        result[f"ext_{country}_curve_curvature"] = (
            2 * rates[f"{country}_yield_5y_pct"] - rates[f"{country}_yield_2y_pct"] - rates[f"{country}_yield_10y_pct"]
        ) / 100
    for prefix in ("shibor", "cn_yield", "us_yield", "gdp"):
        field = "gdp_release_date" if prefix == "gdp" else f"{prefix}_date"
        result[f"ext_{prefix}_source_age"] = _age(cutoff, _date(day, field)).where(_known(day, prefix))
        result[f"ext_{prefix}_known_age"] = _age(cutoff, _date(day, prefix + "_available_at"))
    gdp = _num(day, "cn_gdp_yoy_pct").where(_known(day, "gdp"))
    result["ext_gdp_growth"] = gdp / 100
    result["ext_gdp_change_20d"] = gdp.diff(20) / 100
    result["ext_gdp_age_log"] = np.log1p(_num(day, "gdp_age_days").where(_known(day, "gdp")))
    result["ext_gdp_release_event"] = _event(_date(day, "gdp_available_at").where(_known(day, "gdp")))
    for field, period in (("weekday", 5), ("month", 12), ("quarter", 4), ("day_of_month", 31)):
        angle = 2 * np.pi * (_num(day, field) - 1) / period
        result[f"ext_{field}_sin"] = np.sin(angle)
        result[f"ext_{field}_cos"] = np.cos(angle)
    for field in CALENDAR_FIELDS[6:8] + CALENDAR_FIELDS[8:10] + CALENDAR_FIELDS[12:14]:
        result[f"ext_{field}"] = _num(day, field)
    for field in ("days_since_prev_trade", "days_to_next_trade", "closed_weekdays_before", "closed_weekdays_after"):
        series = _num(day, field)
        result[f"ext_{field}_log"] = np.log1p(series.where(series >= 0))
    result["ext_monday"] = (_num(day, "weekday") == 1).astype(float)
    result["ext_friday"] = (_num(day, "weekday") == 5).astype(float)
    result["ext_quarter_start"] = ((_num(day, "month") % 3 == 1) & (_num(day, "is_month_first_trade") == 1)).astype(float)
    result["ext_quarter_end"] = ((_num(day, "month") % 3 == 0) & (_num(day, "is_month_last_trade") == 1)).astype(float)
    for name in GLOBAL_NAMES:
        prefix = f"global_{name}_"
        target = f"ext_global_{name}_"
        known = _known(day, "global_" + name)
        source_age = _age(cutoff, _date(day, prefix + "source_date"))
        known &= source_age.notna()
        raw_missing = day[prefix + "missing_flag"].fillna(True).astype(bool)
        valid = known & ~raw_missing
        close = _num(day, prefix + "close").where(valid).where(lambda s: s > 0)
        open_price = _num(day, prefix + "open").where(valid)
        high = _num(day, prefix + "high").where(valid)
        low = _num(day, prefix + "low").where(valid)
        volume = _num(day, prefix + "volume").where(valid).where(lambda s: s > 0)
        returns = close.pct_change(fill_method=None)
        result[target + "known_return"] = returns
        for period in (5, 20):
            result[target + f"momentum_{period}d"] = close / close.shift(period) - 1
        result[target + "volatility_20d"] = returns.rolling(20, min_periods=15).std() * np.sqrt(20)
        result[target + "intraday"] = _ratio(close, open_price, True) - 1
        result[target + "amplitude"] = _ratio(high - low, open_price, True)
        sessions = _num(day, prefix + "new_sessions").where(known)
        result[target + "open_gap"] = (_ratio(open_price, close.shift(), True) - 1).where(sessions > 0)
        result[target + "volume_ratio_20d"] = _ratio(volume, volume.rolling(20, min_periods=15).mean(), True)
        result[target + "new_sessions"] = sessions
        result[target + "missing_flag"] = (raw_missing | ~known | close.isna()).astype(float)
        result[target + "staleness_days"] = (_num(day, prefix + "staleness_hours") / 24).where(known)
    output = _frame(result, day.index)
    if set(output) != set(DAILY_FEATURES):
        raise AssertionError("新增日频因子注册表与计算结果不一致")
    return output[DAILY_FEATURES]


def build_stock_factors(stock):
    """只向前看历史观察，不滚动填充报告，不把YTD财报当作TTM。"""
    cutoff = _date(stock, "signal_time")
    known = _known(stock, "financial") & stock["financial_currency"].eq("CNY")
    for field in ("financial_report_date", "financial_notice_date", "financial_update_date"):
        timestamp = _date(stock, field)
        known &= timestamp.isna() | timestamp.le(cutoff)
    annual_known = _known(stock, "annual") & known & _date(stock, "annual_report_date").le(cutoff)
    n = lambda column: _num(stock, column).where(known)
    price = _num(stock, "unadjusted_close").where(lambda s: s > 0)
    eps = n("eps_ytd")
    annual_eps = n("eps_annual").where(annual_known)
    cashflow = n("operating_cashflow_per_share_ytd")
    result = {name: n(source) / 100 for name, (source, _, _) in FUNDAMENTAL_LEVELS.items()}
    result.update({
        "ext_earnings_yield": _ratio(pd.Series(1., index=stock.index), n("pe_static"), True).where(annual_known),
        "ext_log_pb": _log_positive(n("pb")),
        "ext_value_quality": n("book_to_market") * n("roe_ytd_pct") / 100,
        "ext_log_float_cap": _log_positive(price * _num(stock, "float_shares_estimate")),
        "ext_log_float_shares": _log_positive(_num(stock, "float_shares_estimate")),
        "ext_eps_ytd_log": np.sign(eps) * np.log1p(eps.abs()),
        "ext_annual_roe_proxy": _ratio(annual_eps, n("book_value_per_share"), True),
        "ext_revenue_log": np.sign(n("revenue_ytd")) * np.log1p(n("revenue_ytd").abs()),
        "ext_profit_log": np.sign(n("net_profit_ytd")) * np.log1p(n("net_profit_ytd").abs()),
        "ext_cashflow_yield_ytd": _ratio(cashflow, price, True),
        "ext_cash_profit_ratio_ytd": _ratio(cashflow, eps, True),
        "ext_accrual_proxy_ytd": _ratio(eps - cashflow, price, True),
        "ext_margin_gap": (n("gross_margin_pct") - n("net_margin_pct")) / 100,
        "ext_delevered_roe": n("roe_ytd_pct") / 100 * (1 - n("debt_to_assets_pct") / 100),
        "ext_growth_gap": (n("net_profit_yoy_pct") - n("revenue_yoy_pct")) / 100,
        "ext_profit_growth_change_20d": n("net_profit_yoy_pct").diff(20) / 100,
        "ext_revenue_growth_change_20d": n("revenue_yoy_pct").diff(20) / 100,
        "ext_dividend_coverage": _ratio(n("cash_dividend_365d"), annual_eps, True),
        "ext_report_age": n("financial_age_days").where(n("financial_age_days") >= 0),
        "ext_report_known_days": _age(cutoff, _date(stock, "financial_available_at")),
        "ext_notice_age": _age(cutoff, _date(stock, "financial_notice_date")).where(known),
        "ext_update_age": _age(cutoff, _date(stock, "financial_update_date")).where(known),
        "ext_disclosure_delay": _age(_date(stock, "financial_notice_date"), _date(stock, "financial_report_date")).where(known & _date(stock, "financial_notice_date").le(cutoff)),
        "ext_annual_report_age": _age(cutoff, _date(stock, "annual_report_date")).where(annual_known),
        "ext_annual_known_days": _age(cutoff, _date(stock, "annual_available_at")),
        "ext_report_quarter": _date(stock, "financial_report_date").dt.quarter.where(known),
        "ext_report_event": _event(_date(stock, "financial_available_at").where(known)),
        "ext_financial_missing": (~known | stock[["eps_ytd", "revenue_ytd", "roe_ytd_pct"]].isna().any(axis=1)).astype(float),
        "ext_valuation_missing": (n("pe_static").isna() | n("pb").isna() | (n("pe_static") <= 0) | (n("pb") <= 0)).astype(float),
        "ext_price_source_changed": _event(stock["unadjusted_price_source"]),
    })
    for prefix in ("market", "industry"):
        open_price = _num(stock, prefix + "_open")
        close = _num(stock, prefix + "_close")
        result[f"ext_{prefix}_intraday"] = _ratio(close, open_price, True) - 1
        result[f"ext_{prefix}_amplitude"] = _ratio(_num(stock, prefix + "_high") - _num(stock, prefix + "_low"), open_price, True)
        result[f"ext_{prefix}_open_gap"] = _ratio(open_price, close.shift(), True) - 1
    output = _frame(result, stock.index)
    if set(output) != set(STOCK_FEATURES):
        raise AssertionError("新增个股因子注册表与计算结果不一致")
    return output[STOCK_FEATURES]


def add_cross_factors(features, industries):
    """行业内排名只用当日横截面；宏观交互让共同冲击具备个股差异。"""
    result = {}
    keys = [features["date"], industries]
    for factor in RANK_FACTORS:
        grouped = features[factor].groupby(keys, dropna=False)
        count = grouped.transform("count")
        result[f"ext_industry_rank_{factor.removeprefix('ext_')}"] = (
            grouped.rank(pct=True) - 0.5
        ).where((count >= 3) & industries.notna())
    result.update({
        "ext_rate_beta": features["market_beta_60d"] * features["ext_cn_yield_10y_change_20d_bp"] / 100,
        "ext_growth_beta": features["market_beta_60d"] * features["ext_gdp_growth"],
        "ext_sp500_beta": features["market_beta_60d"] * features["ext_global_sp500_known_return"],
        "ext_vix_beta": features["market_beta_60d"] * features["ext_global_vix_known_return"],
        "ext_monthend_momentum": features["ext_is_month_last_trade"] * features["momentum_5d"],
        "ext_holiday_reversal": features["ext_is_after_holiday"] * features["reversal_1d"],
        "ext_market_volume_response": features["_raw_market_return"] * features["volume_ratio_20d"],
        "ext_industry_volume_response": features["_raw_industry_return"] * features["volume_ratio_20d"],
        "ext_value_rate_gap": features["ext_earnings_yield"] - features["ext_cn_yield_10y_level"],
    })
    return pd.concat([features, _frame(result, features.index)], axis=1)


def field_usage(prices):
    """每个原始字段都有用途说明；身份/来源字段不做任意整数编码。"""
    legacy = {"date", "code", "open", "close", "high", "low", "volume", "amount", "turnover",
              "market_close", "market_volume", "industry_close", "industry_volume", "industry_amount"}
    metadata = {"name": "股票名称与结果展示", "code": "股票身份、分组与排序",
                "date": "时间排序、滚动窗口与横截面分组", "industry": "同行业分组与排名",
                "market": "市场标识；常量不作为数值因子", "financial_currency": "仅CNY财报进入新增财务比率",
                "unadjusted_price_source": "来源变更标志与数据追溯"}
    rows = []
    for column in prices:
        names = [name for name, spec in CATALOG.items() if column in spec["source_fields"].split(",")]
        uses = []
        if column in legacy:
            uses.append("原137因子/标签的输入或分组")
        if names:
            uses.append("新增因子输入/可用时间检查")
        if column in metadata:
            uses.append(metadata[column])
        rows.append({"field": column, "usage": "；".join(uses) or "未使用，需要检查",
                     "new_factor_count": len(names), "new_factors": ",".join(names),
                     "missing_pct": round(prices[column].isna().mean() * 100, 4)})
    return pd.DataFrame(rows)
