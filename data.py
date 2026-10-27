"""在线下载并合并行情、财报、宏观、日历和全球指数，生成根目录 prices.parquet。"""

from pathlib import Path
import os
import stat
import tempfile
import time

import akshare as ak
import numpy as np
import pandas as pd
import requests
from price_source import PRICE_FILE, validate_prices, load_prices as read_parquet_prices


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "download_cache"
PRICE_DIR = CACHE_DIR / "prices"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"
START_DATE = "20200101"
MARKET_SYMBOL = "sh000300"
MARKET_NAME = "沪深300"

# 名称：(Yahoo代码，交易所日历)。指数保持原币种，不与ETF或期货混用。
GLOBAL_INDICES = {
    "sp500": ("^GSPC", "XNYS"),
    "nasdaq100": ("^NDX", "XNYS"),
    "russell2000": ("^RUT", "XNYS"),
    "semiconductor": ("^SOX", "XNYS"),
    "vix": ("^VIX", "XNYS"),
    "hangseng": ("^HSI", "XHKG"),
    "nikkei225": ("^N225", "XTKS"),
    "kospi": ("^KS11", "XKRX"),
    "taiwan": ("^TWII", "XTAI"),
    "eurostoxx50": ("^STOXX50E", "XETR"),
    "dax": ("^GDAXI", "XETR"),
    "ftse100": ("^FTSE", "XLON"),
}

FINANCIAL_FIELDS = {
    "REPORT_DATE": "financial_report_date",
    "NOTICE_DATE": "financial_notice_date",
    "UPDATE_DATE": "financial_update_date",
    "EPSJB": "eps_ytd",
    "BPS": "book_value_per_share",
    "ROEJQ": "roe_ytd_pct",
    "ZZCJLL": "roa_ytd_pct",
    "TOTALOPERATEREVE": "revenue_ytd",
    "PARENTNETPROFIT": "net_profit_ytd",
    "TOTALOPERATEREVETZ": "revenue_yoy_pct",
    "PARENTNETPROFITTZ": "net_profit_yoy_pct",
    "MGJYXJJE": "operating_cashflow_per_share_ytd",
    "XSMLL": "gross_margin_pct",
    "XSJLL": "net_margin_pct",
    "ZCFZL": "debt_to_assets_pct",
}


def prepare_folders() -> None:
    """创建项目运行需要的目录。"""
    for folder in (DATA_DIR, CACHE_DIR, PRICE_DIR, MODEL_DIR, RESULT_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def get_end_date(end_date: str | None = None) -> str:
    """收盘前不取当天日线；节假日由接口返回的交易日决定。"""
    now = pd.Timestamp.now(tz="Asia/Shanghai")
    latest = now.normalize()
    if now.hour < 15 or (now.hour == 15 and now.minute < 30):
        latest -= pd.Timedelta(days=1)
    requested = pd.Timestamp(end_date or latest.strftime("%Y%m%d"))
    return min(requested, latest.tz_localize(None)).strftime("%Y%m%d")


def get_stock_pool(n_stocks: int = 300, refresh: bool = False) -> pd.DataFrame:
    """每次获取沪深300最新成分股，默认保留全部300只。"""
    prepare_folders()
    cache = CACHE_DIR / "stock_pool.csv"

    raw = ak.index_stock_cons_csindex(symbol="000300")
    pool = raw.rename(columns={
        "成分券代码": "code",
        "成分券名称": "name",
        "交易所": "exchange",
        "日期": "constituent_date",
    })[["code", "name", "exchange", "constituent_date"]].copy()
    pool["code"] = pool["code"].astype(str).str.zfill(6)
    pool = pool.drop_duplicates("code").sort_values("code").head(n_stocks)
    pool.to_csv(cache, index=False, encoding="utf-8-sig")
    return pool.reset_index(drop=True)


def clean_price_data(raw: pd.DataFrame, code: str, name: str) -> pd.DataFrame:
    """将 AkShare 日线整理为项目统一字段。"""
    columns = ["date", "open", "close", "high", "low", "volume", "amount", "turnover"]
    frame = raw.rename(columns=lambda x: str(x).lower())[columns].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    numeric = ["open", "close", "high", "low", "volume", "amount", "turnover"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame["code"] = str(code).zfill(6)
    frame["name"] = name

    frame = frame.dropna(subset=["date", "open", "close"]).drop_duplicates("date")
    return frame[["date", "code", "name", *numeric]].sort_values("date")


def download_prices(
    pool: pd.DataFrame,
    start_date: str = START_DATE,
    end_date: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """下载前复权日线；过期缓存重下，所有记录限制在指定日期范围。"""
    prepare_folders()
    end_date = get_end_date(end_date)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    frames = []

    for i, row in pool.reset_index(drop=True).iterrows():
        code = str(row["code"]).zfill(6)
        name = str(row["name"])
        cache = PRICE_DIR / f"{code}.csv"
        print(f"[{i + 1:02d}/{len(pool):02d}] {code} {name}", flush=True)

        if cache.exists() and not refresh:
            frame = pd.read_csv(cache, parse_dates=["date"], dtype={"code": str})
        else:
            frame = pd.DataFrame()

        if frame.empty or refresh or frame["date"].max() < end:
            market_code = ("sh" if code.startswith("6") else "sz") + code
            raw = ak.stock_zh_a_daily(
                symbol=market_code,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )
            frame = clean_price_data(raw, code, name)
            time.sleep(0.12)

        frame = frame[frame["date"].between(start, end)].copy()
        frame["name"] = name
        frame.to_csv(cache, index=False, encoding="utf-8-sig")
        frames.append(frame)

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.drop_duplicates(["date", "code"]).sort_values(["date", "code"])
    return prices


def get_stock_industry(pool: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """查询每只股票当前所属的申万一级行业。"""
    cache = CACHE_DIR / "stock_industry.csv"
    pool = pool.copy()
    pool["code"] = pool["code"].astype(str).str.zfill(6)

    if cache.exists() and not refresh:
        mapping = pd.read_csv(cache, dtype={"code": str, "industry_code": str})
        selected = mapping[mapping["code"].isin(pool["code"])]
        if len(selected) == len(pool):
            selected = selected.drop_duplicates("code", keep="last").copy()
            selected["industry"] = selected["industry"].astype(str).str.strip()
            selected["industry_code"] = selected["industry_code"].astype(str).str.strip()
            selected.to_csv(cache, index=False, encoding="utf-8-sig")
            return selected.reset_index(drop=True)
    else:
        mapping = pd.DataFrame(columns=["code", "name", "industry", "industry_code"])

    industry_index = ak.sw_index_first_info()[["行业名称", "行业代码"]].copy()
    industry_index["industry_code"] = industry_index["行业代码"].str.replace(".SI", "", regex=False)

    missing_codes = set(pool["code"]) - set(mapping["code"])
    rows = mapping.to_dict("records")
    for _, item in industry_index.iterrows():
        industry = item["行业名称"]
        industry_code = item["industry_code"]
        members = ak.index_component_sw(symbol=industry_code)
        members["证券代码"] = members["证券代码"].astype(str).str.zfill(6)
        matched = members[members["证券代码"].isin(missing_codes)]

        for _, stock in matched.iterrows():
            rows.append(
                {
                    "code": stock["证券代码"],
                    "name": stock["证券名称"],
                    "industry": industry,
                    "industry_code": industry_code,
                }
            )
            missing_codes.remove(stock["证券代码"])

        if not matched.empty:
            pd.DataFrame(rows).to_csv(cache, index=False, encoding="utf-8-sig")
        if not missing_codes:
            break
        time.sleep(0.12)

    mapping = pd.DataFrame(rows).drop_duplicates("code", keep="last")
    mapping["industry"] = mapping["industry"].astype(str).str.strip()
    mapping["industry_code"] = mapping["industry_code"].astype(str).str.strip()
    mapping.to_csv(cache, index=False, encoding="utf-8-sig")
    return mapping[mapping["code"].isin(pool["code"])].reset_index(drop=True)


def clean_industry_data(raw: pd.DataFrame, industry: str) -> pd.DataFrame:
    """整理行业指数日线并计算行业当日收益率。"""
    rename = {
        "日期": "date",
        "开盘": "industry_open",
        "收盘": "industry_close",
        "最高": "industry_high",
        "最低": "industry_low",
        "成交量": "industry_volume",
        "成交额": "industry_amount",
    }
    frame = raw.rename(columns=rename)[list(rename.values())].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    numeric = [column for column in rename.values() if column != "date"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame["industry"] = industry
    frame = frame.dropna(subset=["date", "industry_close"]).sort_values("date")
    frame["industry_return_1d"] = frame["industry_close"].pct_change(fill_method=None)
    return frame[["date", "industry", *numeric, "industry_return_1d"]]


def download_industry_prices(
    stock_industry: pd.DataFrame,
    start_date: str = START_DATE,
    end_date: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """下载股票池所涉及行业的日频行情。"""
    cache = CACHE_DIR / "industry_prices.csv"
    end_date = get_end_date(end_date)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    industries = stock_industry[["industry", "industry_code"]].copy()
    industries["industry"] = industries["industry"].astype(str).str.strip()
    industries["industry_code"] = industries["industry_code"].astype(str).str.strip()
    industries = industries.drop_duplicates().reset_index(drop=True)

    if cache.exists() and not refresh:
        data = pd.read_csv(cache, parse_dates=["date"], dtype={"industry_code": str})
        data = data[data["industry_code"].isin(industries["industry_code"])]
        latest_dates = data.groupby("industry_code")["date"].max()
        complete = set(industries["industry_code"]).issubset(set(latest_dates.index))
        if complete and latest_dates.min() >= end:
            data = data[data["date"].between(start, end)].drop_duplicates(["date", "industry_code"])
            data.to_csv(cache, index=False, encoding="utf-8-sig")
            return data.reset_index(drop=True)

    frames = []
    for i, row in industries.reset_index(drop=True).iterrows():
        industry = row["industry"]
        industry_code = row["industry_code"]
        print(f"[行业 {i + 1:02d}/{len(industries):02d}] {industry}", flush=True)
        raw = ak.index_hist_sw(symbol=industry_code, period="day")
        frame = clean_industry_data(raw, industry)
        frame = frame[frame["date"].between(start, end)].copy()
        frame["industry_code"] = industry_code
        frames.append(frame)
        time.sleep(0.12)

    data = pd.concat(frames, ignore_index=True)
    data.to_csv(cache, index=False, encoding="utf-8-sig")
    return data


def download_market_prices(
    start_date: str = START_DATE,
    end_date: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """下载沪深300日频行情并计算大盘当日收益率。"""
    cache = CACHE_DIR / "market_prices.csv"
    end_date = get_end_date(end_date)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if cache.exists() and not refresh:
        data = pd.read_csv(cache, parse_dates=["date"])
        if data["date"].max() >= end:
            data = data[data["date"].between(start, end)].copy()
            data.to_csv(cache, index=False, encoding="utf-8-sig")
            return data

    raw = ak.stock_zh_index_daily(symbol=MARKET_SYMBOL)
    rename = {
        "date": "date",
        "open": "market_open",
        "close": "market_close",
        "high": "market_high",
        "low": "market_low",
        "volume": "market_volume",
    }
    data = raw.rename(columns=rename)[list(rename.values())].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    numeric = [column for column in rename.values() if column != "date"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["date", "market_close"]).sort_values("date")
    data["market"] = MARKET_NAME
    data["market_return_1d"] = data["market_close"].pct_change(fill_method=None)
    data = data[data["date"].between(start, end)].copy()
    data.to_csv(cache, index=False, encoding="utf-8-sig")
    return data


def add_market_context(
    prices: pd.DataFrame,
    pool: pd.DataFrame,
    end_date: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """为个股行情匹配行业及沪深300的日频市场环境数据。"""
    stock_industry = get_stock_industry(pool, refresh=refresh)
    industry_prices = download_industry_prices(
        stock_industry,
        end_date=end_date,
        refresh=refresh,
    )
    market_prices = download_market_prices(
        end_date=end_date,
        refresh=refresh,
    )

    context_columns = [
        "industry",
        "industry_open",
        "industry_close",
        "industry_high",
        "industry_low",
        "industry_volume",
        "industry_amount",
        "industry_return_1d",
        "market",
        "market_open",
        "market_close",
        "market_high",
        "market_low",
        "market_volume",
        "market_return_1d",
    ]
    prices = prices.drop(
        columns=[column for column in context_columns if column in prices.columns]
    )

    prices = prices.merge(
        stock_industry[["code", "industry"]],
        on="code",
        how="left",
    )

    industry_columns = [
        "date",
        "industry",
        "industry_open",
        "industry_close",
        "industry_high",
        "industry_low",
        "industry_volume",
        "industry_amount",
        "industry_return_1d",
    ]
    prices = prices.merge(
        industry_prices[industry_columns].drop_duplicates(["date", "industry"]),
        on=["date", "industry"],
        how="left",
    )

    market_columns = [
        "date",
        "market",
        "market_open",
        "market_close",
        "market_high",
        "market_low",
        "market_volume",
        "market_return_1d",
    ]
    prices = prices.merge(
        market_prices[market_columns].drop_duplicates("date"),
        on="date",
        how="left",
    )

    return add_extra_data(prices, pool, end_date=end_date, refresh=refresh)


def load_prices() -> pd.DataFrame:
    """兼容入口；最终合并数据统一读Parquet，不再读旧行情CSV。"""
    return read_parquet_prices()


def fresh_cache(path: Path, end_date: str, refresh: bool) -> bool:
    """同一天已下载的文件直接复用；失败的下载不会写成空缓存。"""
    return (path.exists() and not refresh
            and pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").tz_convert("Asia/Shanghai")
            .normalize().tz_localize(None) >= pd.Timestamp(end_date))


def download_company_data(pool: pd.DataFrame, end_date: str, refresh: bool = False):
    """每股一份缓存；基本面单位为元、比率单位为百分数，EPS/ROE等为年初至今。"""
    folder = CACHE_DIR / "fundamentals"
    folder.mkdir(exist_ok=True)
    reports, raw_prices, dividends = [], [], []
    for i, item in pool.reset_index(drop=True).iterrows():
        code = str(item["code"]).zfill(6)
        market = "sh" if code.startswith("6") else "sz"
        print(f"[基本面 {i + 1:03d}/{len(pool)}] {code}", flush=True)

        path = folder / f"{code}_reports.csv"
        if fresh_cache(path, end_date, refresh):
            report = pd.read_csv(path, dtype={"code": str})
        else:
            raw = ak.stock_financial_analysis_indicator_em(symbol=f"{code}.{market.upper()}")
            report = raw.rename(columns=FINANCIAL_FIELDS)[list(FINANCIAL_FIELDS.values())].copy()
            report["code"] = code
            report["financial_currency"] = raw["CURRENCY"].to_numpy()
            report.to_csv(path, index=False, encoding="utf-8-sig")
        reports.append(report)

        path = folder / f"{code}_raw_prices.csv"
        if fresh_cache(path, end_date, refresh):
            unadjusted = pd.read_csv(path, dtype={"code": str})
            if "unadjusted_price_source" not in unadjusted:
                unadjusted["unadjusted_price_source"] = "AkShare/新浪"
        else:
            # 腾讯免费接口一只股票一次请求；与原有前复权行情分开，不覆盖原OHLC。
            response = requests.get("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
                params={"param": f"{market + code},day,{pd.Timestamp(START_DATE).date()},{pd.Timestamp(end_date).date()},2000,"}, timeout=30)
            response.raise_for_status()
            rows = response.json()["data"][market + code]["day"]
            unadjusted = pd.DataFrame([row[:3] for row in rows], columns=["date", "raw_open", "unadjusted_close"])
            unadjusted = unadjusted.drop(columns="raw_open")
            unadjusted["unadjusted_close"] = pd.to_numeric(unadjusted["unadjusted_close"], errors="coerce")
            unadjusted["code"] = code
            unadjusted["unadjusted_price_source"] = "腾讯免费日线"
            unadjusted.to_csv(path, index=False, encoding="utf-8-sig")
        raw_prices.append(unadjusted[["date", "code", "unadjusted_close", "unadjusted_price_source"]])

        path = folder / f"{code}_dividends.csv"
        if fresh_cache(path, end_date, refresh):
            dividend = pd.read_csv(path, dtype={"code": str})
        else:
            raw = ak.stock_history_dividend_detail(symbol=code, indicator="分红")
            rename = {"公告日期": "dividend_notice_date", "除权除息日": "ex_dividend_date",
                      "派息": "cash_per_10_shares", "送股": "bonus_per_10_shares",
                      "转增": "transfer_per_10_shares", "进度": "dividend_status"}
            dividend = raw.rename(columns=rename).reindex(columns=list(rename.values()))
            dividend["code"] = code
            dividend.to_csv(path, index=False, encoding="utf-8-sig")
        dividends.append(dividend)
        time.sleep(0.12)

    reports = pd.concat(reports, ignore_index=True)
    for column in ("financial_report_date", "financial_notice_date", "financial_update_date"):
        reports[column] = pd.to_datetime(reports[column], errors="coerce")
    numeric = list(FINANCIAL_FIELDS.values())[3:]
    reports[numeric] = reports[numeric].apply(pd.to_numeric, errors="coerce")
    # 免费接口可能返回修订值：取较晚日期，并推迟到次日09:00，不假装是原始披露版本。
    reports["financial_available_at"] = reports[["financial_notice_date", "financial_update_date"]].max(axis=1)
    reports["financial_available_at"] += pd.Timedelta(days=1, hours=9)
    reports = reports.dropna(subset=["financial_report_date", "financial_available_at"])
    reports = reports[reports["financial_currency"] == "CNY"]
    reports = reports.drop_duplicates(["code", "financial_report_date"], keep="last")

    raw_prices = pd.concat(raw_prices, ignore_index=True)
    raw_prices["date"] = pd.to_datetime(raw_prices["date"])
    raw_prices = raw_prices.drop_duplicates(["date", "code"], keep="last")
    dividends = pd.concat(dividends, ignore_index=True)
    for column in ("dividend_notice_date", "ex_dividend_date"):
        dividends[column] = pd.to_datetime(dividends[column], errors="coerce")
    dividends["cash_per_share"] = pd.to_numeric(dividends["cash_per_10_shares"], errors="coerce") / 10
    reports.to_csv(CACHE_DIR / "financial_reports.csv", index=False, encoding="utf-8-sig")
    raw_prices.to_csv(CACHE_DIR / "valuation_prices.csv", index=False, encoding="utf-8-sig")
    dividends.to_csv(CACHE_DIR / "dividend_events.csv", index=False, encoding="utf-8-sig")
    return reports, raw_prices, dividends


def latest_report_events(reports: pd.DataFrame) -> pd.DataFrame:
    """后补的旧报告不能覆盖已经可用的新报告。输入为一只股票的报告。"""
    reports = reports.sort_values(["financial_available_at", "financial_report_date"])
    reports = reports[reports["financial_report_date"] == reports["financial_report_date"].cummax()]
    return reports.drop_duplicates("financial_available_at", keep="last")


def match_company_data(prices, reports, raw_prices, dividends):
    """按09:00时点回看财报；估值用当日未复权收盘价，保持原项目收盘后使用的场景。"""
    prices["signal_time"] = pd.to_datetime(prices["signal_time"]).astype("datetime64[ns]")
    reports["financial_available_at"] = pd.to_datetime(reports["financial_available_at"]).astype("datetime64[ns]")
    prices = prices.merge(raw_prices, on=["date", "code"], how="left", validate="one_to_one")
    # 新浪原始成交量/换手率可还原流通股本；这是估计流通股数，不是总股本。
    prices["float_shares_estimate"] = prices["volume"] / prices["turnover"].where(prices["turnover"] > 0)
    pieces = []
    for code, stock in prices.groupby("code", sort=False):
        stock = stock.sort_values("signal_time").copy()
        report = reports[reports["code"] == code].drop(columns="code")
        stock = pd.merge_asof(stock, latest_report_events(report), left_on="signal_time",
                              right_on="financial_available_at", direction="backward")
        # PE采用静态口径：最后一份当时已经可用的年报EPS，不把季报EPS当全年EPS。
        annual = report[report["financial_report_date"].dt.month == 12]
        annual = latest_report_events(annual)[["financial_report_date", "financial_available_at", "eps_ytd"]]
        annual = annual.rename(columns={"financial_report_date": "annual_report_date",
                                        "financial_available_at": "annual_available_at", "eps_ytd": "eps_annual"})
        stock = pd.merge_asof(stock, annual, left_on="signal_time", right_on="annual_available_at",
                              direction="backward")
        events = dividends[(dividends["code"] == code) & (dividends["dividend_status"] == "实施")].copy()
        events = events.dropna(subset=["dividend_notice_date", "ex_dividend_date", "cash_per_share"])
        events["known_date"] = pd.concat([events["dividend_notice_date"] + pd.Timedelta(days=1),
                                          events["ex_dividend_date"]], axis=1).max(axis=1)
        # 除息口径、过去365个自然日的税前现金分红；成功查询且没有分红事件才记0。
        dates = pd.date_range(stock["date"].min() - pd.Timedelta(days=366), stock["date"].max())
        cash = events.groupby("known_date")["cash_per_share"].sum().reindex(dates, fill_value=0)
        stock["cash_dividend_365d"] = stock["date"].map(cash.rolling("365D").sum())
        pieces.append(stock)

    prices = pd.concat(pieces, ignore_index=True).copy()
    positive_price = prices["unadjusted_close"].where(prices["unadjusted_close"] > 0)
    positive_eps = prices["eps_annual"].where(prices["eps_annual"] > 0)
    positive_book = prices["book_value_per_share"].where(prices["book_value_per_share"] > 0)
    prices["pe_static"] = positive_price / positive_eps
    prices["pb"] = positive_price / positive_book
    prices["book_to_market"] = positive_book / positive_price
    prices["dividend_yield_365d_pct"] = prices["cash_dividend_365d"] / positive_price * 100
    prices["financial_age_days"] = (prices["date"] - prices["financial_report_date"]).dt.days
    return prices


def download_macro_data(end_date: str, refresh: bool = False):
    """原始利率保持百分数；GDP只匹配有发布日期的序列，不猜测新季度的公布日期。"""
    paths = {name: CACHE_DIR / f"{name}.csv" for name in
             ("shibor", "treasury_yields", "gdp_releases", "gdp_quarterly")}
    if fresh_cache(paths["shibor"], end_date, refresh):
        shibor = pd.read_csv(paths["shibor"])
    else:
        shibor = ak.macro_china_shibor_all().rename(columns={"日期": "shibor_date",
            "O/N-定价": "shibor_overnight_pct", "1W-定价": "shibor_1w_pct",
            "1M-定价": "shibor_1m_pct", "3M-定价": "shibor_3m_pct", "1Y-定价": "shibor_1y_pct"})
        shibor = shibor[["shibor_date", "shibor_overnight_pct", "shibor_1w_pct",
                         "shibor_1m_pct", "shibor_3m_pct", "shibor_1y_pct"]]
        shibor.to_csv(paths["shibor"], index=False, encoding="utf-8-sig")
    if fresh_cache(paths["treasury_yields"], end_date, refresh):
        bonds = pd.read_csv(paths["treasury_yields"])
    else:
        bonds = ak.bond_zh_us_rate(start_date="20190101")
        rename = {"日期": "yield_date"}
        for country, prefix in (("中国", "cn"), ("美国", "us")):
            for years in (2, 5, 10, 30):
                rename[f"{country}国债收益率{years}年"] = f"{prefix}_yield_{years}y_pct"
        bonds = bonds.rename(columns=rename)[list(rename.values())]
        bonds.to_csv(paths["treasury_yields"], index=False, encoding="utf-8-sig")
    if fresh_cache(paths["gdp_releases"], end_date, refresh):
        gdp = pd.read_csv(paths["gdp_releases"])
    else:
        gdp = ak.macro_china_gdp_yearly().rename(columns={"日期": "gdp_release_date", "今值": "cn_gdp_yoy_pct"})
        gdp = gdp[["gdp_release_date", "cn_gdp_yoy_pct"]]
        gdp.to_csv(paths["gdp_releases"], index=False, encoding="utf-8-sig")
    if not fresh_cache(paths["gdp_quarterly"], end_date, refresh):
        # 最新季度总量另存，缺少发布日期，不并入历史特征。
        ak.macro_china_gdp().to_csv(paths["gdp_quarterly"], index=False, encoding="utf-8-sig")
    return shibor, bonds, gdp


def match_macro_data(daily, shibor, bonds, gdp):
    daily["signal_time"] = pd.to_datetime(daily["signal_time"]).astype("datetime64[ns]")
    shibor["shibor_date"] = pd.to_datetime(shibor["shibor_date"]).astype("datetime64[ns]")
    shibor["shibor_available_at"] = shibor["shibor_date"] + pd.Timedelta(days=1, hours=9)
    daily = pd.merge_asof(daily, shibor.sort_values("shibor_available_at"), left_on="signal_time",
                          right_on="shibor_available_at", direction="backward")
    bonds["yield_date"] = pd.to_datetime(bonds["yield_date"]).astype("datetime64[ns]")
    for country in ("cn", "us"):
        columns = [f"{country}_yield_{years}y_pct" for years in (2, 5, 10, 30)]
        rates = bonds[["yield_date", *columns]].dropna(subset=columns, how="all").copy()
        date_name, time_name = f"{country}_yield_date", f"{country}_yield_available_at"
        rates = rates.rename(columns={"yield_date": date_name}).sort_values(date_name)
        if country == "cn":
            rates[time_name] = rates[date_name] + pd.Timedelta(days=1, hours=9)
        else:
            # 美国当日收益率在纽约18:00之后才视为可用，不能按同名日期并到A股盘前。
            rates[time_name] = (rates[date_name].dt.tz_localize("America/New_York") + pd.Timedelta(hours=18))
            rates[time_name] = rates[time_name].dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        daily = pd.merge_asof(daily, rates.sort_values(time_name), left_on="signal_time",
                              right_on=time_name, direction="backward")
        daily[f"{country}_term_spread_10y_2y_pp"] = daily[columns[2]] - daily[columns[0]]
    daily["cn_us_spread_10y_pp"] = daily["cn_yield_10y_pct"] - daily["us_yield_10y_pct"]
    gdp["gdp_release_date"] = pd.to_datetime(gdp["gdp_release_date"]).astype("datetime64[ns]")
    gdp["cn_gdp_yoy_pct"] = pd.to_numeric(gdp["cn_gdp_yoy_pct"], errors="coerce")
    gdp = gdp.dropna(subset=["gdp_release_date", "cn_gdp_yoy_pct"])
    gdp["gdp_available_at"] = gdp["gdp_release_date"] + pd.Timedelta(days=1, hours=9)
    daily = pd.merge_asof(daily, gdp.sort_values("gdp_available_at"), left_on="signal_time",
                          right_on="gdp_available_at", direction="backward")
    daily["gdp_age_days"] = (daily["date"] - daily["gdp_release_date"]).dt.days
    return daily


def make_calendar_data(dates):
    """使用A股交易所日历，节假日标志区分普通周末；日历尾端未知值不伪装成月末。"""
    calendar = ak.tool_trade_date_hist_sina().rename(columns={"trade_date": "date"})
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar = calendar.drop_duplicates("date").sort_values("date")
    calendar["weekday"] = calendar["date"].dt.dayofweek + 1
    calendar["month"] = calendar["date"].dt.month
    calendar["quarter"] = calendar["date"].dt.quarter
    calendar["day_of_month"] = calendar["date"].dt.day
    previous, following = calendar["date"].shift(1), calendar["date"].shift(-1)
    calendar["days_since_prev_trade"] = (calendar["date"] - previous).dt.days
    calendar["days_to_next_trade"] = (following - calendar["date"]).dt.days
    calendar["is_month_first_trade"] = (previous.dt.month != calendar["month"]).astype(int)
    calendar["is_month_last_trade"] = (following.dt.month != calendar["month"]).astype(int)
    calendar["is_after_long_break"] = (calendar["days_since_prev_trade"] > 3).astype(int)
    calendar["is_before_long_break"] = (calendar["days_to_next_trade"] > 3).astype(int)
    for column in ("days_to_next_trade", "is_month_last_trade", "is_before_long_break"):
        calendar.loc[following.isna(), column] = float("nan")
    calendar = calendar[calendar["date"].isin(pd.to_datetime(dates))].copy()
    # 交易日之间若存在周一至周五休市，标为节假日/特殊休市；普通周末不标为节假日。
    day_values = calendar["date"].to_numpy(dtype="datetime64[D]")
    prev_values = previous.loc[calendar.index].fillna(calendar["date"]).to_numpy(dtype="datetime64[D]")
    next_values = following.loc[calendar.index].fillna(calendar["date"]).to_numpy(dtype="datetime64[D]")
    calendar["closed_weekdays_before"] = np.busday_count(prev_values + np.timedelta64(1, "D"), day_values)
    calendar["closed_weekdays_after"] = np.busday_count(day_values + np.timedelta64(1, "D"), next_values)
    calendar["is_after_holiday"] = (calendar["closed_weekdays_before"] > 0).astype(int)
    calendar["is_before_holiday"] = (calendar["closed_weekdays_after"] > 0).astype(int)
    for column in ("closed_weekdays_after", "is_before_holiday"):
        calendar.loc[following.loc[calendar.index].isna(), column] = float("nan")
    calendar["signal_time"] = calendar["date"] + pd.Timedelta(hours=9)
    calendar.to_csv(CACHE_DIR / "calendar.csv", index=False, encoding="utf-8-sig")
    return calendar.reset_index(drop=True)


def download_global_indices(end_date: str, refresh: bool = False):
    """Yahoo价格指数日线；保留应有但缺失的交易日，不用前值冒充下载成功。"""
    import exchange_calendars as xcals

    folder = CACHE_DIR / "global_indices"
    folder.mkdir(exist_ok=True)
    frames = {}
    start = pd.Timestamp(START_DATE) - pd.Timedelta(days=40)
    end = pd.Timestamp(end_date)
    for name, (symbol, calendar_name) in GLOBAL_INDICES.items():
        path = folder / f"{name}.csv"
        print(f"[全球指数] {name} {symbol}", flush=True)
        if fresh_cache(path, end_date, refresh):
            frame = pd.read_csv(path, parse_dates=["source_date", "available_at"])
        else:
            response = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                params={"period1": int(start.tz_localize("UTC").timestamp()),
                        "period2": int((end + pd.Timedelta(days=1)).tz_localize("UTC").timestamp()),
                        "interval": "1d"}, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            response.raise_for_status()
            raw = response.json()["chart"]["result"][0]
            metadata = raw["meta"]
            frame = pd.DataFrame(raw["indicators"]["quote"][0])
            timestamps = pd.to_datetime(raw["timestamp"], unit="s", utc=True)
            frame["source_date"] = timestamps.tz_convert(metadata["exchangeTimezoneName"]).tz_localize(None).normalize()
            calendar = xcals.get_calendar(calendar_name, start=start, end=end)
            sessions = calendar.sessions_in_range(start, end)
            frame = frame.drop_duplicates("source_date").set_index("source_date").reindex(sessions)
            frame.index.name = "source_date"
            frame = frame.reset_index()
            # 正常/提前收市及夏令时由日历处理，再加1小时发布缓冲。VIX等使用保守缓冲。
            closes = pd.DatetimeIndex([calendar.session_close(day) for day in sessions])
            frame["available_at"] = (closes + pd.Timedelta(hours=1)).tz_convert("Asia/Shanghai").tz_localize(None)
            frame = frame[frame["available_at"] <= pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)].copy()
            frame["symbol"] = symbol
            frame["currency"] = metadata.get("currency", "")
            frame["price_basis"] = "Yahoo原指数口径，非统一全收益口径"
            frame.to_csv(path, index=False, encoding="utf-8-sig")
            time.sleep(0.2)
        frames[name] = frame
    return frames


def match_global_data(daily, frames):
    """先对齐统一A股09:00时点，再匹配个股；长假变化可由相邻可见收盘价累计得到。"""
    daily["signal_time"] = pd.to_datetime(daily["signal_time"]).astype("datetime64[ns]")
    for name, frame in frames.items():
        frame = frame.copy()
        frame["available_at"] = pd.to_datetime(frame["available_at"]).astype("datetime64[ns]")
        frame = frame.sort_values("available_at")
        frame["new_sessions"] = range(1, len(frame) + 1)
        frame["missing_count"] = frame["close"].isna().cumsum()
        columns = ["source_date", "available_at", "open", "high", "low", "close", "volume",
                   "new_sessions", "missing_count"]
        prefix = f"global_{name}_"
        frame = frame[columns].rename(columns={column: prefix + column for column in columns})
        daily = pd.merge_asof(daily, frame, left_on="signal_time", right_on=prefix + "available_at",
                              direction="backward")
        missing = daily[prefix + "missing_count"].diff().fillna(0) > 0
        daily[prefix + "missing_flag"] = daily[prefix + "close"].isna() | missing
        daily[prefix + "new_sessions"] = daily[prefix + "new_sessions"].diff()
        daily[prefix + "staleness_hours"] = (daily["signal_time"] - daily[prefix + "available_at"]).dt.total_seconds() / 3600
        daily = daily.drop(columns=prefix + "missing_count")
    return daily


def write_data_structure(prices):
    """输出实际字段、类型、缺失率与最新有效日期，便于检查来源停更或历史缺口。"""
    rows = []
    sources = {"原有个股行情": "AkShare/新浪：前复权OHLC，成交量与成交额保持接口口径",
               "原有行业与大盘": "AkShare：申万一级行业、沪深300",
               "全球指数": "Yahoo Finance日线 + exchange-calendars；收盘加1小时后可用",
               "A股日历": "AkShare/新浪交易日历；临时休市未还原事前公告版本",
               "宏观": "AkShare：Shibor、中美国债、带发布日期的GDP同比",
               "基本面与估值": "AkShare/东方财富财报、新浪未复权价格及分红；PE为静态口径"}
    calendar_fields = set(pd.read_csv(CACHE_DIR / "calendar.csv", nrows=0).columns) - {"date"}
    original = {"date", "code", "name", "open", "high", "low", "close", "volume", "amount", "turnover"}
    for column in prices.columns:
        if column in original:
            group = "原有个股行情"
        elif column.startswith("industry") or column.startswith("market"):
            group = "原有行业与大盘"
        elif column.startswith("global_"):
            group = "全球指数"
        elif column in calendar_fields:
            group = "A股日历"
        elif column.startswith(("shibor", "cn_", "us_", "gdp")):
            group = "宏观"
        else:
            group = "基本面与估值"
        valid = prices[column].notna()
        unit = "接口原始单位"
        if column.endswith("_pct"):
            unit = "百分数，例如5表示5%"
        elif column.endswith("_pp"):
            unit = "百分点"
        elif column in {"pe_static", "pb", "book_to_market"}:
            unit = "倍/比值"
        rows.append({"group": group, "field": column, "dtype": str(prices[column].dtype),
                     "source_and_basis": sources[group], "unit": unit,
                     "missing_pct": round((1 - valid.mean()) * 100, 3),
                     "latest_valid_stock_date": prices.loc[valid, "date"].max()})
    pd.DataFrame(rows).to_csv(CACHE_DIR / "data_structure.csv", index=False, encoding="utf-8-sig")


def add_extra_data(prices, pool, end_date=None, refresh=False):
    """仅增加数据列，不改原OHLC、样本行、因子名单、模型或网页。"""
    prepare_folders()
    end_date = end_date or prices["date"].max().strftime("%Y%m%d")
    # 重跑时清除上次新增列，保留原有个股、行业和大盘数据，避免重复字段。
    original = [column for column in prices.columns if column in
                {"date", "code", "name", "open", "high", "low", "close", "volume", "amount", "turnover"}
                or column.startswith(("industry", "market"))]
    prices = prices[original].copy()
    daily = make_calendar_data(prices["date"].unique())
    daily = match_macro_data(daily, *download_macro_data(end_date, refresh))
    daily = match_global_data(daily, download_global_indices(end_date, refresh))
    daily.to_csv(CACHE_DIR / "daily_context.csv", index=False, encoding="utf-8-sig")
    prices = prices.merge(daily, on="date", how="left", validate="many_to_one")
    prices = match_company_data(prices, *download_company_data(pool, end_date, refresh))
    prices = prices.sort_values(["date", "code"]).reset_index(drop=True)
    write_data_structure(prices)
    print(f"数据补充完成：{len(prices):,}行，{prices['code'].nunique()}只股票，{len(prices.columns)}列。", flush=True)
    return prices


def save_prices_parquet(prices: pd.DataFrame, output_path: Path | None = None) -> Path:
    """所有来源完整合并、字段校验与临时文件回读成功后，原子替换主表。"""
    from extended_factors import CATALOG

    prices = validate_prices(prices)
    required = set().union(*(set(spec["source_fields"].split(",")) for spec in CATALOG.values()))
    required.add("market")
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"合并数据缺少后续因子字段，保留旧Parquet：{sorted(missing)}")
    # 统一日期类型，避免CSV缓存把部分时间戳变为字符串；不更改数据匹配口径。
    for column in prices:
        if column == "date" or column.endswith(("_date", "_available_at")) or column == "signal_time":
            prices[column] = pd.to_datetime(prices[column], errors="raise").astype("datetime64[ns]")
    target = Path(output_path) if output_path is not None else PRICE_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.stem}_", suffix=".parquet", dir=target.parent)
    os.close(descriptor)
    temporary = Path(name)
    old_mode = target.stat().st_mode if target.exists() else None
    made_writable = False
    try:
        prices.to_parquet(temporary, index=False, engine="pyarrow", compression="snappy")
        import pyarrow.parquet as pq
        metadata = pq.ParquetFile(temporary).metadata
        if metadata.num_rows != len(prices) or metadata.num_columns != len(prices.columns):
            raise ValueError("临时Parquet行列数校验失败，保留旧文件")
        # 原项目主表带Windows只读属性。仅在新文件完整写好后解除目标的只读属性。
        if old_mode is not None and not old_mode & stat.S_IWRITE:
            target.chmod(old_mode | stat.S_IWRITE)
            made_writable = True
        os.replace(temporary, target)
    except Exception:
        if made_writable and target.exists():
            target.chmod(old_mode)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return target


def download_dataset(n_stocks: int = 300, end_date: str | None = None,
                     refresh: bool = False, output_path: Path | None = None) -> Path:
    """在线入口：股票池→行情→行业/大盘→财报/宏观/日历/全球→完整Parquet。"""
    if not 1 <= n_stocks <= 300:
        raise ValueError("下载股票数必须为1至300")
    prepare_folders()
    end_date = get_end_date(end_date)
    if pd.Timestamp(end_date) < pd.Timestamp(START_DATE):
        raise ValueError(f"截止日期不能早于{START_DATE}")
    pool = get_stock_pool(n_stocks=n_stocks, refresh=refresh)
    if pool.empty:
        raise ValueError("在线接口返回空股票池，保留旧Parquet")
    prices = download_prices(pool, end_date=end_date, refresh=refresh)
    prices = add_market_context(prices, pool, end_date=end_date, refresh=refresh)
    if set(prices["code"]) != set(pool["code"]):
        raise ValueError("合并数据与股票池不一致，保留旧Parquet")
    path = save_prices_parquet(prices, output_path)
    print(f"下载合并完成：{path}；{len(prices):,}行，{prices['code'].nunique()}只股票，{len(prices.columns)}列。", flush=True)
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="在线下载完整数据并生成根目录prices.parquet，不训练模型")
    parser.add_argument("--stocks", type=int, default=300, help="沪深300成分股数量，默认300，按代码排序")
    parser.add_argument("--end-date", default=None, help="下载截止日期，格式YYYYMMDD")
    parser.add_argument("--refresh", action="store_true", help="忽略原始缓存，重新下载全部来源")
    args = parser.parse_args()
    download_dataset(args.stocks, args.end_date, args.refresh)


if __name__ == "__main__":
    main()
