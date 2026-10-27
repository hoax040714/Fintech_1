"""只用 AkShare 完成个股、申万一级行业和沪深300的日频匹配。"""

from pathlib import Path
import time

import akshare as ak
import pandas as pd


START_DATE = "20160101"
END_DATE = None
N_STOCKS = 50
OUTPUT_FILE = Path(__file__).with_name("stock_industry_market_matched.csv")


def get_stock_pool(n_stocks=50):
    """取沪深300当前成分股中权重最大的若干只股票。"""
    raw = ak.index_stock_cons_weight_csindex(symbol="000300")
    pool = raw.iloc[:, [4, 5, 9]].copy()
    pool.columns = ["code", "name", "weight"]
    pool["code"] = pool["code"].astype(str).str.zfill(6)
    pool["weight"] = pd.to_numeric(pool["weight"], errors="coerce")
    return pool.sort_values("weight", ascending=False).head(n_stocks).reset_index(drop=True)


def get_stock_industry(pool):
    """从申万一级行业成分股中反查每只股票所属行业。"""
    industry_list = ak.sw_index_first_info()[["行业名称", "行业代码"]].copy()
    industry_list["industry_code"] = industry_list["行业代码"].str.replace(".SI", "", regex=False)

    rows = []
    stock_codes = set(pool["code"])

    for _, industry in industry_list.iterrows():
        industry_name = industry["行业名称"]
        industry_code = industry["industry_code"]
        members = ak.index_component_sw(symbol=industry_code)
        members["证券代码"] = members["证券代码"].astype(str).str.zfill(6)
        members = members[members["证券代码"].isin(stock_codes)]

        for _, stock in members.iterrows():
            rows.append(
                {
                    "code": stock["证券代码"],
                    "industry": industry_name,
                    "industry_code": industry_code,
                }
            )

    return pd.DataFrame(rows).drop_duplicates("code")


def download_stock_prices(pool, start_date, end_date):
    """下载个股前复权日线。"""
    frames = []

    for i, stock in pool.iterrows():
        code = stock["code"]
        name = stock["name"]
        market_code = ("sh" if code.startswith("6") else "sz") + code
        print(f"个股 {i + 1:02d}/{len(pool):02d}: {code} {name}")

        frame = ak.stock_zh_a_daily(
            symbol=market_code,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )[["date", "open", "close", "high", "low", "volume", "amount"]].copy()

        frame["date"] = pd.to_datetime(frame["date"])
        frame["code"] = code
        frame["name"] = name
        frame["stock_return_1d"] = frame["close"].pct_change(fill_method=None)
        frames.append(frame)
        time.sleep(0.1)

    return pd.concat(frames, ignore_index=True)


def download_industry_prices(stock_industry, start_date, end_date):
    """下载股票池涉及的申万一级行业指数日线。"""
    frames = []
    industries = stock_industry[["industry", "industry_code"]].drop_duplicates()
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    for _, industry in industries.iterrows():
        name = industry["industry"]
        code = industry["industry_code"]
        print(f"行业: {name}")

        raw = ak.index_hist_sw(symbol=code, period="day")
        frame = raw.rename(
            columns={
                "日期": "date",
                "开盘": "industry_open",
                "收盘": "industry_close",
                "最高": "industry_high",
                "最低": "industry_low",
            }
        )[["date", "industry_open", "industry_close", "industry_high", "industry_low"]].copy()

        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
        frame["industry"] = name
        frame["industry_return_1d"] = frame["industry_close"].pct_change(fill_method=None)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def download_market_prices(start_date, end_date):
    """下载沪深300日线。"""
    frame = ak.stock_zh_index_daily(symbol="sh000300").rename(
        columns={
            "open": "market_open",
            "close": "market_close",
            "high": "market_high",
            "low": "market_low",
        }
    )

    frame["date"] = pd.to_datetime(frame["date"])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    frame = frame[(frame["date"] >= start) & (frame["date"] <= end)].copy()
    frame["market_return_1d"] = frame["market_close"].pct_change(fill_method=None)

    return frame[
        ["date", "market_open", "market_close", "market_high", "market_low", "market_return_1d"]
    ]


def main():
    end_date = END_DATE or pd.Timestamp.today().strftime("%Y%m%d")

    pool = get_stock_pool(N_STOCKS)
    stock_industry = get_stock_industry(pool)
    stock_prices = download_stock_prices(pool, START_DATE, end_date)
    industry_prices = download_industry_prices(stock_industry, START_DATE, end_date)
    market_prices = download_market_prices(START_DATE, end_date)

    data = stock_prices.merge(stock_industry, on="code", how="left")
    data = data.merge(industry_prices, on=["date", "industry"], how="left")
    data = data.merge(market_prices, on="date", how="left")

    data["excess_return_vs_industry"] = data["stock_return_1d"] - data["industry_return_1d"]
    data["excess_return_vs_market"] = data["stock_return_1d"] - data["market_return_1d"]
    data = data.sort_values(["date", "code"])
    data.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\n完成：{len(data):,} 行，{data['code'].nunique()} 只股票")
    print(f"输出文件：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
