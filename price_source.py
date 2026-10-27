"""下游统一读取根目录 prices.parquet；在线更新由 data.py 负责。"""

from pathlib import Path
import hashlib

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"
PRICE_FILE = BASE_DIR / "prices.parquet"
FEATURE_FILE = DATA_DIR / "features.parquet"
FACTOR_CATALOG_FILE = DATA_DIR / "factor_catalog_extended.csv"
PRICE_COLUMNS = [
    "date", "code", "name", "open", "close", "high", "low", "volume", "amount", "turnover",
]


def prepare_folders() -> None:
    for folder in (DATA_DIR, MODEL_DIR, RESULT_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def source_signature() -> str:
    """把训练结果绑定到本次源文件，避免旧CSV预测与新Parquet行情混用。"""
    digest = hashlib.sha256()
    with PRICE_FILE.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prices(columns: list[str] | None = None) -> pd.DataFrame:
    """读取并检查源表；不访问网络，不回退到 CSV，不改写源文件。"""
    if not PRICE_FILE.is_file():
        raise FileNotFoundError(f"缺少唯一行情数据源：{PRICE_FILE}")
    selected = None if columns is None else list(dict.fromkeys([*PRICE_COLUMNS, *columns]))
    frame = pd.read_parquet(PRICE_FILE, columns=selected, engine="pyarrow")
    return validate_prices(frame)


def validate_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """供下载落盘和下游读取共用的基础表检查；不填充或重算行情。"""
    missing = set(PRICE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"prices.parquet 缺少基础字段：{sorted(missing)}")
    if frame.empty or frame[["date", "code", "name"]].isna().any().any():
        raise ValueError("行情为空，或 date/code/name 存在缺失值")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    if not frame["code"].str.fullmatch(r"\d{6}").all():
        raise ValueError("股票代码必须为六位数字")
    if frame.duplicated(["date", "code"]).any():
        raise ValueError("prices.parquet 存在重复的 date/code，必须先确认源数据")
    for column in PRICE_COLUMNS[3:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if (frame[["open", "close", "high", "low"]] <= 0).any().any():
        raise ValueError("基础 OHLC 价格存在非正值")
    return frame.sort_values(["date", "code"]).reset_index(drop=True)
