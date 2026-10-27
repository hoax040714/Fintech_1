"""Streamlit 展示页：Nocturne Glassmorphism 夜航玻璃拟态。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from models import MODEL_LABELS


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULT_DIR = BASE_DIR / "results"

st.set_page_config(page_title="股票收益预测 · Stock Return Lab", page_icon="◈", layout="wide")

st.markdown(
    """
<style>
:root {
  --night: #0B1322;
  --night-deep: #060A13;
  --moon: #7C9CC4;
  --gold: #E4B863;
  --gold-light: #F3DCA8;
  --text: rgba(255,255,255,.94);
  --text-soft: rgba(255,255,255,.76);
  --text-muted: rgba(255,255,255,.54);
  --glass: rgba(255,255,255,.08);
  --glass-soft: rgba(255,255,255,.055);
  --glass-hover: rgba(255,255,255,.12);
  --line: rgba(255,255,255,.15);
  --line-hover: rgba(255,255,255,.30);
  --radius: 24px;
  --spring: cubic-bezier(.16,1,.3,1);
  --glass-shadow: 0 16px 40px rgba(3,7,18,.50), inset 0 1px 0 rgba(255,255,255,.22), inset 0 -1px 0 rgba(2,6,16,.35);
}
html, body, [class*="css"] {
  font-family: "Aptos", "Segoe UI Variable", "Microsoft YaHei UI", sans-serif;
}
[data-testid="stMain"] {
  overflow-y: scroll !important;
  scrollbar-gutter: stable;
  overscroll-behavior-y: auto;
}
[data-testid="stMain"]::-webkit-scrollbar { width: 12px; }
[data-testid="stMain"]::-webkit-scrollbar-track { background: rgba(255,255,255,.04); }
[data-testid="stMain"]::-webkit-scrollbar-thumb {
  background: rgba(255,255,255,.28);
  border: 3px solid transparent;
  border-radius: 12px;
  background-clip: padding-box;
}
[data-testid="stMain"]::-webkit-scrollbar-thumb:hover { background-color: rgba(255,255,255,.42); }
[data-testid="stAppViewContainer"] {
  color: var(--text);
  background:
    radial-gradient(circle at 88% 4%, rgba(124,156,196,.28) 0, rgba(124,156,196,.12) 24%, transparent 57%),
    radial-gradient(circle at 7% 88%, rgba(51,81,122,.38) 0, rgba(51,81,122,.14) 28%, transparent 62%),
    radial-gradient(circle at 54% 42%, rgba(228,184,99,.11) 0, transparent 34%),
    linear-gradient(145deg, var(--night-deep) 0%, var(--night) 48%, #0A101C 100%);
  background-attachment: fixed;
}
[data-testid="stAppViewContainer"]::before {
  content: "";
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  opacity: .025;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.95'/%3E%3C/svg%3E");
}
[data-testid="stAppViewContainer"] > .main { position: relative; z-index: 1; }
[data-testid="stHeader"] {
  background: rgba(6,10,19,.18);
  backdrop-filter: blur(60px) saturate(180%);
  -webkit-backdrop-filter: blur(60px) saturate(180%);
  border-bottom: 1px solid rgba(255,255,255,.08);
}
[data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
.block-container {
  width: 100%;
  max-width: none;
  padding: 1.5rem clamp(1rem, 3vw, 3.5rem) 4rem;
}
.block-container > div { width: 100%; min-width: 0; }
h1, h2, h3, h4 {
  color: var(--text);
  letter-spacing: -.02em;
  font-family: "Iowan Old Style", "Noto Serif SC", "Songti SC", serif;
}
p, label, [data-testid="stMarkdownContainer"] { color: var(--text-soft); }
p { line-height: 1.72; }
.project-title {
  color: var(--text);
  font-family: "Iowan Old Style", "Noto Serif SC", "Songti SC", serif;
  font-size: 34px;
  font-weight: 650;
  letter-spacing: -.035em;
  margin: 10px 0 28px;
  text-shadow: 0 8px 30px rgba(3,7,18,.65);
}
.section-kicker {
  color: var(--gold);
  font: 650 12px ui-monospace, "Cascadia Mono", Consolas, monospace;
  letter-spacing: .07em;
}
.section-title {
  color: var(--text);
  font-family: "Iowan Old Style", "Noto Serif SC", "Songti SC", serif;
  font-size: 25px;
  font-weight: 650;
  margin: 3px 0 4px;
}
.section-desc { color: var(--text-muted); font-size: 13px; margin-bottom: 18px; }
.note-box {
  color: var(--text-soft);
  background: var(--glass-soft);
  backdrop-filter: blur(40px) saturate(180%);
  -webkit-backdrop-filter: blur(40px) saturate(180%);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 13px 16px;
  font-size: 12px;
  margin-top: 14px;
  box-shadow: 0 8px 24px rgba(3,7,18,.45), inset 0 1px 0 rgba(255,255,255,.18), inset 0 -1px 0 rgba(2,6,16,.30);
}
.footer-note {
  border-top: 1px solid rgba(255,255,255,.11);
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.8;
  padding-top: 20px;
  margin-top: 34px;
}
.footer-note b { color: var(--text-soft); }

/* Glass navigation sits beside the title; the content area stays unboxed. */
.stTabs { position: relative; min-width: 0; }
.stTabs [data-baseweb="tab-list"] {
  position: absolute;
  z-index: 5;
  right: 0;
  top: -83px;
  width: max-content;
  max-width: min(62vw, 720px);
  gap: 6px;
  background: var(--glass-soft);
  backdrop-filter: blur(60px) saturate(180%);
  -webkit-backdrop-filter: blur(60px) saturate(180%);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 6px;
  box-shadow: var(--glass-shadow);
}
.stTabs [data-baseweb="tab"] {
  color: var(--text-muted);
  border: 1px solid transparent;
  border-radius: 16px;
  padding: 12px 20px;
  font-size: 13px;
  transition: all 500ms var(--spring);
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--text);
  background: rgba(255,255,255,.08);
  border-color: var(--line);
}
.stTabs [aria-selected="true"] {
  color: var(--gold-light) !important;
  font-weight: 700;
  background: rgba(255,255,255,.10) !important;
  border-color: rgba(228,184,99,.40) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.20), inset 0 -1px 0 rgba(2,6,16,.28);
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding: 28px 2px 8px; }

/* Inputs */
div[data-baseweb="select"] > div,
[data-testid="stDateInput"] input {
  color: var(--text);
  background: rgba(255,255,255,.06);
  backdrop-filter: blur(40px) saturate(180%);
  -webkit-backdrop-filter: blur(40px) saturate(180%);
  border: 1px solid var(--line);
  border-radius: 16px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.15), inset 0 -1px 0 rgba(2,6,16,.30);
  transition: all 500ms var(--spring);
}
div[data-baseweb="select"] > div:hover,
[data-testid="stDateInput"] input:hover {
  background: rgba(255,255,255,.10);
  border-color: var(--line-hover);
}
div[data-baseweb="select"] > div:focus-within,
[data-testid="stDateInput"] input:focus {
  outline: none;
  background: rgba(255,255,255,.10);
  border-color: rgba(255,255,255,.35);
  box-shadow: 0 0 0 3px rgba(228,184,99,.15), inset 0 1px 0 rgba(255,255,255,.25), inset 0 -1px 0 rgba(2,6,16,.30);
}
div[data-baseweb="select"] * { color: var(--text-soft); }
[data-baseweb="popover"] [role="listbox"], [data-baseweb="calendar"] {
  color: var(--text);
  background: rgba(255,255,255,.10);
  backdrop-filter: blur(60px) saturate(180%);
  -webkit-backdrop-filter: blur(60px) saturate(180%);
  border: 1px solid var(--line);
  border-radius: 20px;
  box-shadow: var(--glass-shadow);
}
[data-baseweb="popover"] [role="option"] { color: var(--text-soft); }
[data-baseweb="popover"] [role="option"]:hover { background: rgba(255,255,255,.10); color: var(--text); }
[data-testid="stWidgetLabel"] p { color: var(--text-soft); }
[data-testid="stForm"] { border: 0; padding: 0; }

/* Champagne is reserved for the primary action and key figures. */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
  min-height: 42px;
  color: var(--gold-light);
  background: rgba(228,184,99,.15);
  backdrop-filter: blur(40px) saturate(180%);
  -webkit-backdrop-filter: blur(40px) saturate(180%);
  border: 1px solid rgba(228,184,99,.40);
  border-radius: 16px;
  font-weight: 700;
  box-shadow: 0 4px 20px rgba(228,184,99,.15), inset 0 1px 0 rgba(243,220,168,.35), inset 0 -1px 0 rgba(2,6,16,.30);
  transition: all 500ms var(--spring);
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
  color: var(--gold-light);
  background: rgba(228,184,99,.22);
  border-color: rgba(228,184,99,.55);
  box-shadow: 0 10px 32px rgba(3,7,18,.55), 0 4px 24px rgba(228,184,99,.18), inset 0 1px 0 rgba(255,255,255,.35);
  transform: translateY(-2px);
}
.stButton > button:active, [data-testid="stFormSubmitButton"] > button:active { transform: scale(.97); }
.stButton > button:focus-visible, [data-testid="stFormSubmitButton"] > button:focus-visible {
  outline: none;
  box-shadow: 0 0 0 3px rgba(228,184,99,.18), 0 8px 24px rgba(3,7,18,.45);
}

/* Neutral colorless glass surfaces. */
div[data-testid="stMetric"] {
  position: relative;
  overflow: hidden;
  background: var(--glass);
  background-image: linear-gradient(to bottom, rgba(255,255,255,.12), transparent 50%);
  backdrop-filter: blur(60px) saturate(180%);
  -webkit-backdrop-filter: blur(60px) saturate(180%);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 17px 18px;
  box-shadow: var(--glass-shadow);
  transition: all 500ms var(--spring);
}
div[data-testid="stMetric"]:hover {
  background-color: var(--glass-hover);
  border-color: var(--line-hover);
  transform: translateY(-4px);
  box-shadow: 0 24px 64px rgba(3,7,18,.60), inset 0 1px 0 rgba(255,255,255,.32), inset 0 -1px 0 rgba(2,6,16,.35);
}
[data-testid="stMetricLabel"] { color: var(--text-muted); font-size: 12px; }
[data-testid="stMetricValue"] { color: var(--gold-light); font-size: 25px; }
[data-testid="stDataFrame"],
[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"],
[data-testid="stPlotlyChart"],
[data-testid="stExpander"] {
  overflow: hidden;
  background: var(--glass-soft);
  background-image: linear-gradient(to bottom, rgba(255,255,255,.10), transparent 50%);
  backdrop-filter: blur(60px) saturate(180%);
  -webkit-backdrop-filter: blur(60px) saturate(180%);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--glass-shadow);
}
[data-testid="stExpander"] summary { color: var(--text-soft); }
[data-testid="stDataFrame"] { color: var(--text-soft); }
[data-testid="stPlotlyChart"],
[data-testid="stPlotlyChart"] .js-plotly-plot,
[data-testid="stPlotlyChart"] .plot-container,
[data-testid="stPlotlyChart"] .svg-container {
  overscroll-behavior: auto;
}
hr { border-color: rgba(255,255,255,.11); }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; animation-duration: .01ms !important; }
  div[data-testid="stMetric"]:hover, .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { transform: none; }
}
@media(max-width:820px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .project-title { font-size: 28px; }
  .stTabs [data-baseweb="tab-list"] {
    position: relative;
    top: auto;
    right: auto;
    width: 100%;
    max-width: none;
    gap: 2px;
    padding: 4px;
    border-radius: 18px;
  }
  .stTabs [data-baseweb="tab"] { padding: 10px 8px; font-size: 12px; border-radius: 14px; }
  .stTabs [data-baseweb="tab-panel"] { padding: 22px 0 8px; }
  div[data-testid="stMetric"] { backdrop-filter: blur(30px) saturate(180%); -webkit-backdrop-filter: blur(30px) saturate(180%); }
}
@media(min-width:821px) and (max-width:1080px) {
  .project-title { font-size: 29px; }
  .stTabs [data-baseweb="tab-list"] { top: -79px; max-width: 66vw; }
  .stTabs [data-baseweb="tab"] { padding-left: 14px; padding-right: 14px; }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, ttl=300)
def read_result(name: str) -> pd.DataFrame:
    path = RESULT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    kwargs = {"dtype": {"code": str}} if name in {"latest_predictions.csv", "test_predictions.csv"} else {}
    return pd.read_csv(path, **kwargs)


@st.cache_data(show_spinner=False, ttl=300)
def read_prices() -> pd.DataFrame:
    path = DATA_DIR / "prices.csv"
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path, parse_dates=["date"], dtype={"code": str})
    data["code"] = data["code"].str.zfill(6)
    return data


@st.cache_data(show_spinner=False, ttl=300)
def read_factor_catalog() -> pd.DataFrame:
    path = DATA_DIR / "factor_catalog.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def pct(value: float) -> str:
    return f"{value:+.2%}" if pd.notna(value) else "—"


def section_header(number: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="section-kicker">{number}</div><div class="section-title">{title}</div>'
        f'<div class="section-desc">{description}</div>', unsafe_allow_html=True,
    )


def price_chart(history: pd.DataFrame, prediction: pd.Series, horizon: int) -> go.Figure:
    """股票交易图：K 线、成交量、非连续交易日和预测终点。"""
    history = history[["date", "open", "high", "low", "close", "volume"]].copy()
    last_date = history["date"].iloc[-1]
    last_close = history["close"].iloc[-1]
    future_date = last_date + pd.offsets.BDay(horizon)
    target_price = prediction["reference_target_price"]
    initial_start = last_date - pd.DateOffset(months=6)
    initial_end = future_date + pd.offsets.BDay(5)
    initial_history = history[history["date"] >= initial_start]
    visible_low = min(initial_history["low"].min(), last_close, target_price)
    visible_high = max(initial_history["high"].max(), last_close, target_price)
    price_padding = max((visible_high - visible_low) * 0.10, visible_high * 0.02)
    visible_volume = initial_history["volume"].max() * 1.15

    trading_dates = pd.DatetimeIndex(history["date"].dt.normalize())
    weekdays = pd.date_range(trading_dates.min(), trading_dates.max(), freq="B")
    missing_trade_dates = weekdays.difference(trading_dates)
    date_breaks = [{"bounds": ["sat", "mon"]}]
    if len(missing_trade_dates):
        date_breaks.append({"values": missing_trade_dates.strftime("%Y-%m-%d").tolist()})

    volume_colors = [
        "rgba(168,58,58,0.76)" if close >= open_price else "rgba(31,110,85,0.76)"
        for open_price, close in zip(history["open"], history["close"])
    ]

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.76, 0.24],
        vertical_spacing=0.03,
    )
    figure.add_trace(
        go.Candlestick(
            x=history["date"],
            open=history["open"],
            high=history["high"],
            low=history["low"],
            close=history["close"],
            name="历史 K 线",
            increasing={"line": {"color": "#A83A3A", "width": 1.4}, "fillcolor": "#A83A3A"},
            decreasing={"line": {"color": "#1F6E55", "width": 1.4}, "fillcolor": "#1F6E55"},
            hovertext=[
                f"日期：{r.date:%Y-%m-%d}<br>开盘：¥ {r.open:.2f}<br>最高：¥ {r.high:.2f}"
                f"<br>最低：¥ {r.low:.2f}<br>收盘：¥ {r.close:.2f}"
                for r in history.itertuples()
            ],
            hoverinfo="text",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=[last_date, future_date],
            y=[last_close, target_price],
            mode="lines+markers",
            name=f"{horizon} 日预测",
            line={"color": "#E4B863", "width": 4, "dash": "dash"},
            marker={"color": "#E4B863", "size": [6, 11]},
            hovertemplate="日期：%{x|%Y-%m-%d}<br>预测价格：¥ %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=history["date"],
            y=history["volume"],
            name="成交量",
            marker={"color": volume_colors, "line": {"width": 0}},
            hovertemplate="日期：%{x|%Y-%m-%d}<br>成交量：%{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.add_annotation(
        x=future_date,
        y=target_price,
        text=f"¥ {target_price:.2f}",
        showarrow=False,
        xanchor="left",
        xshift=9,
        yshift=12,
        font={"color": "#F3DCA8", "size": 12},
        row=1,
        col=1,
    )
    figure.update_layout(
        height=620,
        margin={"l": 60, "r": 40, "t": 18, "b": 55},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.035)",
        hovermode="closest",
        dragmode="pan",
        legend={"orientation": "h", "y": 1.03, "x": 0, "font": {"color": "rgba(255,255,255,0.72)"}},
        bargap=0.08,
        font={"color": "rgba(255,255,255,0.72)", "family": "Aptos, Segoe UI, sans-serif"},
        hoverlabel={"bgcolor": "#111A28", "bordercolor": "rgba(255,255,255,0.20)", "font": {"color": "#F3F6FA"}},
    )
    figure.update_xaxes(
        range=[initial_start, initial_end],
        rangebreaks=date_breaks,
        rangeslider_visible=False,
        showgrid=False,
        showspikes=True,
        spikecolor="rgba(255,255,255,0.38)",
        spikedash="dot",
        spikemode="across",
        spikesnap="cursor",
    )
    figure.update_xaxes(title="交易日", row=2, col=1)
    figure.update_yaxes(
        title="价格 / CNY",
        range=[visible_low - price_padding, visible_high + price_padding],
        fixedrange=True,
        gridcolor="rgba(255,255,255,0.09)",
        showspikes=True,
        spikecolor="rgba(255,255,255,0.38)",
        spikedash="dot",
        spikemode="across",
        spikesnap="cursor",
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title="成交量",
        range=[0, visible_volume],
        fixedrange=True,
        showgrid=False,
        rangemode="tozero",
        tickformat="~s",
        row=2,
        col=1,
    )
    return figure


def show_price_chart(figure: go.Figure) -> None:
    """使用原生 Plotly 容器显示交易图，让页面接管滚轮与双指手势。"""
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={
            "scrollZoom": False,
            "displaylogo": False,
            "responsive": True,
            "doubleClick": "reset+autosize",
        },
    )


st.markdown('<div class="project-title">股票收益预测课程项目</div>', unsafe_allow_html=True)

metrics = read_result("model_metrics.csv")
latest = read_result("latest_predictions.csv")
factor_summary = read_result("factor_ic_summary.csv")
factor_daily = read_result("factor_ic_daily.csv")
backtest = read_result("backtest_nav.csv")
backtest_summary = read_result("backtest_summary.csv")
importance = read_result("feature_importance.csv")
factor_catalog = read_factor_catalog()

if metrics.empty or latest.empty:
    st.error("尚未发现训练结果。请先在项目目录运行 `python main.py --refresh`。")
    st.stop()

prices = read_prices()
if not prices.empty and pd.to_datetime(latest["date"]).max() < prices["date"].max():
    st.info("行情范围已更新，当前预测与评估仍来自上次训练。完整运行 main.py 后将显示新范围的结果。")

tab1, tab2, tab3, tab4 = st.tabs(["首页预测", "模型表现", "因子表现", "策略回测"])

with tab1:
    section_header("01 / PREDICTION", "最新收益预测", "选择股票、预测周期与模型，查看收盘后预测和参考目标价。")
    with st.form("prediction_form"):
        f1, f2, f3, f4 = st.columns([1.1, 1.1, 1.5, 0.8], vertical_alignment="bottom")
        horizons = sorted(int(x) for x in latest["horizon"].unique())
        horizon = f1.selectbox("预测周期", horizons, format_func=lambda x: f"{x} 个交易日")
        labels = MODEL_LABELS
        model_names = [m for m in labels if m in set(latest["model"])]
        model = f2.selectbox("模型", model_names, format_func=lambda x: labels[x])
        subset0 = latest[(latest["horizon"] == horizon) & (latest["model"] == model)].copy()
        subset0["code"] = subset0["code"].astype(str).str.zfill(6)
        stock_labels = [f"{r.code}　{r.name}" for r in subset0.sort_values("code").itertuples()]
        stock_label = f3.selectbox("股票", stock_labels)
        f4.form_submit_button("查看预测", use_container_width=True)

    code = stock_label.split("　", 1)[0]
    subset = latest[(latest["horizon"] == horizon) & (latest["model"] == model)].copy()
    subset["code"] = subset["code"].astype(str).str.zfill(6)
    row = subset[subset["code"] == code].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最近收盘价 / CNY", f"¥ {row['close']:.2f}")
    c2.metric(f"预测 {horizon} 日收益率", pct(row["predicted_return"]))
    c3.metric("参考目标价 / CNY", f"¥ {row['reference_target_price']:.2f}")
    c4.metric("预测方向", "上涨" if row["predicted_return"] > 0 else "下跌或持平")

    st.markdown(f"#### {code} {row['name']} · 历史 K 线与预测终点")
    prices = read_prices()
    history = prices[prices["code"] == code].sort_values("date")[
        ["date", "open", "high", "low", "close", "volume"]
    ]
    if not history.empty:
        show_price_chart(price_chart(history, row, int(horizon)))
    st.markdown(
        '<div class="note-box">上方为 K 线，下方为对应交易日的成交量；横轴会跳过没有交易记录的日期。'
        '主图默认展示最近 6 个月；普通滚轮和双指手势由页面接管，浏览器可正常滚动与缩放。'
        'K 线仍可左右拖动，并可使用图表右上角工具栏缩放或复位。'
        '鼠标悬停显示 OHLC 价格，'
        '深红 K 线表示上涨、深绿 K 线表示下跌，横线与竖线用于定位。'
        '香槟金虚线连接最新收盘价与预测终点，不表示逐日预测路径。</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### 股票池最新预测排名")
    ranked = subset.sort_values("predicted_return", ascending=False).copy()
    ranked["预测收益率"] = ranked["predicted_return"].map(lambda x: f"{x:+.2%}")
    ranked["参考目标价"] = ranked["reference_target_price"].map(lambda x: f"¥ {x:.2f}")
    st.dataframe(
        ranked[["code", "name", "close", "预测收益率", "参考目标价"]].rename(
            columns={"code": "代码", "name": "股票", "close": "最新收盘价"}),
        use_container_width=True, hide_index=True, height=330,
    )

with tab2:
    section_header("02 / MODELS", "模型表现比较", f"在相同股票池和测试区间中，比较 {metrics['model'].nunique()} 个模型。")
    horizon2 = st.selectbox(
        "预测周期", sorted(int(x) for x in metrics["horizon"].unique()),
        format_func=lambda x: f"{x} 个交易日", key="metric_horizon")
    view = metrics[metrics["horizon"] == horizon2].copy().sort_values("mae")
    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown("#### 测试集 R² 与预测相关系数")
        chart_view = view.set_index("model")[["r2", "prediction_correlation"]].rename(
            columns={"r2": "R²", "prediction_correlation": "预测相关系数"})
        st.bar_chart(chart_view, color=["#E4B863", "#7C9CC4"], height=300)
    with right:
        st.markdown("#### 展示口径")
        st.markdown(
            "**R²**：相对均值基线的改善程度，允许为负。  \n"
            "**MAE**：平均绝对预测误差。  \n"
            "**方向准确率**：预测涨跌方向与真实方向一致的比例。  \n"
            "**预测相关系数**：预测值与真实收益率的线性相关性。")
        st.markdown('<div class="note-box">超参数只使用验证集选择，测试集不参与调参。</div>', unsafe_allow_html=True)

    table = view.copy()
    table["MAE"] = table["mae"].map(lambda x: f"{x:.2%}")
    table["RMSE"] = table["rmse"].map(lambda x: f"{x:.2%}")
    table["方向准确率"] = table["direction_accuracy"].map(lambda x: f"{x:.1%}")
    table["模型"] = table["model"].map(MODEL_LABELS)
    st.dataframe(
        table[["模型", "best_params", "r2", "MAE", "RMSE", "方向准确率", "prediction_correlation", "test_start", "test_end"]].rename(
            columns={"best_params": "验证集最优参数", "r2": "R²", "prediction_correlation": "预测相关系数", "test_start": "测试开始", "test_end": "测试结束"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("#### 模型因子权重 / 重要性")
    i1, i2, i3 = st.columns(3)
    chosen_model = i1.selectbox(
        "查看模型", list(view["model"]), key="importance_model",
        format_func=lambda x: MODEL_LABELS.get(x, x))
    factor_types = list(factor_catalog["factor_type"].drop_duplicates()) if not factor_catalog.empty else []
    chosen_type = i2.selectbox("因子类型", [*factor_types, "全部因子"], key="importance_type")
    top_count = i3.selectbox("显示数量", [10, 20, 30], index=1, key="importance_count")
    imp = importance[(importance["horizon"] == horizon2) & (importance["model"] == chosen_model)].copy()
    if "importance_type" in imp.columns and not imp.empty:
        st.caption(f"权重口径：{imp['importance_type'].iloc[0]}")
    if not factor_catalog.empty:
        imp = imp.merge(factor_catalog, on="factor", how="left")
    else:
        imp["factor_type"] = "未分类"
        imp["definition"] = ""
    if chosen_type != "全部因子":
        imp = imp[imp["factor_type"] == chosen_type]
    imp["abs_importance"] = imp["importance"].abs()
    imp = imp.nlargest(top_count, "abs_importance").sort_values("abs_importance")
    st.bar_chart(imp.set_index("factor")["importance"], horizontal=True, color="#E4B863", height=390)
    with st.expander("查看这些因子的类型与定义"):
        st.dataframe(
            imp.sort_values("abs_importance", ascending=False)[
                ["factor", "factor_type", "definition", "importance"]
            ].rename(columns={"factor": "因子", "factor_type": "类型", "definition": "定义", "importance": "权重/重要性"}),
            use_container_width=True, hide_index=True,
        )

with tab3:
    section_header("03 / FACTORS", "因子预测能力", "按所选区间的 |平均 IC| 排名，保留相关方向的正负号。")
    horizon3 = st.selectbox(
        "预测周期", sorted(int(x) for x in factor_summary["horizon"].unique()),
        format_func=lambda x: f"{x} 个交易日", key="factor_horizon")
    daily = factor_daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    available = daily[daily["horizon"] == horizon3]
    factor_types = list(factor_catalog["factor_type"].drop_duplicates()) if not factor_catalog.empty else []
    chosen_ic_type = st.selectbox("因子类型", [*factor_types, "全部因子"], key="factor_type")
    if chosen_ic_type != "全部因子":
        factors_in_type = factor_catalog.loc[
            factor_catalog["factor_type"] == chosen_ic_type, "factor"
        ]
        available = available[available["factor"].isin(factors_in_type)]
    min_date, max_date = available["date"].min().date(), available["date"].max().date()
    start_default = max(min_date, (pd.Timestamp(max_date) - pd.DateOffset(years=2)).date())
    date_range = st.date_input("因子形成日期范围", value=(start_default, max_date), min_value=min_date, max_value=max_date)
    if len(date_range) == 2:
        selected = available[(available["date"].dt.date >= date_range[0]) & (available["date"].dt.date <= date_range[1])]
        ranked_ic = selected.groupby("factor")["ic"].agg(mean_ic="mean", ic_std="std", n_dates="count").reset_index()
        if not factor_catalog.empty:
            ranked_ic = ranked_ic.merge(factor_catalog, on="factor", how="left")
        ranked_ic["abs_mean_ic"] = ranked_ic["mean_ic"].abs()
        ranked_ic = ranked_ic.sort_values("abs_mean_ic", ascending=False)
        left, right = st.columns([1.2, 0.8])
        with left:
            st.markdown("#### 平均 IC 排名")
            st.bar_chart(ranked_ic.set_index("factor")["mean_ic"], horizontal=True, color="#E4B863", height=390)
        with right:
            st.markdown("#### 排名明细")
            show_ic = ranked_ic.copy()
            show_ic["平均 IC"] = show_ic["mean_ic"].map(lambda x: f"{x:+.4f}")
            show_ic["IC 波动"] = show_ic["ic_std"].map(lambda x: f"{x:.4f}")
            st.dataframe(
                show_ic[["factor", "平均 IC", "IC 波动", "n_dates"]].rename(columns={"factor": "因子", "n_dates": "有效日期"}),
                use_container_width=True, hide_index=True, height=390)
        st.markdown(
            f'<div class="note-box">有效区间：{date_range[0]} 至 {date_range[1]}　·　'
            f'{selected["date"].nunique()} 个交易日。每日在股票间计算 Pearson IC，再对日期等权平均；负 IC 表示反向关系。</div>',
            unsafe_allow_html=True)
        with st.expander("查看所选类型的完整因子定义"):
            definitions = factor_catalog
            if chosen_ic_type != "全部因子":
                definitions = definitions[definitions["factor_type"] == chosen_ic_type]
            st.dataframe(
                definitions.rename(columns={"factor": "因子", "factor_type": "类型", "definition": "定义"}),
                use_container_width=True, hide_index=True,
            )

with tab4:
    section_header("04 / BACKTEST", "样本外策略回测", "用测试集历史预测构建固定规则策略，并与股票池等权基准比较。")
    b1, b2 = st.columns(2)
    h4 = b1.selectbox(
        "预测周期", sorted(int(x) for x in backtest["horizon"].unique()),
        format_func=lambda x: f"{x} 个交易日", key="bt_horizon")
    available_models = list(backtest[backtest["horizon"] == h4]["model"].unique())
    m4 = b2.selectbox(
        "模型", available_models, key="bt_model",
        format_func=lambda x: MODEL_LABELS.get(x, x))
    nav = backtest[(backtest["horizon"] == h4) & (backtest["model"] == m4)].copy()
    nav["date"] = pd.to_datetime(nav["date"])
    summary = backtest_summary[(backtest_summary["horizon"] == h4) & (backtest_summary["model"] == m4)].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("策略累计收益", pct(summary["total_return"]))
    c2.metric("等权基准收益", pct(summary["benchmark_total_return"]))
    c3.metric("年化 Sharpe", f"{summary['annualized_sharpe']:.2f}")
    c4.metric("最大回撤", pct(summary["max_drawdown"]))
    st.markdown("#### 策略与基准净值")
    nav_chart = nav.set_index("date")[["strategy_nav", "benchmark_nav"]].rename(
        columns={"strategy_nav": "策略净值", "benchmark_nav": "等权基准"})
    st.line_chart(nav_chart, color=["#E4B863", "#7C9CC4"], height=360)
    st.markdown(
        '<div class="note-box">每隔预测周期调仓，等权持有预测收益最高的 10 只股票，每次调仓扣 0.1% 成本。'
        '回测只使用测试集预测；60 日周期样本较少，结果仅作课程展示。</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note"><b>方法与风险提示</b><br>'
    '股票池为沪深300最新一期的全部300只成分股，行情从2020年开始；固定当前成分会带来生存偏差。'
    '前复权历史、模型预测、因子 IC 与回测结果仅用于课程项目，不构成投资建议。</div>',
    unsafe_allow_html=True)
